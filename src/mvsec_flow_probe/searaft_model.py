"""Event-JEPA feature adaptor + genuine SEA-RAFT iterative optical-flow decoder.

Uses actual ScaleEvent/sea_raft/core/{update,layer,loss} at runtime,
plus a namespaced implementation of its correlation pyramid. Replaces its RGB ResNet-FPN encoder with Stage1+Stage2 event
features projected onto the SEA-RAFT 1/8 feature grid. This is an adapter,
not an unmodified ScaleEvent architecture; results must be labelled so.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def add_searaft_core(core: str | Path):
    core=Path(core).resolve()
    for fn in ['corr.py','update.py','layer.py','loss.py','utils/utils.py']:
        if not (core/fn).is_file():
            raise FileNotFoundError(f'Missing SEA-RAFT core {core/fn}. See README sparse checkout.')
    if str(core) not in sys.path:
        sys.path.insert(0,str(core))


def coords_grid(batch,h,w,device):
    yy,xx=torch.meshgrid(torch.arange(h,device=device),
                         torch.arange(w,device=device),indexing='ij')
    return torch.stack((xx,yy),dim=0).float()[None].expand(batch,-1,-1,-1)


def training_policy(protocol):
    if protocol not in {'frozen','ft10','full','stage2-only','stage1-only'}:
        raise ValueError(protocol)
    return {'stage1':protocol in {'ft10','full','stage1-only'},
            'stage2':protocol in {'ft10','full','stage2-only'},
            'decoder':True}


class SEAEventDecoder(nn.Module):
    """Adapted SEA-RAFT CorrBlock + original BasicUpdateBlock, NLL and upsampling.

    Input f1,f2: B,2*dim,28,28; two adjacent event intervals (t-1,t],(t,t+1].
    Output flow: B,2,224,224, displaced in resized-model pixels.
    """
    def __init__(self, *, dim=128, radius=4, iters=4, num_blocks=2):
        super().__init__()
        from update import BasicUpdateBlock
        from layer import conv3x3
        self.cfg=SimpleNamespace(dim=dim, radius=radius, iters=iters,
            num_blocks=num_blocks, corr_levels=4,corr_radius=radius,
            corr_channel=4*(2*radius+1)**2,
            use_var=True,var_max=10.0,var_min=0.0)
        self.cnet_adapter=nn.Sequential(nn.Conv2d(4*dim,2*dim,3,padding=1),
                                        nn.GELU(),nn.Conv2d(2*dim,2*dim,3,padding=1))
        self.init_conv=conv3x3(2*dim,2*dim)
        self.upsample_weight=nn.Sequential(
            nn.Conv2d(dim,2*dim,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(2*dim,64*9,1))
        self.flow_head=nn.Sequential(
            nn.Conv2d(dim,2*dim,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(2*dim,6,3,padding=1))
        self.update_block=BasicUpdateBlock(self.cfg,hdim=dim,cdim=dim)

    @staticmethod
    def upsample_data(flow,info,mask):
        b,c,h,w=info.shape
        mask=torch.softmax(mask.reshape(b,1,9,8,8,h,w),dim=2)
        up_flow=F.unfold(8*flow,[3,3],padding=1).view(b,2,9,1,1,h,w)
        up_info=F.unfold(info,[3,3],padding=1).view(b,c,9,1,1,h,w)
        up_flow=torch.sum(mask*up_flow,dim=2).permute(0,1,4,2,5,3)
        up_info=torch.sum(mask*up_info,dim=2).permute(0,1,4,2,5,3)
        return up_flow.reshape(b,2,8*h,8*w),up_info.reshape(b,c,8*h,8*w)

    @staticmethod
    def _nf_loss(flow_gt,flow_pred,info,var_min=0.,var_max=10.):
        # Same mixture-of-Laplace training term as SEA-RAFT/core/raft.py
        raw_b=info[:,2:]
        log_b=torch.stack((raw_b[:,0].clamp(0,var_max),raw_b[:,1].clamp(var_min,0)),dim=1)
        weight=info[:,:2]
        term2=(flow_gt-flow_pred).abs().unsqueeze(2)*torch.exp(-log_b).unsqueeze(1)
        term1=weight-math.log(2)-log_b
        return torch.logsumexp(weight,dim=1,keepdim=True)-torch.logsumexp(term1.unsqueeze(1)-term2,dim=2)

    def forward(self,f1,f2,flow_gt=None):
        from mvsec_flow_probe.searaft_corr import CorrBlock
        if f1.shape != f2.shape or f1.ndim != 4:
            raise ValueError(f'feature pair dimensions mismatch: {f1.shape}, {f2.shape}')
        b, c, h, w=f1.shape
        if c != 2*self.cfg.dim or (h,w)!=(28,28):
            raise ValueError(f'expected B,{2*self.cfg.dim},28,28; got {tuple(f1.shape)}')
        if flow_gt is not None and flow_gt.shape != (b,2,h*8,w*8):
            raise ValueError(f'flow GT wrong size: {flow_gt.shape}')
        cnet=self.cnet_adapter(torch.cat((f1,f2),dim=1))
        cnet=self.init_conv(cnet)
        net,context=torch.split(cnet,[self.cfg.dim,self.cfg.dim],dim=1)
        flow_update=self.flow_head(net)
        weights=.25*self.upsample_weight(net)
        flow_8x=flow_update[:,:2]
        info_8x=flow_update[:,2:]
        flow_up,info_up=self.upsample_data(flow_8x,info_8x,weights)
        flows=[flow_up];infos=[info_up]
        corr_fn=CorrBlock(f1.float(),f2.float(),self.cfg)
        dilation=torch.ones(b,1,h,w,device=f1.device)
        for _ in range(self.cfg.iters):
            flow_8x=flow_8x.detach()
            coords2=(coords_grid(b,h,w,f1.device)+flow_8x).detach()
            corr=corr_fn(coords2,dilation=dilation)
            net=self.update_block(net,context,corr,flow_8x)
            flow_update=self.flow_head(net)
            weights=.25*self.upsample_weight(net)
            flow_8x=flow_8x+flow_update[:,:2]
            info_8x=flow_update[:,2:]
            flow_up,info_up=self.upsample_data(flow_8x,info_8x,weights)
            flows.append(flow_up);infos.append(info_up)
        nfs=[]
        if flow_gt is not None:
            for p,info in zip(flows,infos):
                # FP32 prevents extreme BF16 loss/NLL overflows
                nfs.append(self._nf_loss(flow_gt.float(),p.float(),info.float(),
                                         self.cfg.var_min,self.cfg.var_max))
        return {'flow':flows,'nf':nfs,'final':flows[-1]}


class EventJEPASEARAFT(nn.Module):
    def __init__(self,stage1,stage2,protocol, *, dim=128,
                 checkpoint_encoder=True, iters=4):
        super().__init__()
        self.stage1=stage1
        self.stage2=stage2
        self.protocol=protocol
        self.policy=training_policy(protocol)
        self.stage1.requires_grad_(self.policy['stage1'])
        self.stage2.requires_grad_(self.policy['stage2'])
        self.checkpoint_encoder=checkpoint_encoder
        # Shared feature extractor for both timestamps
        self.fmap_adapter=nn.Sequential(nn.Conv2d(384,2*dim,1),
                    nn.GELU(),nn.Conv2d(2*dim,2*dim,3,padding=1))
        self.decoder=SEAEventDecoder(dim=dim,iters=iters)

    def train(self,mode=True):
        super().train(mode)
        if not self.policy['stage1']:
            self.stage1.eval()
        if not self.policy['stage2']:
            self.stage2.eval()
        return self

    def stage1_tokens(self,events):
        b,t,c,h,w=events.shape
        if (t,c,h,w)!=(4,3,224,224):
            raise ValueError(f'bad event tensor {events.shape}')
        tokens=[]
        # GEP-only dense flow needs two event intervals; skip two unused
        # Stage-1 passes rather than spending VRAM on discarded gradients.
        frame_ids=(2,3) if self.protocol=='stage1-only' else range(4)
        for i in frame_ids:
            x=events[:,i]
            def encode(inp):
                return self.stage1.forward_features(inp)['x_norm_patchtokens']
            if self.policy['stage1']:
                if self.checkpoint_encoder and self.training:
                    z=checkpoint(encode,x,use_reentrant=False)
                else:
                    z=encode(x)
            else:
                # Avoid computing or tracking frozen Stage1 activations
                with torch.no_grad():
                    with torch.autocast(device_type='cuda',enabled=False):
                        z=encode(x.float())
            if z.shape[1:]!=(256,384):
                raise RuntimeError(f'bad Stage-1 token geometry {z.shape}')
            tokens.append(z)
        return torch.stack(tokens,dim=1)

    def forward(self,events,flow_gt=None):
        tokens=self.stage1_tokens(events)
        b=events.shape[0]
        if self.protocol == 'stage1-only':
            # E: GEP Stage-1-only + full fine-tune. Stage-2 is BYPASSED,
            # not merely frozen: no JEPA features are used.
            # Observed intervals: the last two tokens end at t_i and t_(i+1).
            # Same feature adapter / SEA-RAFT decoder structure as A/B/C/D.
            pair=tokens[:,-2:]
        else:
            if self.policy['stage2']:
                if self.checkpoint_encoder and self.training:
                    z=checkpoint(self.stage2,tokens,use_reentrant=False)
                else:
                    z=self.stage2(tokens)
            else:
                with torch.no_grad():
                    z=self.stage2(tokens)
            if z.shape!=(b,4*256,384):
                raise RuntimeError(f'bad Stage2 output {z.shape}')
            pair=z.reshape(b,4,256,384)[:,-2:]
        pair=pair.reshape(b,2,16,16,384)
        pair=pair.permute(0,1,4,2,3).reshape(b*2,384,16,16)
        fmap=self.fmap_adapter(pair)
        fmap=F.interpolate(fmap,size=(28,28),mode='bilinear',align_corners=False)
        fmap=fmap.reshape(b,2,-1,28,28)
        return self.decoder(fmap[:,0],fmap[:,1],flow_gt=flow_gt)
