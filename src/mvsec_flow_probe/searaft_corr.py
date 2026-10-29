"""Namespaced adaptation of SEA-RAFT/core/corr.py CorrBlock.

The upstream file imports `utils.utils`, which conflicts with Event-JEPA's
`src/utils.py`. Correlation pyramid and sampling mathematics remain identical.
Original SEA-RAFT BSD-3-Clause copyright/licensing apply to adapted code.
See third_party/ScaleEvent/sea_raft/LICENSE.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def bilinear_sampler(img,coords):
    h,w=img.shape[-2:]
    x,y=coords.split([1,1],dim=-1)
    grid=torch.cat([2*x/(w-1)-1,2*y/(h-1)-1],dim=-1)
    return F.grid_sample(img,grid,align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, args):
        self.num_levels=args.corr_levels
        self.radius=args.corr_radius
        self.pyramid=[]
        b,c,h,w=fmap1.shape
        for _ in range(self.num_levels):
            corr=self.corr(fmap1,fmap2)
            _,h1,w1,_,h2,w2=corr.shape
            self.pyramid.append(corr.reshape(b*h1*w1,1,h2,w2))
            fmap2=F.interpolate(fmap2,scale_factor=.5,mode='bilinear',align_corners=False)

    def __call__(self,coords,dilation=None):
        r=self.radius
        coords=coords.permute(0,2,3,1)
        b,h,w,_=coords.shape
        if dilation is None:
            dilation=torch.ones(b,1,h,w,device=coords.device)
        dx=torch.linspace(-r,r,2*r+1,device=coords.device)
        dy=torch.linspace(-r,r,2*r+1,device=coords.device)
        delta=torch.stack(torch.meshgrid(dy,dx,indexing='ij'),dim=-1)
        delta=delta.reshape(1,2*r+1,2*r+1,2)
        offsets=delta*dilation.reshape(b*h*w,1,1,1)
        outs=[]
        for level,corr in enumerate(self.pyramid):
            centroid=coords.reshape(b*h*w,1,1,2)/(2**level)
            sample=bilinear_sampler(corr,centroid+offsets)
            outs.append(sample.reshape(b,h,w,-1))
        return torch.cat(outs,dim=-1).permute(0,3,1,2).contiguous().float()

    @staticmethod
    def corr(f1,f2):
        b,d,h,w=f1.shape
        h2,w2=f2.shape[-2:]
        f1=f1.reshape(b,1,d,h*w)
        f2=f2.reshape(b,1,d,h2*w2)
        c=f1.transpose(2,3)@f2
        return c.reshape(b,h,w,1,h2,w2)/math_sqrt(d,f1.device)


def math_sqrt(x,device):
    # python scalar: avoiding a CPU tensor in upstream CorrBlock.corr
    return x**0.5
