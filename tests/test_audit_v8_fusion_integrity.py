from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from torch import nn

TARGET = Path(__file__).parents[1] / 'scripts' / 'audit_v8_fusion_integrity.py'
spec=importlib.util.spec_from_file_location('audit_module', TARGET)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_exact_raft_match_and_missing_detection():
    raw={'fnet.weight':torch.tensor([1.,2.]), 'cnet.num_batches_tracked':torch.tensor(7)}
    state={'raft.'+k:v.clone() for k,v in raw.items()}
    assert audit.compare_base_tensors(raw,state)['exact_match'] is True
    bad={**state, 'raft.fnet.weight': torch.tensor([1.,3.])}
    report=audit.compare_base_tensors(raw,bad)
    assert not report['exact_match'] and 'fnet.weight' in report['changed']
    del bad['raft.fnet.weight']
    assert audit.compare_base_tensors(raw,bad)['missing'] == ['fnet.weight']


class ToyRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet=nn.Conv2d(3,256,kernel_size=3,stride=2,padding=1)
        self.cnet=nn.Conv2d(6,256,kernel_size=3,stride=2,padding=1)

    def forward(self, image1, image2, flow_gt=None, test_mode=True):
        c=self.cnet(torch.cat((image1,image2),dim=1))
        a=self.fnet(image1)
        b=self.fnet(image2)
        feat=(a-b+c)
        return {'final': torch.nn.functional.interpolate(feat[:,:2],size=image1.shape[-2:],mode='bilinear',align_corners=False)}


class ToyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.raft=ToyRAFT()
        self.gate_fnet=nn.Parameter(torch.tensor(0.2))
        self.gate_cnet=nn.Parameter(torch.tensor(0.3))
        self.f_adapter=nn.Conv2d(3,256,1)
        self.c_adapter=nn.Conv2d(6,256,1)

    def forward(self, pair, context, test_mode=True):
        residual_f=self.f_adapter(pair[:,0].float())
        residual_c=self.c_adapter(torch.cat((pair[:,0],pair[:,1]),dim=1).float())
        count=[0]
        def fh(_, inputs, out):
            count[0]+=1
            return out + self.gate_fnet*torch.nn.functional.interpolate(residual_f,size=out.shape[-2:])
        def ch(_,inputs,out):
            return out + self.gate_cnet*torch.nn.functional.interpolate(residual_c,size=out.shape[-2:])
        h1=self.raft.fnet.register_forward_hook(fh)
        h2=self.raft.cnet.register_forward_hook(ch)
        try:
            x=self.raft(pair[:,0].float(),pair[:,1].float(),test_mode=test_mode)
        finally:
            h1.remove();h2.remove()
        assert count[0]==2
        return x


def test_three_pass_gate_parity_and_restoration():
    torch.manual_seed(0)
    model=ToyFusion().eval()
    pair=torch.randn(1,2,3,16,16)
    gt=torch.zeros(1,2,16,16)
    raw,zero,actual=audit.audited_predictions(model,pair,pair,gt)
    assert torch.equal(raw,zero), float((raw-zero).abs().max())
    assert float((raw-actual).abs().max())>0
    assert model.gate_fnet.item()>0.19 and model.gate_cnet.item()>0.29


def test_flow_stats_zero_epe_on_same_mask():
    pred=torch.zeros(1,2,4,4)
    gt=torch.ones(1,2,4,4)
    mask=torch.ones(1,4,4,dtype=torch.bool)
    es, _, _, _, n=audit.flow_statistics(pred,gt,mask)
    assert abs(es/n - 2**0.5)<1e-5
