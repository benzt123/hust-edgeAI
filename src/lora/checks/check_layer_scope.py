"""验证层范围、零增量等价及真实更新边界。"""
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'math_layers_runtime'))
from adapters import setup_scoped,core

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.ModuleDict({'mlp':nn.ModuleDict({n:nn.Linear(4,4) for n in sorted(core.TARGET_NAMES)})}) for _ in range(28)])
    def forward(self,x):
        for block in self.model.layers:
            for layer in block['mlp'].values():x=x+0.05*layer(x)
        return x

for keep in [28,14,7]:
    torch.manual_seed(42);m=Toy();x=torch.randn(2,4);before=m(x).detach()
    stats=setup_scoped(m,dict(rank=2,alpha=4,train_last_layers=keep))
    assert torch.equal(before,m(x).detach())
    assert len(stats['active_adapter_paths'])==keep*3
    frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    m(x).square().mean().backward();opt.step()
    assert all(torch.equal(p,frozen[n]) and p.grad is None for n,p in m.named_parameters() if n in frozen)
    assert any(torch.count_nonzero(p) for n,p in m.named_parameters() if p.requires_grad and n.endswith('.A'))
    print('PASS',keep,stats['trainable'])
