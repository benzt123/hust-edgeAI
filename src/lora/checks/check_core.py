"""填写TODO后执行python checks/check_core.py -v。只需CPU和PyTorch。"""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import LoRALinear,inject_lora,trainable_parameters,setup_lora


class CoreChecks(unittest.TestCase):
    def test_initial_equivalence_and_metadata(self):
        for bias in [True,False]:
            base=nn.Linear(5,3,bias=bias).double()
            x=torch.randn(2,4,5,dtype=torch.float64)
            expected=base(x).detach().clone()
            layer=LoRALinear(base,rank=2,alpha=4)
            self.assertIs(layer.base,base)
            self.assertEqual(layer.A.shape,(3,2));self.assertEqual(layer.B.shape,(2,5))
            self.assertEqual(layer.A.dtype,base.weight.dtype)
            self.assertEqual(layer.A.device,base.weight.device)
            self.assertTrue(layer.A.requires_grad and layer.B.requires_grad)
            self.assertFalse(any(p.requires_grad for p in base.parameters()))
            torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)

    def test_nonzero_branch_matches_full_matrix(self):
        layer=LoRALinear(nn.Linear(4,3),rank=2,alpha=6)
        with torch.no_grad():layer.A.fill_(.2);layer.B.fill_(.3)
        x=torch.randn(5,4)
        expected=nn.functional.linear(x,layer.base.weight+3*(layer.A@layer.B),layer.base.bias)
        torch.testing.assert_close(layer(x),expected)

    def test_gradients_update_and_input_path(self):
        layer=LoRALinear(nn.Linear(4,3),rank=2,alpha=4)
        with torch.no_grad():layer.B.fill_(.25)
        original=layer.base.weight.detach().clone()
        x=torch.ones(2,4,requires_grad=True)
        layer(x).sum().backward()
        self.assertGreater(layer.A.grad.abs().sum().item(),0)
        self.assertEqual(layer.B.grad.abs().sum().item(),0)
        self.assertIsNone(layer.base.weight.grad)
        torch.testing.assert_close(x.grad,original.sum(0).expand_as(x))
        opt=torch.optim.SGD(trainable_parameters(layer),lr=.01)
        opt.step();opt.zero_grad(set_to_none=True)
        torch.testing.assert_close(layer.base.weight,original,rtol=0,atol=0)
        layer(x).sum().backward()
        self.assertGreater(layer.B.grad.abs().sum().item(),0)

    def test_injection_scope_and_repeat(self):
        model=nn.ModuleDict({'ffn':nn.ModuleDict({n:nn.Linear(4,4) for n in
                              ['gate_proj','up_proj','down_proj']}),'q_proj':nn.Linear(4,4)})
        stats=setup_lora(model,rank=2,alpha=4)
        self.assertEqual(set(stats['replaced']),{'ffn.gate_proj','ffn.up_proj','ffn.down_proj'})
        self.assertEqual(stats['trainable'],3*2*(4+4))
        self.assertIsInstance(model['q_proj'],nn.Linear)
        self.assertTrue(all(name.endswith(('.A','.B')) for name,p in model.named_parameters() if p.requires_grad))
        params=list(trainable_parameters(model))
        self.assertEqual(inject_lora(model,rank=2),[])
        self.assertEqual([id(p) for p in params],[id(p) for p in trainable_parameters(model)])
        with self.assertRaises(ValueError):setup_lora(model,rank=2)

    def test_invalid_arguments(self):
        for rank in [0,-1,True,1.5,5]:
            with self.assertRaises(ValueError):LoRALinear(nn.Linear(4,3),rank=rank)
        for alpha in [0,-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):LoRALinear(nn.Linear(4,3),rank=2,alpha=alpha)
        with self.assertRaises(TypeError):LoRALinear(nn.ReLU(),rank=2)
        with self.assertRaises(ValueError):trainable_parameters(nn.Linear(2,2).requires_grad_(False))


if __name__=='__main__':unittest.main()
