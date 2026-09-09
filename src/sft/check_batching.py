"""检查右侧padding、梯度等价和不足一个完整batch的加权。"""
import unittest
from types import SimpleNamespace
import torch
from core import response_loss
from batching import forward_batch


class BatchTests(unittest.TestCase):
    def test_padding_loss_and_gradient(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(10,6)
                self.head=torch.nn.Linear(6,10)
            def forward(self,input_ids,attention_mask):
                self.last_attention=attention_mask
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
        items=[{'windows':[([1,2,3,4],[0,0,1,1])]},
               {'windows':[([2,3,5],[0,1,1])]},
               {'windows':[([1,3,4,5,6],[0,0,0,1,1])]}]
        torch.manual_seed(1)
        model=Toy()
        for item in items:
            logits,labels,mask=forward_batch(model,[item],0,'cpu')
            (response_loss(logits,labels,mask)/3).backward()
        expected=[p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        for chunk in [items[:2],items[2:]]:
            logits,labels,mask=forward_batch(model,chunk,0,'cpu')
            (response_loss(logits,labels,mask)*len(chunk)/3).backward()
        for p,g in zip(model.parameters(),expected): torch.testing.assert_close(p.grad,g)
        _,_,mask=forward_batch(model,items,0,'cpu')
        self.assertEqual(mask.tolist(),[[0,1,1,0],[1,1,0,0],[0,0,1,1]])
        self.assertEqual(model.last_attention.tolist(),[[1,1,1,1],[1,1,1,0],[1,1,1,1]])


if __name__=='__main__': unittest.main()
