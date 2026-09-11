"""CPU 学习检查：python checks/check_core.py -v。TODO 未填时预期报错。"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import group_advantages, response_token_log_probs, probability_ratio, grpo_loss


class CoreChecks(unittest.TestCase):
    def test_advantages(self):
        rewards = torch.tensor([[1.,1.,0.,0.], [1.,1.,1.,1.], [0.,0.,0.,0.]], requires_grad=True)
        actual = group_advantages(rewards)
        torch.testing.assert_close(actual, torch.tensor([[.5,.5,-.5,-.5],[0.,0.,0.,0.],[0.,0.,0.,0.]]))
        self.assertFalse(actual.requires_grad)
        for bad in [torch.zeros(4), torch.zeros(1,1), torch.zeros(0,4)]:
            with self.assertRaises(ValueError):
                group_advantages(bad)

    def test_tokens_shift_and_gather(self):
        logits = torch.tensor([[[1.,2.,0.],[3.,-1.,2.],[0.,0.,0.],[1.,1.,1.]]], requires_grad=True)
        ids = torch.tensor([[0,2,1,0]])
        mask = torch.tensor([[0,0,1,0]])
        values, shifted = response_token_log_probs(logits,ids,mask)
        expected = torch.stack([logits[0,i,j]-torch.logsumexp(logits[0,i],0)
                                for i,j in enumerate([2,1,0])])[None]
        torch.testing.assert_close(values,expected)
        torch.testing.assert_close(shifted,torch.tensor([[0,1,0]]))
        (values*shifted).sum().backward()
        self.assertEqual(logits.grad[0,0].abs().sum().item(),0)
        self.assertGreater(logits.grad[0,1].abs().sum().item(),0)
        with self.assertRaises(ValueError):
            response_token_log_probs(logits,ids,torch.zeros_like(mask))

    def test_ratio(self):
        new = torch.tensor([[.3,.1]]).log().requires_grad_()
        old = torch.tensor([[.2,.2]]).log().requires_grad_()
        ratio = probability_ratio(new,old)
        torch.testing.assert_close(ratio,torch.tensor([[1.5,.5]]))
        ratio.sum().backward()
        self.assertIsNotNone(new.grad)
        self.assertIsNone(old.grad)

    def test_clip_four_cases(self):
        # positive advantage: upper clip; negative advantage: lower clip.
        new = torch.tensor([[1.5],[.5],[.5],[1.5]]).log().requires_grad_()
        old = torch.zeros_like(new,requires_grad=True)
        advantage = torch.tensor([1.,1.,-1.,-1.],requires_grad=True)
        loss = grpo_loss(new,old,advantage,torch.ones_like(new))
        self.assertAlmostEqual(loss.item(),.15,places=6)
        loss.backward()
        self.assertEqual(new.grad[0].item(),0)
        self.assertLess(new.grad[1].item(),0)
        self.assertEqual(new.grad[2].item(),0)
        self.assertGreater(new.grad[3].item(),0)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)

    def test_sum_not_length_mean_and_mask(self):
        new = torch.tensor([[0.,0.,1000.],[0.,1000.,1000.]],requires_grad=True)
        mask = torch.tensor([[1,1,0],[1,0,0]])
        loss = grpo_loss(new,torch.zeros_like(new),torch.ones(2),mask)
        self.assertAlmostEqual(loss.item(),-1.5,places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertEqual(new.grad[mask==0].abs().sum().item(),0)

    def test_zero_loss_can_have_gradient(self):
        new = torch.zeros(2,1,requires_grad=True)
        loss = grpo_loss(new,torch.zeros_like(new),torch.tensor([.5,-.5]),torch.ones_like(new))
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertLess(new.grad[0].item(),0)
        self.assertGreater(new.grad[1].item(),0)

    def test_reject_empty_response(self):
        with self.assertRaises(ValueError):
            grpo_loss(torch.zeros(1,2),torch.zeros(1,2),torch.ones(1),torch.zeros(1,2))


if __name__ == '__main__':
    unittest.main()
