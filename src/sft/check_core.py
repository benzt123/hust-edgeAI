"""本地 CPU 测试；不下载模型、不使用 vLLM。python check_core.py"""
import unittest
import torch
from core import shift_batch, response_loss, backward_microbatch, optimizer_update


class CoreTests(unittest.TestCase):
    def test_1_shift(self):
        ids = torch.tensor([[1, 2, 3, 4, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 0]])
        x, y, m = shift_batch(ids, mask)
        self.assertEqual(x.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(y.tolist(), [[2, 3, 4, 0]])
        self.assertEqual(m.tolist(), [[0, 1, 1, 0]])

    def test_2_mask_and_loss(self):
        logits = torch.zeros(2, 3, 4, requires_grad=True)
        labels = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([[0, 1, 0], [0, 1, 1]])
        loss = response_loss(logits, labels, mask)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(4.)).item(), places=6)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[mask == 0] == 0))
        self.assertAlmostEqual(logits.grad[0, 1, 1].item(), -0.375, places=6)
        self.assertAlmostEqual(logits.grad[1, 1, 2].item(), -0.1875, places=6)
        with self.assertRaises(ValueError):
            response_loss(logits.detach(), labels, torch.zeros_like(mask))

    def test_3_accumulation(self):
        # 不同大小累积组，检查参数在整组结束前不更新，最终梯度正确。
        for n in [1, 3]:
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1.)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            for i in range(n):
                backward_microbatch(model(torch.tensor([[float(i + 1)]])).sum(), n)
            self.assertAlmostEqual(model.weight.item(), 1.)
            self.assertAlmostEqual(model.weight.grad.item(), (n + 1) / 2, places=6)
            optimizer_update(model, opt, max_grad_norm=100.)
            self.assertAlmostEqual(model.weight.item(), 1 - 0.1 * (n + 1) / 2, places=6)
            self.assertTrue(model.weight.grad is None or torch.all(model.weight.grad == 0))

    def test_4_clipping(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        backward_microbatch(model(torch.tensor([[100.]])).sum(), 1)
        optimizer_update(model, opt, max_grad_norm=1.)
        self.assertAlmostEqual(model.weight.item(), 0.9, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
