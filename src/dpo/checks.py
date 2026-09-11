"""填写 TODO 后运行：python checks.py。只需 PyTorch，不加载模型。"""
import math
import unittest

import torch

from core import build_preference_pairs, dpo_loss, sequence_log_probs


class CoreChecks(unittest.TestCase):
    def test_loss_and_gradient_direction(self):
        w = torch.tensor([-12.0], requires_grad=True)
        l = torch.tensor([-10.0], requires_grad=True)
        rw = w.detach().clone().requires_grad_()
        rl = l.detach().clone().requires_grad_()
        loss = dpo_loss(w, l, rw, rl)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(w.grad.item(), 0)  # 梯度下降会增加 chosen 得分
        self.assertGreater(l.grad.item(), 0)
        self.assertIsNone(rw.grad)
        self.assertIsNone(rl.grad)

    def test_improved_margin_and_numerical_stability(self):
        t = lambda x: torch.tensor([float(x)])
        self.assertAlmostEqual(dpo_loss(t(-11), t(-12), t(-12), t(-10)).item(),
                               0.55435524, places=6)
        self.assertTrue(torch.isfinite(dpo_loss(t(-100000), t(0), t(0), t(0))))

    def test_sequence_sum_shift_and_mask(self):
        logits = torch.zeros(2, 5, 4, requires_grad=True)
        ids = torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 0, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]])
        scores = sequence_log_probs(logits, ids, mask)
        torch.testing.assert_close(scores, torch.tensor([-2 * math.log(4), -math.log(4)]))
        scores.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 3:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            sequence_log_probs(logits, ids, torch.zeros_like(mask))

    def test_pairs(self):
        def row(q, text, correct, parsed=True):
            return dict(question_id=q, prompt=q, response=text,
                        answer_correct=correct, parsed=parsed,
                        format_ok=True, truncated=False)
        rows = [row('q1', 'good', True), row('q1', 'bad', False),
                row('q1', 'bad', False), row('q2', 'only good', True),
                row('q3', 'good', True), row('q3', 'unparsed', False, False)]
        before = [dict(r) for r in rows]
        self.assertEqual(build_preference_pairs(rows), [dict(
            question_id='q1', prompt='q1', chosen='good', rejected='bad')])
        self.assertEqual(rows, before)
        self.assertEqual(build_preference_pairs([]), [])
        with self.assertRaises(ValueError):
            build_preference_pairs([row('q1', 'same', True), row('q1', 'same', False)])


if __name__ == '__main__':
    unittest.main()
