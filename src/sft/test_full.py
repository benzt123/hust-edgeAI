"""CPU 检查：不下载模型，不需要服务器。"""
import unittest
import tempfile
import random
from pathlib import Path
from types import SimpleNamespace
import torch
from core import response_loss,backward_microbatch
from full_data import build_windows,make_split
from full_scoring import score,number


class FullTests(unittest.TestCase):
    def test_windows_cover_every_response_once(self):
        for prefix in [1,4,12,25]:
            ids=list(range(prefix+23))
            windows=build_windows(ids,prefix,10,'window',3)
            targets=[token for tokens,mask in windows for token,flag in zip(tokens,mask) if flag]
            self.assertEqual(targets,ids[prefix:])
            self.assertTrue(all(len(tokens)<=10 and mask[0]==0 for tokens,mask in windows))
        with self.assertRaises(ValueError):build_windows(list(range(30)),3,10,'error',3)

    def test_weighted_chunks_match_original_gradient(self):
        # 独立 logits 排除上下文改变，验证分段权重本身等价。
        torch.manual_seed(0)
        logits=torch.randn(1,7,5,requires_grad=True)
        labels=torch.tensor([[0,1,2,3,4,0,1]])
        response_loss(logits,labels,torch.ones_like(labels)).backward()
        expected=logits.grad.clone()
        copy=logits.detach().clone().requires_grad_(True)
        for a,b in [(0,2),(2,7)]:
            loss=response_loss(copy[:,a:b],labels[:,a:b],torch.ones_like(labels[:,a:b]))
            backward_microbatch(loss*((b-a)/7),1)
        torch.testing.assert_close(copy.grad,expected)

    def test_split_no_duplicate_question_leak(self):
        rows=[{'question':str(i//2),'id':i} for i in range(20)]
        train,val=make_split(rows,0.2,42)
        self.assertFalse({r['question'] for r in train}&{r['question'] for r in val})
        self.assertEqual(len(train)+len(val),20)

    def test_scoring_boundaries(self):
        self.assertEqual(number('2/4'),number('0.5'))
        self.assertEqual(number(r'\boxed{\frac{1}{2}}'),number('0.5'))
        self.assertNotEqual(number('5%'),number('5'))
        self.assertEqual(number('5%'),number('0.05'))
        self.assertIsNone(number('The answer might be 5 or 7'))
        self.assertIsNone(number('1,2'))
        self.assertIsNone(number('1/0'))
        self.assertTrue(score('work</think>\n<answer>5</answer>','5')['joint_correct'])
        self.assertFalse(score('</answer>work</think> <answer>5','5')['joint_correct'])
        self.assertFalse(score('work</think> <answer>5</answer>junk','5')['format_ok'])
        self.assertFalse(score('work</think> <answer>5</answer><answer>5</answer>','5')['format_ok'])
        self.assertTrue(score('5','5')['answer_correct'])
        self.assertFalse(score('5','5')['format_ok'])

    def test_validation_restores_training_and_random_state(self):
        from full_validation import evaluate
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=torch.nn.Linear(4,4)
                self.config=SimpleNamespace(max_position_embeddings=100)
            def forward(self,input_ids):
                return SimpleNamespace(logits=self.layer(torch.nn.functional.one_hot(input_ids,4).float()))
            def generate(self,inputs,**kwargs):
                torch.rand(3)
                return torch.cat([inputs,torch.tensor([[2,3]])],dim=1)
        class Tokenizer:
            eos_token_id=3
            def encode(self,text,**kwargs):return [0,1]
            def decode(self,tokens,**kwargs):return 'work</think> <answer>5</answer>'
        model=Toy().train()
        item=dict(row={'id':'val/1','answer':'5'},prompt='question',
                  windows=[([0,1,2,3],[0,0,1,1])],response_tokens=2)
        config=dict(seed=42,generation_limit=0,generation_max_new_tokens=10,
                    generation_temperature=1.,generation_top_p=1.)
        before=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            result=evaluate(model,Tokenizer(),[item],config,Path(temp)/'eval')
            self.assertEqual(result['answer_correct_rate'],1.)
            self.assertEqual(result['format_ok_rate'],1.)
            self.assertEqual(result['loss_samples'],1)
            self.assertTrue((Path(temp)/'eval'/'predictions.jsonl').exists())
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main(verbosity=2)
