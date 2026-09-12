"""测试编排、token/奖励对应，以及微型模型的实际更新。"""
import sys
import math
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from pipeline import prepare_round,train_records,self_play_round


def grade(response,answer):
    return dict(answer_correct=response==answer,format_ok=True,parsed=True)


class PipelineChecks(unittest.TestCase):
    def test_order_and_update(self):
        seen=[]
        def solve(requests,g):
            seen.extend(requests)
            q=requests[0]['question_id']
            return [dict(question_id=q,candidate_index=i,response=str(5+i),truncated=False,
                         ids=[0,1+i],mask=[0,1],old=[-math.log(4)]) for i in [1,0]]
        result=prepare_round([dict(text='Problem: What is 2+3?\nAnswer: 5',truncated=False)],
                             solve,grade,str.isdigit,'User: {question}',2)
        self.assertEqual(seen[0]['prompt'],'User: What is 2+3?')
        self.assertEqual(set(seen[0]),{'question_id','prompt'})
        records=result['records']
        self.assertEqual([x['advantage'] for x in records],[.5,-.5])
        self.assertEqual([x['ids'] for x in records],[[0,1],[0,2]])
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weights=torch.nn.Parameter(torch.zeros(4,4))
            def forward(self,input_ids,attention_mask,use_cache=False):
                return SimpleNamespace(logits=self.weights[input_ids])
        model=Tiny();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        updates=train_records(model,optimizer,records,0,epochs=1,effective_batch=2,micro_batch=1)
        self.assertEqual(len(updates),1)
        self.assertGreater(model.weights[0,1].item(),0)
        self.assertLess(model.weights[0,2].item(),0)

    def test_empty_round_does_not_solve_or_train(self):
        def forbidden(*args):raise AssertionError('空数据不应继续')
        result=self_play_round(lambda prompts:[dict(text='invalid',truncated=False)],
            forbidden,grade,str.isdigit,'{question}',forbidden,n_problems=1,group_size=2)
        self.assertEqual(result['records'],[])
        self.assertEqual(result['updates'],[])


if __name__=='__main__':unittest.main()
