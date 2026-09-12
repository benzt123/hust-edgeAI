import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from backend import extract_completion,TransformersBackend
from state import save,restore


class RuntimeChecks(unittest.TestCase):
    def test_stop_excludes_padding_and_keeps_real_eos(self):
        decode=lambda ids:repr(ids)
        r=extract_completion([7,8],[2,9,0,0],2,decode)
        self.assertEqual(r['ids'],[7,8,2,9])
        self.assertEqual(r['mask'],[0,0,1,1])
        self.assertFalse(r['truncated'])
        r=extract_completion([7],[1,2,3],0,decode)
        self.assertEqual(r['ids'],[7,1,2,3])
        self.assertTrue(r['truncated'])

    def test_solver_caches_same_token_scores(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.zeros(4,4))
            def forward(self,input_ids,attention_mask,use_cache=False):
                return SimpleNamespace(logits=self.w[input_ids])
        model=Tiny();model.train()
        backend=TransformersBackend(model,None,{'solve_max_new_tokens':16})
        def sample(prompts,*args):
            self.assertEqual(prompts,['question','question'])
            return [dict(text=str(i),truncated=False,ids=[0,i],mask=[0,1]) for i in [1,2]]
        backend.sample=sample
        records=backend.solve([dict(question_id='q',prompt='question')],2)
        self.assertEqual([r['candidate_index'] for r in records],[0,1])
        self.assertEqual(records[1]['ids'],[0,2])
        self.assertAlmostEqual(records[0]['old'][0],-1.38629436,places=6)
        self.assertTrue(model.training)
        self.assertIsNone(model.w.grad)

    def test_checkpoint_resume_matches_next_update(self):
        torch.manual_seed(42);random.seed(42)
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        def step(m,o):
            o.zero_grad();x=torch.randn(3,2)*random.random()
            loss=m(x).square().mean();loss.backward();o.step()
        step(model,optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            save(path,model,optimizer,'identity',dict(round=1,seen=['q']))
            step(model,optimizer)
            expected=copy.deepcopy(model.state_dict())
            resumed=torch.nn.Linear(2,1)
            opt=torch.optim.AdamW(resumed.parameters(),lr=.9)
            metadata=restore(path,resumed,opt,'identity')
            self.assertEqual(metadata,dict(round=1,seen=['q']))
            step(resumed,opt)
            for k,v in expected.items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
            with self.assertRaises(ValueError):restore(path,resumed,opt,'other')


if __name__=='__main__':unittest.main()
