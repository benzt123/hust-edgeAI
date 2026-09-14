"""CPU检查：实际梯度更新、adapter往返、合并、数据隔离。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from adapters import core,save_adapter,load_adapter,merge_lora,adapter_payload
from data import read_rows,split_rows,encode_rows
from run import response_loss


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(8,4)
        self.up_proj=nn.Linear(4,4)
        self.head=nn.Linear(4,8)

    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=self.head(self.up_proj(self.embedding(input_ids)).tanh()))


class RuntimeChecks(unittest.TestCase):
    def test_update_reload_and_merge(self):
        torch.manual_seed(42)
        model=TinyLM();original=copy.deepcopy(model)
        core.setup_lora(model,rank=2,alpha=4)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW(core.trainable_parameters(model),lr=.01)
        items=[dict(ids=[1,2,3,4],mask=[0,0,1,1]),dict(ids=[2,3,1],mask=[0,1,1])]
        loss=response_loss(model,items,0,torch.device('cpu'))
        self.assertTrue(torch.isfinite(loss))
        loss.backward();optimizer.step()
        self.assertGreater(model.up_proj.A.abs().sum().item(),0)
        for name,expected in frozen.items():
            torch.testing.assert_close(dict(model.named_parameters())[name],expected,rtol=0,atol=0)
        x=torch.tensor([[1,2,3]])
        expected=model(x).logits.detach()
        core.setup_lora(original,rank=2,alpha=4)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.pt'
            save_adapter(model,path,'test-base')
            payload=torch.load(path,weights_only=True)
            load_adapter(original,payload,'test-base')
        torch.testing.assert_close(original(x).logits,expected,rtol=0,atol=0)
        merge_lora(original)
        self.assertFalse(any(isinstance(m,core.LoRALinear) for m in original.modules()))
        torch.testing.assert_close(original(x).logits,expected)

    def test_reject_mismatched_and_invalid_adapter_before_copy(self):
        model=TinyLM();core.setup_lora(model,rank=2)
        payload=adapter_payload(model,'base')
        with self.assertRaises(ValueError):load_adapter(model,payload,'other-base')
        bad=copy.deepcopy(payload);bad['weights']['up_proj.A'].fill_(1)
        bad['weights']['up_proj.B'].fill_(float('nan'))
        with self.assertRaises(ValueError):load_adapter(model,bad,'base')
        torch.testing.assert_close(model.up_proj.A,payload['weights']['up_proj.A'])

    def test_split_and_duplicate_rejection(self):
        rows=[dict(id=str(i),question=f'physics {i}',answer='solution') for i in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            splits=split_rows(read_rows(path))
            self.assertEqual(splits,split_rows(list(reversed(rows))))
            sets=[{r['id'] for r in group} for group in splits.values()]
            self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
            rows.append(dict(id='extra',question=' physics   0 ',answer='different'))
            path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            with self.assertRaises(ValueError):read_rows(path)

    def test_answer_mask_eos_and_length(self):
        tokenizer=SimpleNamespace(eos_token_id=0,encode=lambda text,**kwargs:[1]*len(text))
        rows=[dict(id='p',question='question',answer='answer')]
        item=encode_rows(rows,tokenizer,512)[0]
        self.assertEqual(item['ids'][-1],0)
        self.assertEqual(sum(item['mask']),len('answer')+1)
        self.assertEqual(item['mask'][:len(item['prompt'])],[0]*len(item['prompt']))
        with self.assertRaises(ValueError):encode_rows(rows,tokenizer,10)


if __name__=='__main__':unittest.main()
