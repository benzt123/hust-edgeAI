"""独立评估训练后的最终adapter；不改变原选优结果和训练身份。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from adapters import core,load_adapter
from data import read_rows,split_rows,encode_rows

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    c=json.loads(parser.parse_args().config.read_text(encoding='utf-8-sig'))
    out=Path(c['output']);base=Path(c['base_model'])
    assert (out/'training_complete.json').exists()
    destination=out/'test_final'
    if (destination/'metrics.json').exists():return
    if destination.exists():destination.rename(out/('test_final_interrupted_'+str(time.time_ns())))
    files=sorted(base.glob('*.safetensors'))+sorted(base.glob('*.json'))+sorted(base.glob('*.jinja'))
    identity=hashlib.sha256(json.dumps({p.name:digest(p) for p in files},sort_keys=True).encode()).hexdigest()
    assert identity==json.loads((out/'identity.json').read_text())['base']
    sys.path.insert(0,c['sft_code'])
    import full_validation
    sys.path.insert(0,c['grader_code'])
    import numeric_grader
    full_validation.score=numeric_grader.score;full_validation.VERSION=numeric_grader.VERSION
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(8);torch.manual_seed(c['seed'])
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to(c['device'])
    core.setup_lora(model,c['rank'],c['alpha'])
    adapter=out/'final_adapter.pt'
    load_adapter(model,torch.load(adapter,map_location='cpu',weights_only=True),identity)
    items=encode_rows(split_rows(read_rows(c['data']),c['seed'])['test'],tok,c['max_length'])
    rows=[dict(row=dict(id=x['id'],answer=x['gold']),prompt=x['prompt'],windows=[(x['ids'],x['mask'])],response_tokens=sum(x['mask'])) for x in items]
    ec=dict(seed=c['seed'],generation_limit=0,generation_max_new_tokens=c['generation_max_new_tokens'],generation_temperature=1.,generation_top_p=1.)
    metrics=full_validation.evaluate(model,tok,rows,ec,destination)
    assert metrics['generation_samples']==len(items)==1319
    report=dict(metrics=metrics,base_identity=identity,adapter_sha256=digest(adapter),selection_changed=False,time=time.time())
    (out/'final_test_complete.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
