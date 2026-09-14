"""Paired validation-only token-budget diagnostic; no training or selection."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'math_runtime'))
from adapters import core,load_adapter,atomic_save
from check_grader import Grader
from run import digest_file

def main():
    c=json.loads((HERE/'math_runtime/config.math.json').read_text())
    source=Path(c['output']);out=source.parent/'lora-math-token-diagnostic-20260914'
    out.mkdir(exist_ok=False)
    state=torch.load(source/'latest.pt',map_location='cpu',weights_only=False)
    identity=json.loads((source/'identity.json').read_text())
    assert state['identity']==identity['sha256']
    atomic_save(out/'step_adapter.pt',state['adapter']);step=state['step'];del state
    base=Path(c['base_model'])
    files=sorted(base.glob('*.safetensors'))+sorted(base.glob('*.json'))+sorted(base.glob('*.jinja'))
    actual=hashlib.sha256(json.dumps({p.name:digest_file(p) for p in files},sort_keys=True).encode()).hexdigest()
    assert actual==identity['base']
    rows=[r for r in map(json.loads,open(c['data'])) if r['split']=='validation']
    chosen=random.Random(42).sample(rows,32)
    (out/'manifest.json').write_text(json.dumps(dict(ids=[r['id'] for r in chosen],step=step,base_identity=actual,adapter_sha256=digest_file(out/'step_adapter.pt'),seed=42,caps=[1024,2048],method='Same 2048-token trajectory, prefix truncated at 1024; validation only'),indent=2))
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    torch.manual_seed(42);core.setup_lora(model,c['rank'],c['alpha'])
    grader=Grader(HERE/'math_runtime/course_grader.py',5)
    records=[]
    try:
        for arm in ['baseline','step_adapter']:
            if arm=='step_adapter':load_adapter(model,torch.load(out/'step_adapter.pt',weights_only=True),actual)
            for index,row in enumerate(chosen):
                tick=time.monotonic();torch.manual_seed(42+index)
                ids=tok.encode(row['prompt'],add_special_tokens=False)
                assert len(ids)+2048<=model.config.max_position_embeddings
                x=torch.tensor([ids],device='cuda')
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=model.generate(x,attention_mask=torch.ones_like(x),do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=2048,use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,stop_strings=['</answer>'],tokenizer=tok)[0,len(ids):].tolist()
                for cap in [1024,2048]:
                    tokens=generated[:cap];response=tok.decode(tokens,skip_special_tokens=True)
                    record=dict(arm=arm,id=row['id'],question=row['question'],gold=row['gold'],cap=cap,response=response,tokens=len(tokens),truncated=len(tokens)==cap and '</answer>' not in response and tokens[-1]!=tok.eos_token_id,seconds_2048=time.monotonic()-tick)
                    # Save generated evidence even if grading fails.
                    with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    grade=grader.score(response,row['gold']);record['grade']=grade
                    with (out/'answers.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    if grade['status']!='ok':raise RuntimeError('Grader failure; diagnostic stopped, never counted as incorrect')
                    records.append(record)
                print('PROGRESS',arm,index+1,len(chosen),flush=True)
        summary=[]
        for arm in ['baseline','step_adapter']:
            for cap in [1024,2048]:
                rr=[r for r in records if r['arm']==arm and r['cap']==cap]
                summary.append(dict(arm=arm,cap=cap,count=len(rr),correct=sum(r['grade']['result']['answer_reward']==1 for r in rr),truncated=sum(r['truncated'] for r in rr)))
        (out/'complete.json').write_text(json.dumps(dict(step=step,summary=summary,time=time.time()),indent=2))
        print('COMPLETE',summary,flush=True)
    finally:grader.close()

if __name__=='__main__':main()
