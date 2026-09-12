"""服务器 CPU 前置审计：真实 Parquet、答案提取、划分和完整序列长度。

不启动训练、不静默丢弃样本。python audit_data.py --output /path/to/report
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--grader',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8'))
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    spec=importlib.util.spec_from_file_location('course_grader',args.grader)
    grader=importlib.util.module_from_spec(spec);spec.loader.exec_module(grader)
    tokenizer=AutoTokenizer.from_pretrained(c['model']['initial_weights'],local_files_only=True)
    template=(HERE/c['prompt']['file']).read_text(encoding='utf-8').rstrip()
    sets={};manifest=[];errors=[]
    for split in ['train','test']:
        rows=[]
        for subject in c['dataset']['subjects']:
            files=sorted((Path(c['dataset']['path'])/subject).glob(split+'-*.parquet'))
            if not files:errors.append('missing '+subject+'/'+split)
            offset=0
            for f in files:
                data=f.read_bytes()
                if data[:4]!=b'PAR1':
                    errors.append('not_parquet '+str(f));continue
                table=pq.read_table(f).to_pylist()
                manifest.append(dict(path=str(f),rows=len(table),sha256=hashlib.sha256(data).hexdigest()))
                for i,r in enumerate(table):
                    gold=grader.extract_boxed_answer(r['solution'])
                    row=dict(id=f'{subject}/{split}/{offset+i}',subject=subject,level=r.get('level','unknown'),
                             problem=r['problem'],solution=r['solution'],gold=gold)
                    if not gold or not gold.strip():errors.append('missing_gold '+row['id'])
                    rows.append(row)
                offset+=len(table)
        sets[split]=rows
    key=lambda r:re.sub(r'\s+',' ',r['problem']).strip()
    test_keys={key(r) for r in sets['test']}
    overlap=[r['id'] for r in sets['train'] if key(r) in test_keys]
    if overlap:errors.append('train_test_overlap '+str(len(overlap)))
    groups=defaultdict(list)
    for r in sets['train']:groups[key(r)].append(r)
    strata=defaultdict(list)
    for problem,group in groups.items():
        first=group[0];strata[(first['subject'],first['level'])].append(problem)
    val_keys=set()
    rng=random.Random(c['seed'])
    for stratum in sorted(strata):
        keys=sorted(strata[stratum]);rng.shuffle(keys)
        count=round(len(keys)*c['dataset']['validation_fraction'])
        if len(keys)>1:count=max(1,min(len(keys)-1,count))
        else:count=0
        val_keys.update(keys[:count])
    train=[r for r in sets['train'] if key(r) not in val_keys]
    val=[r for r in sets['train'] if key(r) in val_keys]
    random.Random(c['seed']+1).shuffle(val)
    lengths={};oversize=[];prompt_oversize=[]
    for name,rows in [('train',train),('validation',val),('test',sets['test'])]:
        values=[]
        for r in rows:
            if not r['gold']:continue
            prefix=tokenizer.encode(template.replace('{question}',r['problem']),add_special_tokens=False)
            response=r['solution']+'</think> <answer>'+r['gold']+'</answer>'
            suffix=tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id]
            n=len(prefix)+len(suffix);values.append(n)
            if n>c['runtime']['initial_max_sequence_length']:oversize.append(dict(id=r['id'],tokens=n,split=name))
            if len(prefix)+c['generation']['max_new_tokens']>c['runtime']['initial_max_sequence_length']:
                prompt_oversize.append(dict(id=r['id'],prompt_tokens=len(prefix),split=name))
        values.sort()
        lengths[name]=dict(count=len(values),max=max(values,default=0),
                           p95=values[min(len(values)-1,int(len(values)*.95))] if values else 0,
                           mean=sum(values)/len(values) if values else 0)
    if oversize:errors.append('full_solution_exceeds_context '+str(len(oversize)))
    if prompt_oversize:errors.append('prompt_plus_generation_exceeds_context '+str(len(prompt_oversize)))
    report=dict(train_original=len(sets['train']),train=len(train),validation=len(val),test=len(sets['test']),
                manifest=manifest,grader_sha256=hashlib.sha256(args.grader.read_bytes()).hexdigest(),
                subject_level_counts={name:dict(Counter(r['subject']+'/'+r['level'] for r in rs)) for name,rs in [('train',train),('validation',val),('test',sets['test'])]},
                lengths=lengths,oversize=oversize,prompt_oversize=prompt_oversize,train_test_overlap_ids=overlap,
                errors=errors,ready=not errors)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.output/'split.json').write_text(json.dumps(dict(train_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in val],test_ids=[r['id'] for r in sets['test']]),indent=2))
    print(json.dumps({k:report[k] for k in ['train_original','train','validation','test','lengths','oversize','prompt_oversize','errors','ready']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
