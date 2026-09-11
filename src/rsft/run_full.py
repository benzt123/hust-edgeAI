"""正式RSFT运行配套：分批采样落盘、每题上限、可恢复训练、独立测试。"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('student_rsft',HERE/'core.py')
student=importlib.util.module_from_spec(spec)
spec.loader.exec_module(student)


def dump(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','sample','train','test'])
    parser.add_argument('--config',type=Path,default=HERE/'full_config.json')
    args=parser.parse_args()
    c=read(args.config)
    sys.path.insert(0,c['sft_code'])
    from full_data import load_rows,encode_rows
    from full_scoring import score
    from transformers import AutoTokenizer
    run=Path(c['output']); source=Path(c['source_run'])
    original=read(source/'config.json'); split=read(source/'split.json')
    assert not set(split['train_ids']) & set(split['validation_ids'])
    train_ids=list(split['train_ids']); random.Random(c['seed']).shuffle(train_ids)
    chosen=train_ids[:c['questions']]
    if len(chosen)!=c['questions']: raise ValueError('训练题数量不符')
    val_ids=split['validation_ids'][:c['validation_questions']]
    template=(source/'user.txt').read_text()
    mapping={r['id']:r for r in load_rows(original['data_dir'])}
    fingerprint=hashlib.sha256((json.dumps(c,sort_keys=True)+template+
        json.dumps([mapping[i] for i in chosen+val_ids],sort_keys=True)+
        (HERE/'core.py').read_text()+Path(__file__).read_text()).encode()).hexdigest()
    if args.phase=='prepare':
        run.mkdir(parents=True,exist_ok=False)
        (run/'chunks').mkdir()
        dump(run/'config.json',c); dump(run/'ids.json',dict(train=chosen,validation=val_ids))
        dump(run/'identity.json',dict(sha256=fingerprint))
        for filename in ['core.py','run_full.py']:
            (run/('snapshot_'+filename)).write_bytes((HERE/filename).read_bytes())
        dump(run/'status.json',dict(phase='prepared',time=time.time()))
        print('PREPARED',run,flush=True)
        return
    if read(run/'identity.json')['sha256']!=fingerprint:
        raise RuntimeError('代码、配置或数据变化，拒绝不一致的恢复')
    dump(run/'status.json',dict(phase=args.phase,time=time.time()))
    if args.phase=='sample':
        from vllm import LLM,SamplingParams
        llm=None
        for start in range(0,len(chosen),c['sample_chunk']):
            destination=run/'chunks'/f'{start:06d}.json'
            if destination.exists(): continue
            if llm is None:
                llm=LLM(model=str(source/'best_model'),dtype='bfloat16',
                    gpu_memory_utilization=0.65,max_model_len=4096,max_num_seqs=128,
                    enforce_eager=True,seed=c['seed'])
            ids=chosen[start:start+c['sample_chunk']]
            prompts=[template.replace('{question}',mapping[i]['question']) for i in ids]
            params=SamplingParams(n=c['group_size'],temperature=1.,top_p=1.,top_k=-1,
                max_tokens=c['max_new_tokens'],min_tokens=4,stop=['</answer>'],
                include_stop_str_in_output=True,seed=c['seed']+start)
            tick=time.monotonic()
            outputs=llm.generate(prompts,params,use_tqdm=False)
            records=[]
            for qid,prompt,output in zip(ids,prompts,outputs):
                if len(output.outputs)!=c['group_size']: raise RuntimeError('候选数目不完整')
                for k,item in enumerate(output.outputs):
                    result=score(item.text,mapping[qid]['answer'])
                    records.append(dict(question_id=qid,prompt=prompt,response=item.text,
                        gold=mapping[qid]['answer'],candidate_index=k,
                        answer_correct=bool(result['answer_correct']),format_ok=bool(result['format_ok']),
                        truncated=item.finish_reason=='length',finish_reason=item.finish_reason))
            if len(records)!=len(ids)*c['group_size']: raise RuntimeError('采样结果不完整')
            dump(destination,records)
            dump(run/'progress.json',dict(phase='sample',questions=min(start+len(ids),len(chosen)),
                total=len(chosen),chunk_seconds=time.monotonic()-tick,time=time.time()))
            print('SAMPLED',min(start+len(ids),len(chosen)),len(chosen),'seconds',time.monotonic()-tick,flush=True)
        candidates=[r for f in sorted((run/'chunks').glob('*.json')) for r in read(f)]
        if len(candidates)!=len(chosen)*c['group_size']: raise RuntimeError('候选总数不符')
        qualified=student.select_correct_samples(candidates)
        selected=student.select_up_to_k_per_question(qualified,k=c['keep_per_question'],seed=c['seed'])
        dump(run/'quality_stats.json',student.summarize_candidates(candidates,qualified))
        dump(run/'selection_stats.json',student.summarize_candidates(candidates,selected))
        dump(run/'selected.json',selected)
        if not selected: raise RuntimeError('没有合格训练样本')
        print('SAMPLE_COMPLETE',len(selected),flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM
    from full_validation import evaluate
    from batching import backward_group
    from core import optimizer_update
    torch.manual_seed(c['seed']); random.seed(c['seed'])
    tokenizer=AutoTokenizer.from_pretrained(str(source/'best_model'),local_files_only=True)
    ec=original | dict(generation_limit=c['validation_questions'])
    validation,_=encode_rows([mapping[i] for i in val_ids],tokenizer,template,ec)
    def load_model(path):
        return AutoModelForCausalLM.from_pretrained(str(path),torch_dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).to('cuda')
    def evaluate_to(model,items,config,name):
        destination=run/name
        if (destination/'metrics.json').exists(): return read(destination/'metrics.json')
        if destination.exists(): destination.rename(run/(name+'_interrupted_'+str(time.time_ns())))
        return evaluate(model,tokenizer,items,config,destination)
    if args.phase=='test':
        if not (run/'training_complete.json').exists(): raise RuntimeError('训练尚未完成')
        ec['generation_limit']=0
        test,_=encode_rows(load_rows(original['data_dir'],'test'),tokenizer,template,ec)
        model=load_model(run/'best_model')
        result=evaluate_to(model,test,ec,'test_best')
        dump(run/'complete.json',dict(test=result,time=time.time()))
        dump(run/'status.json',dict(phase='complete',time=time.time()))
        print('COMPLETE',run,flush=True)
        return
    if (run/'training_complete.json').exists(): return
    selected=read(run/'selected.json'); items=[]
    chosen_set=set(chosen)
    for row in selected:
        if row['question_id'] not in chosen_set: raise RuntimeError('发现非训练题')
        prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
        suffix=tokenizer.encode(row['response'],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prefix)+len(suffix)>original['max_length']: raise RuntimeError('样本超长，未丢弃')
        items.append(dict(windows=[(prefix+suffix,[0]*len(prefix)+[1]*len(suffix))],response_tokens=len(suffix)))
    model=load_model(source/'best_model'); model.config.use_cache=False
    model.gradient_checkpointing_enable()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0.)
    order=list(range(len(items))); random.Random(c['seed']).shuffle(order)
    step=offset=0; best=(-1.,float('-inf'))
    checkpoint_path=run/'latest.pt'
    if checkpoint_path.exists():
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if state['identity']!=fingerprint: raise RuntimeError('断点不匹配')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        step,offset,best=state['step'],state['offset'],tuple(state['best'])
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        torch.cuda.empty_cache()
        for name in ['train.jsonl','validation.jsonl']:
            path=run/name
            if path.exists():
                old=[json.loads(x) for x in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(x)+'\n' for x in old if x['step']<=step))
        # 重做断点后的验证，避免复用未提交状态产生的指标。
        for d in run.glob('validation_*'):
            if d.is_dir() and d.name[11:].isdigit() and int(d.name[11:])>step:
                d.rename(run/(d.name+'_interrupted_'+str(time.time_ns())))
    else:
        evaluate_to(model,validation,ec,'before')
    def save_checkpoint(end):
        tmp=run/'latest.pt.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            offset=end,best=best,identity=fingerprint,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()),tmp)
        os.replace(tmp,checkpoint_path)
    model.train()
    total=math.ceil(len(items)/c['effective_batch'])
    with (run/'train.jsonl').open('a') as stream:
        for start in range(offset,len(order),c['effective_batch']):
            tick=time.monotonic(); group=[items[i] for i in order[start:start+c['effective_batch']]]
            optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
            loss,_=backward_group(model,group,tokenizer.eos_token_id,c['micro_batch'])
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('非有限梯度')
            optimizer_update(model,optimizer,1.); step+=1
            record=dict(step=step,total=total,samples=len(group),loss=loss/len(group),
                seconds=time.monotonic()-tick,allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            print('TRAIN',record,flush=True)
            end=min(start+len(group),len(order))
            if step%c['eval_every']==0 or end==len(order):
                torch.cuda.empty_cache()
                metrics=evaluate_to(model,validation,ec,f'validation_{step:06d}')
                with (run/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**metrics))+'\n')
                candidate=(metrics['answer_correct_rate'],-metrics['loss'])
                if candidate>best:
                    best=candidate; model.save_pretrained(run/'best_model'); tokenizer.save_pretrained(run/'best_model')
                    dump(run/'best.json',dict(step=step,metrics=metrics))
                save_checkpoint(end)
                torch.cuda.empty_cache()
                print('CHECKPOINT',step,flush=True)
    del optimizer
    torch.cuda.empty_cache()
    model.save_pretrained(run/'final_model'); tokenizer.save_pretrained(run/'final_model')
    dump(run/'training_complete.json',dict(steps=step,samples=len(items),time=time.time()))
    print('TRAIN_COMPLETE',flush=True)


if __name__=='__main__': main()
