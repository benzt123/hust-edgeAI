"""RSFT 小规模运行配套：核心筛选与统计来自用户的 core.py。

生成、训练在独立进程中运行，避免模型和优化器同时常驻显存。
复用已有SFT组件；不使用Trainer，不改变标准答案或拆分解答标签。
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rsft_student_core',HERE/'core.py')
student=importlib.util.module_from_spec(spec)
spec.loader.exec_module(student)


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['sample','train'])
    parser.add_argument('--config',type=Path,default=HERE/'trial_config.json')
    args=parser.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8-sig'))
    sys.path.insert(0,c['sft_code'])
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from full_data import load_rows,encode_rows
    from full_validation import evaluate
    from full_scoring import score
    from batching import backward_group
    from core import optimizer_update
    torch.manual_seed(c['seed'])
    random.seed(c['seed'])
    run=Path(c['output'])
    source=Path(c['source_run'])
    original=json.loads((source/'config.json').read_text())
    split=json.loads((source/'split.json').read_text())
    assert not set(split['train_ids']) & set(split['validation_ids'])
    mapping={r['id']:r for r in load_rows(original['data_dir'])}
    tokenizer=AutoTokenizer.from_pretrained(str(source/'best_model'),local_files_only=True)
    template=(source/'user.txt').read_text()
    train_ids=list(split['train_ids'])
    random.Random(c['seed']).shuffle(train_ids)
    chosen=train_ids[:c['questions']]
    val_ids=split['validation_ids'][:c['validation_questions']]
    evaluation_config=original | dict(generation_limit=c['validation_questions'])
    validation,_=encode_rows([mapping[i] for i in val_ids],tokenizer,template,evaluation_config)
    if args.phase=='sample':
        run.mkdir(parents=True,exist_ok=False)
        dump(run/'config.json',c)
        dump(run/'ids.json',dict(train=chosen,validation=val_ids))
        (run/'student_core.py').write_bytes((HERE/'core.py').read_bytes())
        (run/'run_trial.py').write_bytes(Path(__file__).read_bytes())
        model=AutoModelForCausalLM.from_pretrained(str(source/'best_model'),
            torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        evaluate(model,tokenizer,validation,evaluation_config,run/'before')
        candidates=[]
        with (run/'candidates.jsonl').open('w',encoding='utf-8') as stream:
            for index,qid in enumerate(chosen):
                row=mapping[qid]
                prompt=template.replace('{question}',row['question'])
                inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=model.generate(**inputs,do_sample=True,num_return_sequences=c['group_size'],
                        temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,
                        max_new_tokens=c['max_new_tokens'],min_new_tokens=4,use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,
                        stop_strings=['</answer>'],tokenizer=tokenizer)
                for k,output in enumerate(outputs):
                    tokens=output[inputs['input_ids'].shape[1]:].tolist()
                    eos=tokenizer.eos_token_id
                    end=tokens.index(eos) if eos in tokens else len(tokens)
                    response=tokenizer.decode(tokens[:end],skip_special_tokens=True)
                    metrics=score(response,row['answer'])
                    truncated=end>=c['max_new_tokens'] and '</answer>' not in response and eos not in tokens
                    candidate=dict(question_id=qid,prompt=prompt,response=response,
                        answer_correct=bool(metrics['answer_correct']),format_ok=bool(metrics['format_ok']),
                        truncated=bool(truncated),candidate_index=k,gold=row['answer'])
                    candidates.append(candidate)
                    stream.write(json.dumps(candidate,ensure_ascii=False)+'\n')
                stream.flush()
                print('SAMPLED',index+1,len(chosen),flush=True)
        selected=student.select_correct_samples(candidates)
        stats=student.summarize_candidates(candidates,selected)
        dump(run/'sampling_stats.json',stats)
        with (run/'selected.jsonl').open('w',encoding='utf-8') as stream:
            for row in selected: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        print('SAMPLING_COMPLETE',stats,flush=True)
        if not selected: raise RuntimeError('没有合格样本，停止训练')
        return
    if (run/'student_core.py').read_bytes() != (HERE/'core.py').read_bytes():
        raise RuntimeError('筛选代码在采样后变化，停止')
    selected=[json.loads(s) for s in (run/'selected.jsonl').read_text().splitlines()]
    if not selected: raise RuntimeError('没有训练样本')
    items=[]
    for row in selected:
        if row['question_id'] not in chosen: raise RuntimeError('发现非训练题')
        prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
        suffix=tokenizer.encode(row['response'],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prefix)+len(suffix)>original['max_length']: raise RuntimeError('样本超长，未静默丢弃')
        items.append(dict(windows=[(prefix+suffix,[0]*len(prefix)+[1]*len(suffix))],response_tokens=len(suffix)))
    model=AutoModelForCausalLM.from_pretrained(str(source/'best_model'),torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda')
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0.)
    tracked=next(p for n,p in model.named_parameters() if 'layers.0.mlp.down_proj.weight' in n)
    before=tracked.detach().clone()
    order=list(range(len(items)))
    random.Random(c['seed']).shuffle(order)
    with (run/'train.jsonl').open('w') as stream:
        for step,start in enumerate(range(0,len(order),c['effective_batch']),1):
            tick=time.monotonic()
            group=[items[i] for i in order[start:start+c['effective_batch']]]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            loss,_=backward_group(model,group,tokenizer.eos_token_id,c['micro_batch'])
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('梯度非有限')
            optimizer_update(model,optimizer,1.)
            record=dict(step=step,samples=len(group),loss=loss/len(group),seconds=time.monotonic()-tick,
                allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            print('TRAIN',record,flush=True)
    changed=not torch.equal(before,tracked)
    if not changed: raise RuntimeError('参数未更新')
    del optimizer,before,tracked
    torch.cuda.empty_cache()
    model.save_pretrained(run/'final_model')
    tokenizer.save_pretrained(run/'final_model')
    # 实际重载保存的模型，再运行同一验证集。
    del model
    torch.cuda.empty_cache()
    model=AutoModelForCausalLM.from_pretrained(str(run/'final_model'),torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda')
    after=evaluate(model,tokenizer,validation,evaluation_config,run/'after')
    dump(run/'complete.json',dict(parameter_changed=changed,selected_samples=len(items),validation=after))
    print('COMPLETE',run,flush=True)


if __name__=='__main__': main()
