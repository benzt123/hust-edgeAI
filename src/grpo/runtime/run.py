"""手写在线 GRPO：同题采样、固定 old 概率、两轮更新、可恢复检查点。

核心算法直接调用学生 core.py。采样保留原 token ID，不解码后重新分词。
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('student_grpo', HERE.parent/'core.py')
student = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare','smoke','train','test'])
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    args = parser.parse_args()
    c = read(args.config)
    out, source, sft = Path(c['output']),Path(c['source_model']),Path(c['sft_run'])
    sys.path.insert(0,c['sft_code'])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, StopStringCriteria
    from full_data import load_rows,encode_rows
    import numeric_grader
    torch.set_num_threads(8)
    torch.manual_seed(c['seed'])
    random.seed(c['seed'])
    if not (source.parent/'complete.json').exists():
        raise RuntimeError('DPO 完整测试尚未结束')
    tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=True)
    original = read(sft/'config.json')
    split = read(sft/'split.json')
    template = (sft/'user.txt').read_text()
    rows = {x['id']:x for x in load_rows(original['data_dir'])}
    assert not set(split['train_ids']) & set(split['validation_ids'])
    files = [Path(__file__),HERE.parent/'core.py',HERE/'numeric_grader.py',sft/'split.json',sft/'user.txt',sft/'config.json']
    files += sorted(source.glob('*.json'))+sorted(source.glob('*.safetensors'))
    files += [Path(c['sft_code'])/name for name in ['full_data.py','full_validation.py','full_scoring.py','core.py']]
    identity = dict(config=c,files={str(p):sha(p) for p in files})
    identity['rows_sha256'] = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if args.phase == 'prepare':
        out.mkdir(parents=True,exist_ok=False)
        (out/'rollouts').mkdir()
        dump(out/'config.json',c)
        dump(out/'identity.json',dict(sha256=fingerprint,**identity))
        order = list(split['train_ids'])
        random.Random(c['seed']).shuffle(order)
        schedule = []
        cursor = 0
        for _ in range(c['rollout_rounds']):
            selected = []
            while len(selected)<c['rollout_questions']:
                if cursor == len(order):
                    random.Random(c['seed']+len(schedule)+1).shuffle(order)
                    cursor = 0
                selected.append(order[cursor]);cursor += 1
            schedule.append(selected)
        dump(out/'schedule.json',schedule)
        (out/'snapshot_core.py').write_bytes((HERE.parent/'core.py').read_bytes())
        (out/'snapshot_run.py').write_bytes(Path(__file__).read_bytes())
        dump(out/'status.json',dict(phase='prepared',time=time.time()))
        print('PREPARED',dict(rounds=len(schedule),questions_per_round=c['rollout_questions'],
                              group_size=c['group_size'],responses=c['rollout_rounds']*c['rollout_questions']*c['group_size']),flush=True)
        return
    if read(out/'identity.json')['sha256'] != fingerprint:
        raise RuntimeError('代码、模型、数据或配置变更；请使用新的实验目录')
    dump(out/'status.json',dict(phase=args.phase,time=time.time()))

    def load_model(path=source):
        model = AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).to('cuda')
        model.config.use_cache=False
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return model

    def collate(items):
        length = max(len(x['ids']) for x in items)
        ids,mask,attention,old = [],[],[],[]
        for x in items:
            pad = length-len(x['ids'])
            ids.append(x['ids']+[tokenizer.eos_token_id]*pad)
            mask.append(x['mask']+[0]*pad)
            attention.append([1]*len(x['ids'])+[0]*pad)
            if 'old' in x:old.append(x['old']+[0.]*pad)
        batch = {key:torch.tensor(value,device='cuda') for key,value in
                 [('input_ids',ids),('response_mask',mask),('attention_mask',attention)]}
        if old:batch['old_log_probs']=torch.tensor(old,device='cuda')
        return batch

    def token_scores(model,batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
        return student.response_token_log_probs(logits,batch['input_ids'],batch['response_mask'])

    class CaptureEnd(StoppingCriteria):
        """记录真正停止位置，排除 generate 给已结束样本追加的 padding。"""
        def __init__(self,n,prefix_length):
            self.stop = StopStringCriteria(tokenizer,['</answer>'])
            self.ends = torch.zeros(n,dtype=torch.long,device='cuda')
            self.prefix_length = prefix_length

        def __call__(self,input_ids,scores,**kwargs):
            done = self.stop(input_ids,scores) | (input_ids[:,-1] == tokenizer.eos_token_id)
            newly = done & (self.ends == 0)
            self.ends[newly] = input_ids.shape[1]-self.prefix_length
            return done

    def rollout(model,qids,round_id):
        model.eval()
        records = []
        tick=time.monotonic()
        torch.manual_seed(c['seed']+round_id)
        with torch.no_grad():
            for start in range(0,len(qids),c['sample_questions_per_batch']):
                group_ids=qids[start:start+c['sample_questions_per_batch']]
                prefixes=[tokenizer.encode(template.replace('{question}',rows[q]['question']),add_special_tokens=False) for q in group_ids]
                width=max(map(len,prefixes))
                if width+c['max_new_tokens']>c['max_length']:raise ValueError('生成预算超出上下文上限')
                ids,attention=[],[]
                for prefix in prefixes:
                    for _ in range(c['group_size']):
                        ids.append([tokenizer.eos_token_id]*(width-len(prefix))+prefix)
                        attention.append([0]*(width-len(prefix))+[1]*len(prefix))
                x=torch.tensor(ids,device='cuda')
                stopper=CaptureEnd(len(ids),width)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=model.generate(input_ids=x,attention_mask=torch.tensor(attention,device='cuda'),
                        do_sample=True,temperature=1.,top_p=1.,top_k=0,typical_p=1.,min_p=None,
                        repetition_penalty=1.,no_repeat_ngram_size=0,min_new_tokens=0,
                        max_new_tokens=c['max_new_tokens'],eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,forced_eos_token_id=None,use_cache=True,
                        stopping_criteria=StoppingCriteriaList([stopper]))
                generated=generated[:,width:].cpu().tolist()
                ends=stopper.ends.cpu().tolist()
                for i,(tokens,end) in enumerate(zip(generated,ends)):
                    qindex=i//c['group_size'];qid=group_ids[qindex];prefix=prefixes[qindex]
                    suffix=tokens[:end] if end else tokens
                    text=tokenizer.decode(suffix,skip_special_tokens=True)
                    graded=numeric_grader.score(text,rows[qid]['answer'])
                    truncated=end==0
                    # 只奖励格式完整、未截断且最终数值正确的回答；坏回答仍参与组内训练。
                    reward=float(graded['answer_correct'] and graded['format_ok'] and not truncated)
                    records.append(dict(question_id=qid,response=text,gold=rows[qid]['answer'],
                        ids=prefix+suffix,mask=[0]*len(prefix)+[1]*len(suffix),reward=reward,
                        truncated=truncated,**graded))
                del generated,x
                print('ROLLOUT_GENERATION',round_id,start+len(group_ids),len(qids),'seconds',time.monotonic()-tick,flush=True)
            rewards=torch.tensor([r['reward'] for r in records]).reshape(len(qids),c['group_size'])
            advantages=student.group_advantages(rewards).reshape(-1).tolist()
            for r,a in zip(records,advantages):r['advantage']=a
            # 以相同 micro-batch 划分缓存旧分数，整个 rollout 的两轮更新不刷新。
            for start in range(0,len(records),c['micro_batch']):
                group=records[start:start+c['micro_batch']]
                values,_=token_scores(model,collate(group))
                for r,value in zip(group,values):
                    r['old']=value[:len(r['ids'])-1].cpu().tolist()
        model.train()
        return records,dict(seconds=time.monotonic()-tick,reward=rewards.mean().item(),
                            informative_groups=int(((rewards.max(1).values-rewards.min(1).values)>0).sum()),
                            truncated=sum(r['truncated'] for r in records))

    def update(model,optimizer,items):
        optimizer.zero_grad(set_to_none=True)
        loss_sum=clip_sum=token_count=0.
        for start in range(0,len(items),c['micro_batch']):
            group=items[start:start+c['micro_batch']]
            batch=collate(group)
            new,mask=token_scores(model,batch)
            advantage=torch.tensor([x['advantage'] for x in group],device='cuda')
            loss=student.grpo_loss(new,batch['old_log_probs'],advantage,mask,c['clip_eps'])
            if not torch.isfinite(loss):raise RuntimeError('非有限 GRPO loss')
            (loss*len(group)/len(items)).backward()
            loss_sum+=loss.item()*len(group)
            ratio=(new.detach()-batch['old_log_probs']).masked_fill(~mask.bool(),0.).exp()
            clip_sum+=(((ratio-1).abs()>c['clip_eps'])*mask).sum().item()
            token_count+=mask.sum().item()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return dict(loss=loss_sum/len(items),grad_norm=norm.item(),clip_fraction=clip_sum/max(token_count,1))

    if args.phase=='smoke':
        model=load_model()
        qids=read(out/'schedule.json')[0][:c['smoke_questions']]
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),weight_decay=0.,foreach=False)
        torch.cuda.reset_peak_memory_stats()
        items,stats=rollout(model,qids,0)
        if not stats['informative_groups']:raise RuntimeError('试跑题目全为同奖励，需更多题检查梯度')
        model.eval()
        with torch.no_grad():
            for start in range(0,len(items),c['micro_batch']):
                batch=collate(items[start:start+c['micro_batch']])
                new,mask=token_scores(model,batch)
                torch.testing.assert_close(new[mask.bool()],batch['old_log_probs'][mask.bool()],rtol=0,atol=.002)
        model.train()
        tick=time.monotonic()
        updates=[]
        for epoch in range(c['epochs_per_rollout']):
            for start in range(0,len(items),c['effective_batch']):
                updates.append(update(model,optimizer,items[start:start+c['effective_batch']]))
        if not any(x['grad_norm']>0 for x in updates):raise RuntimeError('试跑没有有效梯度')
        result=dict(rollout=stats,updates=updates,update_seconds=time.monotonic()-tick,
                    allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    note='Temporary smoke updates discarded; formal training reloads DPO weights.')
        dump(out/'smoke.json',result)
        print('SMOKE_OK',result,flush=True)
        return

    import full_validation
    full_validation.score=numeric_grader.score
    full_validation.VERSION=numeric_grader.VERSION
    ec=original | dict(generation_limit=c['validation_questions'])
    validation,_=encode_rows([rows[q] for q in split['validation_ids'][:c['validation_questions']]],tokenizer,template,ec)

    def evaluate(model,encoded,name,config=ec):
        path=out/name
        if (path/'metrics.json').exists():return read(path/'metrics.json')
        if path.exists():path.rename(out/(name+'_interrupted_'+str(time.time_ns())))
        return full_validation.evaluate(model,tokenizer,encoded,config,path)

    if args.phase=='test':
        if not (out/'training_complete.json').exists():raise RuntimeError('GRPO 训练未完成')
        model=load_model(out/'best_model')
        test_config=ec | dict(generation_limit=0)
        test,_=encode_rows(load_rows(original['data_dir'],'test'),tokenizer,template,test_config)
        metrics=evaluate(model,test,'test_best',test_config)
        dump(out/'complete.json',dict(metrics=metrics,time=time.time()))
        dump(out/'status.json',dict(phase='complete',time=time.time()))
        print('COMPLETE',metrics,flush=True)
        return
    if (out/'training_complete.json').exists():return
    model=load_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),weight_decay=0.,foreach=False)
    schedule=read(out/'schedule.json')
    latest=out/'latest.pt'
    completed=steps=0
    best=(-1.,float('-inf'))
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity']!=fingerprint:raise RuntimeError('断点不匹配')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        completed,steps,best=state['round'],state['steps'],tuple(state['best'])
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        for name in ['train.jsonl','validation.jsonl']:
            path=out/name
            if path.exists():
                records=[json.loads(x) for x in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(r)+'\n' for r in records if r['round']<=completed))
        for path in out.glob('validation_*'):
            if path.is_dir() and path.name[11:].isdigit() and int(path.name[11:])>completed:
                path.rename(out/(path.name+'_interrupted_'+str(time.time_ns())))
        torch.cuda.empty_cache()
    else:
        metrics=evaluate(model,validation,'before')
        best=(metrics['answer_correct_rate'],-metrics['loss'])
        model.save_pretrained(out/'best_model');tokenizer.save_pretrained(out/'best_model')
        dump(out/'best.json',dict(round=0,steps=0,metrics=metrics))

    def save(round_id):
        tmp=out/'latest.pt.tmp'
        torch.save(dict(identity=fingerprint,round=round_id,steps=steps,best=best,
            model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()),tmp)
        os.replace(tmp,latest)

    if completed==0:save(0)
    for round_id in range(completed+1,c['rollout_rounds']+1):
        tick=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        items,stats=rollout(model,schedule[round_id-1],round_id)
        dump(out/'rollouts'/f'{round_id:04d}.json',dict(round=round_id,policy_steps=steps,stats=stats,records=items))
        lr=c['learning_rate']*min(1.,round_id/c['warmup_rounds'])
        for group in optimizer.param_groups:group['lr']=lr
        updates=[]
        for epoch in range(c['epochs_per_rollout']):
            order=list(range(len(items)))
            random.Random(c['seed']+round_id*100+epoch).shuffle(order)
            for start in range(0,len(order),c['effective_batch']):
                group=[items[i] for i in order[start:start+c['effective_batch']]]
                updates.append(update(model,optimizer,group));steps+=1
        record=dict(round=round_id,total_rounds=c['rollout_rounds'],steps=steps,lr=lr,
                    reward=stats['reward'],informative_groups=stats['informative_groups'],truncated=stats['truncated'],
                    rollout_seconds=stats['seconds'],seconds=time.monotonic()-tick,
                    loss=sum(x['loss'] for x in updates)/len(updates),
                    clip_fraction=sum(x['clip_fraction'] for x in updates)/len(updates),
                    max_grad_norm=max(x['grad_norm'] for x in updates),
                    allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30,time=time.time())
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        dump(out/'progress.json',record)
        print('TRAIN_ROUND',record,flush=True)
        if round_id%c['eval_every']==0 or round_id==c['rollout_rounds']:
            torch.cuda.empty_cache()
            metrics=evaluate(model,validation,f'validation_{round_id:04d}')
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(round=round_id,**metrics))+'\n')
            quality=(metrics['answer_correct_rate'],-metrics['loss'])
            if quality>best:
                best=quality
                model.save_pretrained(out/'best_model');tokenizer.save_pretrained(out/'best_model')
                dump(out/'best.json',dict(round=round_id,steps=steps,metrics=metrics))
            save(round_id)
        elif round_id%c['save_every']==0:save(round_id)
    model.save_pretrained(out/'final_model');tokenizer.save_pretrained(out/'final_model')
    dump(out/'training_complete.json',dict(rounds=c['rollout_rounds'],steps=steps,time=time.time()))
    print('TRAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
