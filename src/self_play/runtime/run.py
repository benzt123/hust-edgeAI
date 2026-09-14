"""Self-Play命令行：prepare、audit、smoke、train、test。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

from pipeline import prepare_round,train_records,student,grpo
from backend import TransformersBackend
from state import dump,save,restore

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','audit','smoke','train','test'])
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    args=parser.parse_args();c=read(args.config)
    for key in ['rounds','problems_per_round','generation_batch','epochs_per_rollout','effective_batch',
                'micro_batch','validation_questions','eval_every','warmup_rounds','audit_problems','max_empty_rounds']:
        if type(c[key]) is not int or c[key]<1:raise ValueError(f'{key}必须为正整数')
    if type(c['group_size']) is not int or c['group_size']<2:raise ValueError('group_size必须>=2')
    if c['learning_rate']<=0 or not 0<c['clip_eps']<1:raise ValueError('学习率或clip参数非法')
    sys.path.insert(0,c['sft_code'])
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from full_data import load_rows,encode_rows
    import full_validation
    import numeric_grader
    torch.set_num_threads(8);torch.manual_seed(c['seed']);random.seed(c['seed'])
    out=Path(c['output']);source=Path(c['source_model']);sft=Path(c['sft_run'])
    if not (source.parent/'complete.json').exists():raise RuntimeError('上一阶段尚未完成')
    original=read(sft/'config.json');split=read(sft/'split.json');template=(sft/'user.txt').read_text()
    rows=load_rows(original['data_dir']);test_rows=load_rows(original['data_dir'],'test')
    mapping={x['id']:x for x in rows}
    if set(split['train_ids']) & set(split['validation_ids']):raise ValueError('训练验证ID重叠')
    if any(i not in mapping for i in split['train_ids']+split['validation_ids']):raise ValueError('split ID缺失')
    if c['validation_questions']>len(split['validation_ids']):raise ValueError('验证数量超出固定划分')
    # 只对题干做精确排重；测试答案从不进入出题/奖励路径。
    blocked={student.problem_key(x['question']) for x in rows+test_rows}
    files=list(HERE.glob('*.py'))+[Path(student.__file__),Path(grpo.__file__),sft/'split.json',sft/'user.txt']
    files+=sorted(source.glob('*.json'))+sorted(source.glob('*.safetensors'))
    files+=[Path(c['sft_code'])/n for n in ['full_data.py','full_validation.py','full_scoring.py','core.py']]
    identity=dict(config=c,files={str(p):file_hash(p) for p in files},
        data=hashlib.sha256(json.dumps([rows,test_rows,original],sort_keys=True).encode()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if args.phase=='prepare':
        out.mkdir(parents=True,exist_ok=False);(out/'rounds').mkdir();(out/'models').mkdir()
        dump(out/'identity.json',dict(sha256=fingerprint,**identity));dump(out/'config.json',c)
        dump(out/'status.json',dict(phase='prepared',time=time.time()))
        print('PREPARED',out,flush=True);return
    if read(out/'identity.json')['sha256']!=fingerprint:raise ValueError('输入改变，请创建新实验目录')
    dump(out/'status.json',dict(phase=args.phase,time=time.time()))
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    def load_model(path):
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda')
        model.config.use_cache=False
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return model
    full_validation.score=numeric_grader.score;full_validation.VERSION=numeric_grader.VERSION
    ec=original|dict(generation_limit=c['validation_questions'])
    validation,_=encode_rows([mapping[i] for i in split['validation_ids'][:c['validation_questions']]],tokenizer,template,ec)
    def evaluate(model,items,name,config=ec):
        path=out/name
        if (path/'metrics.json').exists():return read(path/'metrics.json')
        if path.exists():path.rename(out/(name+'_interrupted_'+str(time.time_ns())))
        return full_validation.evaluate(model,tokenizer,items,config,path)
    if args.phase=='test':
        if not (out/'training_complete.json').exists():raise ValueError('训练未完成')
        metadata=read(out/'training_complete.json');best=metadata['best']
        model=load_model(best['path'])
        test_config=ec|dict(generation_limit=0)
        test,_=encode_rows(test_rows,tokenizer,template,test_config)
        metrics=evaluate(model,test,'test_best',test_config)
        if metrics['generation_samples']!=len(test_rows):raise ValueError('完整测试未覆盖全部题目')
        dump(out/'complete.json',dict(best=best,metrics=metrics,time=time.time()))
        dump(out/'status.json',dict(phase='complete',time=time.time()));return
    if args.phase=='train' and (out/'training_complete.json').exists():return
    model=load_model(source);backend=TransformersBackend(model,tokenizer,c)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),weight_decay=0.,foreach=False)
    def make_round(count,seen,round_id,attempt=0):
        sample_seed=c['seed']+round_id+attempt*1000003
        artifact=f'{round_id:04d}' if attempt==0 else f'{round_id:04d}_attempt_{attempt}'
        torch.manual_seed(sample_seed)
        prompts=[c.get('problem_gen_prompt',student.PROBLEM_GEN_PROMPT)]*count
        if c.get('use_training_seeds',False):
            seed_ids=random.Random(sample_seed).sample(sorted(split['train_ids']),count)
            prompts=[c['problem_gen_prompt'].replace('{seed_problem}',mapping[i]['question']) for i in seed_ids]
            dump(out/'rounds'/f'{artifact}_seed_ids.json',seed_ids)
        if c.get('verified_arithmetic',False):
            from verified_arithmetic import generate
            generations=generate(count,sample_seed)
        else:
            generations=backend.generate(prompts)
        # 原始出题文本先保存，后续评分失败仍可审查。
        dump(out/'rounds'/f'{artifact}_generations.json',generations)
        eligible=[]
        for generation in generations:
            question,_=student.parse_problem_and_answer(generation['text'])
            if question and re.search(r'<<|\b(?:solution|explanation|dep\.\s*steps)\s*[:\d]|\d\s*[+*/=]\s*\d',question,re.I):
                continue
            eligible.append(generation)
        result=prepare_round(eligible,backend.solve,numeric_grader.score,
            lambda a:numeric_grader.number(a) is not None,template,c['group_size'],blocked|set(seen))
        result['generations']=generations
        result['sampling_attempt']=attempt
        dump(out/'rounds'/f'{artifact}_rollout.json',result)
        return result
    def update(records,round_id):
        return train_records(model,optimizer,records,tokenizer.eos_token_id,
            c['epochs_per_rollout'],c['effective_batch'],c['micro_batch'],c['clip_eps'],c['seed']+round_id)
    if args.phase in ['audit','smoke']:
        torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
        result=make_round(c['audit_problems'],[],0)
        if args.phase=='smoke':
            if not result['stats']['informative_groups']:raise RuntimeError('试跑无奖励差异，不能确认更新链路')
            updates=update(result['records'],0)
            if not any(x['grad_norm']>0 for x in updates):raise RuntimeError('试跑无有效梯度')
            result['updates']=updates
        result.update(seconds=time.monotonic()-tick,allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        dump(out/(args.phase+'.json'),result)
        print(args.phase.upper(),result['stats'],flush=True);return
    metadata=dict(round=0,steps=0,seen=[],empty_rounds=0,best=None)
    checkpoint=Path(c.get('checkpoint_path',str(out/'latest.pt')))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():
        metadata=restore(checkpoint,model,optimizer,fingerprint)
        # checkpoint之后的验证不允许复用；旧产物保留为中断记录。
        for p in out.glob('validation_*'):
            if p.is_dir() and p.name[11:].isdigit() and int(p.name[11:])>metadata['round']:
                p.rename(out/(p.name+'_interrupted_'+str(time.time_ns())))
        dump(out/'best.json',metadata['best'])
    else:
        baseline=evaluate(model,validation,'before')
        metadata['best']=dict(round=0,path=str(source),metrics=baseline)
        save(checkpoint,model,optimizer,fingerprint,metadata);dump(out/'best.json',metadata['best'])
    for round_id in range(metadata['round']+1,c['rounds']+1):
        tick=time.monotonic();result=make_round(c['problems_per_round'],metadata['seen'],round_id)
        retry_seen=list(metadata['seen'])
        for attempt in range(1,3):
            if result['stats']['informative_groups']>0:break
            retry_seen.extend(student.problem_key(p['problem']) for p in result['problems'])
            result=make_round(c['problems_per_round'],retry_seen,round_id,attempt)
        informative=result['stats']['informative_groups']>0
        metadata['empty_rounds']=0 if informative else metadata['empty_rounds']+1
        if metadata['empty_rounds']>=c['max_empty_rounds']:
            raise RuntimeError('连续多轮没有可学习的奖励差异，停止以便检查出题质量')
        for group in optimizer.param_groups:group['lr']=c['learning_rate']*min(1.,round_id/c['warmup_rounds'])
        updates=update(result['records'],round_id)
        metadata['seen']+= [student.problem_key(p['problem']) for p in result['problems']]
        metadata['steps']+=len(updates);metadata['round']=round_id
        result['updates']=updates
        dump(out/'rounds'/f'{round_id:04d}_result.json',result)
        if round_id%c['eval_every']==0 or round_id==c['rounds']:
            metrics=evaluate(model,validation,f'validation_{round_id:04d}')
            best=metadata['best']['metrics']
            if (metrics['answer_correct_rate'],-metrics['loss'])>(best['answer_correct_rate'],-best['loss']):
                # 新目录保存，checkpoint提交失败也不会覆盖原best权重。
                path=out/'models'/f'best_{round_id:04d}_{time.time_ns()}'
                model.save_pretrained(path);tokenizer.save_pretrained(path)
                metadata['best']=dict(round=round_id,path=str(path),metrics=metrics)
        save(checkpoint,model,optimizer,fingerprint,metadata)
        dump(out/'best.json',metadata['best'])
        progress=dict(round=round_id,total=c['rounds'],steps=metadata['steps'],stats=result['stats'],
                      seconds=time.monotonic()-tick,time=time.time())
        dump(out/'progress.json',progress);print('ROUND_COMPLETE',progress,flush=True)
    model.save_pretrained(out/'final_model');tokenizer.save_pretrained(out/'final_model')
    dump(out/'training_complete.json',metadata|dict(time=time.time()))


if __name__=='__main__':main()
