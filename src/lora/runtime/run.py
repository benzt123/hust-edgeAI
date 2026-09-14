"""手写物理LoRA训练/评估/合并入口。评分报告NLL，不冒充物理正确率。"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time
import sys
import torch
from adapters import core,adapter_payload,atomic_save,save_adapter,load_adapter,merge_lora
from data import read_rows,split_rows,encode_rows

HERE=Path(__file__).resolve().parent


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def response_loss(model,items,pad_token_id,device):
    width=max(len(x['ids']) for x in items)
    ids=torch.tensor([x['ids']+[pad_token_id]*(width-len(x['ids'])) for x in items],device=device)
    mask=torch.tensor([x['mask']+[0]*(width-len(x['ids'])) for x in items],device=device)
    attention=torch.tensor([[1]*len(x['ids'])+[0]*(width-len(x['ids'])) for x in items],device=device)
    ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with ctx:logits=model(input_ids=ids[:,:-1],attention_mask=attention[:,:-1],use_cache=False).logits
    loss=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),ids[:,1:],reduction='none')
    counts=mask[:,1:].sum(1)
    if (counts==0).any():raise ValueError('空回答mask')
    return ((loss*mask[:,1:]).sum(1)/counts).mean()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','smoke','train','test','merge'])
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    args=parser.parse_args();c=json.loads(args.config.read_text(encoding='utf-8-sig'))
    from transformers import AutoModelForCausalLM,AutoTokenizer
    for key in ['epochs','micro_batch','effective_batch','max_length','save_every','generation_max_new_tokens']:
        if type(c[key]) is not int or c[key]<1:raise ValueError(f'非法参数{key}')
    if not 0<c['learning_rate']<1:raise ValueError('学习率无效')
    base=Path(c['base_model']);out=Path(c['output']);device=torch.device(c['device'])
    weights=sorted(base.glob('*.safetensors'))
    if not weights:raise ValueError('请配置存在的本地基座checkpoint，要求safetensors格式')
    base_files=weights+sorted(base.glob('*.json'))+sorted(base.glob('*.jinja'))
    base_identity=hashlib.sha256(json.dumps({p.name:digest_file(p) for p in base_files},sort_keys=True).encode()).hexdigest()
    rows=read_rows(c['data']);splits=split_rows(rows,c['seed'])
    fingerprint=hashlib.sha256(json.dumps(dict(config=c,base=base_identity,data=digest_file(c['data']),
        code={p.name:digest_file(p) for p in list(HERE.glob('*.py'))+[Path(core.__file__)]}),sort_keys=True).encode()).hexdigest()
    if args.phase=='prepare':
        out.mkdir(parents=True,exist_ok=False)
        dump(out/'config.json',c);dump(out/'identity.json',dict(sha256=fingerprint,base=base_identity))
        dump(out/'split.json',{k:[r['id'] for r in v] for k,v in splits.items()})
        print('PREPARED',{k:len(v) for k,v in splits.items()});return
    if json.loads((out/'identity.json').read_text())['sha256']!=fingerprint:raise ValueError('配置/数据/代码变化，拒绝混用断点')
    torch.manual_seed(c['seed']);random.seed(c['seed'])
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to(device)
    model.config.use_cache=False
    stats=core.setup_lora(model,c['rank'],c['alpha']);dump(out/'parameter_stats.json',stats)
    if any(p.requires_grad and not n.endswith(('.A','.B')) for n,p in model.named_parameters()):
        raise ValueError('发现非LoRA可训练参数')
    optimizer=torch.optim.AdamW(core.trainable_parameters(model),lr=c['learning_rate'],weight_decay=0.)
    if hasattr(model,'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    encoded={k:encode_rows(v,tokenizer,c['max_length']) for k,v in splits.items()}
    if c.get('dataset')=='gsm8k':
        encoded['validation']=encoded['validation'][:c['validation_limit']]
        sys.path.insert(0,c['sft_code'])
        import full_validation
        sys.path.insert(0,c['grader_code'])
        import numeric_grader
        full_validation.score=numeric_grader.score
        full_validation.VERSION=numeric_grader.VERSION
    def math_evaluate(items,name):
        path=out/name
        if (path/'metrics.json').exists():return json.loads((path/'metrics.json').read_text())
        if path.exists():path.rename(out/(name+'_interrupted_'+str(time.time_ns())))
        eval_rows=[dict(row=dict(id=x['id'],answer=x['gold']),prompt=x['prompt'],
            windows=[(x['ids'],x['mask'])],response_tokens=sum(x['mask'])) for x in items]
        ec=dict(seed=c['seed'],generation_limit=0,generation_max_new_tokens=c['generation_max_new_tokens'],
                generation_temperature=1.,generation_top_p=1.)
        return full_validation.evaluate(model,tokenizer,eval_rows,ec,path)
    @torch.no_grad()
    def evaluate(items):
        was=model.training;model.eval()
        try:return sum(response_loss(model,[x],tokenizer.eos_token_id,device).item() for x in items)/len(items)
        finally:model.train(was)
    def update(items):
        optimizer.zero_grad(set_to_none=True);value=0.
        for start in range(0,len(items),c['micro_batch']):
            batch=items[start:start+c['micro_batch']]
            loss=response_loss(model,batch,tokenizer.eos_token_id,device)
            if not torch.isfinite(loss):raise ValueError('非有限loss')
            (loss*len(batch)/len(items)).backward();value+=loss.item()*len(batch)
        norm=torch.nn.utils.clip_grad_norm_(core.trainable_parameters(model),1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        return value/len(items),norm.item()
    if args.phase in ['test','merge']:
        if not (out/'training_complete.json').exists():raise ValueError('训练未完成')
        payload=torch.load(out/'best_adapter.pt',map_location='cpu',weights_only=True)
        load_adapter(model,payload,base_identity)
        if args.phase=='merge':
            destination=out/'merged_model'
            if destination.exists():raise ValueError('导出目录已存在，请先检查已有结果')
            merge_lora(model).save_pretrained(destination);tokenizer.save_pretrained(destination);return
        if c.get('dataset')=='gsm8k':
            metrics=math_evaluate(encoded['test'],'test_best')
            if metrics['generation_samples']!=len(encoded['test']):raise ValueError('测试覆盖不完整')
            dump(out/'test_metrics.json',metrics)
            dump(out/'complete.json',dict(metrics=metrics,time=time.time(),base_identity=base_identity))
            return
        model.eval();baseline=json.loads((out/'baseline.json').read_text())
        metrics=dict(test_response_nll=evaluate(encoded['test']),samples=len(encoded['test']),
                     baseline_test_response_nll=baseline['test_response_nll'],physics_accuracy=None)
        path=out/'test_predictions.jsonl'
        with path.open('w',encoding='utf-8') as f,torch.no_grad():
            for item in encoded['test']:
                ids=tokenizer.encode(item['prompt'],add_special_tokens=False)
                if len(ids)+c['generation_max_new_tokens']>model.config.max_position_embeddings:raise ValueError('生成预算超限')
                x=torch.tensor([ids],device=device)
                ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
                with ctx:
                    generated=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,
                        max_new_tokens=c['generation_max_new_tokens'],use_cache=True,pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id)[0,len(ids):]
                response=tokenizer.decode(generated,skip_special_tokens=True)
                f.write(json.dumps(dict(id=item['id'],prompt=item['prompt'],reference=item['answer'],response=response,
                    truncated=len(generated)==c['generation_max_new_tokens'] and int(generated[-1])!=tokenizer.eos_token_id),ensure_ascii=False)+'\n');f.flush()
        dump(out/'test_metrics.json',metrics);dump(out/'complete.json',dict(metrics=metrics,time=time.time()));return
    model.train()
    if args.phase=='smoke':
        selected=sorted(encoded['train'],key=lambda x:len(x['ids']),reverse=True)[:c['effective_batch']]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        value,norm=update(selected)
        if norm==0:raise ValueError('LoRA试跑梯度为0')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('冻结参数存在梯度')
        if not any(torch.count_nonzero(m.A).item() for m in model.modules() if isinstance(m,core.LoRALinear)):
            raise ValueError('LoRA A矩阵未更新')
        dump(out/'smoke.json',dict(loss=value,grad_norm=norm,parameters=stats,
            allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
            note='临时更新未保存；正式训练重载基座'))
        return
    if (out/'training_complete.json').exists():return
    completed=0;best=float('inf');best_step=0
    latest=out/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity']!=fingerprint:raise ValueError('断点不匹配')
        load_adapter(model,state['adapter'],base_identity);optimizer.load_state_dict(state['optimizer'])
        completed,best,best_step=state['step'],state['best'],state['best_step']
        torch.set_rng_state(state['rng']);random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        # best内容也保存在原子checkpoint中，恢复时纠正可能提前写出的best文件。
        atomic_save(out/'best_adapter.pt',state['best_adapter'])
        del state
        for filename in ['train.jsonl','validation.jsonl']:
            p=out/filename
            if p.exists():
                kept=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
                p.write_text(''.join(json.dumps(r)+'\n' for r in kept if r['step']<=completed))
    else:
        if c.get('dataset')=='gsm8k':
            metrics=math_evaluate(encoded['validation'],'before')
            best=[metrics['answer_correct_rate'],-metrics['loss']]
            dump(out/'baseline.json',metrics)
        else:
            best=evaluate(encoded['validation'])
            dump(out/'baseline.json',dict(validation_response_nll=best,test_response_nll=evaluate(encoded['test'])))
        save_adapter(model,out/'best_adapter.pt',base_identity)
    def checkpoint(step):
        atomic_save(latest,dict(identity=fingerprint,adapter=adapter_payload(model,base_identity),
            best_adapter=torch.load(out/'best_adapter.pt',map_location='cpu',weights_only=True),optimizer=optimizer.state_dict(),
            step=step,best=best,best_step=best_step,rng=torch.get_rng_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None))
    if completed==0:checkpoint(0)
    step=0
    for epoch in range(c['epochs']):
        order=list(encoded['train']);random.Random(c['seed']+epoch).shuffle(order)
        for start in range(0,len(order),c['effective_batch']):
            step+=1
            if step<=completed:continue
            tick=time.monotonic();value,norm=update(order[start:start+c['effective_batch']])
            record=dict(step=step,epoch=epoch+1,loss=value,grad_norm=norm,seconds=time.monotonic()-tick)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            dump(out/'progress.json',record);print('TRAIN',record,flush=True)
            end=start+c['effective_batch']>=len(order)
            if end:
                if c.get('dataset')=='gsm8k':
                    metrics=math_evaluate(encoded['validation'],f'validation_{step}')
                    nll=metrics['loss'];candidate=[metrics['answer_correct_rate'],-nll]
                    improved=candidate>best
                else:
                    nll=evaluate(encoded['validation']);candidate=nll;improved=nll<best
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,response_nll=nll))+'\n')
                if improved:
                    best=candidate;best_step=step;save_adapter(model,out/'best_adapter.pt',base_identity)
            if end or step%c['save_every']==0:checkpoint(step)
    save_adapter(model,out/'final_adapter.pt',base_identity)
    dump(out/'training_complete.json',dict(steps=step,best_step=best_step,selection_score=best,time=time.time()))


if __name__=='__main__':main()
