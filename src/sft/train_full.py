"""全量 SFT：accumulation 保留为每次更新的样本数，micro-batch 可独立调整。"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
import torch
from core import response_loss, backward_microbatch, optimizer_update
from full_data import load_rows, make_split, encode_rows, digest
from full_validation import evaluate, forward_window, response_entropy


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('full_config.json')))
    parser.add_argument('--resume',type=Path,help='本程序生成的可信 latest.pt')
    parser.add_argument('--prepare-only',action='store_true',help='CPU 检查全部数据、长度和划分，不加载模型')
    parser.add_argument('--micro-batch-size',type=int,default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if not 1 <= args.micro_batch_size <= config['accumulation']:
        raise ValueError('micro-batch-size 必须介于1与有效batch之间')
    for key in ['epochs','accumulation','max_length','eval_every','save_every','entropy_every']:
        if config[key]<1: raise ValueError(f'{key} 必须为正')
    if config['generation_limit']<0 or config['generation_max_new_tokens']<1 or not 0<=config['warmup_ratio']<1:
        raise ValueError('无效的验证长度/数量或 warmup_ratio')
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    template = Path(__file__).with_name('user.txt').read_text(encoding='utf-8-sig').strip()
    if '{question}' not in template or not template.endswith('Assistant: <think>'):
        raise ValueError('user.txt 缺少 question 占位符或正确的 assistant 前缀')
    tokenizer = AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    capacity = AutoConfig.from_pretrained(config['model'],local_files_only=True).max_position_embeddings
    if config['max_length']>capacity: raise ValueError('max_length 超过模型上下文容量')
    rows = load_rows(config['data_dir'])
    train_rows, val_rows = make_split(rows,config['validation_fraction'],config['seed'])
    # 固定验证生成子集，包含哪些题保存到每次验证报告，不依据结果挑选。
    random.Random(config['seed']+1).shuffle(val_rows)
    train, train_stats = encode_rows(train_rows,tokenizer,template,config)
    validation, val_stats = encode_rows(val_rows,tokenizer,template,config)
    if args.micro_batch_size > 1 and any(len(x['windows']) != 1 for x in train):
        raise ValueError('批量训练暂不支持分段样本')
    for row in rows:
        from full_scoring import number
        if number(row['answer']) is None: raise ValueError(f"无法解析标准数值 {row['id']}")
    identity = digest(dict(data=rows,config=config,template=template,
        core=Path(__file__).with_name('core.py').read_text(encoding='utf-8')))
    report = dict(train=train_stats,validation=val_stats,total=len(rows),identity=identity)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if args.prepare_only:
        write_json(Path(args.config).with_suffix('.data_report.json'),report)
        return
    if not torch.cuda.is_available(): raise RuntimeError('正式训练需要 GPU；可用 --prepare-only 检查数据')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('此配置要求 BF16 GPU')
    model = AutoModelForCausalLM.from_pretrained(config['model'],torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda')
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    updates_per_epoch = math.ceil(len(train)/config['accumulation'])
    total_steps = updates_per_epoch*config['epochs']
    warmup = max(1,round(total_steps*config['warmup_ratio'])) if config['warmup_ratio'] else 0
    state = None
    start_epoch=start_offset=step=0
    best=(-1.,float('-inf'))
    if args.resume:
        # 仅加载自己保存的 checkpoint；torch pickle 不适合加载未知来源文件。
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['identity']!=identity: raise ValueError('数据/配置/模板/core.py 已变化，拒绝不一致的断点续训')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch,start_offset,step=state['epoch'],state['offset'],state['step']
        best=tuple(state['best'])
        run=args.resume.resolve().parent
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        del state
        # 丢弃日志中未进入 checkpoint 的更新，避免恢复后出现重复 step。
        for name in ['train.jsonl','validation.jsonl']:
            path=run/name
            if path.exists():
                records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                path.write_text(''.join(json.dumps(r)+'\n' for r in records if r['step']<=step),encoding='utf-8')
    else:
        run=Path(config['output_root'])/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'config.json',config)
        write_json(run/'data_report.json',report)
        write_json(run/'split.json',dict(train_ids=[r['id'] for r in train_rows],validation_ids=[r['id'] for r in val_rows]))
        write_json(run/'versions.json',{p:version(p) for p in ['torch','transformers','datasets']})
        (run/'user.txt').write_text(template,encoding='utf-8')
        for name in ['core.py','train_full.py','full_data.py','full_validation.py','full_scoring.py']:
            (run/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    writer=None
    with (run/'execution_changes.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time=datetime.now().isoformat(),resume_step=step,
            micro_batch_size=args.micro_batch_size,effective_batch_size=config['accumulation']))+'\n')
    for name in ['train_full.py','batching.py']:
        source=Path(__file__).with_name(name)
        if source.exists(): (run/f'active_{name}').write_bytes(source.read_bytes())
    if config['tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(run/'tensorboard'),purge_step=step+1 if args.resume else None)

    def validate():
        nonlocal best
        destination=run/f'validation_{step:07d}'
        if destination.exists():
            # 中断可能发生在验证完成、保存 checkpoint 之前，保留未提交的诊断文件。
            destination.rename(run/(destination.name+'_interrupted_'+datetime.now().strftime('%H%M%S_%f')))
        metrics=evaluate(model,tokenizer,validation,config,destination)
        with (run/'validation.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(step=step,**metrics))+'\n')
        if writer:
            for key in ['loss','entropy','answer_correct_rate','format_ok_rate','joint_correct_rate','truncated_rate']:
                writer.add_scalar('validation/'+key,metrics[key],step)
        # 优先看答案正确率，同分选验证 loss 更低的 checkpoint。
        candidate=(metrics['answer_correct_rate'],-metrics['loss'])
        if candidate>best:
            best=candidate
            model.save_pretrained(run/'best_model')
            tokenizer.save_pretrained(run/'best_model')
            write_json(run/'best.json',dict(step=step,metrics=metrics))
        print('VALIDATION',step,metrics,flush=True)

    def checkpoint(epoch,offset):
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,offset=offset,
            step=step,best=best,identity=identity,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate())
        temp=run/'latest.pt.tmp'
        torch.save(payload,temp)
        os.replace(temp,run/'latest.pt')

    try:
        if not args.resume: validate()  # 同一协议下的训练前 baseline。
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch,config['epochs']):
            order=list(range(len(train)))
            random.Random(config['seed']+epoch).shuffle(order)
            first=start_offset if epoch==start_epoch else 0
            for start in range(first,len(order),config['accumulation']):
                update_started = time.monotonic()
                group=order[start:start+config['accumulation']]
                if warmup and step<warmup: scale=(step+1)/warmup
                else: scale=max(0.,(total_steps-step)/max(1,total_steps-warmup))
                for pg in optimizer.param_groups: pg['lr']=config['learning_rate']*scale
                torch.cuda.reset_peak_memory_stats()
                loss_sum=entropy_sum=0.
                log_entropy=(step+1)%config['entropy_every']==0
                if args.micro_batch_size > 1:
                    from batching import backward_group
                    loss_sum,entropy_sum=backward_group(model,[train[i] for i in group],
                        tokenizer.eos_token_id,args.micro_batch_size,log_entropy)
                for index in (group if args.micro_batch_size == 1 else []):
                    item=train[index]
                    for window in item['windows']:
                        logits,labels,mask=forward_window(model,window,'cuda')
                        loss=response_loss(logits.float(),labels,mask)
                        # 分段先按该段的回答 token 占比加权，再按原始样本数平均。
                        weight=sum(window[1])/item['response_tokens']
                        if not torch.isfinite(loss): raise RuntimeError('非有限 loss，停止')
                        loss_sum+=loss.item()*weight
                        if log_entropy:
                            with torch.no_grad(): entropy_sum+=response_entropy(logits,mask)*weight
                        backward_microbatch(loss*weight,len(group))
                        del logits,loss
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('非有限梯度，停止，未更新参数')
                grad_norm=math.sqrt(sum(p.grad.detach().float().norm().item()**2 for p in model.parameters() if p.grad is not None))
                optimizer_update(model,optimizer,config['max_grad_norm'])
                step+=1
                record=dict(step=step,epoch=epoch+1,samples=len(group),loss=loss_sum/len(group),
                    update_seconds=time.monotonic()-update_started,total_steps=total_steps,
                    micro_batch_size=args.micro_batch_size,
                    lr=optimizer.param_groups[0]['lr'],grad_norm=grad_norm,
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                if log_entropy: record['entropy']=entropy_sum/len(group)
                with (run/'train.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
                if writer:
                    for key,value in record.items(): writer.add_scalar('train/'+key,value,step)
                print(record,flush=True)
                end=min(start+len(group),len(order))
                if step%config['eval_every']==0 or end==len(order): validate()
                if step%config['save_every']==0 or step%config['eval_every']==0 or end==len(order):
                    checkpoint(epoch+1 if end==len(order) else epoch,0 if end==len(order) else end)
        model.save_pretrained(run/'final_model')
        tokenizer.save_pretrained(run/'final_model')
        print('COMPLETE',run,flush=True)
    finally:
        if writer: writer.close()


if __name__=='__main__': main()
