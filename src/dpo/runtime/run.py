"""手写 DPO 配套：偏好数据、参考缓存、显存试跑、训练及独立评估。

核心概率和损失直接调用学生 core.py；无 Trainer。阶段分别运行以释放 GPU。
"""
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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('student_dpo', HERE.parent / 'core.py')
student = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'smoke', 'cache', 'train', 'test'])
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    args = parser.parse_args()
    c = read(args.config)
    out = Path(c['output'])
    source = Path(c['source_model'])
    sft = Path(c['sft_run'])
    sys.path.insert(0, c['sft_code'])
    from full_data import load_rows, encode_rows
    import numeric_grader
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(c['seed'])
    random.seed(c['seed'])
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    original = read(sft / 'config.json')
    split = read(sft / 'split.json')
    template = (sft / 'user.txt').read_text()
    # Snapshot all inputs used by the cache and training; never reuse across identities.
    identity_files = [HERE / 'run.py', HERE / 'numeric_grader.py', HERE.parent / 'core.py',
                      sft / 'split.json', sft / 'user.txt', sft / 'config.json']
    chunks = sorted(Path(c['candidates']).glob('*.json'))
    if not chunks:
        raise ValueError('服务器原始候选文件缺失')
    identity_files += chunks
    identity_files += sorted(source.glob('*.json')) + sorted(source.glob('*.safetensors'))
    identity = dict(config=c, files={str(p): sha(p) for p in identity_files})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.phase == 'prepare':
        out.mkdir(parents=True, exist_ok=False)
        rows = {r['id']: r for r in load_rows(original['data_dir'])}
        train_ids = set(split['train_ids'])
        val_ids = set(split['validation_ids'])
        assert not train_ids & val_ids
        candidates = []
        for chunk in chunks:
            for row in read(chunk):
                qid = row['question_id']
                if qid not in train_ids:
                    raise ValueError('偏好候选含非训练题')
                if row['prompt'] != template.replace('{question}', rows[qid]['question']):
                    raise ValueError('候选 prompt 与原始训练模板不一致')
                if row['gold'] != rows[qid]['answer']:
                    raise ValueError('候选 gold 不一致')
                row.update(numeric_grader.score(row['response'], row['gold']))
                candidates.append(row)
        pairs = student.build_preference_pairs(candidates, seed=c['seed'])
        encoded = []
        for pair in pairs:
            prefix = tokenizer.encode(pair['prompt'], add_special_tokens=False)
            item = dict(question_id=pair['question_id'])
            for side in ['chosen', 'rejected']:
                suffix = tokenizer.encode(pair[side], add_special_tokens=False)
                if not suffix or suffix[-1] != tokenizer.eos_token_id:
                    suffix.append(tokenizer.eos_token_id)
                ids = prefix + suffix
                if not prefix or len(ids) > c['max_length']:
                    raise ValueError('空提示或超长样本，停止而非静默截断')
                item[side] = dict(input_ids=ids, response_mask=[0]*len(prefix)+[1]*len(suffix))
            encoded.append(item)
        if not encoded:
            raise ValueError('没有偏好对')
        dump(out / 'pairs.json', pairs)
        dump(out / 'encoded.json', encoded)
        dump(out / 'identity.json', dict(sha256=fingerprint, **identity))
        dump(out / 'config.json', c)
        (out / 'snapshot_core.py').write_bytes((HERE.parent / 'core.py').read_bytes())
        (out / 'snapshot_run.py').write_bytes(Path(__file__).read_bytes())
        lengths = [len(item[s]['input_ids']) for item in encoded for s in ['chosen', 'rejected']]
        stats = dict(candidates=len(candidates), pairs=len(pairs), train_questions=len(train_ids),
                     question_coverage=len(pairs)/len(train_ids), max_tokens=max(lengths),
                     mean_tokens=sum(lengths)/len(lengths), source='SFT offline candidates, unified numeric v2',
                     policy_reference_checkpoint=str(source), dropped_pairs=0)
        dump(out / 'data_stats.json', stats)
        print('PREPARED', stats, flush=True)
        return
    if read(out / 'identity.json')['sha256'] != fingerprint:
        raise ValueError('代码、数据、模型或配置改变，拒绝复用缓存或断点')
    dump(out / 'status.json', dict(phase=args.phase, time=time.time()))
    items = read(out / 'encoded.json')

    def load_model(path=source):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
            attn_implementation='sdpa', local_files_only=True).to('cuda')
        model.config.use_cache = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        return model

    def batch_side(group, side):
        length = max(len(item[side]['input_ids']) for item in group)
        ids, mask, attention = [], [], []
        for item in group:
            data = item[side]
            pad = length - len(data['input_ids'])
            ids.append(data['input_ids'] + [tokenizer.eos_token_id]*pad)
            mask.append(data['response_mask'] + [0]*pad)
            attention.append([1]*len(data['input_ids']) + [0]*pad)
        return {key: torch.tensor(value, device='cuda') for key, value in
                [('input_ids', ids), ('response_mask', mask), ('attention_mask', attention)]}

    def score(model, group, side):
        batch = batch_side(group, side)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           use_cache=False).logits
        return student.sequence_log_probs(logits, batch['input_ids'], batch['response_mask'])

    if args.phase == 'cache':
        model = student.freeze_reference(load_model())
        path = out / 'reference.json'
        cached = read(path) if path.exists() else dict(identity=fingerprint, scores=[])
        if cached['identity'] != fingerprint:
            raise ValueError('参考缓存身份不符')
        values = cached['scores']
        tick = time.monotonic()
        with torch.no_grad():
            for i in range(len(values), len(items)):
                w = score(model, [items[i]], 'chosen').item()
                l = score(model, [items[i]], 'rejected').item()
                if not math.isfinite(w+l):
                    raise ValueError('参考分数非有限')
                values.append([w, l])
                if (i+1) % 25 == 0 or i+1 == len(items):
                    dump(path, cached)
                    print('CACHE', i+1, len(items), 'seconds', time.monotonic()-tick, flush=True)
        return

    if args.phase == 'smoke':
        # Include the longest pair to exercise the largest activation allocation.
        ordered = sorted(items, key=lambda x: sum(len(x[s]['input_ids']) for s in ['chosen','rejected']), reverse=True)
        model = load_model()
        model.eval()
        selected = ordered[:2] + items[:2]
        with torch.no_grad():
            refs = [(score(model, [x], 'chosen').detach(), score(model, [x], 'rejected').detach()) for x in selected]
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], foreach=False)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        losses = []
        for item, (rw, rl) in zip(selected, refs):
            w, l = score(model, [item], 'chosen'), score(model, [item], 'rejected')
            loss = student.dpo_loss(w, l, rw, rl, c['beta'])
            losses.append(loss.item())
            (loss/len(selected)).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if norm.item() == 0 or any(abs(x-math.log(2)) > 0.002 for x in losses):
            raise RuntimeError('初始 loss 或梯度异常')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            after = [student.dpo_loss(score(model,[x],'chosen'), score(model,[x],'rejected'),rw,rl,c['beta']).item()
                     for x,(rw,rl) in zip(selected,refs)]
        if sum(after) >= sum(losses):
            raise RuntimeError('微型更新后同批 DPO loss 未下降')
        result = dict(initial_losses=losses, after_losses=after, grad_norm=norm.item(),
                      seconds=time.monotonic()-tick, allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      note='Disposable smoke update; formal training reloads unchanged RSFT weights.')
        dump(out / 'smoke.json', result)
        print('SMOKE_OK', result, flush=True)
        return

    import full_validation
    full_validation.score = numeric_grader.score
    full_validation.VERSION = numeric_grader.VERSION
    evaluate = full_validation.evaluate
    mapping = {r['id']: r for r in load_rows(original['data_dir'])}
    ec = original | dict(generation_limit=c['validation_questions'])
    val_rows = [mapping[i] for i in split['validation_ids'][:c['validation_questions']]]
    validation, _ = encode_rows(val_rows, tokenizer, template, ec)

    def evaluate_to(model, encoded, config, name):
        path = out / name
        if (path / 'metrics.json').exists():
            return read(path / 'metrics.json')
        if path.exists():
            path.rename(out / (name + '_interrupted_' + str(time.time_ns())))
        # Previous generation settings, with the audited unified numeric grader.
        return evaluate(model, tokenizer, encoded, config, path)

    if args.phase == 'test':
        if not (out / 'training_complete.json').exists():
            raise RuntimeError('训练尚未完成')
        model = load_model(out / 'best_model')
        ec['generation_limit'] = 0
        test, _ = encode_rows(load_rows(original['data_dir'], 'test'), tokenizer, template, ec)
        metrics = evaluate_to(model, test, ec, 'test_best')
        dump(out / 'complete.json', dict(metrics=metrics, time=time.time()))
        dump(out / 'status.json', dict(phase='complete', time=time.time()))
        print('COMPLETE', metrics, flush=True)
        return

    if (out / 'training_complete.json').exists():
        return
    refs = read(out / 'reference.json')
    if refs['identity'] != fingerprint or len(refs['scores']) != len(items):
        raise ValueError('参考缓存不完整或不匹配')
    model = load_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=0., foreach=False)
    total = math.ceil(len(items)/c['effective_batch'])*c['epochs']
    warmup = max(1, math.ceil(total*c['warmup_ratio']))
    completed = 0
    best = (-1., float('-inf'))
    latest = out / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['identity'] != fingerprint:
            raise ValueError('断点身份不符')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        completed, best = state['step'], tuple(state['best'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        torch.cuda.empty_cache()
        for name in ['train.jsonl','validation.jsonl']:
            path = out / name
            if path.exists():
                records = [json.loads(x) for x in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(x)+'\n' for x in records if x['step']<=completed))
        for path in out.glob('validation_*'):
            if path.is_dir() and path.name[11:].isdigit() and int(path.name[11:])>completed:
                path.rename(out / (path.name+'_interrupted_'+str(time.time_ns())))
    else:
        metrics = evaluate_to(model, validation, ec, 'before')
        best = (metrics['answer_correct_rate'], -metrics['loss'])
        model.save_pretrained(out / 'best_model')
        tokenizer.save_pretrained(out / 'best_model')
        dump(out / 'best.json', dict(step=0, metrics=metrics))

    def save(step):
        tmp = out / 'latest.pt.tmp'
        torch.save(dict(identity=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        step=step, best=best, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), tmp)
        os.replace(tmp, latest)

    if completed == 0:
        save(0)
    step = 0
    model.train()
    for epoch in range(c['epochs']):
        order = list(range(len(items)))
        random.Random(c['seed']+epoch).shuffle(order)
        for offset in range(0,len(order),c['effective_batch']):
            step += 1
            if step <= completed:
                continue
            indexes = order[offset:offset+c['effective_batch']]
            lr = c['learning_rate']*min(1.,step/warmup)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            loss_sum = margin_sum = correct = 0.
            for start in range(0,len(indexes),c['micro_batch']):
                batch_ids = indexes[start:start+c['micro_batch']]
                batch = [items[i] for i in batch_ids]
                rw, rl = torch.tensor([refs['scores'][i] for i in batch_ids], device='cuda').unbind(1)
                w, l = score(model,batch,'chosen'), score(model,batch,'rejected')
                loss = student.dpo_loss(w,l,rw,rl,c['beta'])
                if not torch.isfinite(loss):
                    raise RuntimeError('非有限 loss')
                (loss*len(batch_ids)/len(indexes)).backward()
                margin = ((w-l)-(rw-rl)).detach()
                loss_sum += loss.item()*len(batch_ids)
                margin_sum += margin.sum().item()
                correct += (margin>0).sum().item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = dict(step=step,total=total,epoch=epoch+1,loss=loss_sum/len(indexes),
                          relative_margin=margin_sum/len(indexes),preference_accuracy=correct/len(indexes),
                          lr=lr,grad_norm=norm.item(),seconds=time.monotonic()-tick,
                          allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            with (out / 'train.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            dump(out / 'progress.json', record | dict(time=time.time()))
            print('TRAIN',record,flush=True)
            if step % c['eval_every'] == 0 or step == total:
                torch.cuda.empty_cache()
                metrics = evaluate_to(model,validation,ec,f'validation_{step:06d}')
                with (out / 'validation.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step,**metrics))+'\n')
                quality = (metrics['answer_correct_rate'],-metrics['loss'])
                if quality > best:
                    best = quality
                    model.save_pretrained(out / 'best_model')
                    tokenizer.save_pretrained(out / 'best_model')
                    dump(out / 'best.json',dict(step=step,metrics=metrics))
                save(step)
            elif step % c['save_every'] == 0:
                save(step)
    model.save_pretrained(out / 'final_model')
    tokenizer.save_pretrained(out / 'final_model')
    dump(out / 'training_complete.json',dict(steps=total,pairs=len(items),time=time.time()))
    print('TRAIN_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
