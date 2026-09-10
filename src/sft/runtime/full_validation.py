"""与主程序共用的完整验证：teacher-forced loss + 独立生成准确率。"""
import json
import random
from contextlib import nullcontext
from pathlib import Path
import torch
from core import shift_batch, response_loss
from full_scoring import score, VERSION, number


def forward_window(model, window, device):
    ids, mask = window
    x, labels, mask = shift_batch(torch.tensor([ids],device=device), torch.tensor([mask],device=device))
    # FP32 参数及 AdamW 状态，矩阵运算使用 BF16 autocast。
    with torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext():
        logits = model(input_ids=x).logits
    return logits, labels, mask


def response_entropy(logits, mask):
    # 分块处理词表分布，避免为整段序列额外保留多个 float32 张量。
    selected = logits[mask.bool()]
    total = 0.0
    for block in selected.split(32):
        logp = block.float().log_softmax(-1)
        total += (-(logp.exp()*logp).sum(-1)).sum().item()
    return total / len(selected)


@torch.no_grad()
def evaluate(model, tokenizer, encoded, config, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    was_training = model.training
    py_state = random.getstate()
    device = next(model.parameters()).device
    totals = dict(answer_correct=0, format_ok=0, joint_correct=0, parsed=0, truncated=0)
    losses, entropies = [], []
    # 验证不改变训练的随机序列；每轮生成使用同样固定的 seed。
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config['seed'])
            for item_index, item in enumerate(encoded):
                value = entropy = 0.0
                for window in item['windows']:
                    logits, labels, mask = forward_window(model, window, device)
                    weight = sum(window[1]) / item['response_tokens']
                    value += response_loss(logits.float(), labels, mask).item()*weight
                    entropy += response_entropy(logits, mask)*weight
                losses.append(value)
                entropies.append(entropy)
                if (item_index+1)%100==0:
                    print('VALIDATION_LOSS_PROGRESS',item_index+1,len(encoded),flush=True)
            limit = config['generation_limit'] or len(encoded)
            chosen = encoded[:limit]
            with (out_dir/'predictions.jsonl').open('w',encoding='utf-8') as stream:
                for item_index, item in enumerate(chosen):
                    tokens = tokenizer.encode(item['prompt'],add_special_tokens=False)
                    if len(tokens)+config['generation_max_new_tokens'] > int(model.config.max_position_embeddings):
                        raise ValueError('验证问题加生成预算超过模型上下文容量，未静默截断')
                    inputs = torch.tensor([tokens],device=device)
                    kwargs = dict(max_new_tokens=config['generation_max_new_tokens'], do_sample=True,
                        temperature=config['generation_temperature'],top_p=config['generation_top_p'],
                        top_k=0,repetition_penalty=1.0,eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,use_cache=True,
                        stop_strings=['</answer>'],tokenizer=tokenizer)
                    with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
                        output = model.generate(inputs,attention_mask=torch.ones_like(inputs),**kwargs)[0,len(tokens):]
                    response = tokenizer.decode(output,skip_special_tokens=True)
                    metrics = score(response,item['row']['answer'])
                    truncated = len(output)==config['generation_max_new_tokens'] and '</answer>' not in response and int(output[-1])!=tokenizer.eos_token_id
                    for key in ['answer_correct','format_ok','joint_correct','parsed']:
                        totals[key] += int(metrics[key])
                    totals['truncated'] += int(truncated)
                    stream.write(json.dumps(dict(id=item['row']['id'],prompt=item['prompt'],
                        gold=item['row']['answer'],response=response,truncated=truncated,**metrics),ensure_ascii=False)+'\n')
                    stream.flush()
                    print('VALIDATION_GENERATION_PROGRESS',item_index+1,len(chosen),flush=True)
            n = len(chosen)
            result = dict(loss=sum(losses)/len(losses),entropy=sum(entropies)/len(entropies),
                loss_samples=len(encoded),generation_samples=n,grader=VERSION,
                generation_ids=[x['row']['id'] for x in chosen],
                **{key+'_rate':value/n for key,value in totals.items()})
            (out_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            return result
    finally:
        random.setstate(py_state)
        model.train(was_training)
