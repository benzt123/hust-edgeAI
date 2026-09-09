"""右侧补齐批量输入；mask 始终与预测目标对齐。"""
import torch
from core import shift_batch, response_loss
from full_validation import response_entropy


def forward_batch(model, items, pad_id, device):
    if any(len(item['windows']) != 1 for item in items):
        raise ValueError('批量路径仅支持未分段样本')
    windows = [item['windows'][0] for item in items]
    length = max(len(ids) for ids, _ in windows)
    ids = torch.full((len(items), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    attention = torch.zeros_like(ids)
    for i, (tokens, flags) in enumerate(windows):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, :len(tokens)] = torch.tensor(flags, device=device)
        attention[i, :len(tokens)] = 1
    inputs, labels, mask = shift_batch(ids, mask)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        logits = model(input_ids=inputs, attention_mask=attention[:, :-1]).logits
    return logits, labels, mask


def backward_group(model, items, pad_id, micro_batch_size, log_entropy=False):
    loss_sum = entropy_sum = 0.
    for start in range(0, len(items), micro_batch_size):
        chunk = items[start:start+micro_batch_size]
        logits, labels, mask = forward_batch(model, chunk, pad_id, 'cuda')
        loss = response_loss(logits.float(), labels, mask)
        if not torch.isfinite(loss):
            raise RuntimeError('非有限 loss')
        loss_sum += loss.item()*len(chunk)
        if log_entropy:
            with torch.no_grad():
                for i in range(len(chunk)):
                    entropy_sum += response_entropy(logits[i:i+1], mask[i:i+1])
        # 按实际样本数加权，包括 epoch 末不足8条的累积组。
        (loss*(len(chunk)/len(items))).backward()
    return loss_sum, entropy_sum
