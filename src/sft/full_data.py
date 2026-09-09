"""GSM8K 数据、无泄漏划分，以及不丢回答 token 的可选分段。"""
import hashlib
import json
import math
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_rows(data_dir, split='train'):
    from datasets import load_dataset
    files = sorted(Path(data_dir).glob(f'{split}-*.parquet'))
    if not files:
        raise FileNotFoundError(f'缺少 {data_dir}/{split}-*.parquet，请先下载真实数据')
    for file in files:
        with file.open('rb') as stream:
            if stream.read(4) != b'PAR1':
                raise ValueError(f'{file} 不是真实 Parquet，可能还是 Git LFS 指针')
    table = load_dataset('parquet', data_files=[str(f) for f in files], split='train')
    rows = []
    for i, row in enumerate(table):
        if '####' not in row['answer']:
            raise ValueError(f'{split}/{i} 缺少 ####，停止而不是静默丢样本')
        reasoning, answer = row['answer'].rsplit('####', 1)
        if not reasoning.strip() or not answer.strip():
            raise ValueError(f'{split}/{i} 解答为空')
        rows.append(dict(id=f'{split}/{i}', question=row['question'],
                         reasoning=reasoning.strip(), answer=answer.strip()))
    if not rows:
        raise ValueError('数据集为空')
    return rows


def make_split(rows, fraction, seed):
    # 同一问题即使重复出现，也必须全部在划分的同一侧。
    keys = sorted({r['question'].strip() for r in rows})
    if len(keys) < 2 or not 0 < fraction < 1:
        raise ValueError('至少需要两个不同问题，validation_fraction 必须在 0 和 1 之间')
    random.Random(seed).shuffle(keys)
    n = min(len(keys)-1, max(1, math.ceil(len(keys)*fraction)))
    val_keys = set(keys[:n])
    return ([r for r in rows if r['question'].strip() not in val_keys],
            [r for r in rows if r['question'].strip() in val_keys])


def build_windows(ids, response_start, max_length, policy='error', overlap=512):
    if not 1 <= response_start < len(ids) or max_length < 2:
        raise ValueError('无有效回答或长度预算无效')
    if len(ids) <= max_length:
        return [(ids, [0]*response_start + [1]*(len(ids)-response_start))]
    if policy == 'error':
        raise ValueError(f'样本长 {len(ids)} > max_length={max_length}；增大预算或显式选择 long_policy=window')
    if policy != 'window' or not 1 <= overlap < max_length:
        raise ValueError('window 模式要求 1 <= overlap < max_length')
    # 每个回答 token 只监督一次；后续窗口保留最近 overlap 个上下文 token。
    windows = []
    next_target = response_start
    while next_target < len(ids):
        start = max(0, next_target-overlap)
        end = min(len(ids), start+max_length)
        windows.append((ids[start:end], [0]*(next_target-start) + [1]*(end-next_target)))
        next_target = end
    return windows


def encode_rows(rows, tokenizer, template, config):
    encoded, lengths, long_ids = [], [], []
    for row in rows:
        prompt = template.replace('{question}', row['question'])
        response = row['reasoning'] + '</think> <answer>' + row['answer'] + '</answer>'
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffix = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = prefix + suffix
        lengths.append(len(ids))
        if len(ids) > config['max_length']:
            long_ids.append(row['id'])
        windows = build_windows(ids, len(prefix), config['max_length'], config['long_policy'], config['overlap'])
        assert sum(sum(mask) for _, mask in windows) == len(suffix)
        encoded.append(dict(row=row, prompt=prompt, windows=windows, response_tokens=len(suffix)))
    return encoded, dict(samples=len(encoded), max_tokens=max(lengths), long_ids=long_ids,
                        window_count=sum(len(x['windows']) for x in encoded), dropped_samples=0,
                        context_is_truncated=bool(long_ids and config['long_policy']=='window'))
