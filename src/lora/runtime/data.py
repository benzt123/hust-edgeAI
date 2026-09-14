"""物理文本问答数据接口：固定无重复划分；不假装支持未解析的图像题。"""
import json
import random


def normalize(text):return ' '.join(text.split())


def read_rows(path):
    rows=[];ids=set();questions=set()
    with open(path,encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ['id','question','answer']):
                raise ValueError('每条需包含非空字符串id/question/answer')
            if row.get('image') or row.get('images'):raise ValueError('仅支持文本物理题，请先处理图像依赖')
            q=normalize(row['question'])
            if row['id'] in ids or q in questions:raise ValueError('重复ID或题目，先去重再划分')
            ids.add(row['id']);questions.add(q);rows.append(row)
    if len(rows)<10:raise ValueError('正式数据至少10条；小模型检查不使用此入口')
    return rows


def split_rows(rows,seed=42):
    if any('split' in r for r in rows):
        if any(r.get('split') not in ('train','validation','test') for r in rows):
            raise ValueError('固定划分标签缺失或非法')
        result={k:[r for r in rows if r['split']==k] for k in ('train','validation','test')}
        if any(not v for v in result.values()):raise ValueError('固定划分不能为空')
        return result
    ordered=sorted(rows,key=lambda r:r['id']);random.Random(seed).shuffle(ordered)
    n=max(1,round(len(rows)*.1))
    return dict(test=ordered[:n],validation=ordered[n:2*n],train=ordered[2*n:])


def encode_rows(rows,tokenizer,max_length):
    result=[]
    for row in rows:
        # 提示模板、回答边界与EOS全程一致，单位/公式保留原文本。
        prompt=row.get('prompt','User: Solve this physics problem. Explain your reasoning and include units where applicable.\n'+row['question']+'\nAssistant: ')
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        suffix=tokenizer.encode(row['answer'],add_special_tokens=False)
        if not suffix or suffix[-1]!=tokenizer.eos_token_id:suffix.append(tokenizer.eos_token_id)
        if not prefix or len(prefix)+len(suffix)>max_length:raise ValueError(f"样本超长或无效：{row['id']}；未静默截断")
        result.append(dict(id=row['id'],prompt=prompt,answer=row['answer'],gold=row.get('gold'),ids=prefix+suffix,
                           mask=[0]*len(prefix)+[1]*len(suffix)))
    return result
