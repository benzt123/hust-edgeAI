"""Self-Play 配套：回调接采样器，更新直接复用学生 GRPO。

采样器必须同步当前 policy，保留生成 token 与更新前 old 分数。
此模块不自行连接服务器或加载大模型。
"""
import importlib.util
from pathlib import Path
import math


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
student = load_module('selfplay_student', ROOT/'self_play/core.py')
grpo = load_module('grpo_student', ROOT/'grpo/core.py')


def prepare_round(generations, solve, grade, answer_supported, template,
                  group_size=8, blocked_problem_keys=()):
    """solve(requests, group_size) 返回完整候选，每条额外保留 ids/mask/old。

    requests 每条只含 question_id、prompt，不包含 proposer 的参考答案。
    ids/mask 长度T；old长度T-1，必须在更新前按实际生成策略计算。
    ids是去除padding后的原始token，禁止解码后重新分词或追加未采样EOS。
    返回记录顺序为题目优先、candidate_index升序。
    """
    problems = student.prepare_problems(generations, answer_supported, blocked_problem_keys)
    stats = dict(generated=len(generations), accepted=len(problems),
                 filtered=len(generations)-len(problems), responses=0,
                 matched_fraction=0., informative_groups=0)
    if not problems:
        return dict(problems=[], records=[], stats=stats)
    requests = [dict(question_id=p['question_id'],
                     prompt=student.format_solve_prompt(p['problem'],template)) for p in problems]
    candidates = solve(requests, group_size)
    groups = student.build_reward_groups(problems,candidates,grade,group_size)
    by_id = {(x['question_id'],x['candidate_index']):x for x in candidates}
    records=[]
    for group in groups:
        rewards=group['rewards']
        mean=sum(rewards)/len(rewards)
        stats['informative_groups'] += int(min(rewards)!=max(rewards))
        for index,reward in enumerate(rewards):
            candidate=by_id[(group['question_id'],index)]
            ids,mask,old=candidate['ids'],candidate['mask'],candidate['old']
            if len(ids)<2 or len(mask)!=len(ids) or len(old)!=len(ids)-1:
                raise ValueError('生成 token、mask 与 old 分数长度不一致')
            if any(type(x) is not int or x<0 for x in ids):
                raise ValueError('ids 必须是非负整数 token ID')
            if any(x not in (0,1) for x in mask) or not sum(mask[1:]):
                raise ValueError('回答 mask 无效')
            if not all(math.isfinite(x) for x in old):
                raise ValueError('old 分数存在非有限值')
            # 新字典及列表，避免修改采样器的原记录。
            records.append(dict(question_id=group['question_id'],candidate_index=index,
                response=candidate['response'],truncated=candidate['truncated'],
                ids=list(ids),mask=list(mask),old=list(old),reward=reward,
                advantage=reward-mean))
    stats['responses']=len(records)
    stats['matched_fraction']=sum(x['reward'] for x in records)/len(records)
    return dict(problems=problems,records=records,stats=stats)


def train_records(policy, optimizer, records, pad_token_id, epochs=2,
                  effective_batch=32, micro_batch=1, clip_eps=.2, seed=42):
    """真实 PyTorch 更新，支持CPU检查与CUDA BF16 autocast。

    records只来自同一rollout；两遍更新不刷新old。advantage按完整组预计算。
    CUDA显存与吞吐需在部署时实测；本函数不负责恢复断点。
    """
    import random
    import torch
    from contextlib import nullcontext
    if any(type(x) is not int or x<1 for x in [epochs,effective_batch,micro_batch]):
        raise ValueError('epoch和batch必须为正整数')
    if not records or not any(x['advantage']!=0 for x in records):
        return []  # 全零优势不执行step，避免Adam历史动量单独改变参数。
    device=next(policy.parameters()).device
    updates=[]
    was_training=policy.training
    policy.train()
    try:
        for epoch in range(epochs):
            order=list(range(len(records)))
            random.Random(seed+epoch).shuffle(order)
            for start in range(0,len(order),effective_batch):
                indexes=order[start:start+effective_batch]
                optimizer.zero_grad(set_to_none=True)
                loss_sum=0.
                for offset in range(0,len(indexes),micro_batch):
                    items=[records[i] for i in indexes[offset:offset+micro_batch]]
                    width=max(len(x['ids']) for x in items)
                    ids=[];masks=[];attention=[];old=[]
                    for x in items:
                        pad=width-len(x['ids'])
                        ids.append(x['ids']+[pad_token_id]*pad)
                        masks.append(x['mask']+[0]*pad)
                        attention.append([1]*len(x['ids'])+[0]*pad)
                        old.append(x['old']+[0.]*pad)
                    batch={k:torch.tensor(v,device=device) for k,v in
                           [('input_ids',ids),('response_mask',masks),('attention_mask',attention),
                            ('old_log_probs',old),('advantages',[x['advantage'] for x in items])]}
                    ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
                    with ctx:
                        loss=grpo.compute_batch_loss(policy,batch,clip_eps)
                    if not torch.isfinite(loss):raise ValueError('非有限训练loss')
                    (loss*len(items)/len(indexes)).backward()
                    loss_sum+=loss.item()*len(items)
                norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates.append(dict(epoch=epoch+1,loss=loss_sum/len(indexes),grad_norm=norm.item()))
    finally:
        policy.train(was_training)
    return updates


def self_play_round(generate, solve, grade, answer_supported, template, train,
                    n_problems=32, group_size=8, blocked_problem_keys=()):
    """一次完整编排。回调generate(prompts)和solve需使用当前policy。

    train(records) 可绑定train_records；结果必须由外层保存为可审查记录。
    空数据或全零优势时跳过更新；匹配率不是独立数学正确率。
    """
    if type(n_problems) is not int or n_problems<1:
        raise ValueError('n_problems必须为正整数')
    if type(group_size) is not int or group_size<2:
        raise ValueError('group_size必须至少为2')
    generations=generate([student.PROBLEM_GEN_PROMPT]*n_problems)
    if len(generations)!=n_problems:raise ValueError('出题采样返回数量不完整')
    result=prepare_round(generations,solve,grade,answer_supported,template,group_size,blocked_problem_keys)
    result['generations']=generations
    result['updates']=train(result['records']) if result['stats']['informative_groups'] else []
    return result
