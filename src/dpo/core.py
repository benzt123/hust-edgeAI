"""DPO 核心练习：填写三个 TODO。阅读 README.md 后开始。

这不是可直接启动全量训练的入口；外层负责模型加载、分词、优化器和验证。
"""
import random

import torch
import torch.nn.functional as F


def build_preference_pairs(candidates, seed=42):
    """TODO 1：每题最多构建一对，返回字典列表。

    输入字段：question_id, prompt, response, answer_correct,
    format_ok, truncated, parsed。后四项必须为 bool。
    parsed 表示成功提取答案；旧数据缺失时须先重新评分补齐。

    步骤：
    1. 按 question_id 分组，同一 ID 的 prompt 不一致时抛 ValueError。
    2. 仅使用格式合格、未截断、成功解析的候选。
    3. 正确回答进 chosen 池，错误回答进 rejected 池；各池按文本去重。
       同一文本同时进入两池时抛 ValueError，提示评分冲突。
    4. 两池都非空才用 rng.choice 各抽一条；否则跳过该题。
    5. 输出 question_id, prompt, chosen, rejected；保持题目首次出现顺序。
       不修改输入字典；空输入返回 []；不补造回答。
    """
    rng = random.Random(seed)
    raise NotImplementedError("TODO 1：同题偏好配对")


def sequence_log_probs(logits, input_ids, response_mask):
    """TODO 2：返回每条回答的 log 概率之和，形状 [B]，保留梯度。

    logits: [B,T,V]，未经 causal shift 的模型原始输出。
    input_ids: [B,T]，真实 token ID（包括 PAD ID，不得用 -100）。
    response_mask: [B,T]，回答 token 为 1，提示和 padding 为 0。
    EOS 若属于有效回答也标为 1；mask 按 token 自身的位置定义。

    步骤：
    1. logits 去最后一个位置；input_ids 和 mask 去第一个位置。
    2. 对移位后的 logits.float() 在词表维做 F.log_softmax。
    3. 用 gather 取出目标 token 的 log 概率，得到 [B,T-1]。
    4. 乘移位后的 mask，沿时间维求和，得到 [B]。

    每条移位后的 mask 至少有一个有效 token，否则抛 ValueError。
    不按回答长度平均、不取负号、不调用 item 或 detach。
    """
    raise NotImplementedError("TODO 2：回答序列 log 概率")


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """TODO 3：输入均为 [B]，返回标量 batch 平均 loss。

    policy_margin = policy_chosen - policy_rejected
    reference_margin = ref_chosen - ref_rejected
    z = beta * (policy_margin - reference_margin)
    loss = mean(-F.logsigmoid(z))

    reference 两个输入先 detach，避免对参考模型反传。
    使用 logsigmoid，不要先 sigmoid 再 log（数值稳定性）。
    检查 beta > 0，输入为同形状、非空的一维 tensor。
    """
    raise NotImplementedError("TODO 3：DPO loss")


def freeze_reference(reference):
    """独立加载的参考模型只打分，不更新。"""
    reference.eval()
    reference.requires_grad_(False)
    return reference


def compute_batch_loss(policy, reference, batch, beta=0.1):
    """已连接的前向步骤；backward、梯度累积、step 由外层负责。

    batch['chosen'] 和 batch['rejected'] 各包含同设备上的：
    input_ids、attention_mask、response_mask，形状 [B,T]。
    每一对的 prompt 必须相同；右侧 padding；两边 T 可以不同。
    外层 collator 应剔除被截断的回答，不能将残缺样本视为完整回答。
    """
    if policy is reference:
        raise ValueError("policy 与 reference 必须是独立模型")

    def score(model, side):
        output = model(input_ids=side['input_ids'],
                       attention_mask=side['attention_mask'], use_cache=False)
        return sequence_log_probs(output.logits, side['input_ids'], side['response_mask'])

    reference.eval()
    with torch.no_grad():
        ref_chosen = score(reference, batch['chosen'])
        ref_rejected = score(reference, batch['rejected'])
    policy_chosen = score(policy, batch['chosen'])
    policy_rejected = score(policy, batch['rejected'])
    return dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
