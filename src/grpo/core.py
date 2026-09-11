"""GRPO 核心练习：按题目文档“实现细节注意事项”填写四个 TODO。

本版：组内奖励减均值、不除标准差；token 求和、回答平均；不加 KL。
不加载模型，不采样，不启动训练。学习步骤与例子见 README.md。
"""
import torch
import torch.nn.functional as F


def group_advantages(rewards):
    """TODO 1：计算同一道题内部的相对优势。

    输入 rewards：[B,G]，B 道题，每题 G>=2 个回答；有限浮点 tensor。
    输出：[B,G]，与输入同设备，但不参与反向传播。

    步骤：
    1. 检查输入为非空二维 tensor，G>=2，数值有限。
    2. rewards.detach()，奖励是固定评分，不对评分器求导。
    3. 沿 dim=1 求均值，keepdim=True，得到 [B,1]。
    4. 每条奖励减去该题均值，返回 [B,G]。不要除标准差！

    示例：[1,1,0,0] -> [0.5,0.5,-0.5,-0.5]。
    全对或全错 -> 全0，不能人为给这种组制造正负优势。
    无效输入抛 ValueError，不修改原始 rewards。
    """
    if rewards.ndim != 2:
        raise ValueError("rewards 必须是 [B, G] 二维 tensor")
    if rewards.shape[0] == 0 or rewards.shape[1] < 2:
        raise ValueError("至少需要一道题，每题至少两个回答")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards 中存在 NaN 或 Inf")
    rewards_detached = rewards.detach()
    group_mean = rewards_detached.mean(dim=1, keepdim=True)
    advantages = rewards_detached - group_mean
    return advantages


def response_token_log_probs(logits, input_ids, response_mask):
    """TODO 2：取每个实际回答 token 的 log 概率，保留 token 维。

    输入 logits：[N,T,V]，未经 shift 的模型输出；N=B*G。
    input_ids：[N,T]，真实 token ID，不允许用 -100 代替 PAD。
    response_mask：[N,T]，回答及有效 EOS 为1，prompt/PAD为0。
    输入已由外层校验形状与合法 token ID。
    输出二元组：(token_log_probs, shifted_mask)，均为 [N,T-1]。

    步骤：
    1. logits 去最后位置；input_ids、response_mask 去第一位置。
    2. 每条移位后的 mask 至少有一个1，否则抛 ValueError。
    3. 对 logits.float() 在词表维做 F.log_softmax。
    4. gather 取真实下一个 token 的 log 概率，squeeze 最后一维。
    5. 返回概率与移位后的 mask；此处不乘 mask、不求和、不 detach。

    与 DPO 的区别：DPO 返回每个回答一个数；这里每个位置一个数。
    prompt/PAD 对应的分数会在后面的 loss 中通过 mask 排除。
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    if (shift_mask.sum(dim=1) == 0).any():
        raise ValueError("每条移位后的 mask 至少有一个有效 token")

    all_log_probs = F.log_softmax(
        shift_logits.float(), 
        dim=-1
        )

    token_log_probs = all_log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs, shift_mask


def probability_ratio(new_log_probs, old_log_probs):
    """TODO 3：返回逐 token 的新旧概率比，形状不变 [N,L]。

    输入是同一批回答、同样 token 位置的 log 概率，形状相同且有限。
    ratio = exp(new_log_probs - old_log_probs.detach())。

    保留 new 的梯度，切断 old 的梯度。不在这里 clamp，TODO 4 需要原值。
    不要先对 token 求和再 exp：那会变成整段回答的比值。
    遇到非有限输出抛 ValueError，不能静默截断比值掩盖异常。
    """
    log_diff = new_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_diff)
    if not torch.isfinite(ratio).all():
        raise ValueError("概率比中存在 NaN 或 Inf")
    return ratio


def grpo_loss(new_log_probs, old_log_probs, advantages, response_mask, clip_eps=0.2):
    """TODO 4：实现文档的 clipped surrogate loss，返回标量 tensor。

    new_log_probs、old_log_probs、response_mask：[N,L]，已 shift 对齐。
    advantages：[N]，展平顺序与回答顺序一致；每个回答一个优势值。
    L 是移位后的长度，不再进行第二次 shift。

    步骤：
    1. 检查三个二维输入同形状且非空；advantages.shape==(N,)；
       0<clip_eps<1，mask 为0/1且每条至少有一个有效位置。错误抛 ValueError。
    2. 在计算 exp 之前，用 mask 把 new/old 的无效位置置0。
       可用 masked_fill 或 torch.where，避免无效位置 exp 溢出。
    3. 调用 probability_ratio 得到 ratio；advantages.detach().unsqueeze(-1)。
    4. raw = ratio * advantage；clipped = clamp(ratio,1-eps,1+eps)*advantage。
    5. 取 torch.minimum(raw,clipped)，加负号转换为需要最小化的 loss。
    6. 乘 mask，沿 token 维 sum，再沿回答维 mean，返回标量。

    不除回答长度，不加 KL；这是本题明确给出的版本。
    正负 advantage 都必须使用 minimum，不能直接只用 clamp 后的目标。
    new 的梯度要保留；old 和 advantage 不得收到梯度。
    """
    // 检查输入
    if new_log_probs.ndim != 2 or new_log_probs.numel() == 0:
        raise ValueError("log 概率必须是非空二维 tensor")
    if (
        old_log_probs.shape != new_log_probs.shape
        or response_mask.shape != new_log_probs.shape
    ):
        raise ValueError("new、old 和 mask 的形状必须相同")
    n = new_log_probs.shape[0]
    if advantages.shape != (n,):
        raise ValueError("advantages 的形状必须是 (N,)")
    if not (0 < clip_eps < 1):
        raise ValueError("clip_eps 必须在 (0,1) 之间")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response_mask 必须是 0/1 tensor")
    if (response_mask.sum(dim=1) == 0).any():
        raise ValueError("每条回答至少有一个有效 token")
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages 中存在 NaN 或 Inf")

    valid = response_mask.bool()

    safe_new = new_log_probs.masked_fill(~valid, 0.0)
    safe_old = old_log_probs.masked_fill(~valid, 0.0)

    ratio = probability_ratio(safe_new, safe_old)

    advantage = advantages.detach().unsqueeze(-1)
    raw_objective = ratio * advantage
    clipped_objective = torch.clamp(
        ratio, 
        1 - clip_eps, 
        1 + clip_eps
    ) * advantage
    objective = torch.minimum(
        raw_objective, 
        clipped_objective
    )
    token_loss = -objective
    mask = response_mask.to(token_loss.dtype)
    response_losses = (token_loss * mask).sum(dim=-1)
    return response_losses.mean()


def compute_batch_loss(policy, batch, clip_eps=0.2):
    """配套前向已接好；外层采样、缓存和优化器尚待实现。

    batch：input_ids/attention_mask/response_mask [N,T]；
           old_log_probs [N,T-1]（采样策略固定分数）；advantages [N]。
    所有 tensor 需同设备；调用者负责 autocast 与梯度累积。
    """
    output = policy(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    use_cache=False)
    new_log_probs, mask = response_token_log_probs(
        output.logits, batch['input_ids'], batch['response_mask'])
    return grpo_loss(new_log_probs, batch['old_log_probs'], batch['advantages'], mask, clip_eps)
