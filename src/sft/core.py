"""你只需要先修改这个文件。TODO 没完成时会明确报错。"""
import torch
import torch.nn.functional as F


def shift_batch(input_ids, response_mask):
    """TODO 1：构造 next-token prediction 数据。

    输入形状均为 [B, T]。
    返回 inputs、labels、loss_mask，形状均为 [B, T-1]。
    inputs 去掉最后一个 token；labels 去掉第一个 token。
    loss_mask 应与 labels 对齐，而不是与 inputs 对齐。
    """
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    loss_mask = response_mask[:, 1:]
    return inputs, labels, loss_mask


def response_loss(logits, labels, loss_mask):
    """TODO 2：只对回答 token 计算平均交叉熵。

    logits: [B, T, V]；labels、loss_mask: [B, T]。
    先计算每个位置的 loss（reduction='none'），再乘 mask。
    为让后续梯度累积在不同回答长度下也有明确含义：
    先对每条样本的有效回答 token 求平均，再对样本求平均。
    任一样本的 mask 全零时抛出 ValueError，不要产生 NaN。
    返回标量 Tensor，保留梯度。
    """
    B, T, V = logits.shape
    token_loss = F.cross_entropy(
        logits.reshape(-1,V),
        labels.reshape(-1),
        reduction='none',
    ).reshape(B, T)
    counts = loss_mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("存在没有回答token的样本")
    sample_loss = (token_loss * loss_mask).sum(dim=1) / counts
    return sample_loss.mean()


def backward_microbatch(loss, accumulation_count):
    """TODO 3：先将 loss 除以当前累积组大小，再 backward。

    不在这里 step 或 zero_grad。服务器脚本使用等大的 micro-batch=1。
    accumulation_count 是这一组实际的批次数，最后一组可能不足配置值。
    """
    scaled_loss = loss / accumulation_count
    scaled_loss.backward()


def optimizer_update(model, optimizer, max_grad_norm=1.0):
    """TODO 4：裁剪梯度 → optimizer.step → optimizer.zero_grad。

    梯度裁剪使用 torch.nn.utils.clip_grad_norm_。
    zero_grad 推荐 set_to_none=True。
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)