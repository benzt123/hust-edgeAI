# GRPO 核心练习

按 `Technical Challenge 2026.md` 的 GRPO「实现细节注意事项」实现。此前口头讲解包括其他版本；本练习统一为：advantage 只减组均值、loss 按 token 求和再按回答平均、不添加 reference KL。

## 填写顺序

1. `group_advantages`：奖励 [B,G] → 优势 [B,G]。例如 [1,1,0,0] 对应 [.5,.5,-.5,-.5]。每行是一道题，不能跨题计算均值。
2. `response_token_log_probs`：沿用 DPO 的 shift、log_softmax、gather；返回每个 token 的概率和 mask，不求和。第 t 个位置预测第 t+1 个 token。
3. `probability_ratio`：同一回答、同一位置的 exp(new_logp-old_logp)。old 固定，new 保留梯度，不提前 clip。
4. `grpo_loss`：广播回答级优势到每个 token，比较原始目标与 clipped 目标，取较小值，加负号、mask、求和与平均。

`core.py` 每个 TODO 都有形状说明与具体步骤，保留 NotImplementedError 等你填写；配套前向 `compute_batch_loss` 已连接。

## 模型角色与外层流程

本题基础版本只需要 policy 与 old policy 的角色，不要求额外的固定 reference。old 是生成本轮回答时的 policy 快照，不能每次 optimizer.step 后就更新缓存。新一轮 rollout 才重新生成回答与对应 old 分数。

```text
当前 policy 为每题采样 G 条完整回答
    → 评分器给奖励 [B,G]
    → TODO 1 得到优势，按题目优先顺序展平为 [B*G]
    → 更新前缓存相同 token/mask 对应的 old_log_probs
    → TODO 2 计算当前 policy 的逐 token 分数
    → TODO 3/4 计算 loss
    → backward、梯度累积、裁剪、optimizer.step
    → 同批可更新多轮，old 缓存保持固定
    → 下一轮重新采样
```

old 与 new 比较的必须是同一批现成回答，不能各自重新生成再比较。采样分布与用于计算 old 分数的分布须一致；最初接入采用 temperature=1、不做 top-k/top-p 截断，避免忽略分布变化。缓存必须绑定策略版本、token ID、模板和 mask。

组内全对或全错时优势为0，这组没有奖励区分信号。loss 等于0也不一定没有梯度：等长的一正一负优势在 ratio=1 时，数值可抵消，但对应 token 的梯度仍不同。

本题 token 求和会使长回答获得不同的权重，这是文档公式的选择；不要在填写过程中悄悄换成按回答长度归一化。clip 限制目标的收益，不是硬性保证参数或 ratio 永远不越界。

## CPU 检查

在 `src/grpo` 目录运行：

```bash
python checks/check_core.py -v
```

也可以只检查已完成的 TODO，例如：

```bash
python checks/check_core.py CoreChecks.test_advantages -v
```

未填写时预期报 NotImplementedError。检查覆盖组内均值、token 对齐、新旧概率比、正负优势四种 clip 情况、长度求和、mask 和梯度隔离。

此骨架不会启动 GPU 训练。完成后还需接入采样、奖励、旧策略缓存、优化器与独立验证；文档超参数是起点，需通过真实显存试跑确定运行配置。不要直接复用 DPO 的静态偏好对作为在线 rollout。
