# DPO 核心练习

填写 `core.py` 的三个 TODO。此次提供核心骨架与前向连接，不启动训练；模型加载、分词、优化器、保存与评估入口尚待接入。

## 从 RSFT 到 DPO

RSFT 只模仿合格回答。DPO 为同一道题提供两个完整回答，让模型学习哪个更值得偏好。数学题先用“正确、格式合格、未截断”的回答作 chosen，用“成功解析但错误、格式合格、未截断”的回答作 rejected。最终答案的正确性不保证推理每一步正确。

流程：原始候选 → 同题偏好对 → 两个模型分别给同一对回答打分 → DPO loss → 只更新 policy。

- policy：继续训练的 RSFT 模型。
- reference：从同一个 RSFT checkpoint 独立加载，调用 `freeze_reference` 后始终固定。
- chosen/rejected：已有的完整回答。采用 teacher forcing 计算概率，不是让两模型各生成一个回答。

不能写 `reference = policy`：这会引用同一个模型，冻结 reference 也会冻结 policy。

## 建议填写顺序：TODO 3 → TODO 2 → TODO 1

### TODO 3：先理解优化目标

设 policy 对好、坏回答的序列 log 概率为 p_w、p_l，reference 对应为 r_w、r_l：

```text
policy_margin = p_w - p_l
reference_margin = r_w - r_l
z = beta * (policy_margin - reference_margin)
loss = -log(sigmoid(z))，再对 batch 取平均
```

例如 reference 的好、坏回答得分为 -12、-10，margin 为 -2；policy 更新后为 -11、-12，margin 为 +1。相对参考模型改善了 3；beta=0.1 时 z=0.3，loss 约 0.5544。

初始两模型相同，z=0，loss 约 0.6931；这是正常起点，梯度并不为零。训练直接推动的是相对参考模型的偏好差距，不保证每次更新 chosen 的绝对概率都上升。DPO 不需要另外训练奖励模型。

beta 影响目标和梯度尺度；先用 0.1 作为待验证起点，不能简单认为越大越好。

### TODO 2：再理解回答概率

token 为 `[提示A, 提示B, 回答C, EOS, PAD]`，mask 为 `[0,0,1,1,0]`。提示B位置的 logits 预测回答C，回答C位置预测 EOS。移位后的 mask 为 `[0,1,1,0]`。

先对词表做 log_softmax，再取真实下一个 token 的值，最后加总回答位置。这里不是取 argmax。Prompt 不计入求和，但仍参与上下文计算。

与之前 SFT 练习不同：**这里按回答 token 求和，不按回答长度平均。**

### TODO 1：最后接入偏好数据

每题最多一对，各池去重后随机抽取，固定 seed；只有正确或只有错误回答的题目跳过。这样控制每题的权重，但仍有题目覆盖偏差。不要用不同题目的回答拼成一对。

示例输入两条均来自 q1、同一个 prompt：正确完整回答 A 和错误完整回答 B。输出应为：

```python
{'question_id': 'q1', 'prompt': '原始提示', 'chosen': '完整回答A', 'rejected': '完整回答B'}
```

旧 RSFT 的 selected.json 只有合格回答，不能提供 rejected；须使用原始候选。若没有 parsed 字段，先重新评分补齐，不能把解析失败直接当作数学错误。SFT 生成的旧候选可以作离线偏好数据，但应记录来源，不能称为 RSFT 新生成数据。只用训练集构造训练对，验证和测试题保持隔离。

## 接下来如何训练

`compute_batch_loss` 已接好两模型前向。外层加载两份 RSFT 权重、冻结 reference、构造回答 mask，再调用此函数。随后按此前手写训练循环做 backward、梯度累积、裁剪、step、zero_grad。最后不足一个累计组时按实际组大小缩放。有效 batch 的单位是“偏好对”，一对包含两个回答。

reference 不建梯度图仍占权重显存；policy 要为两个回答保存计算图，不能照搬此前 SFT 的 batch。正式接入时可以先缓存固定 reference 的 log 概率再卸载它；缓存必须对应同一 checkpoint、tokenizer、模板、token ID 和 mask。

完成 TODO 后运行 `python checks.py`，用 CPU 检查核心算法。当前 TODO 未填写，报 NotImplementedError 是预期行为。通过后再接训练入口，用小批量验证显存、梯度与保存恢复，随后启动正式实验。
