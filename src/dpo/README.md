# DPO 核心练习

## 从 RSFT 到 DPO

RSFT 只模仿合格回答。DPO 为同一道题提供两个完整回答，让模型学习哪个更值得偏好。数学题先用“正确、格式合格、未截断”的回答作 chosen，用“成功解析但错误、格式合格、未截断”的回答作 rejected。最终答案的正确性不保证推理每一步正确。

流程：原始候选 → 同题偏好对 → 两个模型分别给同一对回答打分 → DPO loss → 只更新 policy。

- policy：继续训练的 RSFT 模型。
- reference：从同一个 RSFT checkpoint 独立加载，调用 `freeze_reference` 后始终固定。
- chosen/rejected：已有的完整回答。采用 teacher forcing 计算概率，不是让两模型各生成一个回答。

## 接下来如何训练

`compute_batch_loss` 已接好两模型前向。外层加载两份 RSFT 权重、冻结 reference、构造回答 mask，再调用此函数。随后按此前手写训练循环做 backward、梯度累积、裁剪、step、zero_grad。最后不足一个累计组时按实际组大小缩放。有效 batch 的单位是“偏好对”，一对包含两个回答。

reference 不建梯度图仍占权重显存；policy 要为两个回答保存计算图，不能照搬此前 SFT 的 batch。正式接入时可以先缓存固定 reference 的 log 概率再卸载它；缓存必须对应同一 checkpoint、tokenizer、模板、token ID 和 mask。

完成 TODO 后运行 `python checks.py`，用 CPU 检查核心算法。当前 TODO 未填写，报 NotImplementedError 是预期行为。通过后再接训练入口，用小批量验证显存、梯度与保存恢复，随后启动正式实验。

## 训练结果

| 模型 | 答对 / 1,319 题 | 正确率 |
|---|---:|---:|
| SFT | 878 | 66.57% |
| RSFT | 860 | 65.20% |
| **本轮 DPO** | **940** | **71.27%** |

当前所取的参数为beta=0.1，相对更新策略没有那么激进，还需要更多测试尝试。