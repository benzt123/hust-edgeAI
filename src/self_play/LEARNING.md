# Self-Play + GRPO：把数据来源也放进循环

core.py 的四个 TODO 已完成。本阶段复用已写好的 GRPO 概率、优势和损失，不发明新的优化目标。运行接口、恢复与评估现已接入，见 runtime/README.md；尚未启动服务器训练。

## 推演

上一阶段：真实训练题 → 每题G个回答 → 可验证奖励 → GRPO更新。

本阶段：模型生成题目及参考答案 → 解析与过滤 → 每题G个回答 → 与模型参考答案比较 → GRPO更新 → 用更新后的模型开始下一轮。

新增的是数据生产过程，而非新的 loss。proposer 和 solver 是角色，可以由同一个模型配合不同提示承担。本题选择只对 solver 回答的目标反传，不额外实现 proposer 的奖励或双模型对抗训练。

## 四个核心块

1. **解析**：从 Problem/Answer 两个字段取得题干与参考答案。格式错误就跳过，不能凭猜测补出标签。本骨架要求标签位于行首，比文档示例的宽松正则更严格，便于审查。
2. **过滤与去重**：剔除截断、解析失败、当前数值评分器不支持的答案。按题干折叠空白后去重，首次有效记录保留。稳定 ID 用于将回答关联到题目。此过滤不证明题目有效或参考答案正确。
3. **解题提示**：只给 solver 题干，绝不把 proposer 的参考答案拼进去，否则容易变成照抄答案。
4. **奖励分组**：保留每题完整G条回答。答案匹配、格式完整、可解析、未截断才奖励1，其余0。错误回答不能像 RSFT 一样删除，否则组内优势信号会被破坏。

例如某题奖励 [1,1,0,0]，复用 GRPO 得到 [.5,.5,-.5,-.5]；全为1或全为0时优势为0。不能先只保留正确回答再调用 group_advantages。

## 接到现有代码

```text
出题采样（保留截断状态）
  → prepare_problems(generations, answer_supported)
  → format_solve_prompt(problem, 原r1_zero模板)
  → 每题采样G个回答，保存question_id/index及原始token
  → build_reward_groups(problems, candidates, grade, G)
  → rewards转为[B,G] tensor
  → 原GRPO group_advantages → 固定old分数 → grpo_loss → 更新policy
```

解码文本只用于评分；训练依然使用采样时的原始token与mask。分组顺序必须与token批次一致。新一轮重新出题、采样，并刷新old缓存。

## 最重要的局限

proposer 给出的答案是伪标签。两个角色可能犯同一种错误，答案相同并不证明数学正确。正式训练前先小规模审查题目是否可解、答案是否可信；可加入独立算术验证或人工抽检，不把多数一致误认为证明。

blocked_problem_keys 只能排除规范化后完全相同的题干，不保证排除改写、语义重复或题干内隐含答案。不得用固定验证或测试题作生成种子。种子只能来自训练集；验证始终使用原固定数据，不拿自生成题的匹配率作为模型能力提升证据。

题目文档建议MATH评估；现有项目实验使用GSM8K数值任务。本次骨架沿用该范围以保持阶段对比，符号题扩展需另行配置评分器与评估，不能声称已覆盖MATH要求。

## 检查

在 src/self_play 下运行 `python checks/check_core.py -v`，也可单独运行 `python checks/check_core.py CoreChecks.test_parser -v`。TODO未填写时预期报 NotImplementedError。通过后再接出题采样、质量审查和训练入口。
