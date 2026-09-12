# GRPO 远程运行配套

`run.py` 直接调用学生 core.py，使用 Transformers 采样和手写 PyTorch 更新，不使用 Trainer。全部训练数据来自服务器已存在的 GSM8K 训练划分，不上传题目和回答。

## 实验设置

- 起点：DPO 最佳 step 284，完整测试 940/1319（71.2661%）。
- 200 轮 rollout，每轮 32 道题，每题 8 条回答，共 51200 条新生成回答。
- 每批 rollout 更新 2 遍，有效 batch 为32条回答，共计划3200次 optimizer 更新。
- 奖励为最终数值正确、格式合格且未截断时1，否则0；错误回答不筛掉。
- advantage 只减组均值，token loss 求和后回答平均，clip_eps=0.2，不加 KL。
- 学习率1e-6，前10轮warmup；AdamW betas=(0.9,0.95)，梯度裁剪1。
- FP32 权重和优化器状态、BF16 autocast、梯度检查点、micro batch=1。
- 每次并行生成4题×8回答，temperature=1，top_p=1，top_k=0；不强制最少生成长度，避免修改初始token的采样分布。

## 关键数据约定

生成时直接保留 token ID，不能解码后重新分词作为训练动作。自定义停止条件记录每条回答首次结束位置，排除 generate 给已结束样本追加的 padding；仅把实际采样的 EOS 算作回答 token，不人为追加 EOS。达到长度上限的回答记录 truncated 并奖励0，已采样 token 仍参与本题组内目标。

old 分数在本轮任何更新之前按原模型缓存，两遍更新始终固定。下一轮才用更新后的模型重新采样和缓存。这些 old 分数保存在每轮的 rollouts/*.json 中，以便审查。

## 运行顺序

```bash
conda activate posttrain
python checks/check_core.py -v
python runtime/run.py prepare
python runtime/run.py smoke
# smoke通过后，后台执行：
bash runtime/run.sh
```

smoke 从初始权重产生真实回答，检查 old/current 分数一致性和两遍更新的有限梯度、显存。临时更新不保存，正式训练重新加载DPO。

train 每10轮保存完整恢复断点，每50轮及最后一轮评估固定200题；保存初始DPO基线作为best候选。若最终best仍是round=0，完整测试代表保留的DPO，不能声称GRPO改进。test 自动生成全部1319道测试题，完成后写 complete.json。

结果目录：`/root/autodl-fs/posttrain/grpo-20260911-r1`。run.log、progress.json、train.jsonl 用于进度和耗时；训练中loss不是解题准确率，最终效果以独立生成评估为准。中断后只能按已保存轮次恢复，重复轮次重新采样而非复用不匹配的旧分数。

本轮比静态DPO更耗时：训练过程中持续生成51200条回答。以真实试跑和首轮耗时估计总时间，不直接沿用DPO的耗时估算。
