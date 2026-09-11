# 正式 DPO 运行配套

本目录调用上一级学生 `core.py` 的偏好配对、序列 log 概率和 DPO loss，使用手写 PyTorch 训练循环，不使用 Trainer。

服务器上的固定步骤：

```bash
conda activate posttrain
python checks.py -v
python runtime/run.py prepare
python runtime/run.py smoke
# 检查 smoke.json 后，从 runtime 目录以后台进程执行 run.sh
bash runtime/run.sh
```

`config.json` 记录服务器路径和所有训练参数。原始候选留在服务器，prepare 使用与统一重评分相同的数值评分器，检查题目 ID、prompt、gold 和 train/validation 隔离后，每题抽一对。初始数据为 2,265 对，最长 791 token，未截断丢弃；候选来自 SFT，policy/reference 初始化来自完成 RSFT 的模型。

smoke 使用最长的两对与最前面的两对做一次临时更新，检查初始 loss、有限非零梯度、更新后损失和显存。该进程不保存更新的权重，正式训练重新加载原 RSFT 模型。

cache 固定参考模型，逐对缓存两个回答的序列 log 概率；进程退出后释放参考模型显存。train 只加载 policy，使用 FP32 权重及 AdamW 状态、BF16 autocast、梯度检查点。有效 batch 按偏好对计数，最后不足一组时按实际数量缩放梯度。

每 50 次更新保存可恢复断点，每 100 次及最后一次更新评估固定 200 道验证题；best_model 包括 step 0 基线作为候选，按验证正确率、再按验证 loss 选取。final_model 保留最后 DPO 权重。训练完成后自动评估最佳模型的全部 1,319 道测试题。若最佳仍为 step 0，测试结果代表保留的 RSFT 基线，不能冒称 DPO 获益。

`preference_accuracy` 是训练对相对参考模型的 margin 为正的比例，**不是数学解题正确率**。数学正确率见独立生成评估。生成设置沿用先前 SFT/RSFT，数值评分采用已审计的 v2。

原始文件、模型、配置、代码的哈希绑定参考缓存及断点。改动后需要新实验目录，避免混用。产物默认保存到 `/root/autodl-fs/posttrain/dpo-20260911-r1`：

- `data_stats.json`、`pairs.json`、`encoded.json`：数据统计与实际 token/mask。
- `smoke.json`、`reference.json`：显存试跑和固定参考分数。
- `train.jsonl`、`progress.json`、`validation.jsonl`：进度与曲线数据。
- `latest.pt`：模型、优化器和随机状态断点。
- `best.json`、`best_model/`、`final_model/`：模型选择记录及权重。
- `test_best/`、`complete.json`：完整测试结果及完成标记。

`numeric_grader.py` 是现有 `src/common/rescore_saved.py` 的评分函数快照，仅适配 import 路径；已对全部 26,900 条本地重评分记录确认输出一致。
