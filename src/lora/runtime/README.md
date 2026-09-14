# 物理解题 LoRA 运行配套

这里调用你编写的 `../core.py`，不依赖 PEFT。冻结基座，只训练 FFN 中 gate_proj/up_proj/down_proj 的 A/B；按回答 token 的平均损失训练，再对题目取平均，避免长解答自动获得更大权重。

## 数据与基座

用户已选择物理解题数据，但目前没有实际数据文件。请将至少 10 条**纯文本**题目整理为 JSONL，每行如下（answer 是完整参考解答）：

```json
{"id":"physics-001","question":"质量为2 kg的物体受到6 N的合外力，求加速度。","answer":"根据牛顿第二定律，a=F/m=6/2=3 m/s²。"}
```

拒绝重复 ID、空内容、空白归一化后重复题目、显式图像字段和超长样本，不静默截掉解答。语义近重复题和同一题的不同改写仍需数据整理时去重。数据按固定种子约 80%/10%/10% 划分；划分 ID 会保存。

修改 config.json 中 base_model、data、output。默认基座路径指向之前完成的 GRPO best_model；Self-Play 尚未确认完成，不能假设存在其训练结果。数据路径是占位路径，需准备实际文件。output 必须是新的实验目录。模型要求本地 safetensors 权重和 tokenizer 文件。

默认 rank=8、alpha=16、学习率 1e-4、3 epochs、micro_batch=1、effective_batch=8，即通常累积 8 次后更新；末尾不足 8 题按实际题数归一化。最大总长度 2048，生成上限 1024。基座 FP32，CUDA 前向 BF16，非重入梯度检查点；LoRA 仍需保存基座权重与部分激活，因此必须先试跑观察显存。

## 执行顺序

在服务器项目根目录，使用已有的 PyTorch/Transformers 训练环境：

```bash
python src/lora/checks/check_core.py -v
python src/lora/checks/check_runtime.py -v
python src/lora/runtime/run.py prepare
python src/lora/runtime/run.py smoke
bash src/lora/runtime/run.sh
# 需要完整推理模型时再合并导出：
python src/lora/runtime/run.py merge
```

prepare 检查数据和基座文件，记录配置/代码/数据指纹及划分；smoke 加载模型，用最长训练样本完成一次临时更新并记录峰值显存，不保存这次更新。完整分词和长度校验在加载 tokenizer 后进行。run.sh 顺序训练、测试；失败立即停止。也可以分别运行 train 和 test。自定义配置时对各阶段传入相同的 `--config /path/to/config.json`（run.sh 使用默认配置）。

训练中断后以原配置重新运行 train 会自动从 latest.pt 恢复 A/B、优化器、步数和随机状态。默认每 50 步及每个 epoch 末保存，可能重跑最后未保存的更新。修改代码、配置、数据或基座后须换新 output 并重新 prepare；不混用旧断点。只加载自己生成且可信的训练断点。

## 如何阅读结果

- smoke.json：试跑损失、梯度范数和 PyTorch 峰值 allocated 显存；不等于系统显示的全部显存，也不能保证正式运行绝不 OOM。
- parameter_stats.json：替换层、总参数与可训练参数数量。
- train.jsonl / validation.jsonl：训练曲线和每个 epoch 的验证回答 NLL，越低表示参考答案的预测损失越低。
- best_adapter.pt：验证 NLL 最小的 adapter，包括未训练的初始基线参与比较；若 best_step=0，说明训练没有超过基线。final_adapter.pt 是最后一步。
- latest.pt：完整续训状态；adapter 文件本身不能恢复优化器进度。
- test_metrics.json：训练前后测试回答 NLL。physics_accuracy 为 null：尚无可靠物理判题器，NLL 不是正确率。
- test_predictions.jsonl：独立测试题、参考解答、实际生成与是否截断，可逐题检查公式、数值、单位、推理及停止行为。
- merged_model/：最佳 adapter 合入对应基座后的完整模型，可供通常的 Transformers 推理加载。自定义 adapter.pt 不能直接交给 PEFT 加载。

目前的脚本验证不等于已在服务器完成训练；正式运行时间需要实际数据量与试跑吞吐后估计。
