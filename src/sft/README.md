# SFT 核心代码与可复现实验

2026-09-09 从工作区 SFT 目录归档。后续 SFT 修改以本目录为准；工作区旧 SFT 保留作为迁移前副本，不再双向编辑。

- `core.py`：用户完成的 shift、response loss、梯度累积与参数更新四个核心函数。
- `batching.py`：后续加入的批量执行、padding、attention mask 与累积加权。
- `benchmark_batching.py`：同一有效batch下对比micro-batch 1/2/4；仅在GPU空闲时运行。
- `check_core.py`、`check_batching.py`、`test_full.py`：CPU行为检查。
- `train_full.py`、`full_*.py`、`validate_full.py`：训练、数据、评分与评估支持代码。
- `smoke_full.py`：GPU最长样本与参数更新检查，不保存正式训练结果。
- `user.txt`：本次提示词；`server_run_config.json`：实际实验配置，含原服务器路径。

实验指标、完整测试回答与报告位于 `../../reports/sft_20260908/`。已有原始baseline仍在 `../baseline/`，没有修改。旧baseline评分规则与本次不同，直接对比应使用归档结果中的test_base与test_best。

CPU检查：

```bash
python check_core.py
python check_batching.py
python test_full.py
```

运行训练前检查配置中的机器路径。`run_experiment.sh` 是2026-09-08的运行入口快照，含该次日志/状态路径，不宜原样用作新的实验入口，以免混淆记录。完整说明见 `FULL_TRAINING.md`。模型权重和优化器断点留在服务器，不纳入Git。
