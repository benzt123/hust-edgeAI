# SFT：从这里开始

日常主要阅读 `core.py`（你实现的四个核心函数）和 `batching.py`（批量处理）。

```text
sft/
├── core.py             标签移位、loss、梯度累积、参数更新
├── batching.py         padding、attention mask、批量反传
├── runtime/            完整训练、评估、数据处理、配置、提示词和启动脚本
├── checks/             CPU检查及GPU冒烟检查
├── benchmarks/         GPU批量测速
└── docs/               详细运行说明
```

## 运行入口

从本目录执行，无需移动脚本：

```bash
python checks/check_core.py
python checks/check_batching.py
python checks/test_full.py
python runtime/train_full.py --config runtime/server_run_config.json --prepare-only
python runtime/train_full.py --config runtime/server_run_config.json --micro-batch-size 4
```

正式训练命令需要服务器环境；先检查配置中的模型、数据和输出路径。配置中accumulation=8代表每次更新覆盖8条样本，micro-batch=4时分两次反传。

测速入口：`python benchmarks/benchmark_batching.py`，仅在GPU空闲时运行。GPU冒烟检查：`python checks/smoke_full.py`。

`runtime/run_experiment.sh`保留了9月8日的日志/状态路径，作为历史入口；新实验应先改为新的运行标识，不直接重复使用旧记录路径。

## 实验结果

- [完整结果](../../reports/sft_20260908/REPORT.md)
- [测速结果](../../reports/sft_20260908/BENCHMARK.md)
- `../../reports/sft_20260908/raw/`：原始日志、配置和逐题结果。

| 模型 | 共同题目中答对数 | 正确率 |
|---|---:|---:|
| Base | 334 / 678 | **49.26%** |
| SFT | 455 / 678 | **67.11%** |

从成功被解析的公共题目来看，sft存在18%左右的正确率提升；
格式正确率更是提升到了99%;
说明纯sft的全量训练在输出格式和正确性上效果都是比较明显的




