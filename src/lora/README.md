# src/lora

练习：手写低秩线性模块、冻结基座、注入 FFN、保存 adapter、跨领域评估。

- `core.py`：学习核心算法。
- `checks/`：CPU 核心和运行接口检查。
- `runtime/`：物理解题数据划分、梯度累积训练、断点恢复、adapter 加载、测试生成和合并导出。

具体命令与配置见 [运行说明](runtime/README.md)。课程后续可接入完成的 Self-Play checkpoint；当前默认配置使用此前完成的 GRPO best_model。物理数据路径仍需替换为实际文件。
