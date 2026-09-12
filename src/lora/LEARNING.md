# 手写LoRA练习

完成core.py的四个TODO：初始化LoRALinear、forward、注入FFN、筛选参数。setup_lora已接好冻结和注入顺序。没有使用PEFT库，没有加载模型或启动训练。

## 计算与形状

沿用题目文档命名：W[out,in]冻结，A[out,r]零初始化，B[r,in]随机初始化。

```text
原分支：base(x) = x @ W.T + bias
低秩分支：F.linear(x,B) → F.linear(...,A)
输出：base(x) + (alpha/r) * delta
```

初始A@B=0，因此原模型输出保持不变；A、B不能同时为零，否则两者初始梯度都为零。按此初始化，第一次反传通常A梯度非零、B梯度为零；A更新后B才开始学习，这是正常行为。

冻结base参数不等于把base前向放进no_grad。梯度仍需经过base传到前面注入的LoRA层。forward不能detach输入，不必先构造完整A@B矩阵。

## 注入顺序

setup_lora先冻结全模型，再将gate_proj/up_proj/down_proj包装为LoRALinear。新A/B保持可训练。inject_lora只替换目标，跳过已有LoRALinear；setup_lora则拒绝重复配置，避免把已有A/B冻结。

一个目标层增加r*(in+out)个参数；原bias依旧冻结，低秩分支不加新bias。命名A/B可与其他实现相反，判断时以形状和乘法顺序为准。

## 学习检查

在src/lora执行：

```bash
python checks/check_core.py -v
python checks/check_core.py CoreChecks.test_initial_equivalence_and_metadata -v
```

未填写TODO时会报NotImplementedError，这是预期。检查覆盖初始化等价、非零低秩运算、原权重不更新、输入梯度、A/B学习时序和注入范围。

训练时复用手写SFT回答loss与梯度累积，优化器只接收trainable_parameters(model)。本阶段按题目要求使用非数学领域数据；物理数据尚待准备。当前骨架不含adapter保存/加载与合并、训练入口或数据配置，这些在核心完成后接入。暂不自动选择尚未完成实验的Self-Play权重。
