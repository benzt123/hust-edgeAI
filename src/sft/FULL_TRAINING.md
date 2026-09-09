# 全量 GSM8K SFT

core.py 保留原样。本目录新增的主程序调用你的 shift_batch、response_loss、backward_microbatch、optimizer_update。

## 本地先检查

```bash
python check_core.py
python test_full.py
```

## 服务器准备

将新增文件和原 core.py、check_core.py、user.txt 放在 /root/workspace/test/src/SFT。

```bash
conda activate posttrain
python -m pip install tensorboard
python prepare_data.py --endpoint https://hf-mirror.com
python train_full.py --prepare-only
```

prepare-only 用 CPU/tokenizer 检查全体训练数据，输出 full_config.data_report.json，不加载大模型。
若长度超限，默认报错而不是默默过滤；修改 full_config.json：增大 max_length（不超过模型容量），
或显式设置 long_policy 为 window。window 对全部回答 token 分段监督，不丢样本；但只保留最近 overlap
个上下文 token，因此不能声称与全上下文训练等价。每条原始样本的所有分段按回答 token 数加权。

## 开始全量训练

```bash
bash run_full.sh
```

默认：GSM8K 官方 train 按问题分组划分 90%训练/10%验证；训练部分不限制样本数，完整 1 epoch。
验证集不参与参数更新。训练集每轮确定性打乱，micro-batch=1，累积8条。最后不足8条按实际数量平均。
默认验证 loss/entropy 用全部验证题，生成指标固定抽取50题。generation_limit=0 可生成全部验证题。
这里“全量训练”指全部训练划分，不包括保留的验证集和官方 test。

每200次更新及epoch结束验证并保存；训练开始先按同一协议测原模型。
learning_rate、epochs、max_length、评估间隔均在 full_config.json 中。
学习率先 warmup 再线性衰减；FP32 参数/优化器状态，BF16 autocast + 梯度检查点。
32GB 是否容纳4096长度需实测，不能保证不 OOM；遇到 OOM 可减少长度并明确选择 window。
不自动量化、不冻结参数、不偷偷丢数据。

## 观察训练

```bash
tensorboard --logdir /root/autodl-tmp/sft-full --host 127.0.0.1 --port 6006
```

通过 SSH 转发访问：`ssh -L 6006:127.0.0.1:6006 -p 你的端口 root@你的服务器`。
浏览器访问本机 localhost:6006。

输出包括 train.jsonl（loss、学习率、梯度范数、显存、周期性entropy）、validation.jsonl、
每轮验证的 predictions.jsonl（问题prompt、标准答案、回答、各项评分），TensorBoard 曲线、
数据统计及完整划分ID、代码快照、配置、best_model、final_model 和 latest.pt。
验证 entropy 是标准解答前缀条件下的预测分布熵，不是生成轨迹熵。

## 评分口径

本版明确仅支持 GSM8K 数值答案，full_scoring.py 的 gsm8k_numeric_v1：
- 格式要求先关闭think再包裹唯一答案，允许标签间空白；严格格式率单独统计。
- 答案正确率接受唯一完整answer内的纯数值；无answer标签时允许末尾boxed或整段纯数值。
- 支持整数、小数、科学计数法、分数、千位分隔、百分数；百分数保留数值含义。
- 不从一长段解释里猜一个数字，不使用 SymPy；解析失败按错计并记录。
- joint_correct_rate 是答案正确且格式正确的比例，绝不覆盖原评分器的历史结果。

这套保守提取规则仍可能漏判自然语言最终答案，并非完美评分器。与原评分器/旧vLLM结果不能直接比较。
训练前后都使用本模块、同一prompt、同一验证题、同一Transformers生成设置。相同seed不保证跨硬件位级一致。

## 断点续训

批量执行优化：`train_full.py --micro-batch-size 2` 或 `4`。这里配置中的
`accumulation=8` 继续表示每次参数更新覆盖8条原始样本，因此micro-batch=2对应4次反传，
micro-batch=4对应2次反传。最后不足8条按实际条数加权。此选项只支持未分段的训练数据。
右侧padding带attention mask，padding和prompt均不计入loss。

通过 `--resume .../latest.pt --micro-batch-size N` 可在断点切换执行批量，保持数据、
有效batch、优化器与学习率调度一致。浮点计算顺序变化可能产生细微数值差异，并非位级复现。
每次执行模式记录在 `execution_changes.jsonl`，实际运行代码另存 `active_*.py`。
切换前运行 `check_batching.py`；仅在GPU空闲时运行 `benchmark_batching.py` 做速度与最长样本检查。

```bash
bash run_full.sh --resume /root/autodl-tmp/sft-full/某次运行/latest.pt
```

保存模型、优化器、随机状态、epoch/样本位置；仅加载自己生成的可信文件。
配置/数据/模板/core.py 改变会拒绝原运行续训。断点前未保存的更新需要重做。
checkpoint 较大（FP32参数+Adam状态约18GB），latest.pt替换时还需临时空间。
best/final各约6GB，连同下载缓存需留足数据盘空间；不是每200步无限保存模型副本。

## 独立验证/最终测试

```bash
python validate_full.py --run /root/autodl-tmp/sft-full/某次运行 --model /root/autodl-tmp/sft-full/某次运行/best_model --out /root/autodl-tmp/sft-full/某次运行/recheck
python validate_full.py --run /root/autodl-tmp/sft-full/某次运行 --model /root/autodl-tmp/sft-full/某次运行/best_model --out /root/autodl-tmp/sft-full/某次运行/final_test --split test --all-generation
```

最终test仅在选好模型后使用，原模型也用完全同样的测试协议生成一份结果。
本次在本地完成CPU行为检查；尚未上传服务器、跑真实GPU全量训练或提交仓库。
