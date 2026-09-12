# Self-Play 运行入口

已接入真实 Transformers 采样、学生核心、GRPO更新、每轮恢复断点及独立验证/完整测试。默认从 GRPO 最佳权重开始，先配置20轮小规模实验；当前尚未在GPU执行。

## 目录与部署

保持现有相邻目录关系：src/self_play/core.py、src/self_play/runtime/*、src/grpo/core.py。pipeline.py 按这个关系加载两个学生核心。服务器部署时同时保留两份core，而不是只复制runtime。config.json中的sft_code指向已有SFT运行配套；其中包含full_data.py、full_validation.py、full_scoring.py及SFT core.py。

在src/self_play目录先运行本地检查：

```bash
python checks/check_core.py -v
python checks/check_pipeline.py -v
python checks/check_runtime.py -v
```

在已配置模型和数据的服务器，激活posttrain环境后执行：

```bash
python runtime/run.py prepare
python runtime/run.py audit
python runtime/run.py smoke
bash runtime/run.sh
```

prepare创建新实验目录并绑定代码/模型/数据/配置哈希；重复prepare会拒绝覆盖。audit仅出题、解题、评分，不更新，生成audit.json供观察题目是否可解、参考答案是否可信。smoke实际执行临时更新并记录显存，但不保存临时权重。run.sh从初始模型或latest.pt继续正式训练，随后自动运行全部测试题。audit/smoke不会自动启动正式训练，也不构成人工质量认证。

## 接口与对齐

- backend.generate(prompts)：真实出题，返回text/truncated及原始token记录。
- backend.solve(requests,G)：只接受question_id/prompt，生成G条回答并缓存更新前old概率。
- prepare_round：调用学生解析、过滤、分组和评分，按题目/index恢复顺序。
- train_records：固定同一rollout的old分数，进行两遍GRPO更新，支持梯度累积。

生成采用temperature=1、top_p=1、top_k=0，停止条件捕捉实际结束位置，不把后补padding或未采样EOS作为动作。解码文本只用于评分，训练仍使用原始token。两个采样角色使用同一当前policy，不需要额外的旧模型常驻显存。

## 恢复与选模

每完成一轮保存latest.pt：模型、优化器、Python/PyTorch/CUDA随机状态、轮次、已见题目及最佳模型元数据。恢复后从已提交轮次继续；中断轮次重做，不能混用它的old缓存。配置/代码/数据哈希不一致时拒绝恢复。

独立验证保持原GSM8K固定200题，每5轮及末轮执行。best按正确率、再按验证loss选择，round=0代表保留原GRPO基线。最佳权重存入独立models/best_*目录，避免未提交检查点覆盖旧best。完整测试共1319题，必须生成complete.json才算全部完成。

所有原数据题干及跨轮已见自产题都用于精确排重，禁止作生成种子；题干排重不保证识别语义改写。测试答案只用于最终评估，不传给proposer或训练奖励函数。自生成参考答案仍是伪标签；matched_fraction不是独立解题正确率。连续3轮没有奖励差异时停止并保留原始出题记录，便于排查。

关键产物：audit.json、smoke.json、rounds/*_generations.json、*_rollout.json、*_result.json、latest.pt、best.json、progress.json、validation_*/metrics.json、final_model、test_best、complete.json。

本地检查包含实际CPU优化器更新、停止长度和padding对齐、old分数缓存，以及保存恢复后的下一步更新逐参数一致。GPU吞吐、显存、出题格式成功率和伪标签质量仍需实际audit/smoke验证。
