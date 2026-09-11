# DPO 第一轮运行记录

服务器：connect.bjb1.seetacloud.com:40011。代码目录：/root/workspace/test/src/DPO_20260911。
结果目录：/root/autodl-fs/posttrain/dpo-20260911-r1。

使用已有 RSFT 权重和服务器原始 SFT 候选；没有上传题目或回答数据。代码上传获准后，服务器本地重评分并构造 2265 对，每题一对，覆盖 6725 题中的 33.68%，最长 791 token，未截断丢弃。

配置：beta=0.1，学习率1e-6，前10%更新线性warmup，1epoch，micro batch=1对，有效batch=8对，共284次更新。缓存参考分数后退出参考进程，正式训练只加载policy。每50步保存断点，每100步及末步评估固定200题；训练后对最佳模型测试1319题。

服务器CPU核心检查8项通过。实际模型试跑：初始四对loss均为0.693147；一次临时更新后loss分别为0.242371、0.044363、0.528057、0.246510。梯度范数45.452（裁剪前），峰值allocated24.758GiB、reserved24.854GiB。试跑更新未保存，正式训练重新加载RSFT。

后台流程已提交，完整测试尚未完成。进度以远端run.log、progress.json、status.json和complete.json为准。
