"""Build the human-readable report and verify duplicate/subset coverage."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
summary = json.loads((HERE / 'summary.json').read_text(encoding='utf-8'))
metrics = summary['metrics']

def load(path):
    s = path.read_text(encoding='utf-8')
    return [json.loads(x) for x in s.split('\n') if x.strip()] if path.suffix == '.jsonl' else json.loads(s)

checks = []
for run in ('test_base', 'test_best'):
    a = ROOT / 'SFT/results_20260908' / run / 'predictions.jsonl'
    b = HERE.parent / 'sft_20260908/raw' / run / 'predictions.jsonl'
    assert a.read_bytes() == b.read_bytes()
    checks.append(f'SFT {run} 本地备份与主记录逐字节一致，不重复计数。')
for selected, source in [
    (HERE.parent / 'rsft_trial_20260910/selected.jsonl', HERE / 'trial_candidates.jsonl'),
    (HERE.parent / 'rsft_full_20260910/selected.json', HERE / 'full_candidates.jsonl'),
]:
    candidates = {(x['question_id'], x['response']) for x in load(source)}
    rows = load(selected)
    assert all((x['question_id'], x['response']) in candidates for x in rows)
    checks.append(f'{selected.parent.name} 已选训练回答 {len(rows)} 条全部包含在重评候选中，不重复计数。')

lines = ['# 已保存回答统一重评分（2026-09-11）', '',
    '本次重评找到的 GSM8K 原始回答共 **' + str(sum(x['n'] for x in metrics.values())) + ' 条**。不重新生成、不训练，原记录保留；每条新记录含 original_score 和 unified_score。', '',
    '## 同题完整测试', '',
    '三组各 1319 题，已逐项断言 id、prompt、gold 一致。未解析计入总分分母，不能只在已解析子集上算准确率。', '',
    '| 模型 | 旧数值正确率 | 新数值正确率 | 新解析率 | 原严格格式率 |',
    '|---|---:|---:|---:|---:|']
for name, label in [('base_test','Base'),('sft_test','SFT'),('rsft_test','RSFT')]:
    x = metrics[name]; n = x['n']
    lines.append(f"| {label} | {x['old_answer_correct']}/{n}（{100*x['old_answer_correct']/n:.2f}%） | {x['new_answer_correct']}/{n}（{100*x['new_answer_correct']/n:.2f}%） | {x['new_parsed']}/{n}（{100*x['new_parsed']/n:.2f}%） | {100*x['new_format_ok']/n:.2f}% |")
lines += ['',
    'Base 找回 305 条正确数值答案，旧 2.20% 明显受提取器影响，不能再用作模型数学能力结论。SFT 仍比 Base 高 41.24 个百分点；RSFT 比 SFT 少答对 18 题，低 1.36 个百分点。此次重评分没有改变 RSFT 未超过 SFT 的观察，也不证明差异具有统计显著性。', '',
    'Base 仍有 641 条未解析，SFT 2 条，RSFT 3 条。这里的数值正确率不是人工语义准确率，也不代表推理过程正确。', '',
    '## 其他保存回答（分开统计）', '',
    '| 记录组 | 回答数 | 重评正确数 | 正确率 | 已解析 |',
    '|---|---:|---:|---:|---:|']
for name, x in metrics.items():
    if name in ('base_test','sft_test','rsft_test'):
        continue
    lines.append(f"| {name} | {x['n']} | {x['new_answer_correct']} | {100*x['new_answer_correct']/x['n']:.2f}% | {x['new_parsed']} |")
lines += ['',
    '正式 RSFT 候选数值正确 21766 条，其中格式也合规的 21764 条；不要把这两个计数混为一谈。原始筛选还排除截断。这次没有变动训练集或已训练模型。', '',
    '早期 50 题 GSM8K 的提示和生成设置不同，只作历史补充；old 字段是同一 v1 规则的回算，历史原脚本评分仍保存在原记录中。训练候选正确率不是独立测试准确率。', '',
    '## 提取规则和验证', '',
    '使用 gsm8k_numeric_v2_conservative。extract(response) 不接收标准答案；先提取后比较，禁止从推理中搜索一个碰巧等于 gold 的数字。', '',
    '优先唯一完整 answer 标签；否则检查靠近末尾的完整 boxed、#### 标记或最后非空行。支持单位/简短文字中的唯一数值、显式等式最后右端值、带分数、千位分隔和百分比。重复 answer 标签、未闭合开标签、多个无法消歧数值等保留未解析。格式仍沿用原严格规则。', '',
    '保守启发式仍有局限：末行可能不是本题最终答案，文字歧义可能漏判；最终数值正确不保证中间计算合理。未人工审核全部 1319 条 Base。此前固定种子抽查 15 条新增判对样本，可见末尾 boxed、单位、等式右端等典型修复；也见到推理自相矛盾但最终数值正确的回答。', '',
    '回归检查覆盖末尾 LaTeX 标点、单位、带分数，以及中间正确最终错误、重复标签、表达式、百分比、未闭合标签等反例。3 组测试通过。', '',
    '## 覆盖范围及服务器核对', '',
    *checks, '',
    '已只读连接新服务器端口 40011。远端 SFT 和正式 RSFT 归档 SHA256 与本地一致；试验版 before/after/candidates 三个文件 SHA256 也一致。另下载早期 baseline runs 和 smoke chunks 到 remote_originals。未消耗 GPU。', '',
    '远端发现两组各 50 题 MATH 历史回答（20260905_190505_841360、20260905_190921_328223），含多项式、根式、矩阵和文字答案。本次数值评分器不适用，已保留原始回答但未将其重评或混进 GSM8K 表格。', '',
    '中间检查点若只有 validation.jsonl 汇总指标而没有逐条回答，不能重新评分。本次未找回那些逐条回答，也没有把旧汇总指标冒充新分数。', '',
    'summary.json 含全部纳入文件、计数和 SHA256；各组 jsonl 保存逐条新旧结果；三个 *_test_audit.json 列出正确性变化及未解析回答。', '']
(HERE / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(checks))
print('Total:',sum(x['n'] for x in metrics.values()))
