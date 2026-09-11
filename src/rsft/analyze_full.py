"""从本地保存的原始结果生成核对统计和训练曲线，不运行模型。"""
import json
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

workspace=Path(__file__).resolve().parents[4]
p=workspace/'test/hust-edgeAI/reports/rsft_full_20260910'
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def lines(path): return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines()]
train=lines(p/'train.jsonl'); val=lines(p/'validation.jsonl')
before=read(p/'before/metrics.json')
old=lines(workspace/'SFT/results_20260908/test_best/predictions.jsonl')
new=lines(p/'test_best/predictions.jsonl')
oldmap={r['id']:r for r in old}; newmap={r['id']:r for r in new}
assert len(oldmap)==len(newmap)==len(old)==len(new)==1319
assert oldmap.keys()==newmap.keys()
assert all(oldmap[i]['prompt']==newmap[i]['prompt'] and oldmap[i]['gold']==newmap[i]['gold'] for i in oldmap)
counts=Counter((bool(oldmap[i]['answer_correct']),bool(newmap[i]['answer_correct'])) for i in oldmap)
selected=read(p/'selected.json'); groups=Counter(r['question_id'] for r in selected)
assert max(groups.values())<=2
assert sum(r['samples'] for r in train)==len(selected)
assert [r['step'] for r in train]==list(range(1,len(train)+1))
ids=read(p/'ids.json')
assert not set(ids['train'])&set(ids['validation'])
assert set(groups)<=set(ids['train'])
duplicate_pairs=sum(len({r['response'] for r in selected if r['question_id']==qid})==1
                    for qid,n in groups.items() if n==2)
summary=dict(test_count=len(new),old_correct=sum(r['answer_correct'] for r in old),
    new_correct=sum(r['answer_correct'] for r in new),both_correct=counts[True,True],
    regressions=counts[True,False],improvements=counts[False,True],both_wrong=counts[False,False],
    selected=len(selected),one_answer_questions=sum(n==1 for n in groups.values()),
    two_answer_questions=sum(n==2 for n in groups.values()),identical_two_answer_questions=duplicate_pairs,
    updates=len(train),peak_allocated_gib=max(r['allocated_gib'] for r in train),
    peak_reserved_gib=max(r['reserved_gib'] for r in train),
    finished=datetime.fromtimestamp(read(p/'complete.json')['time'],timezone(timedelta(hours=8))).isoformat())
(p/'audit_summary.json').write_text(json.dumps(summary,indent=2))
changes=[dict(id=i,before_correct=oldmap[i]['answer_correct'],after_correct=newmap[i]['answer_correct'],
              prompt=newmap[i]['prompt'],gold=newmap[i]['gold'],before=oldmap[i]['response'],after=newmap[i]['response'])
         for i in oldmap if oldmap[i]['answer_correct']!=newmap[i]['answer_correct']]
(p/'changed_test_answers.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
window=50
smooth=[sum(r['loss'] for r in train[max(0,i-window+1):i+1])/min(i+1,window) for i in range(len(train))]
axes[0].plot([r['step'] for r in train],smooth,label='Train loss (50-step mean)')
axes[0].plot([0]+[r['step'] for r in val],[before['loss']]+[r['loss'] for r in val],'o-',label='Validation loss')
axes[0].set(xlabel='Update step',ylabel='Loss',title='RSFT training and validation loss')
axes[0].legend()
axes[1].plot([0]+[r['step'] for r in val],[100*before['answer_correct_rate']]+[100*r['answer_correct_rate'] for r in val],'o-')
axes[1].set(xlabel='Update step',ylabel='Accuracy (%)',title='Fixed 200-question validation',ylim=(60,82))
for ax in axes: ax.grid(alpha=.25)
fig.savefig(p/'training_curves.png',dpi=170)
plt.close(fig)
print(json.dumps(summary,ensure_ascii=False))
