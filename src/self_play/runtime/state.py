"""原子checkpoint与随机状态恢复。只加载本实验自行生成的可信checkpoint。"""
import json
import os
import random
from pathlib import Path
import torch


def dump(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def save(path,model,optimizer,identity,metadata):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
        metadata=metadata,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),tmp)
    os.replace(tmp,path)


def restore(path,model,optimizer,identity):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']!=identity:raise ValueError('断点与代码/配置/数据不匹配')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['metadata']
