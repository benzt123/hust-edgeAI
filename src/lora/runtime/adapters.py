"""自定义adapter格式；不依赖PEFT，仅保存A/B及精确基座标识。"""
import importlib.util
from pathlib import Path
import os
import torch

spec=importlib.util.spec_from_file_location('student_lora',Path(__file__).resolve().parents[1]/'core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def adapter_payload(model,base_identity):
    layers={name:dict(rank=m.rank,alpha=m.alpha,in_features=m.base.in_features,out_features=m.base.out_features)
            for name,m in model.named_modules() if isinstance(m,core.LoRALinear)}
    if not layers:raise ValueError('模型未注入LoRA')
    weights={name+'.'+key:getattr(model.get_submodule(name),key).detach().cpu().clone()
             for name in layers for key in ['A','B']}
    return dict(format_version=1,base_identity=base_identity,layers=layers,weights=weights)


def atomic_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def save_adapter(model,path,base_identity):
    atomic_save(path,adapter_payload(model,base_identity))


def load_adapter(model,payload,base_identity):
    """加载到已按相同配置注入的模型；修改参数前先完整验证。"""
    current=adapter_payload(model,base_identity)
    if payload.get('format_version')!=1 or payload.get('base_identity')!=base_identity:
        raise ValueError('adapter版本或基座不匹配')
    if payload['layers']!=current['layers'] or set(payload['weights'])!=set(current['weights']):
        raise ValueError('adapter目标层或配置不匹配')
    for key,value in payload['weights'].items():
        if value.shape!=current['weights'][key].shape or not torch.isfinite(value).all():
            raise ValueError(f'adapter参数无效：{key}')
    with torch.no_grad():
        for name in current['layers']:
            module=model.get_submodule(name)
            for key in ['A','B']:
                getattr(module,key).copy_(payload['weights'][name+'.'+key])


@torch.no_grad()
def merge_lora(model):
    """原地合并并移除包装，供导出推理；不要继续使用此前的优化器。"""
    def visit(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,core.LoRALinear):
                delta=(child.A.float()@child.B.float())*child.scaling
                child.base.weight.add_(delta.to(child.base.weight.dtype))
                setattr(parent,name,child.base)
            else:visit(child)
    visit(model)
    return model
