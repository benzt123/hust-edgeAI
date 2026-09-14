"""手写 LoRA 核心练习：只注入 FFN，不使用 PEFT 库。

沿用课程命名：A[out,r]零初始化，B[r,in]随机初始化，deltaW=(alpha/r)*A@B。
只处理未量化的普通 nn.Linear；运行配套、数据加载与训练入口另行接入。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

TARGET_NAMES = frozenset({'gate_proj', 'up_proj', 'down_proj'})


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0):
        super().__init__()
        """TODO 1：保留原线性层并创建低秩参数。

        1. base_layer必须是nn.Linear，否则抛TypeError。
        2. rank为非bool整数，1<=rank<=min(in_features,out_features)；
           alpha为正有限数值（非bool）；无效参数抛ValueError。
        3. self.base保存传入的原层，不重新随机初始化它；冻结其weight和bias。
        4. 设置self.rank、self.alpha、self.scaling=alpha/rank。
        5. self.A为nn.Parameter，形状[out_features,rank]，初始化全零。
        6. self.B为nn.Parameter，形状[rank,in_features]，使用
           nn.init.kaiming_uniform_(..., a=math.sqrt(5))初始化。

        A/B跟随base.weight的device和dtype，可用weight.new_zeros/new_empty。
        A/B必须可训练；不能两者同时为零。支持原层bias=None。
        """
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer必须是nn.Linear")
        if (
        type(rank) is not int
        or rank < 1
        or rank > min(base_layer.in_features, base_layer.out_features)
        ):
            raise ValueError("rank 超出合法范围")

        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("alpha 必须是正有限数值")

        self.base = base_layer
        self.base.requires_grad_(False)

        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.A = nn.Parameter(
            base_layer.weight.new_zeros(
                base_layer.out_features, 
                rank
            )
        )
        self.B = nn.Parameter(
            base_layer.weight.new_empty(
                rank,
                base_layer.in_features
            )
        )
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        """TODO 2：原分支与低秩分支相加，保留梯度。

        x:[...,in] -> F.linear(x,B):[...,rank] -> F.linear(...,A):[...,out]。
        返回 self.base(x) + self.scaling * delta。
        支持二维/三维输入，不写死batch或sequence维；沿用nn.Linear语义。
        不先构造完整A@B，不detach，不用no_grad包住base分支。
        原层权重被冻结，但梯度仍需要穿过它回传到前面的可训练层。
        """
        base_out = self.base(x)

        low_rank = F.linear(x, self.B)
        delta = F.linear(low_rank, self.A)

        return base_out + self.scaling * delta


def inject_lora(model, rank=8, alpha=16.0):
    """TODO 3：原地替换目标FFN线性层，返回本次新替换的完整模块路径列表。

    本函数只负责替换，不负责冻结整个模型；外层setup_lora已安排冻结顺序。
    递归遍历list(parent.named_children())，用setattr替换子层。
    - 仅名称恰为TARGET_NAMES成员的nn.Linear被替换，其他层不变。
    - 已是LoRALinear则跳过，不进入其base，不重复包装。
    - 目标名称对应其他类型时抛TypeError。
    - 返回示例 ['blocks.0.gate_proj','blocks.0.up_proj',...]。
      无新目标返回[]；重复调用返回[]，不创建新参数。
    提示：内部递归函数接收parent与prefix，将层级名称拼成完整路径。
    不调用任何LoRA/PEFT现成实现。
    """
    replaced = []
    def visit(parent, prefix):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if name in TARGET_NAMES:
                if not isinstance(child, nn.Linear):
                    raise TypeError(f"{full_name}不是nn.Linear")
                new_layer = LoRALinear(child, rank, alpha)
                setattr(parent, name, new_layer)
                replaced.append(full_name)
            else:
                visit(child, full_name)
    visit(model, "")
    return replaced


def trainable_parameters(model):
    """TODO 4：返回requires_grad=True的参数列表，供优化器使用。

    参数对象必须是模型的原Parameter引用，不能clone或detach。
    没有可训练参数时抛ValueError。训练前可检查名称是否仅以.A或.B结尾。
    """
    params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not params:
        raise ValueError("模型没有可训练参数")
    return params


def setup_lora(model, rank=8, alpha=16.0):
    """首次配置接口：必须先冻结，再注入，防止冻结刚创建的A/B。

    已配置模型拒绝再次setup；如只检查重复替换行为，可直接调用inject_lora。
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('模型已配置LoRA，请勿重复setup')
    # 提前核验目标与尺寸，避免预期的配置错误发生在冻结或部分替换之后。
    targets = [(name,m) for name,m in model.named_modules()
               if name.rsplit('.',1)[-1] in TARGET_NAMES]
    if not targets:
        raise ValueError('未发现FFN目标层')
    if type(rank) is not int or rank < 1:
        raise ValueError('rank必须为正整数')
    if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or alpha<=0:
        raise ValueError('alpha必须为正有限数值')
    for name,m in targets:
        if not isinstance(m,nn.Linear):
            raise TypeError(f'{name}不是nn.Linear')
        if rank>min(m.in_features,m.out_features):
            raise ValueError(f'rank超出{name}的输入/输出维度')
    model.requires_grad_(False)
    paths=inject_lora(model,rank,alpha)
    params=trainable_parameters(model)
    return dict(replaced=paths,trainable=sum(p.numel() for p in params),
                total=sum(p.numel() for p in model.parameters()))
