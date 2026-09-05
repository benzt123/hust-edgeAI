#!/usr/bin/env bash
# 任一步骤失败就停止。
set -euo pipefail

# 初始化 Conda，允许脚本激活环境。
source "$(conda info --base)/etc/profile.d/conda.sh"

# 创建独立环境；已存在则跳过。
if ! conda run -n posttrain python --version >/dev/null 2>&1; then
    conda create -n posttrain python=3.12 -y
fi

# 激活环境。
conda activate posttrain

# 安装包管理工具。
python -m pip install --upgrade uv

# 安装 vLLM 及其兼容的 PyTorch、Transformers。
uv pip install --python "$CONDA_PREFIX/bin/python" \
    vllm --torch-backend=auto

# 安装数据处理和日志工具。
uv pip install --python "$CONDA_PREFIX/bin/python" \
    datasets huggingface_hub tensorboard matplotlib

# 检查依赖是否冲突。
python -m pip check

# 检查 GPU 和核心库。
python - <<'PY'
import torch
import transformers
import vllm

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
print("CUDA:", torch.version.cuda)

assert torch.cuda.is_available(), "未检测到可用 GPU"
print("GPU:", torch.cuda.get_device_name(0))
PY

# 保存当前安装版本。
python -m pip freeze > requirements-lock.txt

echo "配置完成。请执行：conda activate posttrain"