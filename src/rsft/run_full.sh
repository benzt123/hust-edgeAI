#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate posttrain
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf-cache
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python check_core.py
if [ ! -f /root/autodl-fs/posttrain/rsft-full-20260910/identity.json ]; then
    python run_full.py prepare
fi
python run_full.py sample
python run_full.py train
python run_full.py test
