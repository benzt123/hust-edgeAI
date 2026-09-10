#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate posttrain
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
cd "$(dirname "$0")"
python ../checks/check_core.py
python ../checks/test_full.py
python train_full.py "$@"
