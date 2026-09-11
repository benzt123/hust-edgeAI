#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate posttrain
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf-cache
export OMP_NUM_THREADS=4
python check_core.py
python run_trial.py sample
python run_trial.py train
