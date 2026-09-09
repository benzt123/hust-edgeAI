#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate posttrain
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
status_file=/root/autodl-tmp/sft_experiment_20260908.status
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED %s %s\n" "$code" "$(date -Is)" > "$status_file"; fi' EXIT
printf 'TRAINING %s\n' "$(date -Is)" > "$status_file"
python train_full.py --config server_run_config.json "$@"
run=$(python -c 'import json; from pathlib import Path; c=json.loads(Path("server_run_config.json").read_text()); print(max((p for p in Path(c["output_root"]).iterdir() if (p/"final_model").is_dir()),key=lambda p:p.name))')
printf 'TESTING_BASE %s\n' "$run" > "$status_file"
python validate_full.py --run "$run" --model /root/autodl-tmp/models/Qwen2.5-Math-1.5B --out "$run/test_base" --split test --all-generation
printf 'TESTING_BEST %s\n' "$run" > "$status_file"
python validate_full.py --run "$run" --model "$run/best_model" --out "$run/test_best" --split test --all-generation
printf 'COMPLETE %s\n' "$run" > "$status_file"
