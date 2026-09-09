"""Explicit download step, only needed if train Parquet is still an LFS pointer."""
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--endpoint', default='https://huggingface.co')
args = parser.parse_args()
os.environ['HF_ENDPOINT'] = args.endpoint
os.environ['HF_HOME'] = '/root/autodl-tmp/hf-cache'
os.environ.pop('HF_HUB_OFFLINE', None)
from huggingface_hub import snapshot_download
snapshot_download('openai/gsm8k', repo_type='dataset',
                  local_dir='/root/autodl-tmp/gsm8k',
                  allow_patterns=['main/train-*.parquet'])
