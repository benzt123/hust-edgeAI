"""离线验证指定模型，复用训练运行的配置与固定验证题 ID。"""
import sys
from pathlib import Path
SFT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_ROOT))
sys.path.insert(0, str(SFT_ROOT / "runtime"))
import argparse
import json
from pathlib import Path
import torch
from full_data import load_rows,encode_rows
from full_validation import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--all-generation',action='store_true')
    args=parser.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    config=json.loads((args.run/'config.json').read_text())
    if args.all_generation: config['generation_limit']=0
    if args.split=='validation':
        mapping={r['id']:r for r in load_rows(config['data_dir'])}
        ids=json.loads((args.run/'split.json').read_text())['validation_ids']
        rows=[mapping[i] for i in ids]
    else: rows=load_rows(config['data_dir'],'test')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    template=(args.run/'user.txt').read_text(encoding='utf-8')
    encoded,_=encode_rows(rows,tokenizer,template,config)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda')
    result=evaluate(model,tokenizer,encoded,config,args.out)
    (args.out/'evaluation_config.json').write_text(json.dumps(dict(model=args.model,split=args.split,config=config),indent=2))
    print(result)


if __name__=='__main__': main()
