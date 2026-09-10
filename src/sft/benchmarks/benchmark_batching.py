"""空闲GPU上测相同8条样本的micro-batch=1/2/4；不保存训练权重。"""
import sys
from pathlib import Path
SFT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_ROOT))
sys.path.insert(0, str(SFT_ROOT / "runtime"))
import json
import time
import statistics
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from full_data import load_rows,make_split,encode_rows
from batching import backward_group


config=json.loads((SFT_ROOT / 'runtime/server_run_config.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
rows,_=make_split(load_rows(config['data_dir']),config['validation_fraction'],config['seed'])
encoded,_=encode_rows(rows,tokenizer,(SFT_ROOT / 'runtime/user.txt').read_text().strip(),config)
model=AutoModelForCausalLM.from_pretrained(config['model'],torch_dtype=torch.float32,
    attn_implementation='sdpa',local_files_only=True).to('cuda')
model.config.use_cache=False
model.gradient_checkpointing_enable()
model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
results=[]
for batch in [1,2,4]:
    durations=[]
    try:
        for repetition in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started=time.monotonic()
            backward_group(model,encoded[repetition*8:repetition*8+8],tokenizer.eos_token_id,batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            torch.cuda.synchronize()
            if repetition: durations.append(time.monotonic()-started)
        # 最大长度压力检查，使用最长的8条样本。
        optimizer.zero_grad(set_to_none=True)
        backward_group(model,sorted(encoded,key=lambda x:len(x['windows'][0][0]),reverse=True)[:8],tokenizer.eos_token_id,batch)
        optimizer.step()
        peak=torch.cuda.max_memory_allocated()/2**30
        result=dict(micro_batch_size=batch,seconds=statistics.mean(durations),peak_gib=peak)
        results.append(result)
        print('BENCHMARK',result,flush=True)
    except torch.cuda.OutOfMemoryError:
        print('OOM',batch,flush=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        break
Path(__file__).with_name('batch_benchmark.json').write_text(json.dumps(results,indent=2))
