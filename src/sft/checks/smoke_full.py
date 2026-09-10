"""正式训练前用同一精度、优化器和数据路径验证最长样本及一次参数更新。"""
import sys
from pathlib import Path
SFT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_ROOT))
sys.path.insert(0, str(SFT_ROOT / "runtime"))
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import response_loss, backward_microbatch, optimizer_update
from full_data import load_rows, make_split, encode_rows
from full_validation import forward_window


def main():
    config = json.loads((SFT_ROOT / 'runtime/server_run_config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    rows, _ = make_split(load_rows(config['data_dir']), config['validation_fraction'], config['seed'])
    template = (SFT_ROOT / 'runtime/user.txt').read_text(encoding='utf-8-sig').strip()
    encoded, _ = encode_rows(rows, tokenizer, template, config)
    item = max(encoded, key=lambda x: max(len(w[0]) for w in x['windows']))
    model = AutoModelForCausalLM.from_pretrained(config['model'], torch_dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True).to('cuda')
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    tracked = next(p for n, p in model.named_parameters() if 'layers.0.mlp.down_proj.weight' in n)
    before = tracked.detach().clone()
    for window in item['windows']:
        logits, labels, mask = forward_window(model, window, 'cuda')
        loss = response_loss(logits.float(), labels, mask)
        assert torch.isfinite(loss)
        print('SMOKE_LOSS', float(loss.detach()), flush=True)
        backward_microbatch(loss * (sum(window[1])/item['response_tokens']), 1)
        del logits, loss
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer_update(model, optimizer, config['max_grad_norm'])
    assert not torch.equal(before, tracked), '参数没有更新'
    print('SMOKE_OK', item['row']['id'], 'peak_memory_gib', torch.cuda.max_memory_allocated()/2**30, flush=True)


if __name__ == '__main__':
    main()
