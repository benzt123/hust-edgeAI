"""单次采样的 MATH baseline；默认固定随机抽取 50 道测试题。"""

import argparse
import hashlib
import json
import re
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def build_prompt(question, system, user):
    # 使用文档文本模板：user 填入 system 的 instruction 占位符。
    # 不额外套聊天模板，vLLM 内部负责分词。
    return system.replace("{instruction}", user.replace("{question}", question))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--data-dir", type=Path, default=Path("/root/autodl-tmp/hendrycks_math"))
    parser.add_argument("--limit", type=int, default=50, help="0 表示全部测试题")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--preview", action="store_true", help="不下载、不使用 GPU，只预览 prompt")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.max_tokens < 1:
        parser.error("limit 必须非负，batch-size 和 max-tokens 必须为正")

    # utf-8-sig 同时兼容普通 UTF-8 和 Windows 编辑器添加的 BOM。
    system = (ROOT / "system.txt").read_text(encoding="utf-8-sig").rstrip()
    user = (ROOT / "user.txt").read_text(encoding="utf-8-sig").rstrip()
    if "{instruction}" not in system or "{question}" not in user:
        raise ValueError("system.txt 必须含 {instruction}，user.txt 必须含 {question}")
    if args.preview:
        print(build_prompt("What is 2 + 3?", system, user))
        return

    # 放在 preview 后导入，因此本地无卡也能预览。
    from datasets import concatenate_datasets, load_dataset
    from drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn
    from vllm import LLM, SamplingParams

    # 合并全部学科再抽样，避免只测到一个学科。
    parts = []
    for subject in SUBJECTS:
        files = sorted((args.data_dir / subject).glob("test-*.parquet"))
        if not files:
            raise FileNotFoundError(f"未找到 {args.data_dir / subject}/test-*.parquet")
        part = load_dataset("parquet", data_files={"test": [str(p) for p in files]}, split="test")
        part = part.add_column("sample_id", [f"{subject}/test/{i}" for i in range(len(part))])
        parts.append(part)
    dataset = concatenate_datasets(parts).shuffle(seed=args.seed)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # 先解析标准解答中的最终数学答案；解析失败直接报错，不悄悄算错。
    golds = []
    for row in dataset:
        gold = extract_boxed_answer(row["solution"])
        if gold is None or not gold.strip():
            raise ValueError(f"标准答案无法解析：{row['sample_id']}")
        golds.append(gold)

    # 每次单独创建结果目录，避免覆盖之前的实验。
    out = args.out_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    prompts = [build_prompt(row["problem"], system, user) for row in dataset]
    (out / "prompt_example.txt").write_text(prompts[0], encoding="utf-8")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        out_dir=str(out), dataset="EleutherAI/hendrycks_math", split="test",
        dataset_fingerprint=dataset._fingerprint,
        sample_ids=list(dataset["sample_id"]), grader="drgrpo_grader.r1_zero_reward_fn", fast=True,
        temperature=1.0, top_p=1.0, samples_per_question=1,
        system_template=system, user_template=user,
        prompt_sha256=hashlib.sha256((system + user).encode()).hexdigest(),
        versions={p: version(p) for p in ["vllm", "torch", "transformers", "datasets", "math-verify"]},
    )
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 加载 Base 模型，--model 也可以传入以后保存的 SFT 模型目录。
    llm = LLM(
        model=args.model, dtype="auto", seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # 不静默截断输入，防止不同阶段实际评估了不同的题面。
    tokenizer = llm.get_tokenizer()
    for row, prompt in zip(dataset, prompts):
        if len(tokenizer.encode(prompt)) + args.max_tokens > args.max_model_len:
            raise ValueError(f"{row['sample_id']} 超过长度预算，请增大 --max-model-len")
    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    correct = formatted = truncated = parse_failures = 0
    # 每批立即写盘，长任务中断时已经完成的回答仍在文件里。
    with (out / "results.jsonl").open("w", encoding="utf-8") as file:
        for start in range(0, len(dataset), args.batch_size):
            outputs = llm.generate(prompts[start:start + args.batch_size], sampling)
            for offset, output in enumerate(outputs):
                index = start + offset
                row = dataset[index]
                generation = output.outputs[0]
                response = generation.text
                # 优先评分 answer 标签内的内容；缺标签时尝试解析完整回答。
                matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.S)
                scores = r1_zero_reward_fn(response, golds[index], fast=True)
                is_correct = scores["answer_reward"] == 1.0
                has_format = scores["format_reward"] == 1.0
                is_truncated = generation.finish_reason == "length"
                correct += is_correct
                formatted += has_format
                truncated += is_truncated
                parse_failures += not bool(matches)
                record = dict(
                    sample_id=row["sample_id"], question=row["problem"],
                    solution=row["solution"], ground_truth=golds[index], response=response,
                    scores=scores,
                    correct=is_correct, format_ok=has_format,
                    answer_tag_missing=not bool(matches),
                    finish_reason=generation.finish_reason,
                    output_tokens=len(generation.token_ids),
                )
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            done = min(start + args.batch_size, len(dataset))
            print(f"已完成 {done}/{len(dataset)}，当前准确率 {correct / done:.2%}")

    summary = dict(
        total=len(dataset), correct=correct, accuracy=correct / len(dataset),
        format_rate=formatted / len(dataset), truncation_rate=truncated / len(dataset),
        answer_tag_missing_count=parse_failures, grader="drgrpo_grader.r1_zero_reward_fn", fast=True,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录：{out}")


if __name__ == "__main__":
    main()
