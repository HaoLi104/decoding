"""
单模型 Baseline 评测脚本
支持 MedQA-USMLE 和 JEC-QA（中国司法考试）两个数据集。

用法：
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=1

  # MedQA — Target-only
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \\
      --dataset medqa --limit 200 \\
      --out results/baseline/target_only_medqa_200.json

  # JEC-QA — Target-only（确认 32B 法律 baseline）
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \\
      --dataset jecqa --limit 200 \\
      --out results/baseline/target_only_jecqa_200.json

  # JEC-QA — Base-3B
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct \\
      --dataset jecqa --limit 200 \\
      --out results/baseline/base_only_jecqa_200.json

  # JEC-QA — Draft-Law-3B（微调后）
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Law \\
      --dataset jecqa --limit 200 \\
      --out results/baseline/draft_law_jecqa_200.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 使用与 run_benchmark.py 完全相同的 prompt 格式与数据加载
from data_loader import format_prompt, load_jecqa, load_medqa


# ---------------------------------------------------------------------------
# 答案抽取（与 run_benchmark.py 对齐，搜索末尾 1500 字符）
# ---------------------------------------------------------------------------

_RE_THINK        = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_STRONG_PATTERNS = [
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"Final\s+answer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\bAnswer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\s*[.)]\s*", re.IGNORECASE),
]
_RE_LASTLINE_LETTER       = re.compile(r"^\s*(?:option\s*)?([A-D])\s*$", re.IGNORECASE)
_RE_LASTLINE_LETTER_PUNCT = re.compile(r"^\s*(?:option\s*)?([A-D])\s*[.)]\s*$", re.IGNORECASE)


def extract_answer(response: str, tail_chars: int = 1500) -> str:
    """从模型输出中抽取 A/B/C/D（与 run_benchmark.py 对齐）。"""
    if not response:
        return ""
    text  = _RE_THINK.sub("", response)
    m     = _RE_ANSWER_BLOCK.search(text)
    scope = m.group(1) if m else text[-tail_chars:]

    for pat in _STRONG_PATTERNS:
        matches = list(pat.finditer(scope))
        if matches:
            return matches[-1].group(1).upper()

    lines = [ln.strip() for ln in scope.splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        mm = _RE_LASTLINE_LETTER.match(ln) or _RE_LASTLINE_LETTER_PUNCT.match(ln)
        if mm:
            return mm.group(1).upper()

    return ""


# ---------------------------------------------------------------------------
# 模型加载（严格单卡 bfloat16，符合 .cursorrules）
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    print(f"[加载模型] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,     # 注意：torch_dtype，不是 dtype
        device_map="cuda:0",
    )
    model.eval()
    return tokenizer, model


def build_prompt(item: dict, tokenizer, dataset_name: str = "medqa") -> str:
    """使用 data_loader.format_prompt()，与 run_benchmark.py 完全相同的 prompt。"""
    return format_prompt(tokenizer, item["question"], item["options"], dataset_name=dataset_name)


# ---------------------------------------------------------------------------
# 评测主循环
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int = 256,
    batch_size: int = 1,   # 严格 batch_size=1，避免 padding 导致准确率下降
    dataset_name: str = "medqa",
) -> dict:
    device = next(model.parameters()).device

    correct      = 0
    total_tokens = 0
    total_time   = 0.0
    results      = []

    for idx in tqdm(range(len(dataset)), desc="评测"):
        item = dataset[idx]

        prompt_text = build_prompt(item, tokenizer, dataset_name=dataset_name)
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids   = enc["input_ids"].to(device)      # [1, L_in]
        prompt_len  = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        gen_ids  = out_ids[0, prompt_len:]             # [L_gen]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 统计 token 数（不含 eos）
        n_gen = (gen_ids != tokenizer.eos_token_id).sum().item()
        total_tokens += n_gen
        total_time   += elapsed

        pred = extract_answer(gen_text)
        # GBaker/MedQA-USMLE-4-options 答案字段为 answer_idx（A/B/C/D）
        gold = str(item.get("answer_idx", item.get("answer", ""))).strip().upper()
        is_correct = (pred == gold) and (pred != "")

        correct += int(is_correct)
        results.append({
            "question": item["question"][:120],
            "gold":     gold,
            "pred":     pred,
            "correct":  is_correct,
            "gen_text": gen_text[:400],
        })

    n          = len(results)
    accuracy   = correct / n if n else 0.0
    tokens_sec = total_tokens / total_time if total_time else 0.0

    return {
        "n_cases":       n,
        "n_correct":     correct,
        "accuracy":      accuracy,
        "tokens_per_sec": tokens_sec,
        "total_tokens":  total_tokens,
        "total_time_s":  round(total_time, 2),
        "samples":       results,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="单模型 Baseline 评测（MedQA / JEC-QA）")
    parser.add_argument("--model",   required=True, help="模型路径")
    parser.add_argument("--dataset", default="medqa", choices=["medqa", "jecqa"],
                        help="评测数据集：medqa（默认）或 jecqa（中国司法考试）")
    parser.add_argument("--limit",   type=int, default=200, help="评测样本数")
    parser.add_argument("--split",   default="test", choices=["train", "test"])
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out",     required=True, help="结果 JSON 输出路径")
    args = parser.parse_args()

    if args.dataset == "jecqa":
        print(f"[加载数据集] JEC-QA（AGIEval KD+CA，单选）limit={args.limit}")
        dataset = load_jecqa(limit=args.limit)
    else:
        print(f"[加载数据集] MedQA-USMLE split={args.split} limit={args.limit}")
        dataset = load_medqa(split=args.split, limit=args.limit)
    print(f"  共 {len(dataset)} 条样本")

    tokenizer, model = load_model(args.model)

    print("[开始评测]")
    result = evaluate(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
    )

    model_name = Path(args.model).name
    result["model"]      = model_name
    result["model_path"] = args.model
    result["dataset"]    = args.dataset

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"  模型:        {model_name}")
    print(f"  数据集:      {args.dataset}")
    print(f"  样本数:      {result['n_cases']}")
    print(f"  Accuracy:    {result['accuracy']:.4f}  ({result['n_correct']}/{result['n_cases']})")
    print(f"  Tokens/sec:  {result['tokens_per_sec']:.1f}")
    print(f"  结果已保存:  {args.out}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
