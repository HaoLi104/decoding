"""
单模型 Baseline 评测脚本
评测 Target / Draft / Base 三个模型在 MedQA-USMLE 上的 Accuracy 与 Tokens/sec。

用法：
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=1

  # Draft-only (Medical-3B)
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Medical \\
      --limit 200 \\
      --out results/baseline/draft_only_medqa.json

  # Base-only (原始 3B)
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct \\
      --limit 200 \\
      --out results/baseline/base_only_medqa.json

  # Target-only (32B, 需要更多显存 -- 建议 CUDA_VISIBLE_DEVICES=0)
  python run_baseline.py \\
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \\
      --limit 200 \\
      --out results/baseline/target_only_medqa.json
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


# ---------------------------------------------------------------------------
# 答案抽取（兼容多种输出格式）
# ---------------------------------------------------------------------------

_RE_ANSWER_LETTER = [
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*\*{0,2}([A-D])\b", re.I),
    re.compile(r"Final\s+answer\s*[:：]?\s*\*{0,2}([A-D])\b", re.I),
    re.compile(r"\bAnswer\s*[:：]?\s*\*{0,2}([A-D])\b", re.I),
    re.compile(r"^\s*([A-D])\s*[.)]\s*$", re.M),
    re.compile(r"^\s*([A-D])\s*$", re.M),
]


def extract_answer(text: str) -> str:
    """从模型输出中抽取 A/B/C/D，无法抽取时返回 ''。"""
    for pat in _RE_ANSWER_LETTER:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
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
        dtype=torch.bfloat16,           # bfloat16，符合规范
        device_map="cuda:0",            # 严格单卡，符合规范
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# 构建 Prompt（使用 Qwen Chat Template）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical expert. Reason concisely in English. "
    "Always end with a single line: 'Final answer: X' where X is A/B/C/D. "
    "Do not add any text after that line."
)


def build_prompt(item: dict, tokenizer) -> str:
    """将一条 MedQA 样本格式化为 Qwen Chat Template 格式的输入字符串。"""
    question = item["question"]
    options  = item["options"]   # {"A": "...", "B": "...", ...}

    opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    content = f"Question: {question}\n\nOptions:\n{opts_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# 评测主循环
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> dict:
    device = next(model.parameters()).device

    correct      = 0
    total_tokens = 0
    total_time   = 0.0
    results      = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="评测"):
        # HF Dataset 切片返回列字典，需按行索引取出 dict 列表
        end = min(start + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start, end)]

        prompts = [build_prompt(item, tokenizer) for item in batch_items]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids      = enc["input_ids"].to(device)        # [B, L_in]
        attention_mask = enc["attention_mask"].to(device)   # [B, L_in]
        prompt_len     = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 贪婪解码
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        # 只解码新生成的 token
        gen_ids    = out_ids[:, prompt_len:]   # [B, L_gen]
        gen_tokens = gen_ids.numel() - (gen_ids == tokenizer.eos_token_id).sum().item()
        total_tokens += gen_tokens
        total_time   += elapsed

        for i, item in enumerate(batch_items):
            gen_text   = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            pred       = extract_answer(gen_text)
            gold       = str(item.get("answer", "")).strip().upper()
            is_correct = (pred == gold) and (pred != "")

            correct += int(is_correct)
            results.append({
                "question":   item["question"][:120],
                "gold":       gold,
                "pred":       pred,
                "correct":    is_correct,
                "gen_text":   gen_text[:300],
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
    parser = argparse.ArgumentParser(description="单模型 MedQA Baseline 评测")
    parser.add_argument("--model",  required=True, help="模型路径")
    parser.add_argument("--limit",  type=int, default=200, help="评测样本数")
    parser.add_argument("--split",  default="test", choices=["train", "test"])
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out",    required=True, help="结果 JSON 输出路径")
    args = parser.parse_args()

    # 加载数据集（HuggingFace，无需本地文件）
    from data_loader import load_medqa
    print(f"[加载数据集] MedQA-USMLE split={args.split} limit={args.limit}")
    dataset = load_medqa(split=args.split, limit=args.limit)
    print(f"  共 {len(dataset)} 条样本")

    tokenizer, model = load_model(args.model)

    print("[开始评测]")
    result = evaluate(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    model_name = Path(args.model).name
    result["model"] = model_name
    result["model_path"] = args.model

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"  模型:        {model_name}")
    print(f"  样本数:      {result['n_cases']}")
    print(f"  Accuracy:    {result['accuracy']:.4f}  ({result['n_correct']}/{result['n_cases']})")
    print(f"  Tokens/sec:  {result['tokens_per_sec']:.1f}")
    print(f"  结果已保存:  {args.out}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
