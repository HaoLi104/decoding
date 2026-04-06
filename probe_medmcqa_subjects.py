"""
探测 MedMCQA 各科目的 Target-32B 与 Base-3B 准确率。

找出 Target-32B acc 最低的科目，作为下一步专项微调的目标领域。

运行（Target-32B，约 60-80 分钟）：
  cd /data/ocean/decoding && conda activate kvner
  export CUDA_VISIBLE_DEVICES=0
  export HF_DATASETS_OFFLINE=1
  python probe_medmcqa_subjects.py \
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --out results/baseline/target32b_medmcqa_by_subject.json

运行（Base-3B，约 15 分钟）：
  python probe_medmcqa_subjects.py \
      --model /data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct \
      --out results/baseline/base3b_medmcqa_by_subject.json

查看对比汇总：
  python probe_medmcqa_subjects.py --summary_only \
      --target_json results/baseline/target32b_medmcqa_by_subject.json \
      --base_json   results/baseline/base3b_medmcqa_by_subject.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data_loader import format_prompt, load_medmcqa

# ---------------------------------------------------------------------------
# 答案抽取
# ---------------------------------------------------------------------------
_RE_THINK        = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_STRONG_PATTERNS = [
    re.compile(r"Final\s+answer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\bAnswer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
]
_RE_LASTLINE = re.compile(r"^\s*(?:option\s*)?([A-D])\s*[.)]*\s*$", re.IGNORECASE)


def extract_answer(response: str) -> str:
    if not response:
        return ""
    text  = _RE_THINK.sub("", response)
    m     = _RE_ANSWER_BLOCK.search(text)
    scope = m.group(1) if m else text[-1500:]
    for pat in _STRONG_PATTERNS:
        hits = list(pat.finditer(scope))
        if hits:
            return hits[-1].group(1).upper()
    lines = [ln.strip() for ln in scope.splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        mm = _RE_LASTLINE.match(ln)
        if mm:
            return mm.group(1).upper()
    return ""


# ---------------------------------------------------------------------------
# 主评测逻辑
# ---------------------------------------------------------------------------

def eval_by_subject(model_path: str, out_path: str, limit_per_subject: int = 80) -> None:
    """加载整个 MedMCQA validation，按 subject_name 分组评测。"""
    print(f"[加载数据集] MedMCQA validation（全量，按科目分组）")
    # 加载全量 validation，不限 limit（~4183 条）
    dataset = load_medmcqa(split="validation", limit=0)
    print(f"  共 {len(dataset)} 条样本")

    # 按科目分组（每科最多 limit_per_subject 条，避免大科目耗时过长）
    by_subject: dict[str, list] = defaultdict(list)
    for item in dataset:
        subj = item.get("subject_name", "Unknown")
        if len(by_subject[subj]) < limit_per_subject:
            by_subject[subj].append(item)

    subjects = sorted(by_subject.keys())
    print(f"  科目数: {len(subjects)}  每科最多 {limit_per_subject} 条")
    print(f"  科目列表: {subjects}")

    print(f"\n[加载模型] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    device = next(model.parameters()).device

    results: dict[str, dict] = {}

    for subj in subjects:
        items   = by_subject[subj]
        correct = 0
        total_tokens = 0
        total_time   = 0.0

        for item in tqdm(items, desc=f"{subj[:30]:<30}", leave=False):
            prompt = format_prompt(tokenizer, item["question"], item["options"],
                                   dataset_name="medmcqa")
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids  = enc["input_ids"].to(device)    # shape: [1, L_in]
            prompt_len = input_ids.shape[1]

            t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - t0

            gen_ids  = out_ids[0, prompt_len:]          # shape: [L_gen]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            n_gen    = (gen_ids != tokenizer.eos_token_id).sum().item()
            total_tokens += n_gen
            total_time   += elapsed

            pred = extract_answer(gen_text)
            gold = str(item.get("answer_idx", "")).strip().upper()
            correct += int(pred == gold and pred != "")

        n   = len(items)
        acc = correct / n if n else 0.0
        tps = total_tokens / total_time if total_time else 0.0
        results[subj] = {"n": n, "correct": correct, "acc": acc, "tps": tps}
        print(f"  {subj:<40} n={n:>3}  acc={acc:.4f}  tps={tps:.1f}")

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}")

    # 打印排序结果
    print_subject_table(results, label=Path(model_path).name)


def print_subject_table(results: dict, label: str = "") -> None:
    print(f"\n{'='*60}")
    print(f"{'科目':<40} {'n':>4} {'acc':>6}")
    print(f"{'='*60}")
    for subj, r in sorted(results.items(), key=lambda x: x[1]["acc"]):
        print(f"  {subj:<38} {r['n']:>4}  {r['acc']:.4f}")
    print(f"{'='*60}")


def summary_only(target_json: str, base_json: str) -> None:
    t_res = json.loads(Path(target_json).read_text(encoding="utf-8"))
    b_res = json.loads(Path(base_json).read_text(encoding="utf-8"))
    common = sorted(set(t_res) & set(b_res))

    print(f"\n{'='*72}")
    print(f"{'科目':<38} {'n':>4} {'Target':>8} {'Base':>8} {'Gap':>8}")
    print(f"{'='*72}")
    for subj in sorted(common, key=lambda s: t_res[s]["acc"]):
        t = t_res[subj]
        b = b_res[subj]
        print(f"  {subj:<36} {t['n']:>4}  {t['acc']:.4f}  {b['acc']:.4f}  {t['acc']-b['acc']:+.4f}")
    print(f"{'='*72}")
    # 高亮 Target acc ≤ 0.65 的科目
    print("\n>>> Target acc ≤ 0.65 的科目（最有希望的专项微调目标）：")
    for subj in sorted(common, key=lambda s: t_res[s]["acc"]):
        if t_res[subj]["acc"] <= 0.65 and t_res[subj]["n"] >= 30:
            t = t_res[subj]
            b = b_res[subj]
            print(f"  ✅ {subj:<36} Target={t['acc']:.4f}  Base={b['acc']:.4f}  Gap={t['acc']-b['acc']:+.4f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="按科目评测 MedMCQA，找 Target-32B 最弱科目")
    parser.add_argument("--model",       default="", help="模型路径（summary_only 时不需要）")
    parser.add_argument("--out",         default="", help="输出 JSON 路径")
    parser.add_argument("--limit_per_subject", type=int, default=80)
    parser.add_argument("--summary_only",  action="store_true")
    parser.add_argument("--target_json",   default="")
    parser.add_argument("--base_json",     default="")
    args = parser.parse_args()

    if args.summary_only:
        summary_only(args.target_json, args.base_json)
    else:
        if not args.model or not args.out:
            parser.error("--model 和 --out 是必填参数（非 --summary_only 模式）")
        eval_by_subject(args.model, args.out, args.limit_per_subject)


if __name__ == "__main__":
    main()
