"""DAF — 单 Target 评测器（M5 Go/No-Go + 飞轮终评）

只加载 Target，禁用 DSSD（不引入 Draft/Base），用 forward+StaticCache 手写贪婪
解码循环（与第一点 [run_baseline.py] 同口径），评测：
  1. MedMCQA Surgery validation（n=200，与第一点对齐）→ surgery_acc
  2. MMLU 子集（默认 5 科目 × 100 = 500 题）→ mmlu_acc（守护通用域 / 防遗忘）

输出：
  eval_round{k}.json：
    {
      "round_id": int,
      "target_model": str,
      "surgery": {"acc": float, "n": int, "tokens_per_sec": float, ...},
      "mmlu":    {"acc": float, "n": int, "subjects": [...]},
      "tps_pure_target": float
    }

用法（远端 H200）：
  cd /data/ocean/decoding
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=0
  python -m daf.run_eval_round \
      --target_model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v1 \
      --round 1 \
      --surgery_limit 200 --mmlu_limit_per_subject 100 \
      --out logs/daf_round0/eval_round1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers.cache_utils import StaticCache

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from data_loader import format_prompt, load_medmcqa, load_mmlu  # noqa: E402
from evaluator import extract_answer  # noqa: E402
from forward_ops import decode_step, prefill  # noqa: E402
from model_loader_v2 import load_single_model, load_tokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.run_eval_round")


# ---------------------------------------------------------------------------
# MMLU 通用域守护：默认 5 科覆盖 STEM/人文/社科/医学/法律，约 500 题
# ---------------------------------------------------------------------------

_DEFAULT_MMLU_SUBJECTS = [
    "high_school_european_history",
    "professional_law",
    "college_computer_science",
    "moral_disputes",
    "clinical_knowledge",
]


# ---------------------------------------------------------------------------
# MMLU prompt 构造（多选 → A/B/C/D，与第一点 evaluator 一致的极简模板）
# ---------------------------------------------------------------------------

def _format_mmlu_prompt(tokenizer, question: str, choices: List[str]) -> str:
    opts = {chr(65 + i): str(c).strip() for i, c in enumerate(choices[:4])}
    sys_prompt = (
        "You are an expert. Reason concisely (within 3 sentences) in English. "
        "Always end with a single line: 'Final answer: X' where X is A/B/C/D. "
        "Do not add any text after that line."
    )
    user_content = (
        question.strip()
        + "\n"
        + "\n".join(f"{k}. {v}" for k, v in opts.items())
        + "\n\nAfter brief reasoning, end with a single line "
        "'Final answer: X' where X is one of A/B/C/D."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# 纯 Target 贪婪生成（与 run_benchmark.run_pure_target_baseline 同口径）
# ---------------------------------------------------------------------------

def _greedy_generate(
    model:        torch.nn.Module,
    tokenizer,
    prompt_text:  str,
    cache:        StaticCache,
    device:       torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """单条样本贪婪生成，返回 {response, tokens, duration_sec, tps}。

    严格使用 model.forward() / forward_ops，不调用 model.generate()。
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: List[int] = []

    t0 = time.perf_counter()
    last_logits = prefill(model, prompt_ids, cache)  # shape: [1, V]
    seq_len = prompt_len

    while len(generated) < max_new_tokens:
        next_token = int(last_logits.argmax(dim=-1).item())
        generated.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        last_logits = decode_step(
            model=model, token_id=token_tensor, cache=cache, position_id=seq_len,
        )
        seq_len += 1

    elapsed = time.perf_counter() - t0
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "response":         response,
        "n_generated":      len(generated),
        "duration_sec":     elapsed,
        "tokens_per_sec":   (len(generated) / elapsed) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 评测：Surgery val
# ---------------------------------------------------------------------------

def eval_surgery(
    model, tokenizer, cache, device,
    limit: int, max_new_tokens: int,
    out_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info("=== 评测 MedMCQA Surgery validation  n=%d ===", limit)
    raw = load_medmcqa(split="validation", limit=limit, subject="Surgery")
    correct = 0
    total_tokens = 0
    total_time   = 0.0
    n            = 0

    fh = out_jsonl.open("w", encoding="utf-8") if out_jsonl else None
    for idx, item in enumerate(raw):
        q  = item.get("question", "")
        opts = item.get("options", {})
        gt = str(item.get("answer_idx", "")).strip().upper()
        if not q or not opts or gt not in {"A", "B", "C", "D"}:
            continue

        prompt = format_prompt(tokenizer, q, opts, dataset_name="medmcqa")
        # 清空 cache 内容（同对象，避免 CUDAGraph 兼容问题）
        # 简化：每次都重新前向 prompt（cache.reset 的语义需 cache_manager；这里用 SeqLen 重置）
        try:
            cache.reset()
        except Exception:
            pass
        gen = _greedy_generate(
            model, tokenizer, prompt, cache, device, max_new_tokens=max_new_tokens,
        )
        pred = extract_answer(gen["response"])
        is_ok = (pred.upper() == gt)
        if is_ok:
            correct += 1
        total_tokens += gen["n_generated"]
        total_time   += gen["duration_sec"]
        n            += 1

        if fh:
            fh.write(json.dumps({
                "idx":  idx, "gt": gt, "pred": pred, "correct": is_ok,
                "n_generated": gen["n_generated"],
                "duration_sec": gen["duration_sec"],
                "tps": gen["tokens_per_sec"],
                "response": gen["response"][:500],
            }, ensure_ascii=False) + "\n")
            fh.flush()

        if (idx + 1) % 20 == 0:
            logger.info("  surgery %d/%d  acc=%.3f  tps=%.1f",
                        idx + 1, limit, correct / max(n, 1),
                        total_tokens / total_time if total_time > 0 else 0.0)

    if fh:
        fh.close()

    return {
        "acc":            correct / max(n, 1),
        "correct":        correct,
        "n":              n,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0.0,
        "total_tokens":   total_tokens,
        "total_time_sec": total_time,
    }


# ---------------------------------------------------------------------------
# 评测：MMLU 多科目
# ---------------------------------------------------------------------------

def eval_mmlu(
    model, tokenizer, cache, device,
    subjects: List[str], limit_per_subject: int, max_new_tokens: int,
    out_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info("=== 评测 MMLU 守护集  subjects=%s  limit_per_subject=%d ===",
                subjects, limit_per_subject)

    fh = out_jsonl.open("w", encoding="utf-8") if out_jsonl else None
    per_subject: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_n       = 0

    for sub in subjects:
        try:
            ds = load_mmlu(sub, split="test", limit=limit_per_subject)
        except Exception as exc:
            logger.warning("MMLU 科目 %s 加载失败: %s", sub, exc)
            continue

        s_correct = 0
        s_n       = 0
        for idx, item in enumerate(ds):
            q       = str(item.get("question", "")).strip()
            choices = list(item.get("choices", []))
            ans_idx = item.get("answer", -1)
            if not q or len(choices) < 2 or not (0 <= int(ans_idx) < len(choices)):
                continue
            gt = chr(65 + int(ans_idx))

            prompt = _format_mmlu_prompt(tokenizer, q, choices)
            try:
                cache.reset()
            except Exception:
                pass
            gen = _greedy_generate(
                model, tokenizer, prompt, cache, device, max_new_tokens=max_new_tokens,
            )
            pred = extract_answer(gen["response"])
            is_ok = (pred.upper() == gt)
            if is_ok:
                s_correct += 1
            s_n += 1

            if fh:
                fh.write(json.dumps({
                    "subject": sub, "idx": idx, "gt": gt, "pred": pred, "correct": is_ok,
                    "n_generated": gen["n_generated"], "duration_sec": gen["duration_sec"],
                    "response": gen["response"][:500],
                }, ensure_ascii=False) + "\n")
                fh.flush()

        per_subject[sub] = {"acc": s_correct / max(s_n, 1), "correct": s_correct, "n": s_n}
        logger.info("  MMLU %s acc=%.3f (%d/%d)", sub, per_subject[sub]["acc"], s_correct, s_n)
        total_correct += s_correct
        total_n       += s_n

    if fh:
        fh.close()

    return {
        "acc":          total_correct / max(total_n, 1),
        "correct":      total_correct,
        "n":            total_n,
        "subjects":     per_subject,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF — 纯 Target 评测 + MMLU 通用域守护")
    p.add_argument("--target_model", required=True, help="待评测的 Target 模型路径（含 v0/v1/v2）")
    p.add_argument("--round",        type=int, required=True, help="飞轮轮次（用于命名 / 元数据）")
    p.add_argument("--out",          required=True, help="eval_round{k}.json 输出路径")

    p.add_argument("--surgery_limit",         type=int, default=200,
                   help="MedMCQA Surgery val 评测样本数（plan: 200）")
    p.add_argument("--mmlu_subjects",         nargs="+", default=_DEFAULT_MMLU_SUBJECTS)
    p.add_argument("--mmlu_limit_per_subject", type=int, default=100,
                   help="每个 MMLU 科目评测样本数（默认 100，5 科 ≈ 500 题）")
    p.add_argument("--max_new_tokens",        type=int, default=256)
    p.add_argument("--max_cache_len",         type=int, default=2048)

    p.add_argument("--skip_mmlu", action="store_true", help="只跑 surgery，跳过 MMLU 守护")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    logger.info("加载评测 Target: %s", args.target_model)
    tokenizer = load_tokenizer(args.target_model)
    model = load_single_model(
        model_path=args.target_model,
        device=device,
        dtype=torch.bfloat16,
        compile_mode=None,
    )
    model.eval()

    # 单 cache 复用整个评测；依赖 StaticCache.reset() 清空
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=args.max_cache_len,
        device=device,
        dtype=torch.bfloat16,
    )

    # ---------- Surgery ----------
    surgery_jsonl = out_path.with_name(f"eval_round{args.round}_surgery.jsonl")
    surgery_res = eval_surgery(
        model, tokenizer, cache, device,
        limit=args.surgery_limit,
        max_new_tokens=args.max_new_tokens,
        out_jsonl=surgery_jsonl,
    )

    # ---------- MMLU ----------
    if args.skip_mmlu:
        mmlu_res = {"acc": None, "n": 0, "subjects": {}, "skipped": True}
    else:
        mmlu_jsonl = out_path.with_name(f"eval_round{args.round}_mmlu.jsonl")
        mmlu_res = eval_mmlu(
            model, tokenizer, cache, device,
            subjects=args.mmlu_subjects,
            limit_per_subject=args.mmlu_limit_per_subject,
            max_new_tokens=args.max_new_tokens,
            out_jsonl=mmlu_jsonl,
        )

    payload: Dict[str, Any] = {
        "round_id":         args.round,
        "target_model":     args.target_model,
        "surgery":          surgery_res,
        "mmlu":             mmlu_res,
        "tps_pure_target":  surgery_res.get("tokens_per_sec"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✓ Round %d 评测写入 %s  surgery_acc=%.4f  mmlu_acc=%s",
                args.round, out_path, surgery_res["acc"],
                f"{mmlu_res['acc']:.4f}" if mmlu_res.get("acc") is not None else "skipped")


if __name__ == "__main__":
    main()
