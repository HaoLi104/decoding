#!/usr/bin/env python3
"""
probe_substitution.py — Target 高熵位置直接替换为 Draft Token 的准确率因果验证

实验目标（因果证据）：
  在生成过程中，当 Target 的逐步 Shannon 熵 H_t > 阈值时，
  直接用 Draft 的贪婪预测 token 替换 Target 的选择。
  - 替换后准确率上升 → Target 高熵 = 领域盲区，Draft 能纠正（直接因果证据）
  - 替换后准确率下降 → 硬替换破坏推理链，需要"软引导"而非"硬替换"

实验变体：
  pure_target   — 全程 Target 贪婪（基准）
  entropy_sub   — H_t > H_thresh 时替换为 Draft token，扫描多阈值
  pure_draft    — 全程 Draft 贪婪（ablation：Draft 单独能力上限）

运行命令（远端）：
  cd /data/ocean/decoding && git pull
  export CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1
  python probe_substitution.py \\
    --subject Surgery --limit 200 --max_new 150 \\
    --h_thresh_grid 0.5 1.0 1.5 2.0 \\
    --out_dir results/entropy_sub

查看结果：
  cat results/entropy_sub/summary.json | python -m json.tool
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s    %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TARGET_PATH = "/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct"
DRAFT_PATH  = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Surgery/checkpoint-1676"


# ──────────────────────────────────────────────────────────────────────────────
# 数据工具
# ──────────────────────────────────────────────────────────────────────────────

def load_data(limit: int, subject: str):
    from data_loader import load_medmcqa
    logger.info(f"Loading MedMCQA validation, subject={subject or 'ALL'}, limit={limit}...")
    ds = load_medmcqa(split="validation", limit=limit, subject=subject)
    logger.info(f"Loaded {len(ds)} questions")
    return list(ds)


def format_prompt(item: dict, tokenizer) -> str:
    from data_loader import format_prompt as dl_fmt
    return dl_fmt(tokenizer, question=item["question"], options=item["options"],
                  dataset_name="medmcqa")


def get_correct_label(item: dict) -> str:
    return item.get("answer_idx", "A")


def extract_label(text: str):
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*[:：]\s*([ABCD])", text)
    if m:
        return m.group(1)
    for ch in reversed(text):
        if ch in "ABCD":
            return ch
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 熵工具
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(logits_1d: torch.Tensor) -> float:
    """H = -Σ P(x) log P(x)  # logits_1d: [V] → scalar (nats)"""
    probs = F.softmax(logits_1d.float(), dim=-1)        # [V]
    return (-torch.sum(probs * torch.log(probs + 1e-12))).item()


# ──────────────────────────────────────────────────────────────────────────────
# 核心生成函数
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_one(
    model_target,
    model_draft,        # None → pure_target 模式
    tokenizer,
    input_ids: torch.Tensor,
    max_new: int,
    H_thresh: float,    # None → pure_target 或 pure_draft（由 always_draft 决定）
    always_draft: bool = False,
) -> dict:
    """
    生成单题答案，按策略在高熵位置替换 Draft token。

    策略说明：
      - always_draft=True, H_thresh=None : pure_draft（全程 Draft）
      - always_draft=False, H_thresh=None: pure_target（全程 Target）
      - always_draft=False, H_thresh=X  : entropy_sub（H_t > X 时用 Draft）

    注意：两个模型始终接收相同的 next_id（无论谁"赢"），
          以保证 KV Cache 的上下文一致性。
    """
    generated_ids   = []
    entropies       = []
    n_substituted   = 0

    past_t = past_d = None
    cur_ids = input_ids.clone()                          # [1, prompt_len]

    for _ in range(max_new):
        is_first = (past_t is None and past_d is None)
        ids_in   = cur_ids if is_first else cur_ids[:, -1:]  # [1, 1] 或 [1, seq]

        # ── Target 前向（pure_draft 模式跳过）─────────────────────────────
        logit_t = None
        H       = 0.0
        if model_target is not None:
            out_t  = model_target(input_ids=ids_in, past_key_values=past_t, use_cache=True)
            logit_t = out_t.logits[0, -1, :]              # [V_target]
            past_t  = out_t.past_key_values
            H       = shannon_entropy(logit_t)

        # ── Draft 前向（pure_target 模式跳过）────────────────────────────
        logit_d = None
        if model_draft is not None:
            out_d  = model_draft(input_ids=ids_in, past_key_values=past_d, use_cache=True)
            logit_d = out_d.logits[0, -1, :]              # [V_draft]
            past_d  = out_d.past_key_values

        # ── 选 token ──────────────────────────────────────────────────────
        if always_draft:
            # pure_draft
            next_id = int(logit_d.argmax().item())
            n_substituted += 1
        elif H_thresh is not None and H > H_thresh:
            # entropy_sub：高熵位置替换为 Draft
            next_id = int(logit_d.argmax().item())
            n_substituted += 1
        else:
            # pure_target 或未触发阈值：用 Target
            next_id = int(logit_t.argmax().item())

        generated_ids.append(next_id)
        entropies.append(H)

        next_tensor = torch.tensor([[next_id]], device=cur_ids.device)
        cur_ids     = torch.cat([cur_ids, next_tensor], dim=-1)  # [1, seq+1]

        if next_id == tokenizer.eos_token_id:
            break

    return {
        "generated_ids":   generated_ids,
        "n_substituted":   n_substituted,
        "sub_rate":        n_substituted / max(1, len(generated_ids)),
        "mean_entropy":    float(np.mean(entropies)) if entropies else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 单策略评估
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(
    model_target, model_draft, tokenizer,
    items, device, max_new,
    H_thresh, always_draft, strategy_name,
):
    """在 items 上跑完整评估，返回 (accuracy, mean_sub_rate, records)。"""
    records = []
    n_correct = 0

    for idx, item in enumerate(items):
        correct_label = get_correct_label(item)
        text      = format_prompt(item, tokenizer)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        gen = generate_one(
            model_target=model_target,
            model_draft=model_draft,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new=max_new,
            H_thresh=H_thresh,
            always_draft=always_draft,
        )

        gen_text   = tokenizer.decode(gen["generated_ids"], skip_special_tokens=True)
        pred_label = extract_label(gen_text)
        is_correct = (pred_label == correct_label) if pred_label else False
        if is_correct:
            n_correct += 1

        records.append({
            "idx":          idx,
            "correct":      correct_label,
            "pred":         pred_label,
            "is_correct":   is_correct,
            "sub_rate":     round(gen["sub_rate"], 4),
            "mean_H":       round(gen["mean_entropy"], 4),
            "n_sub":        gen["n_substituted"],
            "gen_len":      len(gen["generated_ids"]),
        })

        if (idx + 1) % 10 == 0 or idx == len(items) - 1:
            current_acc = n_correct / (idx + 1)
            logger.info(
                f"  [{strategy_name}] {idx+1:>3}/{len(items)} "
                f"acc={current_acc:.3f}  "
                f"sub_rate={gen['sub_rate']:.3f}  "
                f"mean_H={gen['mean_entropy']:.3f}"
            )

    acc      = n_correct / max(1, len(items))
    sub_rate = float(np.mean([r["sub_rate"] for r in records]))
    return acc, sub_rate, records


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device  = "cuda:0"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 tokenizer ─────────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)

    # ── 加载 Target ────────────────────────────────────────────────────────
    logger.info("Loading Target (32B)...")
    model_target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    ).eval()

    # ── 加载 Draft ────────────────────────────────────────────────────────
    logger.info("Loading Draft (Surgery 3B)...")
    model_draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    ).eval()

    # ── 加载数据 ──────────────────────────────────────────────────────────
    items = load_data(args.limit, args.subject)

    # ── 定义实验变体 ──────────────────────────────────────────────────────
    # 每个 variant: (name, H_thresh, always_draft, need_target, need_draft)
    variants = []

    # Baseline：全程 Target
    variants.append(("pure_target", None, False))

    # 扫描熵阈值：高熵位置替换为 Draft token
    for thr in args.h_thresh_grid:
        variants.append((f"entropy_sub_H{thr:.1f}", thr, False))

    # Ablation：全程 Draft（验证 Draft 单独能力上限）
    variants.append(("pure_draft", None, True))

    # ── 逐变体评估 ────────────────────────────────────────────────────────
    all_results = []
    all_records = {}

    for variant_name, H_thresh, always_draft in variants:
        # pure_target 不需要 Draft；pure_draft 不需要 Target
        mt = None if always_draft else model_target
        md = None if (not always_draft and H_thresh is None and not always_draft) else model_draft
        # 修正：pure_target 时无需 Draft，entropy_sub 和 pure_draft 均需 Draft
        if variant_name == "pure_target":
            mt, md = model_target, None
        elif always_draft:
            mt, md = None, model_draft
        else:
            mt, md = model_target, model_draft

        logger.info(f"\n{'='*60}")
        logger.info(f"策略: {variant_name}  H_thresh={H_thresh}  always_draft={always_draft}")
        logger.info(f"{'='*60}")

        acc, sub_rate, records = evaluate_strategy(
            model_target=mt, model_draft=md,
            tokenizer=tokenizer, items=items,
            device=device, max_new=args.max_new,
            H_thresh=H_thresh, always_draft=always_draft,
            strategy_name=variant_name,
        )

        result = {
            "strategy":     variant_name,
            "H_thresh":     H_thresh,
            "always_draft": always_draft,
            "accuracy":     round(acc, 4),
            "sub_rate":     round(sub_rate, 4),
            "n_questions":  len(items),
        }
        all_results.append(result)
        all_records[variant_name] = records

        logger.info(
            f"\n>>> {variant_name}: acc={acc:.4f}  sub_rate={sub_rate:.4f}"
        )

    # ── 打印汇总表 ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("实验汇总（Target 高熵位置 Draft Token 直接替换实验）")
    logger.info("=" * 70)
    header = f"{'策略':<28} {'阈值':>7} {'准确率':>8} {'替换率':>8} {'vs baseline':>12}"
    logger.info(header)
    logger.info("-" * 70)

    baseline_acc = next(r["accuracy"] for r in all_results if r["strategy"] == "pure_target")
    for r in all_results:
        delta = r["accuracy"] - baseline_acc
        delta_str = f"{delta:+.4f}" if r["strategy"] != "pure_target" else "  baseline"
        logger.info(
            f"  {r['strategy']:<26} "
            f"{str(r['H_thresh']):>7} "
            f"{r['accuracy']:>8.4f} "
            f"{r['sub_rate']:>8.4f} "
            f"{delta_str:>12}"
        )

    # ── 保存结果 ──────────────────────────────────────────────────────────
    summary = {
        "experiment":   "entropy_substitution_probe",
        "subject":      args.subject,
        "limit":        args.limit,
        "max_new":      args.max_new,
        "h_thresh_grid": args.h_thresh_grid,
        "baseline_acc": baseline_acc,
        "results":      all_results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\n汇总 → {out_dir / 'summary.json'}")

    with open(out_dir / "records.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    logger.info(f"明细 → {out_dir / 'records.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy-Substitution Accuracy Probe")
    parser.add_argument("--subject",   type=str,   default="Surgery",
                        help="MedMCQA 科目（Surgery/Pharmacology/Anatomy）")
    parser.add_argument("--limit",     type=int,   default=200,
                        help="评测题目数量")
    parser.add_argument("--max_new",   type=int,   default=150,
                        help="每题最大生成 token 数")
    parser.add_argument("--h_thresh_grid", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0],
                        help="扫描的熵阈值列表（nats）")
    parser.add_argument("--out_dir",   type=str,   default="results/entropy_sub",
                        help="结果输出目录")
    args = parser.parse_args()
    main(args)
