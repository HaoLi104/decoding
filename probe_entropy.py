#!/usr/bin/env python3
"""
probe_entropy.py — Target 模型 Shannon 熵分布分析

实验目标：
  1. 对比做对 / 做错题目的序列平均熵（预期：错题熵更高，Target 更不确定）
  2. 对比领域词位置（ΔP > 阈值）vs 通用词位置的逐步熵
  3. 从错题的高熵 token 中找出与领域知识相关的候选词

输出文件（--out_dir）：
  entropy_records.json       — 每题的逐步熵、ΔP、生成文本原始记录
  summary.json               — 核心统计数字（用于论文表格）
  high_entropy_tokens.json   — 错题中高熵且高 ΔP 的 token Top 列表
  entropy_analysis.png       — 可视化图（需 matplotlib）

运行命令（远端）：
  cd /data/ocean/decoding && git pull
  export CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1
  python probe_entropy.py --limit 100 --max_new 150 --out_dir results/entropy_analysis

  # 仅用 Target 运行（不计算 ΔP，速度更快，约 1/3 时间）：
  python probe_entropy.py --limit 100 --max_new 150 --no_delta --out_dir results/entropy_target_only

查看结果：
  cat results/entropy_analysis/summary.json
  cat results/entropy_analysis/high_entropy_tokens.json | python -m json.tool | head -80
"""

import argparse
import json
import logging
import math
import re
from collections import defaultdict
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

# ──────────────────────────────────────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────────────────────────────────────
TARGET_PATH = "/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct"
DRAFT_PATH  = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Surgery/checkpoint-1676"
BASE_PATH   = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct"
MEDMCQA_CACHE = "/data/ocean/decoding/data/medmcqa_cache/"

CHOICE_LABELS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Answer the following multiple choice question. "
    "At the end, clearly state your final answer in the format: Final answer: X"
)

# ──────────────────────────────────────────────────────────────────────────────
# 数据工具
# ──────────────────────────────────────────────────────────────────────────────

def load_surgery_data(limit: int):
    from datasets import load_dataset
    logger.info("Loading MedMCQA validation set...")
    ds = load_dataset(
        "openlifescienceai/medmcqa",
        split="validation",
        cache_dir=MEDMCQA_CACHE,
    )
    items = [
        item for item in ds
        if item.get("subject_name", "").lower() == "surgery"
    ]
    logger.info(f"Found {len(items)} Surgery questions, using first {min(limit, len(items))}")
    return items[:limit]


def format_prompt(item: dict) -> str:
    """构造与 run_benchmark.py 一致的 MCQ prompt。"""
    q = item["question"]
    opts = {
        "A": item.get("opa", ""),
        "B": item.get("opb", ""),
        "C": item.get("opc", ""),
        "D": item.get("opd", ""),
    }
    msg = f"Question: {q}\n"
    for label, text in opts.items():
        msg += f"{label}. {text}\n"
    return msg


def get_correct_label(item: dict) -> str:
    cop = item.get("cop", 0)
    if isinstance(cop, int):
        return CHOICE_LABELS[cop]
    return str(cop).strip().upper()


def extract_predicted_label(text: str):
    """从生成文本中解析最终答案 (A/B/C/D)。"""
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*[:：]\s*([ABCD])", text)
    if m:
        return m.group(1)
    # fallback：倒序找最后一个单独的 ABCD
    for ch in reversed(text):
        if ch in "ABCD":
            return ch
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 熵 / ΔP 计算工具
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(logits_1d: torch.Tensor) -> float:
    """
    计算单步 Shannon 熵（单位：nats）。
    H = -Σ P(x) log P(x)
    # shape: logits_1d [vocab_size] → scalar
    """
    probs = F.softmax(logits_1d.float(), dim=-1)        # [V]
    ent   = -torch.sum(probs * torch.log(probs + 1e-12))
    return ent.item()


def normalized_entropy(logits_1d: torch.Tensor) -> float:
    """归一化熵 H / H_max，∈ [0, 1]。"""
    V     = logits_1d.shape[-1]
    H_max = math.log(V)                                   # log V (nats)
    return shannon_entropy(logits_1d) / H_max


def delta_p_at(logit_draft: torch.Tensor,
               logit_base:  torch.Tensor,
               token_id:    int,
               t: float = 1.0) -> float:
    """
    ΔP = P_draft(token_id) - P_base(token_id)，温度 t=1.0 固定锐化。
    # logit_draft: [V_draft], logit_base: [V_base]
    """
    dv = logit_draft.shape[-1]
    bv = logit_base.shape[-1]
    if token_id >= dv or token_id >= bv:
        return 0.0
    p_d = F.softmax(logit_draft.float() / t, dim=-1)[token_id].item()
    p_b = F.softmax(logit_base.float()  / t, dim=-1)[token_id].item()
    return p_d - p_b

# ──────────────────────────────────────────────────────────────────────────────
# 主生成循环
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_entropy(
    model_target,
    model_draft,       # None if --no_delta
    model_base,        # None if --no_delta
    tokenizer,
    input_ids: torch.Tensor,
    max_new: int,
) -> dict:
    """
    用 Target 贪婪解码生成，同步记录每步的熵（H）和 ΔP。
    返回 dict:
        generated_ids : List[int]
        token_strs    : List[str]
        entropies     : List[float]   # raw H (nats)
        norm_entropies: List[float]   # H / H_max ∈ [0, 1]
        delta_ps      : List[float]   # ΔP，若 no_delta 则全 0
    """
    generated_ids  = []
    token_strs     = []
    entropies      = []
    norm_entropies = []
    delta_ps       = []

    past_t = past_d = past_b = None
    cur_ids = input_ids.clone()                           # [1, seq_len]

    for step in range(max_new):
        # ── Target 前向 ──────────────────────────────────────────────
        is_first = (past_t is None)
        ids_in   = cur_ids if is_first else cur_ids[:, -1:]

        out_t  = model_target(input_ids=ids_in, past_key_values=past_t, use_cache=True)
        logit_t = out_t.logits[0, -1, :]                 # [V_target]
        past_t  = out_t.past_key_values

        # ── 贪婪选 token ─────────────────────────────────────────────
        next_id  = int(logit_t.argmax().item())
        next_tok = tokenizer.decode([next_id])

        # ── 熵 ───────────────────────────────────────────────────────
        H      = shannon_entropy(logit_t)
        H_norm = normalized_entropy(logit_t)

        # ── ΔP（可选）────────────────────────────────────────────────
        dp = 0.0
        if model_draft is not None and model_base is not None:
            out_d   = model_draft(input_ids=ids_in, past_key_values=past_d, use_cache=True)
            logit_d = out_d.logits[0, -1, :]             # [V_draft]
            past_d  = out_d.past_key_values

            out_b   = model_base(input_ids=ids_in, past_key_values=past_b, use_cache=True)
            logit_b = out_b.logits[0, -1, :]             # [V_base]
            past_b  = out_b.past_key_values

            dp = delta_p_at(logit_d, logit_b, next_id)

        generated_ids.append(next_id)
        token_strs.append(next_tok)
        entropies.append(H)
        norm_entropies.append(H_norm)
        delta_ps.append(dp)

        # 追加 token，准备下一步
        next_tensor = torch.tensor([[next_id]], device=cur_ids.device)
        cur_ids     = torch.cat([cur_ids, next_tensor], dim=-1)  # [1, seq+1]

        if next_id == tokenizer.eos_token_id:
            break

    return {
        "generated_ids":   generated_ids,
        "token_strs":      token_strs,
        "entropies":       entropies,
        "norm_entropies":  norm_entropies,
        "delta_ps":        delta_ps,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 分析 & 报告
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_report(records: list, out_dir: Path):
    correct_recs   = [r for r in records if r["is_correct"]]
    incorrect_recs = [r for r in records if not r["is_correct"]]
    H_max_ref      = math.log(152064)                    # Qwen2.5 词表大小

    logger.info("\n" + "=" * 65)
    logger.info(f"总题数: {len(records)} | 做对: {len(correct_recs)} | 做错: {len(incorrect_recs)}")

    # ── 1. 序列平均熵：做对 vs 做错 ──────────────────────────────────────
    corr_H   = [r["mean_entropy"]      for r in correct_recs]
    incorr_H = [r["mean_entropy"]      for r in incorrect_recs]
    corr_Hn  = [r["mean_norm_entropy"] for r in correct_recs]
    incorr_Hn= [r["mean_norm_entropy"] for r in incorrect_recs]

    logger.info("\n【1. 序列平均熵对比（做对 vs 做错）】")
    logger.info(f"  做对题  raw H  = {np.mean(corr_H):.4f} ± {np.std(corr_H):.4f}  (n={len(corr_H)})")
    logger.info(f"  做错题  raw H  = {np.mean(incorr_H):.4f} ± {np.std(incorr_H):.4f}  (n={len(incorr_H)})")
    logger.info(f"  做对题  H/Hmax = {np.mean(corr_Hn):.4f} ± {np.std(corr_Hn):.4f}")
    logger.info(f"  做错题  H/Hmax = {np.mean(incorr_Hn):.4f} ± {np.std(incorr_Hn):.4f}")

    # 统计检验
    pval_str = "N/A"
    if len(corr_H) >= 5 and len(incorr_H) >= 5:
        try:
            from scipy import stats
            _, pval = stats.mannwhitneyu(incorr_H, corr_H, alternative="greater")
            pval_str = f"{pval:.4f}"
            logger.info(f"  Mann-Whitney U（错题熵 > 对题熵）: p = {pval:.4f}")
        except ImportError:
            logger.warning("  scipy 未安装，跳过统计检验")

    # ── 2. 领域词位置（ΔP > 0.05）vs 通用词位置 熵对比 ─────────────────
    domain_H  = []
    general_H = []
    domain_Hn = []
    general_Hn= []
    has_delta = any(any(dp != 0 for dp in r["delta_ps"]) for r in records)

    if has_delta:
        for r in records:
            for H, Hn, dp in zip(r["entropies"], r["norm_entropies"], r["delta_ps"]):
                if dp > 0.05:
                    domain_H.append(H)
                    domain_Hn.append(Hn)
                else:
                    general_H.append(H)
                    general_Hn.append(Hn)

        logger.info("\n【2. 领域词位置（ΔP>0.05）vs 通用词位置 熵对比】")
        logger.info(f"  领域词 raw H  = {np.mean(domain_H):.4f} ± {np.std(domain_H):.4f}  (n={len(domain_H)})")
        logger.info(f"  通用词 raw H  = {np.mean(general_H):.4f} ± {np.std(general_H):.4f}  (n={len(general_H)})")
        logger.info(f"  领域词 H/Hmax = {np.mean(domain_Hn):.4f}")
        logger.info(f"  通用词 H/Hmax = {np.mean(general_Hn):.4f}")
    else:
        logger.info("\n【2. ΔP 未计算（--no_delta 模式），跳过领域词位置对比】")

    # ── 3. 错题中高熵 token 分析 ─────────────────────────────────────────
    logger.info("\n【3. 错题高熵 Token 分析（H > μ+σ，ΔP > 0，token 长度 ≥ 3）】")
    token_stats = defaultdict(lambda: {"count": 0, "H": [], "Hn": [], "dP": []})
    examples    = []

    if incorrect_recs:
        all_H    = [H for r in incorrect_recs for H in r["entropies"]]
        H_thresh = np.mean(all_H) + np.std(all_H)
        logger.info(f"  高熵阈值 H > {H_thresh:.4f} nats")

        for r in incorrect_recs:
            for step, (tok, H, Hn, dp) in enumerate(
                zip(r["token_strs"], r["entropies"], r["norm_entropies"], r["delta_ps"])
            ):
                if H > H_thresh and (dp > 0 or not has_delta):
                    tok_clean = tok.strip()
                    if len(tok_clean) >= 3:
                        token_stats[tok_clean]["count"] += 1
                        token_stats[tok_clean]["H"].append(H)
                        token_stats[tok_clean]["Hn"].append(Hn)
                        token_stats[tok_clean]["dP"].append(dp)

                        if len(examples) < 100:
                            ctx_start = max(0, step - 5)
                            ctx_end   = min(len(r["token_strs"]), step + 6)
                            ctx       = "".join(r["token_strs"][ctx_start:ctx_end])
                            examples.append({
                                "q_idx":   r["idx"],
                                "token":   tok_clean,
                                "H":       round(H, 4),
                                "H_norm":  round(Hn, 4),
                                "dP":      round(dp, 4),
                                "context": ctx,
                                "correct_label": r["correct_label"],
                                "pred_label":    r["pred_label"],
                            })

        sorted_toks = sorted(
            token_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )

        logger.info(f"  满足条件的唯一 token 种数: {len(token_stats)}")
        logger.info(f"\n  Top-25 高熵领域候选 Token:")
        header = f"  {'Token':<28} {'次数':>6} {'均值H':>9} {'H/Hmax':>8} {'均值ΔP':>9}"
        logger.info(header)
        logger.info("  " + "-" * 65)
        for tok, info in sorted_toks[:25]:
            logger.info(
                f"  {tok:<28} {info['count']:>6} "
                f"{np.mean(info['H']):>9.4f} "
                f"{np.mean(info['Hn']):>8.4f} "
                f"{np.mean(info['dP']):>9.4f}"
            )

        # 保存高熵 token 报告
        token_report = {
            "H_threshold":   float(H_thresh),
            "H_max_ref":     float(H_max_ref),
            "top_tokens": [
                {
                    "token":    tok,
                    "count":    info["count"],
                    "mean_H":   float(np.mean(info["H"])),
                    "mean_Hn":  float(np.mean(info["Hn"])),
                    "mean_dP":  float(np.mean(info["dP"])),
                }
                for tok, info in sorted_toks[:100]
            ],
            "examples": examples,
        }
        with open(out_dir / "high_entropy_tokens.json", "w", encoding="utf-8") as f:
            json.dump(token_report, f, ensure_ascii=False, indent=2)
        logger.info(f"\n  高熵 token 报告 → {out_dir / 'high_entropy_tokens.json'}")

    # ── 4. 保存汇总 summary.json ─────────────────────────────────────────
    summary = {
        "n_total":             len(records),
        "n_correct":           len(correct_recs),
        "n_incorrect":         len(incorrect_recs),
        "accuracy":            round(len(correct_recs) / max(1, len(records)), 4),
        "H_max_ref_nats":      round(H_max_ref, 4),
        "correct_mean_H":      round(float(np.mean(corr_H)),    4) if corr_H    else 0,
        "correct_std_H":       round(float(np.std(corr_H)),     4) if corr_H    else 0,
        "correct_mean_Hn":     round(float(np.mean(corr_Hn)),   4) if corr_Hn   else 0,
        "incorrect_mean_H":    round(float(np.mean(incorr_H)),  4) if incorr_H  else 0,
        "incorrect_std_H":     round(float(np.std(incorr_H)),   4) if incorr_H  else 0,
        "incorrect_mean_Hn":   round(float(np.mean(incorr_Hn)), 4) if incorr_Hn else 0,
        "delta_H_incorrect_vs_correct": round(
            float(np.mean(incorr_H)) - float(np.mean(corr_H)), 4
        ) if corr_H and incorr_H else 0,
        "mannwhitney_p":       pval_str,
        "domain_token_mean_H": round(float(np.mean(domain_H)),  4) if domain_H  else "N/A",
        "general_token_mean_H":round(float(np.mean(general_H)), 4) if general_H else "N/A",
        "domain_token_mean_Hn":round(float(np.mean(domain_Hn)), 4) if domain_Hn else "N/A",
        "general_token_mean_Hn":round(float(np.mean(general_Hn)),4) if general_Hn else "N/A",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n汇总报告 → {out_dir / 'summary.json'}")

    # ── 5. 可视化 ─────────────────────────────────────────────────────────
    _plot(
        corr_H, incorr_H, corr_Hn, incorr_Hn,
        domain_H, general_H,
        out_dir,
    )

    return summary


def _plot(corr_H, incorr_H, corr_Hn, incorr_Hn,
          domain_H, general_H, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 3 if (domain_H and general_H) else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

        # 子图 1：raw H 分布
        ax = axes[0]
        if corr_H:
            ax.hist(corr_H,   bins=25, alpha=0.65, color="steelblue", label=f"Correct (n={len(corr_H)})")
        if incorr_H:
            ax.hist(incorr_H, bins=25, alpha=0.65, color="tomato",    label=f"Incorrect (n={len(incorr_H)})")
        if corr_H:
            ax.axvline(np.mean(corr_H),   color="steelblue", lw=2, linestyle="--")
        if incorr_H:
            ax.axvline(np.mean(incorr_H), color="tomato",    lw=2, linestyle="--")
        ax.set_xlabel("Mean Sequence Entropy (nats)")
        ax.set_ylabel("Count")
        ax.set_title("Target Entropy: Correct vs Incorrect")
        ax.legend()

        # 子图 2：H/Hmax 分布
        ax2 = axes[1]
        if corr_Hn:
            ax2.hist(corr_Hn,   bins=25, alpha=0.65, color="steelblue", label=f"Correct")
        if incorr_Hn:
            ax2.hist(incorr_Hn, bins=25, alpha=0.65, color="tomato",    label=f"Incorrect")
        if corr_Hn:
            ax2.axvline(np.mean(corr_Hn),   color="steelblue", lw=2, linestyle="--")
        if incorr_Hn:
            ax2.axvline(np.mean(incorr_Hn), color="tomato",    lw=2, linestyle="--")
        ax2.set_xlabel("Mean Normalized Entropy (H / H_max)")
        ax2.set_ylabel("Count")
        ax2.set_title("Normalized Entropy: Correct vs Incorrect")
        ax2.legend()

        # 子图 3（可选）：领域词 vs 通用词 逐步熵
        if ncols == 3:
            ax3 = axes[2]
            sample_n = 8000
            if general_H:
                ax3.hist(general_H[:sample_n], bins=35, alpha=0.6, color="steelblue",
                         label=f"General tokens (ΔP≤0.05)")
            if domain_H:
                ax3.hist(domain_H[:sample_n],  bins=35, alpha=0.6, color="orange",
                         label=f"Domain tokens (ΔP>0.05)")
            if general_H:
                ax3.axvline(np.mean(general_H), color="steelblue", lw=2, linestyle="--")
            if domain_H:
                ax3.axvline(np.mean(domain_H),  color="orange",    lw=2, linestyle="--")
            ax3.set_xlabel("Step Entropy (nats)")
            ax3.set_ylabel("Count")
            ax3.set_title("Per-step Entropy: Domain vs General Positions")
            ax3.legend()

        plt.tight_layout()
        path = out_dir / "entropy_analysis.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"可视化图 → {path}")
    except Exception as e:
        logger.warning(f"绘图失败（可忽略）: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda:0"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 tokenizer ─────────────────────────────────────────────────────
    logger.info("Loading tokenizer (Target)...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)

    # ── 加载 Target ────────────────────────────────────────────────────────
    logger.info("Loading Target model (32B)...")
    model_target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    # ── 加载 Draft / Base（可选）──────────────────────────────────────────
    model_draft = model_base = None
    if not args.no_delta:
        logger.info("Loading Draft model (Surgery 3B)...")
        model_draft = AutoModelForCausalLM.from_pretrained(
            DRAFT_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        logger.info("Loading Base model (3B)...")
        model_base = AutoModelForCausalLM.from_pretrained(
            BASE_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        ).eval()
    else:
        logger.info("--no_delta 模式：仅用 Target，不计算 ΔP（速度更快）")

    # ── 加载数据 ───────────────────────────────────────────────────────────
    items = load_surgery_data(args.limit)

    # ── 主循环：逐题生成并记录熵 ──────────────────────────────────────────
    records = []
    for idx, item in enumerate(items):
        correct_label = get_correct_label(item)
        user_msg      = format_prompt(item)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        text      = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        gen = generate_with_entropy(
            model_target, model_draft, model_base,
            tokenizer, input_ids,
            max_new=args.max_new,
        )

        gen_text    = tokenizer.decode(gen["generated_ids"], skip_special_tokens=True)
        pred_label  = extract_predicted_label(gen_text)
        is_correct  = (pred_label == correct_label) if pred_label else False

        mean_H  = float(np.mean(gen["entropies"]))       if gen["entropies"]      else 0.0
        mean_Hn = float(np.mean(gen["norm_entropies"]))  if gen["norm_entropies"] else 0.0

        record = {
            "idx":             idx,
            "question":        item["question"][:120],    # 截断节省空间
            "correct_label":   correct_label,
            "pred_label":      pred_label,
            "is_correct":      is_correct,
            "gen_text":        gen_text,
            "token_strs":      gen["token_strs"],
            "entropies":       [round(h, 5) for h in gen["entropies"]],
            "norm_entropies":  [round(h, 5) for h in gen["norm_entropies"]],
            "delta_ps":        [round(d, 5) for d in gen["delta_ps"]],
            "mean_entropy":    round(mean_H,  5),
            "mean_norm_entropy": round(mean_Hn, 5),
        }
        records.append(record)

        logger.info(
            f"[{idx+1:>3}/{len(items)}] "
            f"pred={pred_label or '?'} gt={correct_label} "
            f"{'✓' if is_correct else '✗'}  "
            f"mean_H={mean_H:.3f}  mean_Hn={mean_Hn:.4f}  "
            f"gen_len={len(gen['generated_ids'])}"
        )

    # ── 保存原始记录 ──────────────────────────────────────────────────────
    raw_path = out_dir / "entropy_records.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"\n原始记录 → {raw_path}")

    # ── 分析 & 报告 ────────────────────────────────────────────────────────
    analyze_and_report(records, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target Shannon Entropy Probe")
    parser.add_argument("--limit",    type=int,  default=100,
                        help="分析题目数量（建议 100-200）")
    parser.add_argument("--max_new",  type=int,  default=150,
                        help="每题最大生成 token 数")
    parser.add_argument("--no_delta", action="store_true",
                        help="不加载 Draft/Base，不计算 ΔP（速度约快 3×）")
    parser.add_argument("--out_dir",  type=str,  default="results/entropy_analysis",
                        help="输出目录")
    args = parser.parse_args()
    main(args)
