#!/usr/bin/env python3
"""
probe_entropy.py — Target 模型 Shannon 熵分布分析 + Draft 延续词探针

实验目标：
  1. 对比做对 / 做错题目的序列平均熵（预期：错题熵更高，Target 更不确定）
  2. 对比领域词位置（ΔP > 阈值）vs 通用词位置的逐步熵
  3. 从错题高熵位置找出 Target 最困惑的实义词（纯熵驱动，ΔP 作辅助验证）
  4. Draft 延续词探针：在 Target 高熵位置，让 Draft 贪婪续 3 token，
     揭示 Draft 在该决策分叉口所拥有的领域知识

输出文件（--out_dir）：
  entropy_records.json          — 每题的逐步熵、ΔP、生成文本原始记录
  summary.json                  — 核心统计数字（用于论文表格）
  high_entropy_tokens.json      — 错题高熵实义词 Top 列表（纯熵驱动）
  draft_continuation_probe.json — Draft 在 Target 高熵位置的延续词报告
  entropy_analysis.png          — 可视化图（需 matplotlib）

运行命令（远端，需同时加载 Draft）：
  cd /data/ocean/decoding && git pull
  export CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1
  python probe_entropy.py --limit 100 --max_new 150 --out_dir results/entropy_surgery_full

  # 仅用 Target（不计算 ΔP，不运行 Draft 探针，速度约快 3×）：
  python probe_entropy.py --limit 100 --max_new 150 --no_delta --out_dir results/entropy_target_only

查看结果：
  cat results/entropy_surgery_full/summary.json
  python -m json.tool results/entropy_surgery_full/draft_continuation_probe.json | head -80
"""

import argparse
import copy
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
CHOICE_LABELS = ["A", "B", "C", "D"]

# ──────────────────────────────────────────────────────────────────────────────
# 数据工具：直接复用 data_loader.load_medmcqa（使用正确的缓存路径）
# ──────────────────────────────────────────────────────────────────────────────

def load_surgery_data(limit: int, subject: str = "Surgery"):
    """
    复用 data_loader.load_medmcqa 加载指定科目子集。
    返回的每条记录格式：
      item["question"]   str
      item["options"]    dict  {"A": ..., "B": ..., "C": ..., "D": ...}
      item["answer_idx"] str   "A"/"B"/"C"/"D"
    subject=None 时加载全量（不按科目过滤）。
    """
    from data_loader import load_medmcqa
    logger.info(f"Loading MedMCQA validation set, subject={subject or 'ALL'}...")
    ds = load_medmcqa(split="validation", limit=limit, subject=subject)
    logger.info(f"Loaded {len(ds)} questions (subject={subject or 'ALL'})")
    return list(ds)


def format_prompt(item: dict, tokenizer) -> str:
    """
    复用 data_loader.format_prompt 构造与 run_benchmark.py 完全一致的 prompt。
    """
    from data_loader import format_prompt as dl_format_prompt
    return dl_format_prompt(
        tokenizer,
        question=item["question"],
        options=item["options"],
        dataset_name="medmcqa",
    )


def get_correct_label(item: dict) -> str:
    return item.get("answer_idx", "A")


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
# Draft 延续词探针（核心新增）
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_draft_continuations(
    model_draft,
    tokenizer,
    incorrect_recs: list,
    prompt_ids_list: list,      # List[Tensor]，每个 [1, prompt_len]，在 CPU 上
    H_thresh: float,
    device: str,
    k: int = 3,
    max_probes_per_q: int = 10, # 每道题最多探针次数（取熵最高的 top-N 位置）
) -> list:
    """
    对每道错题，在 Target 高熵位置（H > H_thresh）让 Draft 从同一上下文贪婪续 k token，
    揭示 Draft 在该"决策分叉口"所拥有的领域知识。

    优化：对每道题只做一次 Draft prefill，逐步推进 KV Cache；
         高熵位置到来时 clone KV 再做 k 步探针，避免反复 prefill。

    Returns list of events:
        q_idx, step, H_target, target_token,
        context_prefix, draft_continuation (list[str]), draft_continuation_text (str)
    """
    events = []

    for rec_idx, (r, prompt_ids) in enumerate(zip(incorrect_recs, prompt_ids_list)):
        gen_ids   = r["generated_ids"]   # List[int]
        entropies = r["entropies"]        # List[float]
        tok_strs  = r["token_strs"]       # List[str]

        # 选本题中高熵最高的 top-N 步（而非全部，控制运行时间）
        high_steps = sorted(
            [(step, H) for step, H in enumerate(entropies) if H > H_thresh],
            key=lambda x: x[1], reverse=True,
        )[:max_probes_per_q]
        high_steps_set = {step for step, _ in high_steps}

        if not high_steps_set:
            continue

        logger.info(
            f"  [错题 {rec_idx+1}/{len(incorrect_recs)}] q_idx={r['idx']}  "
            f"高熵位置数={len(high_steps)}"
        )

        # ── 一次性 prefill Draft（处理整个 prompt）─────────────────────────
        prompt_tensor = prompt_ids.to(device)           # [1, prompt_len]
        out = model_draft(
            input_ids=prompt_tensor,
            past_key_values=None,
            use_cache=True,
        )
        past_main  = out.past_key_values
        # out.logits[0, -1, :] = Draft 对第 0 步（第一个生成 token）的预测

        # ── 逐步推进，在高熵位置做探针 ──────────────────────────────────────
        for step in range(len(gen_ids)):
            # 此刻：past_main = KV(prompt + gen_ids[:step])
            #        out.logits[0,-1,:] = Draft 对位置 step 的预测分布

            if step in high_steps_set:
                # Target 在此处高度不确定 → 让 Draft 从同一上下文贪婪续 k 步
                # clone KV，不影响主推进 cache
                # 用 deepcopy 而非手动 tuple clone，确保兼容新版 transformers 的
                # DynamicCache 对象（该对象有 .get_seq_length() 方法，降级为 tuple 会报错）
                past_probe = copy.deepcopy(past_main)
                next_logit = out.logits[0, -1, :].clone()  # [V_draft]

                cont_strs = []
                for ki in range(k):
                    nid = int(next_logit.argmax().item())   # Draft 续词 token id
                    cont_strs.append(tokenizer.decode([nid]))
                    if ki < k - 1:                          # 最后一步无需再推进
                        out_p = model_draft(
                            input_ids=torch.tensor([[nid]], device=device),
                            past_key_values=past_probe,
                            use_cache=True,
                        )
                        past_probe = out_p.past_key_values
                        next_logit = out_p.logits[0, -1, :]

                events.append({
                    "q_idx":                  r["idx"],
                    "step":                   step,
                    "H_target":               round(entropies[step], 4),
                    "target_token":           tok_strs[step].strip(),
                    # 高熵位置前 6 个 token，提供上下文
                    "context_prefix":         "".join(tok_strs[max(0, step - 6):step]),
                    "draft_continuation":     cont_strs,
                    "draft_continuation_text": "".join(cont_strs),
                })

            # ── 推进主 KV Cache：喂入 Target 实际选择的 token ──────────────
            out = model_draft(
                input_ids=torch.tensor([[gen_ids[step]]], device=device),
                past_key_values=past_main,
                use_cache=True,
            )
            past_main = out.past_key_values
            # 循环末尾：past_main = KV(prompt + gen_ids[:step+1])

    logger.info(f"  Draft 延续探针完成：共 {len(events)} 条高熵事件")
    return events


def _report_draft_continuations(draft_events: list, out_dir: Path):
    """报告并保存 Draft 延续词探针结果（Section 4）。"""
    if not draft_events:
        logger.info("\n【4. Draft 延续词探针】（跳过：无高熵事件或 --no_delta 模式）")
        return

    from collections import Counter

    logger.info(f"\n【4. Draft 在 Target 高熵位置的延续词探针】")
    logger.info(f"  共 {len(draft_events)} 个高熵探针事件（每事件 Draft 续 3 token）")

    # ── 4a. 最常见的 Draft 延续词组 ────────────────────────────────────────
    cont_counter = Counter(
        e["draft_continuation_text"].strip() for e in draft_events
    )
    logger.info(f"\n  Top-20 Draft 最常见延续（Target 困惑时 Draft 最想说什么）：")
    logger.info(f"  {'Draft 延续（3 tokens）':<40} {'频次':>5}")
    logger.info("  " + "-" * 48)
    for text, cnt in cont_counter.most_common(20):
        logger.info(f"  {repr(text):<40} {cnt:>5}")

    # ── 4b. Top-20 高熵事件详情（按 H_target 降序）──────────────────────
    sorted_events = sorted(draft_events, key=lambda x: x["H_target"], reverse=True)
    logger.info(
        f"\n  Top-20 高熵位置详情（Target 最困惑时，Draft 想输出的内容）：\n"
        f"  格式：[H=] 上文 | Target实际输出 | Draft续: [3 tokens]"
    )
    logger.info("  " + "-" * 85)
    for e in sorted_events[:20]:
        logger.info(
            f"  [H={e['H_target']:.2f}] "
            f"...{e['context_prefix']!r:.<30} "
            f"Target:『{e['target_token']:.<12}』 "
            f"Draft续:{e['draft_continuation']}"
        )

    # ── 4c. 保存 JSON ───────────────────────────────────────────────────
    report = {
        "total_events":       len(draft_events),
        "k_continuation":     len(draft_events[0]["draft_continuation"]) if draft_events else 0,
        "note": (
            "在 Target 熵 > μ+σ（错题）的位置，让 Draft 从同一上下文贪婪续 k token。"
            "draft_continuation 中出现的领域词，说明 Draft 在 Target 最困惑时拥有该位置的领域知识。"
        ),
        "top_continuations": [
            {"text": t, "count": c} for t, c in cont_counter.most_common(50)
        ],
        "top_events_by_entropy": sorted_events[:50],
    }
    path = out_dir / "draft_continuation_probe.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n  Draft 延续探针报告 → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 分析 & 报告
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_report(records: list, out_dir: Path, draft_events: list = None):
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

    # ── 3. 错题中高熵 token 分析（纯熵驱动，ΔP 作辅助验证，不作筛选条件）────
    # 逻辑：先用 Target 香农熵找出 Target 最不确定的位置，
    #       再去看这些位置上是什么词，验证其是否与领域知识相关。
    #       ΔP 只作为辅助展示，证明熵高位置与领域信号的相关性。
    logger.info(
        "\n【3. 错题高熵 Token 分析（纯熵驱动：H > μ+σ，过滤停用词，ΔP 仅作辅助验证）】"
    )

    # 英文停用词表（功能词、连接词、副词等非实义词，不属于领域知识词汇）
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "not", "no", "nor",
        "that", "this", "these", "those", "it", "its", "which", "who", "what",
        "how", "when", "where", "why", "and", "or", "but", "so", "yet", "for",
        "of", "in", "on", "at", "to", "by", "with", "from", "as", "into",
        "than", "then", "thus", "such", "also", "both", "either", "while",
        "making", "often", "typically", "commonly", "primarily", "generally",
        "usually", "providing", "preserving", "provides", "provide", "during",
        "often", "however", "therefore", "although", "because", "since",
        "between", "through", "after", "before", "about", "over", "under",
        "more", "most", "less", "least", "very", "just", "only", "even",
        "each", "all", "any", "some", "other", "another", "same", "different",
        "new", "first", "second", "third", "one", "two", "three", "four",
        "they", "their", "them", "we", "our", "you", "your", "he", "she",
        "his", "her", "who", "whom", "whose", "there", "here", "where",
        "when", "while", "though", "whether", "both", "either", "neither",
        "without", "within", "along", "across", "among", "between",
    }

    token_stats = defaultdict(lambda: {"count": 0, "H": [], "Hn": [], "dP": []})
    examples    = []

    if incorrect_recs:
        all_H    = [H for r in incorrect_recs for H in r["entropies"]]
        H_thresh = np.mean(all_H) + np.std(all_H)
        logger.info(f"  高熵阈值 H > {H_thresh:.4f} nats（错题熵的 μ+σ）")
        logger.info(f"  筛选条件：① H > 阈值（Target 最不确定的位置）"
                    f"  ② 非英文停用词  ③ token 长度 ≥ 3")
        logger.info(f"  ΔP 作为辅助展示（非筛选条件）：验证高熵位置是否同时具有领域信号")

        for r in incorrect_recs:
            for step, (tok, H, Hn, dp) in enumerate(
                zip(r["token_strs"], r["entropies"], r["norm_entropies"], r["delta_ps"])
            ):
                tok_clean = tok.strip()
                # 筛选条件：仅用 Target 熵 + 非停用词过滤，ΔP 不参与筛选
                if (H > H_thresh
                        and len(tok_clean) >= 3
                        and tok_clean.lower() not in STOP_WORDS):
                    token_stats[tok_clean]["count"] += 1
                    token_stats[tok_clean]["H"].append(H)
                    token_stats[tok_clean]["Hn"].append(Hn)
                    token_stats[tok_clean]["dP"].append(dp)   # 仅记录，不筛选

                    if len(examples) < 100:
                        ctx_start = max(0, step - 5)
                        ctx_end   = min(len(r["token_strs"]), step + 6)
                        ctx       = "".join(r["token_strs"][ctx_start:ctx_end])
                        examples.append({
                            "q_idx":        r["idx"],
                            "token":        tok_clean,
                            "H":            round(H, 4),
                            "H_norm":       round(Hn, 4),
                            "dP":           round(dp, 4),
                            "context":      ctx,
                            "correct_label": r["correct_label"],
                            "pred_label":    r["pred_label"],
                        })

        sorted_toks = sorted(
            token_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )

        # ── 高熵实义词是否具有领域信号？计算 ΔP 分布作为验证 ──────────────────
        all_high_H_dp   = [dp for info in token_stats.values() for dp in info["dP"]]
        # 对照：全量 token 位置的 ΔP 均值
        all_dp_baseline = [dp for r in incorrect_recs for dp in r["delta_ps"]]

        logger.info(f"\n  满足条件的唯一实义 token 种数: {len(token_stats)}")

        if has_delta and all_high_H_dp and all_dp_baseline:
            logger.info(f"\n  ── ΔP 辅助验证（熵驱动选出的实义词，ΔP 是否偏高？）──")
            logger.info(f"  高熵实义词位置  均值ΔP = {np.mean(all_high_H_dp):.4f}")
            logger.info(f"  所有位置（基准）均值ΔP = {np.mean(all_dp_baseline):.4f}")
            ratio = np.mean(all_high_H_dp) / (np.mean(all_dp_baseline) + 1e-9)
            logger.info(f"  比值 = {ratio:.2f}×  （>1 说明高熵实义词位置更倾向于领域词）")

        logger.info(f"\n  Top-25 高熵实义词（Target 最困惑的非停用词位置）：")
        logger.info(f"  说明：均值ΔP 为辅助信息——ΔP 越高，该词越可能是领域专知词汇")
        header = f"  {'Token':<28} {'次数':>6} {'均值H':>9} {'H/Hmax':>8} {'均值ΔP':>9} {'领域信号':>8}"
        logger.info(header)
        logger.info("  " + "-" * 75)
        for tok, info in sorted_toks[:25]:
            mean_dp  = float(np.mean(info["dP"]))
            is_domain = "★" if mean_dp > 0.05 else "·"   # 辅助标记：ΔP>0.05 视为有领域信号
            logger.info(
                f"  {tok:<28} {info['count']:>6} "
                f"{np.mean(info['H']):>9.4f} "
                f"{np.mean(info['Hn']):>8.4f} "
                f"{mean_dp:>9.4f} "
                f"{is_domain:>8}"
            )

        logger.info(f"\n  图例：★ = 均值ΔP > 0.05（有领域信号）  · = 通用词")

        # 保存高熵 token 报告
        token_report = {
            "H_threshold":            float(H_thresh),
            "H_max_ref":              float(H_max_ref),
            "filter_note": (
                "仅用 Target 香农熵筛选（H > μ+σ）+ 停用词过滤。"
                "ΔP 为辅助验证字段，不参与筛选。"
                "均值ΔP > 0.05 的 token 标注 domain_signal=True。"
            ),
            "high_H_dp_mean":         float(np.mean(all_high_H_dp))    if all_high_H_dp    else 0,
            "baseline_dp_mean":       float(np.mean(all_dp_baseline))  if all_dp_baseline  else 0,
            "top_tokens": [
                {
                    "token":        tok,
                    "count":        info["count"],
                    "mean_H":       float(np.mean(info["H"])),
                    "mean_Hn":      float(np.mean(info["Hn"])),
                    "mean_dP":      float(np.mean(info["dP"])),
                    "domain_signal": float(np.mean(info["dP"])) > 0.05,
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

    # ── 4. Draft 延续词探针报告 ──────────────────────────────────────────────
    _report_draft_continuations(draft_events or [], out_dir)

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
    items = load_surgery_data(args.limit, subject=args.subject)

    # ── 主循环：逐题生成并记录熵 ──────────────────────────────────────────
    records         = []
    prompt_ids_list = []    # 与 records 一一对应，保存 prompt input_ids（CPU），供 Draft 探针用

    for idx, item in enumerate(items):
        correct_label = get_correct_label(item)
        # format_prompt 内部已调用 apply_chat_template，直接返回完整 prompt 字符串
        text      = format_prompt(item, tokenizer)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # 保存 prompt_ids（CPU）供后续 Draft 探针重建上下文
        prompt_ids_list.append(input_ids.cpu())

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
            "idx":               idx,
            "question":          item["question"][:120],    # 截断节省空间
            "correct_label":     correct_label,
            "pred_label":        pred_label,
            "is_correct":        is_correct,
            "gen_text":          gen_text,
            "token_strs":        gen["token_strs"],
            "generated_ids":     gen["generated_ids"],      # 原始 token id，供 Draft 探针用
            "entropies":         [round(h, 5) for h in gen["entropies"]],
            "norm_entropies":    [round(h, 5) for h in gen["norm_entropies"]],
            "delta_ps":          [round(d, 5) for d in gen["delta_ps"]],
            "mean_entropy":      round(mean_H,  5),
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
    # generated_ids 仅供本次会话内 Draft 探针使用，不写入 JSON（节省空间）
    records_for_json = [{k: v for k, v in r.items() if k != "generated_ids"} for r in records]
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records_for_json, f, ensure_ascii=False, indent=2)
    logger.info(f"\n原始记录 → {raw_path}")

    # ── Draft 延续词探针（Section 4）────────────────────────────────────────
    # 仅在加载了 Draft 模型时运行（full ΔP 模式）
    draft_events = []
    if model_draft is not None:
        incorrect_recs  = [r for r in records if not r["is_correct"]]
        incorr_pids     = [prompt_ids_list[r["idx"]] for r in incorrect_recs]

        # 用错题整体熵的 μ+σ 作为高熵阈值（与 Section 3 保持一致）
        all_incorr_H = [H for r in incorrect_recs for H in r["entropies"]]
        H_thresh_probe = float(np.mean(all_incorr_H) + np.std(all_incorr_H)) if all_incorr_H else 1.0

        logger.info(
            f"\n开始 Draft 延续词探针（错题 {len(incorrect_recs)} 道，"
            f"高熵阈值={H_thresh_probe:.4f} nats，每题最多探 {args.draft_probe_top_k} 个位置）..."
        )
        draft_events = collect_draft_continuations(
            model_draft=model_draft,
            tokenizer=tokenizer,
            incorrect_recs=incorrect_recs,
            prompt_ids_list=incorr_pids,
            H_thresh=H_thresh_probe,
            device=device,
            k=3,
            max_probes_per_q=args.draft_probe_top_k,
        )

    # ── 分析 & 报告 ────────────────────────────────────────────────────────
    analyze_and_report(records, out_dir, draft_events=draft_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target Shannon Entropy Probe")
    parser.add_argument("--limit",    type=int,  default=100,
                        help="分析题目数量（建议 100-200）")
    parser.add_argument("--max_new",  type=int,  default=150,
                        help="每题最大生成 token 数")
    parser.add_argument("--no_delta", action="store_true",
                        help="不加载 Draft/Base，不计算 ΔP（速度约快 3×）")
    parser.add_argument("--subject",  type=str,  default="Surgery",
                        help="MedMCQA 科目过滤（如 Surgery/Pharmacology/Anatomy，空字符串=全量）")
    parser.add_argument("--out_dir",  type=str,  default="results/entropy_analysis",
                        help="输出目录")
    parser.add_argument("--draft_probe_top_k", type=int, default=10,
                        help="Draft 延续探针：每道错题最多探针的高熵位置数（取熵最高的 top-k）")
    args = parser.parse_args()
    main(args)
