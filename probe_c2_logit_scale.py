"""
诊断脚本：probe_c2_logit_scale.py

目标：统计 C2 Logit 注入的实际尺度，判断 alpha 取值范围是否合理。

统计量：
  1. logit_target 的 mean / std / max（32B 模型的 logit 分布）
  2. ΔLogit_norm 的 mean / std（Z-score 后应约为 0 / 1）
  3. 对 draft 提案 token x：
       - logit_target[x]（未注入时的原始值）
       - injection = alpha * ΔLogit_norm[x]（各 alpha 下的注入量）
       - logit_target_prime[x] = logit_target[x] + injection
       - 注入后是否翻转 argmax（即 argmax(logit_target_prime) == x）
  4. 不同 alpha 下 argmax 翻转率（= 本 alpha 能让 C2 在贪婪模式接受的比例）

用法（远端）：
  cd /data/ocean/decoding && conda activate kvner
  export CUDA_VISIBLE_DEVICES=6
  export HF_DATASETS_OFFLINE=1
  python probe_c2_logit_scale.py --n_steps 500 --alpha_list 1 5 10 20 50 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from config_v2 import HardwareConfig, ModelPaths
from data_loader import load_medmcqa
from domain_signal import compute_delta_logit_normalized
from forward_ops import decode_step, prefill
from model_loader_v2 import load_tri_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def probe(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0")

    paths = ModelPaths()
    hw    = HardwareConfig(compile_mode=None)
    logger.info("加载三模型...")
    bundle = load_tri_models(paths=paths, hw=hw)
    tokenizer = bundle.tokenizer

    logger.info("加载 MedMCQA Surgery 验证集...")
    raw_ds = load_medmcqa(split="validation", limit=20, subject="Surgery")

    from data_loader import format_prompt

    # 统计列表
    stats = {
        "target_logit_mean":   [],   # 每步 logit_target 均值
        "target_logit_std":    [],   # 每步 logit_target 标准差
        "target_logit_max":    [],   # 每步 logit_target 最大值
        "delta_norm_mean":     [],   # ΔLogit_norm 均值（验证 ≈ 0）
        "delta_norm_std":      [],   # ΔLogit_norm 标准差（验证 ≈ 1）
        # 对 draft 提案 token x：
        "draft_token_logit_target":       [],  # logit_target[x]
        "draft_token_delta_norm":         [],  # ΔLogit_norm[x]
        # 每个 alpha：注入量 & argmax 是否翻转
    }
    for a in args.alpha_list:
        stats[f"alpha{a}_injection"]      = []  # alpha * ΔLogit_norm[x]
        stats[f"alpha{a}_logit_prime_x"]  = []  # logit_target[x] + injection
        stats[f"alpha{a}_argmax_flipped"] = []  # bool：argmax(logit_prime)==x?

    n_steps = 0

    for item in raw_ds:
        q    = item.get("question", "")
        opts = item.get("options", {})
        if not q or not opts:
            continue

        prompt_text = format_prompt(tokenizer, q, opts, dataset_name="medmcqa")
        prompt_ids  = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

        # 简单共享 cache（无完整 PrefixSharedCacheManager，直接用独立 StaticCache）
        from transformers.cache_utils import StaticCache
        max_cache_len = 1024

        target_cache = StaticCache(
            config=bundle.target.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.bfloat16,
        )
        draft_cache = StaticCache(
            config=bundle.draft.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.bfloat16,
        )
        base_cache = StaticCache(
            config=bundle.base.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.bfloat16,
        )

        # Prefill 三模型
        logit_target = prefill(bundle.target, prompt_ids, target_cache)  # [1, V_T]
        logit_draft  = prefill(bundle.draft,  prompt_ids, draft_cache)   # [1, V_D]
        logit_base   = prefill(bundle.base,   prompt_ids, base_cache)    # [1, V_B]

        seq_len = prompt_ids.shape[1]

        for step in range(args.n_steps_per_sample):
            if n_steps >= args.n_steps:
                break

            # Draft 提案 token（贪婪）
            draft_token = int(logit_draft.argmax(dim=-1).item())

            # ΔLogit_norm：[1, V_small]
            _v = min(logit_draft.shape[-1], logit_base.shape[-1])
            delta_logit_norm = compute_delta_logit_normalized(
                logit_draft[..., :_v], logit_base[..., :_v]
            )  # shape: [1, _v]，mean≈0, std≈1

            # 对齐 target vocab
            _vt = min(logit_target.shape[-1], _v)
            logit_target_aligned = logit_target[..., :_vt]          # [1, _vt]
            delta_norm_aligned   = delta_logit_norm[..., :_vt]      # [1, _vt]

            # ── 统计 logit_target 分布
            lt = logit_target_aligned[0]  # [V]
            stats["target_logit_mean"].append(float(lt.mean().item()))
            stats["target_logit_std"].append(float(lt.std().item()))
            stats["target_logit_max"].append(float(lt.max().item()))

            # ── 统计 ΔLogit_norm 分布（验证 Z-score）
            dn = delta_norm_aligned[0]    # [V]
            stats["delta_norm_mean"].append(float(dn.mean().item()))
            stats["delta_norm_std"].append(float(dn.std().item()))

            # ── 针对 draft 提案 token x
            x = draft_token
            if x < _vt:
                lt_x  = float(logit_target_aligned[0, x].item())
                dn_x  = float(delta_norm_aligned[0, x].item())
                stats["draft_token_logit_target"].append(lt_x)
                stats["draft_token_delta_norm"].append(dn_x)

                orig_argmax = int(logit_target_aligned.argmax(dim=-1).item())

                for a in args.alpha_list:
                    injection = a * dn_x
                    logit_prime_x = lt_x + injection
                    # 修改 logit 后的新分布 argmax
                    logit_prime = logit_target_aligned + a * delta_norm_aligned   # [1, _vt]
                    new_argmax  = int(logit_prime.argmax(dim=-1).item())
                    flipped     = (new_argmax == x) and (orig_argmax != x)

                    stats[f"alpha{a}_injection"].append(injection)
                    stats[f"alpha{a}_logit_prime_x"].append(logit_prime_x)
                    stats[f"alpha{a}_argmax_flipped"].append(int(flipped))

            n_steps += 1

            # 推进三模型 decode（贪婪，只为获取下一步 logit）
            token_t = torch.tensor([[draft_token]], dtype=torch.long, device=device)
            logit_target = decode_step(bundle.target, token_t, target_cache, position_id=seq_len)
            logit_draft  = decode_step(bundle.draft,  token_t, draft_cache,  position_id=seq_len)
            logit_base   = decode_step(bundle.base,   token_t, base_cache,   position_id=seq_len)
            seq_len += 1

        if n_steps >= args.n_steps:
            break

    # ── 汇总输出
    logger.info("\n" + "="*60)
    logger.info("C2 Logit 尺度诊断报告  (n_steps=%d)", n_steps)
    logger.info("="*60)

    def _stats(lst, name):
        import statistics
        if not lst:
            return
        logger.info("  %-35s  mean=%+8.3f  median=%+8.3f  std=%7.3f  min=%+8.3f  max=%+8.3f",
                    name,
                    statistics.mean(lst),
                    statistics.median(lst),
                    statistics.stdev(lst) if len(lst) > 1 else 0,
                    min(lst),
                    max(lst))

    logger.info("\n── logit_target（32B Target 原始 logit）分布 ──")
    _stats(stats["target_logit_mean"], "target_logit  [per-step mean]")
    _stats(stats["target_logit_std"],  "target_logit  [per-step std]")
    _stats(stats["target_logit_max"],  "target_logit  [per-step max]")

    logger.info("\n── ΔLogit_norm（Z-score 后）分布（期望 mean≈0, std≈1）──")
    _stats(stats["delta_norm_mean"], "ΔLogit_norm   [per-step mean]")
    _stats(stats["delta_norm_std"],  "ΔLogit_norm   [per-step std]")

    logger.info("\n── Draft 提案 token x 的原始数值 ──")
    _stats(stats["draft_token_logit_target"], "logit_target[x]          原始值")
    _stats(stats["draft_token_delta_norm"],   "ΔLogit_norm[x]            注入基底")

    logger.info("\n── 不同 alpha 下的注入量与 argmax 翻转率 ──")
    logger.info("  %-8s  %-20s  %-22s  %-12s", "alpha", "injection[x] mean±std",
                "logit_prime[x] mean±std", "argmax_flip_rate")
    for a in args.alpha_list:
        inj  = stats[f"alpha{a}_injection"]
        lpx  = stats[f"alpha{a}_logit_prime_x"]
        flip = stats[f"alpha{a}_argmax_flipped"]
        if not inj:
            continue
        import statistics
        flip_rate = sum(flip) / len(flip) if flip else 0.0
        logger.info("  %-8s  %+7.2f ± %6.2f        %+7.2f ± %6.2f        %.3f  (%d/%d)",
                    f"α={a}",
                    statistics.mean(inj), statistics.stdev(inj) if len(inj)>1 else 0,
                    statistics.mean(lpx), statistics.stdev(lpx) if len(lpx)>1 else 0,
                    flip_rate, sum(flip), len(flip))

    # 写 JSON
    out_path = Path(args.out) if args.out else Path("results/probe_c2_logit_scale.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    logger.info("\n详细数据已写入: %s", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="C2 Logit 尺度诊断脚本")
    p.add_argument("--n_steps",            type=int,   default=500,
                   help="总采样步数（跨多个 sample）")
    p.add_argument("--n_steps_per_sample", type=int,   default=50,
                   help="每个 sample 最多采样步数")
    p.add_argument("--alpha_list",         type=float, nargs="+",
                   default=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                   help="要测试的 alpha 值列表")
    p.add_argument("--out", default="results/probe_c2_logit_scale.json",
                   help="JSON 输出路径")
    args = p.parse_args()
    probe(args)
