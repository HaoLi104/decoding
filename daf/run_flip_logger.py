"""DAF Round k — flip 事件采集驱动器

复用第一点的三模型投机解码框架（model_loader_v2 / cache_manager / engine_state /
acceptance / decode_loop / telemetry），固定使用：

    --strategy soft_guidance_c9
    --alpha    50         # C9 论文实验的最强配置
    --c4_tau   0.05

执行流程（每个 sample）：
  1. 用 ShadowSyncProposer + SpeculativeDecodeLoop 跑一次完整 C9 解码；
  2. 解码过程中 telemetry 已逐步写入 (final_token_id, is_flip, target_entropy)；
  3. 后处理：根据 telemetry steps + prompt_ids + final_token_id 重建 prefix_ids，
     抽取 flip 事件 (F=True) 写入 flip_events_round{k}.jsonl。

输出：
  flip_events_round{k}.jsonl    （供 fdlp_score / build_flip_sft_data 使用）
  flip_logger_round{k}_summary.json （扫描统计：n_flip / n_qid / mean ΔP / mean H_t）
  per-sample 的 telemetry 文件      （沿用 TelemetryLogger 默认行为，便于回放）

用法（远端 H200）：
  cd /data/ocean/decoding
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=0
  python -m daf.run_flip_logger \
      --target_model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --round 0 \
      --dataset medmcqa --subject Surgery --split train --limit 5000 \
      --out_dir logs/daf_round0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch

# 父目录（项目根）加入 sys.path，使 `python -m daf.run_flip_logger` 直接可用
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from acceptance import create_strategy
from cache_manager import PrefixSharedCacheManager
from config_v2 import (
    DecodeConfig,
    DomainSignalParams,
    ExecutionArch,
    HardwareConfig,
    ModelPaths,
    StrategyType,
)
from data_loader import format_prompt, load_jecqa, load_medmcqa, load_medqa
from decode_loop import DecodeResult, SpeculativeDecodeLoop
from dual_stream_engine import DualStreamProposer
from engine_state import TriModelOrchestrator
from model_loader_v2 import load_tri_models
from shadow_sync_engine import ShadowSyncProposer
from telemetry import TelemetryLogger

from daf.flip_definition import FlipRecord, is_flip, summarize_flip_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.run_flip_logger")


# ---------------------------------------------------------------------------
# 数据集加载（与 run_benchmark._load_dataset 保持口径一致）
# ---------------------------------------------------------------------------

def _load_dataset(dataset_name: str, limit: int, split: str, subject: str) -> List[Dict[str, Any]]:
    if dataset_name == "medmcqa":
        raw_ds = load_medmcqa(split=split, limit=limit, subject=subject if subject else None)
    elif dataset_name == "medqa":
        raw_ds = load_medqa(split=split, limit=limit)
    elif dataset_name == "jecqa":
        raw_ds = load_jecqa(limit=limit)
    else:
        raise ValueError(f"DAF flip_logger 不支持数据集: {dataset_name}")

    cases: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_ds):
        q    = item.get("question", "")
        opts = item.get("options", {})
        gt   = str(item.get("answer_idx", "")).strip().upper()
        if not q or not opts or gt not in {"A", "B", "C", "D"}:
            continue
        cases.append({
            "id":       str(item.get("id", idx)),
            "question": q,
            "options":  opts,
            "gt":       gt,
        })
    return cases[:limit] if limit else cases


# ---------------------------------------------------------------------------
# Proposer 构建
# ---------------------------------------------------------------------------

def _build_proposer(arch: ExecutionArch, orch: TriModelOrchestrator, device: torch.device):
    if arch == ExecutionArch.DUAL_STREAM:
        return DualStreamProposer(
            draft_ctx=orch.draft_ctx, base_ctx=orch.base_ctx, device=device,
        )
    if arch == ExecutionArch.SHADOW_SYNC:
        return ShadowSyncProposer(
            draft_ctx=orch.draft_ctx, base_ctx=orch.base_ctx, device=device,
        )
    if arch == ExecutionArch.DEFERRED_BASE:
        from deferred_base_engine import DeferredBaseProposer
        return DeferredBaseProposer(
            draft_ctx=orch.draft_ctx, base_ctx=orch.base_ctx, device=device,
        )
    raise ValueError(f"未知架构: {arch}")


# ---------------------------------------------------------------------------
# 后处理：根据 telemetry steps 重建 prefix_ids 并抽取 flip 事件
# ---------------------------------------------------------------------------

def _extract_flips_from_telemetry(
    qid:        str,
    round_id:   int,
    prompt_ids: List[int],
    steps:      List[Dict[str, Any]],
) -> List[FlipRecord]:
    """从 telemetry steps（dict 列表）抽取 flip 事件，并附带 prefix_ids。

    prefix_ids 重建规则：
      step 0 的 prefix = prompt_ids
      step t 的 prefix = prompt_ids + [final_token_id 0..t-1]

    若 telemetry 缺 final_token_id（极旧日志），用 draft_token_id（accept 时）
    或 target_top1_id（reject 时）做最佳猜测——但 M0 之后所有新数据应都有该字段。
    """
    flips: List[FlipRecord] = []
    seq: List[int] = list(prompt_ids)

    for s in steps:
        if not is_flip(s):
            # 非 flip 步：仍要推进 seq（用 final_token_id）以保证后续 flip 的 prefix 正确
            ft = s.get("final_token_id")
            if ft is None:
                ft = s.get("draft_token_id") if s.get("accepted") else s.get("target_top1_id")
            if ft is not None:
                seq.append(int(ft))
            continue

        prefix_now = list(seq)  # 本 flip 步生效前的 token 序列
        flips.append(FlipRecord(
            round_id=round_id,
            qid=qid,
            step=int(s.get("step", -1)),
            prefix_ids=prefix_now,
            A=int(s["target_top1_id"]),
            B=int(s["draft_token_id"]),
            F=True,
            delta_p=float(s.get("delta_p", 0.0)),
            h_t=(None if s.get("target_entropy") is None else float(s["target_entropy"])),
            accepted=True,
        ))

        # 推进 seq：flip 步的 final_token = draft (because accepted)
        ft = s.get("final_token_id", s.get("draft_token_id"))
        if ft is not None:
            seq.append(int(ft))

    return flips


def _read_telemetry_steps(telemetry_path: Path) -> List[Dict[str, Any]]:
    """读取一个 sample 的 telemetry.jsonl，返回 step 行列表（按写入顺序）。"""
    steps: List[Dict[str, Any]] = []
    with telemetry_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "step":
                steps.append(obj)
    return steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DAF Round k flip 事件采集驱动器（C9 + 32B Target）",
    )
    p.add_argument("--round", type=int, required=True,
                   help="飞轮轮次编号（0=Round 0；1=Round 1，需配合 --target_model 指 v1）")
    p.add_argument("--target_model", default=None, help="Target 模型路径")
    p.add_argument("--draft_model",  default=None, help="Draft 模型路径")
    p.add_argument("--base_model",   default=None, help="Base 模型路径")

    p.add_argument("--dataset", choices=["medmcqa", "medqa", "jecqa"], default="medmcqa")
    p.add_argument("--subject", default="Surgery",
                   help="medmcqa 专用：按 subject_name 过滤（默认 Surgery）")
    p.add_argument("--split",   default="train",
                   help="数据集分片（默认 train，与 SFT 数据来源对齐）")
    p.add_argument("--limit",   type=int, default=5000,
                   help="解码 sample 数（plan: 5000 题以保证 flip 样本量）")

    p.add_argument("--arch",  choices=["dual_stream", "shadow_sync", "deferred_base"],
                   default="shadow_sync")
    p.add_argument("--gamma", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_cache_len",  type=int, default=2048)

    # C9 固定配置（与 plan 对齐）
    p.add_argument("--alpha",    type=float, default=50.0)
    p.add_argument("--c4_tau",   type=float, default=0.05)
    p.add_argument("--t_sample", type=float, default=0.0)

    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)

    # ---- DAF 第二点专用：prompt 模式切换（不影响第一点 baseline）----
    p.add_argument("--prompt_mode", choices=["baseline", "thinking"], default="baseline",
                   help="baseline=与第一点完全一致的严格 prompt（默认）；"
                        "thinking=DAF 专用长推理 prompt，鼓励 5-8 句话医学推理，显著提升内容 flip 比例")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")

    # ---------- 模型加载 ----------
    paths = ModelPaths(
        TARGET=args.target_model or ModelPaths.TARGET,
        BASE=args.base_model    or ModelPaths.BASE,
        DRAFT=args.draft_model  or ModelPaths.DRAFT,
    )
    hw = HardwareConfig(compile_mode=None)
    logger.info("=== DAF Round %d  flip logger 启动 ===", args.round)
    logger.info("Target = %s", paths.TARGET)
    logger.info("Draft  = %s", paths.DRAFT)
    logger.info("Base   = %s", paths.BASE)

    bundle = load_tri_models(paths=paths, hw=hw)

    target_config = bundle.target.config
    small_config  = bundle.base.config
    cache_mgr = PrefixSharedCacheManager(
        target_config=target_config,
        small_config=small_config,
        max_batch_size=1,
        max_cache_len=args.max_cache_len,
        device=device,
        dtype=torch.bfloat16,
    )
    orch = TriModelOrchestrator(
        target_model=bundle.target,
        draft_model=bundle.draft,
        base_model=bundle.base,
        cache_mgr=cache_mgr,
        device=device,
    )

    # ---------- 数据集 ----------
    logger.info("加载数据集: %s split=%s subject=%s limit=%d",
                args.dataset, args.split, args.subject, args.limit)
    dataset = _load_dataset(args.dataset, args.limit, args.split, args.subject)
    logger.info("数据集就绪，n=%d", len(dataset))

    # ---------- C9 策略 ----------
    signal_params = DomainSignalParams()  # 默认即与第一点 C9 对齐
    config = DecodeConfig(
        strategy=StrategyType.SOFT_GUIDANCE_C9,
        arch=ExecutionArch(args.arch),
        signal_params=signal_params,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        t_sample=args.t_sample,
        alpha=args.alpha,
        c4_tau=args.c4_tau,
    )
    strategy = create_strategy(
        strategy_type=config.strategy,
        signal_params=config.signal_params,
        alpha=config.alpha,
        c4_tau=config.c4_tau,
    )

    # ---------- 输出文件 ----------
    flip_jsonl_path = out_dir / f"flip_events_round{args.round}.jsonl"
    summary_path    = out_dir / f"flip_logger_round{args.round}_summary.json"
    telemetry_dir   = out_dir / f"telemetry_round{args.round}"
    flip_fh = flip_jsonl_path.open("w", encoding="utf-8")

    n_total_steps = 0
    n_total_flips = 0
    n_processed   = 0
    t_start_all   = time.perf_counter()

    for idx, item in enumerate(dataset):
        sample_id = str(item.get("id", idx))

        prompt_text = format_prompt(
            bundle.tokenizer, item["question"], item["options"],
            dataset_name=args.dataset,
            prompt_mode=(None if args.prompt_mode == "baseline" else args.prompt_mode),
        )
        prompt_ids_t = bundle.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        prompt_ids   = prompt_ids_t[0].tolist()

        telemetry = TelemetryLogger(log_dir=telemetry_dir, sample_id=sample_id)
        orch.reset()
        orch.init_from_prompt(prompt_ids_t)

        proposer = _build_proposer(config.arch, orch, device)
        decode_loop = SpeculativeDecodeLoop(
            orchestrator=orch,
            proposer=proposer,
            strategy=strategy,
            telemetry=telemetry,
            tokenizer=bundle.tokenizer,
            config=config,
        )

        try:
            _result: DecodeResult = decode_loop.run(prompt_ids_t)
        except Exception as exc:
            logger.error("sample %s 解码失败：%s", sample_id, exc, exc_info=True)
            orch.reset()
            cache_mgr.reset()
            continue

        # ---------- 抽取 flip + 写入 ----------
        telemetry.flush()  # 落盘 per-sample telemetry
        tel_path = telemetry_dir / f"{sample_id}_telemetry.jsonl"
        steps = _read_telemetry_steps(tel_path)
        flips = _extract_flips_from_telemetry(
            qid=sample_id,
            round_id=args.round,
            prompt_ids=prompt_ids,
            steps=steps,
        )
        for r in flips:
            flip_fh.write(json.dumps({
                "round_id":   r.round_id,
                "qid":        r.qid,
                "step":       r.step,
                "prefix_ids": r.prefix_ids,
                "A":          r.A,
                "B":          r.B,
                "F":          r.F,
                "delta_p":    r.delta_p,
                "h_t":        r.h_t,
                "accepted":   r.accepted,
            }, ensure_ascii=False))
            flip_fh.write("\n")
        flip_fh.flush()

        n_processed   += 1
        n_total_steps += len(steps)
        n_total_flips += len(flips)

        # 重置 cache 为下一 sample 准备
        cache_mgr.reset()

        if (idx + 1) % 50 == 0 or (idx + 1) == len(dataset):
            elapsed = time.perf_counter() - t_start_all
            flip_rate = (n_total_flips / max(n_total_steps, 1))
            logger.info(
                "[round=%d  %d/%d] flips=%d  steps=%d  flip_rate=%.4f  "
                "mean_flip_per_qid=%.2f  elapsed=%.1fs",
                args.round, idx + 1, len(dataset),
                n_total_flips, n_total_steps, flip_rate,
                n_total_flips / max(n_processed, 1), elapsed,
            )

    flip_fh.close()

    # ---------- 汇总 ----------
    summary = summarize_flip_jsonl(flip_jsonl_path)
    summary.update({
        "round_id":      args.round,
        "n_processed":   n_processed,
        "n_total_steps": n_total_steps,
        "n_total_flips": n_total_flips,
        "global_flip_rate": (n_total_flips / max(n_total_steps, 1)),
        "target_model":  paths.TARGET,
        "draft_model":   paths.DRAFT,
        "base_model":    paths.BASE,
        "strategy":      "soft_guidance_c9",
        "alpha":         args.alpha,
        "c4_tau":        args.c4_tau,
        "t_sample":      args.t_sample,
        "dataset":       f"{args.dataset}/{args.subject}/{args.split}",
        "limit":         args.limit,
        "flip_jsonl":    str(flip_jsonl_path),
        "prompt_mode":   args.prompt_mode,
        "max_new_tokens": args.max_new_tokens,
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("✓ Round %d flip logger 完成  flips=%d  rate=%.4f  →  %s",
                args.round, n_total_flips, summary["global_flip_rate"], flip_jsonl_path)


if __name__ == "__main__":
    main()
