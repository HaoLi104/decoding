"""DAF — 飞轮收敛判定（plan 2.4.3）

输入：
  flip_events_round{0,1,...}.jsonl   连续多轮 flip 日志
  eval_round{0,1,...}.json           （可选）每轮 MMLU 守护结果（用于 MMLU 退化阈值）

度量（DAF 文档 2.4.3 节）：
  F̄^(k)  = global flip rate at round k = n_flip_round_k / n_total_steps_round_k
  ρ_k    = 1 - F̄^(k) / F̄^(0)   累计吸收率（k≥1）
  ΔF̄^(k) = F̄^(k-1) - F̄^(k)    跨轮 flip rate 下降量

  停止条件（任一触发即停飞轮）：
    (a) |ΔF̄^(k)| < 0.02         flip rate 不再下降
    (b) ρ_k ≥ 0.90              已吸收 90% 飞行 token
    (c) MMLU 相对 Round 0 退化 > 1.0%   过度灾难性遗忘

输出：
  convergence_round{k}.json：
    {
      "round_id": k,
      "F_bar": [...],              # 各轮 flip rate
      "rho_k":  float,             # 当前轮的累计吸收率
      "delta_F_k": float,          # 与上轮的下降量
      "mmlu_drop_pp": float,       # MMLU 退化（百分点；正数=退化）
      "decision": "stop" | "continue",
      "reason":   str,             # 触发的停止条件
      "thresholds": {...}
    }

用法：
  python -m daf.convergence_check \
      --flip_jsonls logs/daf_round0/flip_events_round0.jsonl \
                    logs/daf_round0/flip_events_round1.jsonl \
      --eval_jsons  logs/daf_round0/eval_round0.json \
                    logs/daf_round0/eval_round1.json \
      --out logs/daf_round0/convergence_round1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.convergence_check")


def _flip_rate_from_jsonl(flip_jsonl: Path) -> Optional[float]:
    """优先读取同目录下 flip_logger_round{k}_summary.json 的 global_flip_rate；
    若不存在，则按 flip_jsonl 行数为分子，但分母（总 steps）需另算——
    所以强烈建议直接读 summary。
    """
    summary_path = flip_jsonl.with_name(
        flip_jsonl.name.replace("flip_events_round", "flip_logger_round_summary_")
    )
    # 兼容 plan 命名： flip_logger_round{k}_summary.json
    if not summary_path.exists():
        # 标准命名
        import re
        m = re.search(r"round(\d+)", flip_jsonl.name)
        if m:
            k = m.group(1)
            cand = flip_jsonl.parent / f"flip_logger_round{k}_summary.json"
            if cand.exists():
                summary_path = cand
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        rate = s.get("global_flip_rate")
        return float(rate) if rate is not None else None
    return None


def _mmlu_acc(eval_json_path: Path) -> Optional[float]:
    if not eval_json_path or not eval_json_path.exists():
        return None
    obj = json.loads(eval_json_path.read_text(encoding="utf-8"))
    mmlu = obj.get("mmlu", {})
    if mmlu.get("skipped"):
        return None
    return mmlu.get("acc")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF 飞轮收敛判定")
    p.add_argument("--flip_jsonls", nargs="+", required=True,
                   help="按轮次顺序的 flip_events_round{k}.jsonl 路径列表")
    p.add_argument("--eval_jsons", nargs="*", default=[],
                   help="按轮次顺序的 eval_round{k}.json 路径（用于 MMLU 退化阈值）")
    p.add_argument("--out", required=True)

    p.add_argument("--delta_F_thresh", type=float, default=0.02,
                   help="跨轮 flip rate 下降量阈值（默认 0.02）")
    p.add_argument("--rho_thresh",     type=float, default=0.90,
                   help="累计吸收率上限（默认 0.90）")
    p.add_argument("--mmlu_drop_thresh_pp", type=float, default=1.0,
                   help="MMLU 退化阈值（百分点，正数=退化；默认 1.0%）")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    flip_paths = [Path(p) for p in args.flip_jsonls]
    eval_paths = [Path(p) for p in args.eval_jsons] if args.eval_jsons else []

    F_bar: List[Optional[float]] = []
    for fp in flip_paths:
        rate = _flip_rate_from_jsonl(fp)
        F_bar.append(rate)
        logger.info("Round %s  F̄=%s  (%s)",
                    fp.stem.replace("flip_events_", ""),
                    f"{rate:.4f}" if rate is not None else "N/A", fp)

    if any(r is None for r in F_bar):
        logger.warning("部分轮次 flip rate 缺失，无法计算 ρ_k；请确认 flip_logger_round{k}_summary.json 存在")
        F_bar = [r for r in F_bar if r is not None]

    if len(F_bar) < 2:
        logger.error("至少需要 2 轮才能计算收敛指标，当前仅 %d 轮", len(F_bar))
        sys.exit(2)

    F0  = F_bar[0]
    Fk  = F_bar[-1]
    Fkm = F_bar[-2]
    round_id = len(F_bar) - 1  # 当前是第几轮

    rho_k     = 1.0 - (Fk / max(F0, 1e-9))
    delta_F_k = Fkm - Fk

    # MMLU 退化（百分点，正数 = 退化）
    mmlu_acc_round0 = _mmlu_acc(eval_paths[0])  if eval_paths else None
    mmlu_acc_roundk = _mmlu_acc(eval_paths[-1]) if len(eval_paths) >= 1 else None
    mmlu_drop_pp: Optional[float] = None
    if (mmlu_acc_round0 is not None) and (mmlu_acc_roundk is not None):
        mmlu_drop_pp = (mmlu_acc_round0 - mmlu_acc_roundk) * 100.0

    # 决策
    reasons: List[str] = []
    decision = "continue"
    if abs(delta_F_k) < args.delta_F_thresh:
        decision = "stop"
        reasons.append(f"|ΔF̄^({round_id})|={abs(delta_F_k):.4f} < {args.delta_F_thresh:.4f}")
    if rho_k >= args.rho_thresh:
        decision = "stop"
        reasons.append(f"ρ_{round_id}={rho_k:.4f} ≥ {args.rho_thresh:.4f}")
    if (mmlu_drop_pp is not None) and (mmlu_drop_pp > args.mmlu_drop_thresh_pp):
        decision = "stop"
        reasons.append(f"MMLU 退化 {mmlu_drop_pp:.2f}pp > {args.mmlu_drop_thresh_pp:.2f}pp")

    out = {
        "round_id":     round_id,
        "F_bar":        F_bar,
        "F0":           F0,
        "F_k":          Fk,
        "F_k_minus_1":  Fkm,
        "rho_k":        rho_k,
        "delta_F_k":    delta_F_k,
        "mmlu_acc_round0": mmlu_acc_round0,
        "mmlu_acc_roundk": mmlu_acc_roundk,
        "mmlu_drop_pp": mmlu_drop_pp,
        "decision":     decision,
        "reasons":      reasons,
        "thresholds":   {
            "delta_F":      args.delta_F_thresh,
            "rho":          args.rho_thresh,
            "mmlu_drop_pp": args.mmlu_drop_thresh_pp,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✓ 收敛分析完成 → %s", out_path)
    logger.info("  ρ_%d=%.4f  ΔF̄_%d=%.4f  MMLU drop=%s pp  →  %s  (%s)",
                round_id, rho_k, round_id, delta_F_k,
                f"{mmlu_drop_pp:.2f}" if mmlu_drop_pp is not None else "N/A",
                decision, "; ".join(reasons) if reasons else "no stop trigger")


if __name__ == "__main__":
    main()
