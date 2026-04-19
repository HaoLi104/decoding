"""DAF — 热点 Top-K 模块跨轮稳定性

输入：
  layer_scores_round{0}.json
  layer_scores_round{1}.json
  ...

度量：
  Jaccard(Top-K_R0, Top-K_R1) = |∩| / |∪|
  Spearman 排序相关（按所有共同模块名的 score 排序）
  Top-K 集合差异：仅 R0 / 仅 R1 / 共同

输出：
  hotspot_stability.json：
    {
      "subset": "fdlp",
      "top_k": 8,
      "rounds": [{"round_id": 0, "top_modules": [...]}, {"round_id": 1, ...}],
      "pairwise": [
        {"r1": 0, "r2": 1, "jaccard": float, "spearman": float, "common": [...], "only_r1": [...], "only_r2": [...]},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.hotspot_stability")


def _load_top_modules(layer_scores_path: Path, subset: str, top_k: int) -> List[str]:
    obj = json.loads(layer_scores_path.read_text(encoding="utf-8"))
    summary = obj.get("summary", {}).get(subset)
    if not summary:
        raise ValueError(f"layer_scores 中找不到 subset='{subset}' (file={layer_scores_path})")
    modules = summary.get("top_k_modules", [])
    return [m["name"] for m in modules[:top_k]]


def _all_module_scores(layer_scores_path: Path, subset: str) -> Dict[str, float]:
    obj = json.loads(layer_scores_path.read_text(encoding="utf-8"))
    raw = obj.get("subsets", {}).get(subset, {})
    return {k: float(v["score"]) for k, v in raw.items()}


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / max(len(union), 1)


def _rankdata(values: List[float]) -> List[float]:
    """Average-rank 实现，避免引入 scipy。"""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(a_map: Dict[str, float], b_map: Dict[str, float]) -> Optional[float]:
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    if len(common) < 3:
        return None
    a_vals = [a_map[k] for k in common]
    b_vals = [b_map[k] for k in common]
    a_r = _rankdata(a_vals)
    b_r = _rankdata(b_vals)
    n = len(common)
    mean_a = sum(a_r) / n
    mean_b = sum(b_r) / n
    num = sum((a_r[i] - mean_a) * (b_r[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((a_r[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((b_r[i] - mean_b) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF — 热点 Top-K 模块跨轮稳定性")
    p.add_argument("--layer_scores", nargs="+", required=True,
                   help="按轮次顺序的 layer_scores_round{k}.json 路径列表")
    p.add_argument("--out",         required=True)
    p.add_argument("--subset",      default="fdlp", help="使用哪个 subset (默认 fdlp)")
    p.add_argument("--top_k",       type=int, default=8)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    rounds_info: List[Dict[str, Any]] = []
    top_modules_per_round: List[List[str]] = []
    score_maps: List[Dict[str, float]] = []
    for k, path_str in enumerate(args.layer_scores):
        path = Path(path_str)
        top = _load_top_modules(path, subset=args.subset, top_k=args.top_k)
        scores = _all_module_scores(path, subset=args.subset)
        rounds_info.append({
            "round_id":      k,
            "layer_scores":  str(path),
            "top_modules":   top,
            "n_modules_scored": len(scores),
        })
        top_modules_per_round.append(top)
        score_maps.append(scores)
        logger.info("Round %d  top-%d=%s", k, args.top_k, top)

    pairwise: List[Dict[str, Any]] = []
    for i in range(len(top_modules_per_round)):
        for j in range(i + 1, len(top_modules_per_round)):
            a, b = top_modules_per_round[i], top_modules_per_round[j]
            jac = _jaccard(a, b)
            sp  = _spearman(score_maps[i], score_maps[j])
            common  = sorted(set(a) & set(b))
            only_a  = sorted(set(a) - set(b))
            only_b  = sorted(set(b) - set(a))
            pairwise.append({
                "r1":       i,
                "r2":       j,
                "jaccard":  jac,
                "spearman": sp,
                "common":   common,
                "only_r1":  only_a,
                "only_r2":  only_b,
            })
            logger.info("Round %d vs %d  Jaccard=%.4f  Spearman=%s  共同=%d 模块",
                        i, j, jac, f"{sp:.4f}" if sp is not None else "N/A",
                        len(common))

    out = {
        "subset":   args.subset,
        "top_k":    args.top_k,
        "rounds":   rounds_info,
        "pairwise": pairwise,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✓ 热点稳定性写入 %s", out_path)


if __name__ == "__main__":
    main()
