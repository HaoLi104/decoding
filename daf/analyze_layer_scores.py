"""DAF — 层位打分结果验收分析脚本

用途：
  - 读取 daf.fdlp_score 产出的 layer_scores_round{k}.json
  - 打印 4 套对照（fdlp / fdlp_top_entropy / fdlp_top_disagreement / fdlp_random_subset）
    的 Top-K 模块、各自事件数、模块类型分布
  - 与另一个 layer_scores 文件做 Top-K 重合度对比（用于 smoke vs formal、Round0 vs Round1）

用法：
  cd /data/ocean/decoding && conda activate kvner

  # 单文件分析
  python -m daf.analyze_layer_scores \
      --layer_scores logs/daf_round0/layer_scores_round0.json

  # 双文件对比（formal vs smoke）
  python -m daf.analyze_layer_scores \
      --layer_scores logs/daf_round0/layer_scores_round0.json \
      --compare_with logs/daf_smoke/layer_scores_round0.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def _load_scores(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"layer_scores 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_top_modules(data: dict, subset: str) -> List[dict]:
    """取出某个 subset 的 top_k_modules（按 score 降序）。"""
    return data.get("summary", {}).get(subset, {}).get("top_k_modules", [])


def _module_kind(name: str) -> str:
    """从模块全名（如 model.layers.41.self_attn.v_proj）提取尾部 kind（v_proj）。"""
    return name.rsplit(".", 1)[-1]


def _layer_idx(name: str) -> Optional[int]:
    """从模块全名抽取 layer 序号（model.layers.{N}.xxx）。"""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _print_top_k(data: dict, subset: str, label: Optional[str] = None) -> List[str]:
    label = label or subset
    top = _get_top_modules(data, subset)
    print(f"\n=== [{label}] Top-{len(top)} 模块（按 score 降序） ===")
    if not top:
        print(f"  (subset 不存在或为空)")
        return []
    names: List[str] = []
    for m in top:
        rank = m.get("rank", "?")
        score = m.get("score", 0.0)
        name = m["name"]
        names.append(name)
        print(f"  rank={rank:>2}  {name:60s}  score={score:.4f}")
    return names


def _print_subset_event_counts(data: dict) -> None:
    print("\n=== 各 subset 事件数 (meta.subset_sizes) ===")
    sizes = data.get("meta", {}).get("subset_sizes") or {}
    if not sizes:
        # 回退路径：从 subsets 字典里找任一模块的 n_event
        subsets = data.get("subsets", {})
        for subset_name, modules in subsets.items():
            n_event = None
            if isinstance(modules, dict):
                for _, info in modules.items():
                    if isinstance(info, dict) and "n_event" in info:
                        n_event = info["n_event"]
                        break
            sizes[subset_name] = n_event if n_event is not None else "(未知)"
    for k, v in sizes.items():
        print(f"  {k:25s}: n_event = {v}")


def _print_kind_distribution(top_modules: List[dict], subset_label: str) -> None:
    if not top_modules:
        return
    print(f"\n=== [{subset_label}] Top-K 模块类型分布 ===")
    c: Counter = Counter()
    for m in top_modules:
        c[_module_kind(m["name"])] += 1
    for k, v in c.most_common():
        print(f"  {k:15s}: {v}")


def _print_layer_distribution(top_modules: List[dict], subset_label: str) -> None:
    if not top_modules:
        return
    layers = sorted({_layer_idx(m["name"]) for m in top_modules if _layer_idx(m["name"]) is not None})
    print(f"\n=== [{subset_label}] Top-K 命中 layer 索引 ===")
    print(f"  layers = {layers}  (共 {len(layers)} 层)")


def _compare_overlap(data_a: dict, data_b: dict, subsets: List[str]) -> None:
    print("\n=== 双文件 Top-K 重合度（A ∩ B） ===")
    print(f"  A = layer_scores 文件 1 (本次)")
    print(f"  B = layer_scores 文件 2 (compare_with)")
    for subset in subsets:
        a = set(m["name"] for m in _get_top_modules(data_a, subset))
        b = set(m["name"] for m in _get_top_modules(data_b, subset))
        if not a or not b:
            print(f"  [{subset:25s}]  跳过 (一方为空)")
            continue
        inter = a & b
        denom = max(len(a), len(b))
        print(f"  [{subset:25s}]  |A∩B|={len(inter):>2}/{denom}  jaccard={len(inter)/len(a|b):.3f}")


def _print_meta(data: dict) -> None:
    meta = data.get("meta", {})
    print("=== meta ===")
    for key in ["round_id", "target_model", "n_flip_total", "max_prefix_len", "flip_jsonl"]:
        if key in meta:
            print(f"  {key:18s} = {meta[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAF layer_scores 验收分析")
    parser.add_argument("--layer_scores", required=True, type=Path,
                        help="主分析的 layer_scores_round{k}.json 路径")
    parser.add_argument("--compare_with", type=Path, default=None,
                        help="可选，与之做 Top-K 重合度对比的另一份 layer_scores 文件")
    parser.add_argument("--subsets", nargs="+",
                        default=["fdlp", "fdlp_top_entropy", "fdlp_top_disagreement", "fdlp_random_subset"],
                        help="要分析的 subset 名称列表")
    args = parser.parse_args()

    data = _load_scores(args.layer_scores)
    print(f"\n## 主文件: {args.layer_scores}")
    _print_meta(data)
    _print_subset_event_counts(data)

    # 主 subset (fdlp) 详细分析
    fdlp_top = _get_top_modules(data, "fdlp")
    _print_top_k(data, "fdlp")
    _print_kind_distribution(fdlp_top, "fdlp")
    _print_layer_distribution(fdlp_top, "fdlp")

    # 其他 subset 仅打印 Top-K 名单
    for subset in args.subsets:
        if subset == "fdlp":
            continue
        _print_top_k(data, subset)

    # subset 间重合度
    print("\n=== 主文件 内 各 subset Top-K 与 fdlp 的重合度 ===")
    fdlp_set = set(m["name"] for m in fdlp_top)
    for subset in args.subsets:
        if subset == "fdlp":
            continue
        s = set(m["name"] for m in _get_top_modules(data, subset))
        if not s:
            continue
        inter = fdlp_set & s
        print(f"  fdlp ∩ {subset:25s} = {len(inter)}/{len(fdlp_set)}  "
              f"jaccard={len(inter)/len(fdlp_set | s):.3f}")

    # 双文件对比
    if args.compare_with is not None:
        data_b = _load_scores(args.compare_with)
        print(f"\n## 对比文件: {args.compare_with}")
        _print_meta(data_b)
        _compare_overlap(data, data_b, args.subsets)

    print("\n## 验收建议")
    print("  - fdlp Top-K 中 v_proj/down_proj 占比 ≥ 50% → 领域知识聚合在 attention/MLP 关键投影")
    print("  - fdlp ∩ fdlp_top_entropy ≥ 5/8 → 高熵子集与全量结果一致，监督信号鲁棒")
    print("  - smoke ∩ formal ≥ 5/8 → hotspot 对样本规模稳定，可进入 M3 训练")


if __name__ == "__main__":
    main()
