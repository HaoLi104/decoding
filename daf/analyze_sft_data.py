"""DAF — SFT 数据集验收分析脚本

用途：
  - 读取 daf.build_flip_sft_data 产出的 alpaca 格式 JSON 数据集
  - 按"是否含 flip 监督标签"自动归类（医学 flip 样本 / 通用 Alpaca 锚点）
  - 完整打印各类样本若干条（不截断 output），确认 token-level 监督正确
  - 统计 output 长度分布、source 分布

用法：
  cd /data/ocean/decoding && conda activate kvner
  python -m daf.analyze_sft_data \
      --data /data/ocean/decoding/data/daf_round0_train.json \
      --val  /data/ocean/decoding/data/daf_round0_val.json \
      --n_show 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _load(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _classify(sample: dict) -> str:
    """根据样本字段判定来源类型。

    规则：
      - 若 sample 含 'meta.source' → 直接用
      - 否则按 instruction / output 启发式推断：
        * instruction 含 '<|im_start|>system' 或 'medical' → flip 类
        * 其他 → general
    """
    meta = sample.get("meta") or {}
    if isinstance(meta, dict) and "source" in meta:
        return str(meta["source"])

    instr = (sample.get("instruction") or "").lower()
    if "<|im_start|>system" in instr or "medical" in instr or "postgraduate" in instr:
        return "flip_or_balance"
    return "general_alpaca"


def _print_sample(idx: int, s: dict, kind: str) -> None:
    print(f"\n--- [{kind}] idx={idx} ---")
    instr = s.get("instruction", "")
    inp = s.get("input", "")
    out = s.get("output", "")
    meta = s.get("meta", {})
    print(f"instruction ({len(instr)} chars):")
    print(f"  {instr[:300]}{' ...' if len(instr) > 300 else ''}")
    print(f"input ({len(inp)} chars):")
    print(f"  {inp[:200]}{' ...' if len(inp) > 200 else ''}")
    print(f"output ({len(out)} chars) [完整]:")
    print(f"  {out!r}")
    if meta:
        print(f"meta: {meta}")


def _stats(samples: List[dict], label: str) -> None:
    print(f"\n=== {label} 统计 ===")
    print(f"  总样本数: {len(samples)}")

    kind_counter: Counter = Counter()
    out_lens: Dict[str, List[int]] = {}
    instr_lens: Dict[str, List[int]] = {}

    for s in samples:
        k = _classify(s)
        kind_counter[k] += 1
        out_lens.setdefault(k, []).append(len(s.get("output", "")))
        instr_lens.setdefault(k, []).append(len(s.get("instruction", "")))

    print(f"  按来源分布:")
    for k, v in kind_counter.most_common():
        ratio = v / len(samples) * 100
        print(f"    {k:25s}  n={v:5d}  ({ratio:5.1f}%)")

    print(f"  output 长度（chars）分布:")
    for k, lens in out_lens.items():
        if not lens:
            continue
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        p50 = lens_sorted[n // 2]
        p10 = lens_sorted[max(0, n // 10)]
        p90 = lens_sorted[min(n - 1, n * 9 // 10)]
        avg = sum(lens) / n
        print(f"    {k:25s}  min={min(lens):4d}  p10={p10:4d}  p50={p50:4d}  "
              f"p90={p90:4d}  max={max(lens):5d}  mean={avg:6.1f}")

    print(f"  instruction 长度（chars）分布:")
    for k, lens in instr_lens.items():
        if not lens:
            continue
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        avg = sum(lens) / n
        print(f"    {k:25s}  min={min(lens):4d}  p50={lens_sorted[n // 2]:4d}  "
              f"max={max(lens):5d}  mean={avg:6.1f}")


def _show_each_kind(samples: List[dict], n_show: int) -> None:
    """每个类别打印 n_show 条完整样本。"""
    by_kind: Dict[str, List[Tuple[int, dict]]] = {}
    for i, s in enumerate(samples):
        by_kind.setdefault(_classify(s), []).append((i, s))

    for kind, items in by_kind.items():
        print(f"\n#######  类别: {kind}  (共 {len(items)} 条，展示前 {min(n_show, len(items))} 条)  #######")
        for idx, sample in items[:n_show]:
            _print_sample(idx, sample, kind)


def _check_flip_balance_pairs(samples: List[dict]) -> None:
    """如样本带 meta.flip_qid + meta.kind ∈ {flip_positive, flip_balance}，
    检查正样本与平衡样本数量是否 1:1。"""
    pos = bal = 0
    for s in samples:
        meta = s.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        kind = meta.get("kind", "")
        if kind == "flip_positive":
            pos += 1
        elif kind == "flip_balance":
            bal += 1
    if pos == 0 and bal == 0:
        print("\n=== flip 正/平衡样本检查: meta.kind 缺失，跳过结构检查 ===")
        return
    print(f"\n=== flip 正/平衡样本平衡度 ===")
    print(f"  flip_positive : {pos}")
    print(f"  flip_balance  : {bal}")
    print(f"  比例 pos:bal  = {pos}/{bal}  → {'OK' if pos == bal else '失衡!'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAF SFT 数据验收分析")
    parser.add_argument("--data", required=True, type=Path,
                        help="主分析的 train json 路径（alpaca 格式）")
    parser.add_argument("--val", type=Path, default=None,
                        help="可选 val 文件，仅做总数 + 来源分布统计")
    parser.add_argument("--n_show", type=int, default=3,
                        help="每类样本展示条数（含完整 output）")
    args = parser.parse_args()

    train = _load(args.data)
    print(f"## 分析文件: {args.data}")
    _stats(train, label="train")
    _check_flip_balance_pairs(train)
    _show_each_kind(train, args.n_show)

    if args.val is not None:
        val = _load(args.val)
        print(f"\n\n## 验证集文件: {args.val}")
        _stats(val, label="val")

    print("\n## 验收建议（继续 M3b 的 Go 标准）")
    print("  1. flip 类样本 output 长度 p50 ≥ 1 且 ≤ 5（应是 token-level 短监督）")
    print("  2. 每条 flip 样本的 instruction 应包含完整 prompt 文本（>50 chars）")
    print("  3. flip_positive 与 flip_balance 数量应严格 1:1")
    print("  4. general_alpaca 比例应接近 25%（22%~30% 区间内）")


if __name__ == "__main__":
    main()
