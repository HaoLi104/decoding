"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-Medical 训练集

混合比例（实验计划 Section 3）：
  75% medalpaca/medical_meadow_medqa  — 医学 MCQ + 解析，注入医学实体
  25% tatsu-lab/alpaca                — 通用指令数据，充当「格式锚点」防止 Chat Template 坍塌

输出格式：LLaMA-Factory alpaca_format（JSON）
  [{"instruction": "...", "input": "...", "output": "..."}]

输出文件（远端机器）：
  /data/ocean/decoding/data/medical_mix_train.json
  /data/ocean/decoding/data/medical_mix_val.json

用法（远端机器）：
  cd /data/ocean/decoding
  conda activate kvner
  python prepare_finetune_data.py \\
    --out_dir /data/ocean/decoding/data \\
    --medical_limit 20000 \\
    --general_ratio 0.25 \\
    --val_size 0.05 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

def load_medical_meadow_medqa(limit: int) -> List[Dict[str, Any]]:
    """加载 medalpaca/medical_meadow_medqa，转换为 alpaca 格式。

    原始字段：
      question (str): 含 A/B/C/D 选项的完整题目文本
      answer   (str): 正确选项字母（A/B/C/D/E）

    转换后：
      instruction: 医学问答系统提示 + 题目
      input:       "" (空，选项已含在 question 中)
      output:      "The correct answer is X."
    """
    from datasets import load_dataset

    ds = load_dataset("medalpaca/medical_meadow_medqa", split="train")
    records = []
    for item in ds:
        question = str(item.get("input", "")).strip()
        answer   = str(item.get("output", "")).strip()
        if not question or not answer:
            continue

        instruction = (
            "You are a medical expert. "
            "Answer the following multiple-choice question by selecting the single best answer. "
            "Provide the answer letter followed by a brief explanation."
        )
        records.append({
            "instruction": instruction,
            "input":       question,
            "output":      answer,
        })
        if len(records) >= limit:
            break

    return records


def load_general_alpaca(limit: int) -> List[Dict[str, Any]]:
    """加载 tatsu-lab/alpaca 通用指令数据，作为格式锚点。

    过滤空 output 的样本，直接复用原始 instruction/input/output 字段。
    """
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    records = []
    for item in ds:
        instruction = str(item.get("instruction", "")).strip()
        inp         = str(item.get("input", "")).strip()
        output      = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        records.append({
            "instruction": instruction,
            "input":       inp,
            "output":      output,
        })
        if len(records) >= limit:
            break

    return records


# ---------------------------------------------------------------------------
# 混合与分割
# ---------------------------------------------------------------------------

def mix_and_split(
    medical_records: List[Dict],
    general_records: List[Dict],
    val_size:        float,
    seed:            int,
) -> tuple[List[Dict], List[Dict]]:
    """按 75/25 比例混合，随机打乱，按 val_size 分割训练/验证集。

    Returns:
        (train_records, val_records)
    """
    all_records = medical_records + general_records
    rng = random.Random(seed)
    rng.shuffle(all_records)

    n_val   = max(1, int(len(all_records) * val_size))
    val     = all_records[:n_val]
    train   = all_records[n_val:]

    return train, val


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Qwen2.5-3B-Instruct-Medical 微调数据集")

    parser.add_argument(
        "--out_dir", type=str,
        default="/data/ocean/decoding/data",
        help="输出目录（会自动创建）",
    )
    parser.add_argument(
        "--medical_limit", type=int, default=20000,
        help="医学数据最大样本数（实际取 min(数据集大小, medical_limit)）",
    )
    parser.add_argument(
        "--general_ratio", type=float, default=0.25,
        help="通用数据占总样本比例（0~1），默认 0.25",
    )
    parser.add_argument(
        "--val_size", type=float, default=0.05,
        help="验证集比例（0~1），默认 0.05",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1：加载医学数据（75%）
    # -----------------------------------------------------------------------
    print(f"[1/4] 加载 medalpaca/medical_meadow_medqa  limit={args.medical_limit}")
    medical_records = load_medical_meadow_medqa(limit=args.medical_limit)
    print(f"      医学样本数: {len(medical_records)}")

    # -----------------------------------------------------------------------
    # Step 2：按比例计算通用数据量（25%）
    # -----------------------------------------------------------------------
    # general / (medical + general) = general_ratio
    # => general = medical * general_ratio / (1 - general_ratio)
    general_limit = int(
        len(medical_records) * args.general_ratio / max(1 - args.general_ratio, 1e-9)
    )
    print(f"[2/4] 加载 tatsu-lab/alpaca（通用格式锚点）  limit={general_limit}")
    general_records = load_general_alpaca(limit=general_limit)
    print(f"      通用样本数: {len(general_records)}")

    actual_ratio = len(general_records) / max(len(medical_records) + len(general_records), 1)
    print(f"      实际混合比例: 医学={1-actual_ratio:.1%}  通用={actual_ratio:.1%}")

    # -----------------------------------------------------------------------
    # Step 3：混合 + 分割
    # -----------------------------------------------------------------------
    print(f"[3/4] 混合打乱，分割训练/验证集（val_size={args.val_size}）")
    train_records, val_records = mix_and_split(
        medical_records=medical_records,
        general_records=general_records,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"      训练集: {len(train_records)}  验证集: {len(val_records)}")

    # -----------------------------------------------------------------------
    # Step 4：写入 JSON
    # -----------------------------------------------------------------------
    train_path = out_dir / "medical_mix_train.json"
    val_path   = out_dir / "medical_mix_val.json"

    print(f"[4/4] 写入文件")
    train_path.write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✓ 完成")
    print(f"  训练集 → {train_path}  ({len(train_records)} 条)")
    print(f"  验证集 → {val_path}  ({len(val_records)} 条)")
    print(f"\n下一步：将以下两行加入 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps({
        "medical_mix_train": {
            "file_name": str(train_path),
            "formatting": "alpaca",
        },
        "medical_mix_val": {
            "file_name": str(val_path),
            "formatting": "alpaca",
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
