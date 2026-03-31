"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-Medical 训练集

混合比例（实验计划 Section 3）：
  75% GBaker/MedQA-USMLE-4-options (train split)
       — 与测试集同源的 MCQ 数据，格式完全对齐（answer_idx: A/B/C/D）
       — 输出统一为 "Final answer: X"，防止格式坍塌
  25% tatsu-lab/alpaca — 通用指令数据，充当「格式锚点」防止 Chat Template 坍塌

关键设计：训练输出格式必须与评测脚本 run_baseline.py 完全一致：
  "Final answer: X"
否则 extract_answer() 无法匹配，评测准确率虚假归零。

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
    --medical_limit 10000 \\
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

def load_gbaker_medqa_mcq(limit: int) -> List[Dict[str, Any]]:
    """加载 GBaker/MedQA-USMLE-4-options 训练集，转换为 alpaca 格式。

    原始字段：
      question   (str):  题干
      options    (dict): {"A": "...", "B": "...", "C": "...", "D": "..."}
      answer_idx (str):  正确选项字母（A/B/C/D）

    转换后输出格式严格对齐 run_baseline.py 的 extract_answer() 期望：
      instruction: 系统提示（与 run_baseline.py SYSTEM_PROMPT 一致）
      input:       题干 + 选项
      output:      "Final answer: X"   ← 关键：与评测脚本完全一致

    与测试集同源（同一数据集的 train/test split），确保：
      1. 格式不会崩塌
      2. ΔP 信号在测试题型上有效
    """
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    SYSTEM_INSTRUCTION = (
        "You are a medical expert. "
        "Answer the following multiple-choice question. "
        "End your response with exactly one line: 'Final answer: X' "
        "where X is the letter (A/B/C/D) of the best answer. "
        "Do not add any text after that line."
    )

    records = []
    for item in ds:
        question   = str(item.get("question", "")).strip()
        options    = item.get("options", {})
        answer_idx = str(item.get("answer_idx", "")).strip().upper()

        if not question or not options or answer_idx not in "ABCD":
            continue

        opts_text = "\n".join(
            "{k}. {v}".format(k=k, v=v) for k, v in sorted(options.items())
        )
        input_text = "Question: {q}\n\nOptions:\n{o}".format(q=question, o=opts_text)

        records.append({
            "instruction": SYSTEM_INSTRUCTION,
            "input":       input_text,
            "output":      "Final answer: {x}".format(x=answer_idx),
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
    print("[1/4] 加载 GBaker/MedQA-USMLE-4-options (train)  limit={lim}".format(lim=args.medical_limit))
    medical_records = load_gbaker_medqa_mcq(limit=args.medical_limit)
    print("      医学 MCQ 样本数: {n}".format(n=len(medical_records)))

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
