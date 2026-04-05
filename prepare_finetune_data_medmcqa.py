"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-MedMCQA 训练集

数据集：medmcqa（印度 PGMEE 医学研究生入学考试，单选题）
  - train：182,822 条，含解释字段 exp（直接用作推理链，解决 Law 域失败的根源）
  - 仅保留 choice_type='single' + cop != -1

混合比例（与医疗实验保持一致）：
  75% MedMCQA train MCQ — 训练/评测格式完全对齐（MCQ in → reasoning + Final answer: X）
  25% tatsu-lab/alpaca  — 通用指令格式锚点，防止 Chat Template 坍塌

输出格式 output（严格对齐 extract_answer 期望）：
  "{exp}\n\nFinal answer: X"
  若 exp 为空则退化为 "Final answer: X"

输出文件（远端机器）：
  /data/ocean/decoding/data/medmcqa_mix_train.json
  /data/ocean/decoding/data/medmcqa_mix_val.json

用法（远端机器）：
  cd /data/ocean/decoding && conda activate kvner
  export HF_ENDPOINT=https://hf-mirror.com
  python prepare_finetune_data_medmcqa.py \\
      --out_dir /data/ocean/decoding/data \\
      --domain_limit 15000 \\
      --general_ratio 0.25 \\
      --val_size 0.05 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_LETTER = ["A", "B", "C", "D"]

SYSTEM_INSTRUCTION = (
    "You are a medical expert specializing in postgraduate medical entrance exams. "
    "Answer the following multiple-choice question. "
    "End your response with exactly one line: 'Final answer: X' "
    "where X is the letter (A/B/C/D) of the best answer. "
    "Do not add any text after that line."
)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_medmcqa_mcq(limit: int) -> List[Dict[str, Any]]:
    """加载 medmcqa train split，转换为 alpaca 格式。

    关键设计：
    - 使用 exp（解释）字段作为推理链 → 输出格式与 extract_answer 完全对齐
    - 仅保留 choice_type='single' 的单选题（与评测口径一致）
    - cop=-1 表示 test 隐藏答案，此处不存在（train 全部有答案）

    原始字段：
      question   (str): 题干
      opa/b/c/d  (str): 四个选项
      cop        (int): 正确选项下标 0-3
      exp        (str): 解题解释（可能为空）
      subject_name (str): 科目名
      choice_type  (str): 'single' 或 'multi'

    输出格式（严格对齐 run_baseline.py extract_answer）：
      instruction: SYSTEM_INSTRUCTION
      input:       "Question: ...\n\nOptions:\nA. ...\nB. ...\nC. ...\nD. ..."
      output:      "{exp}\n\nFinal answer: X"（exp 为空时退化为 "Final answer: X"）
    """
    from datasets import load_dataset

    ds = load_dataset("medmcqa", split="train")

    records: List[Dict[str, Any]] = []
    for item in ds:
        if item.get("choice_type", "single") != "single":
            continue
        cop = item.get("cop", -1)
        if not isinstance(cop, int) or cop not in range(4):
            continue

        question = str(item.get("question", "")).strip()
        opa = str(item.get("opa", "")).strip()
        opb = str(item.get("opb", "")).strip()
        opc = str(item.get("opc", "")).strip()
        opd = str(item.get("opd", "")).strip()
        if not question or not all([opa, opb, opc, opd]):
            continue

        answer_letter = _LETTER[cop]
        exp = str(item.get("exp", "")).strip()

        opts_text = f"A. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"
        input_text = f"Question: {question}\n\nOptions:\n{opts_text}"

        # 优先使用官方解释作为推理链；无解释时退化为仅输出答案
        if exp:
            output_text = f"{exp}\n\nFinal answer: {answer_letter}"
        else:
            output_text = f"Final answer: {answer_letter}"

        records.append({
            "instruction": SYSTEM_INSTRUCTION,
            "input":       input_text,
            "output":      output_text,
        })
        if len(records) >= limit:
            break

    return records


def load_general_alpaca(limit: int) -> List[Dict[str, Any]]:
    """加载 tatsu-lab/alpaca 通用指令数据，充当格式锚点。"""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    records: List[Dict[str, Any]] = []
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
    domain_records:  List[Dict],
    general_records: List[Dict],
    val_size:        float,
    seed:            int,
) -> tuple[List[Dict], List[Dict]]:
    """按指定比例混合后打乱，切出验证集。"""
    all_records = domain_records + general_records
    rng = random.Random(seed)
    rng.shuffle(all_records)

    n_val = max(1, int(len(all_records) * val_size))
    return all_records[n_val:], all_records[:n_val]


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 Qwen2.5-3B-Instruct-MedMCQA 微调数据集（LLaMA-Factory alpaca 格式）"
    )
    parser.add_argument("--out_dir",       type=str,   default="/data/ocean/decoding/data")
    parser.add_argument("--domain_limit",  type=int,   default=15000,
                        help="MedMCQA 领域数据最大样本数（default: 15000）")
    parser.add_argument("--general_ratio", type=float, default=0.25,
                        help="通用数据占总样本比例（default: 0.25）")
    parser.add_argument("--val_size",      type=float, default=0.05,
                        help="验证集比例（default: 0.05）")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1：加载领域数据（75%）
    print(f"[1/4] 加载 medmcqa train（单选）  limit={args.domain_limit}")
    domain_records = load_medmcqa_mcq(limit=args.domain_limit)
    print(f"      领域 MCQ 样本数: {len(domain_records)}")

    # Step 2：计算并加载通用数据（25%）
    # general / (domain + general) = general_ratio
    # => general = domain * general_ratio / (1 - general_ratio)
    general_limit = int(
        len(domain_records) * args.general_ratio / max(1 - args.general_ratio, 1e-9)
    )
    print(f"[2/4] 加载 tatsu-lab/alpaca（通用格式锚点）  limit={general_limit}")
    general_records = load_general_alpaca(limit=general_limit)
    print(f"      通用样本数: {len(general_records)}")
    actual_ratio = len(general_records) / max(len(domain_records) + len(general_records), 1)
    print(f"      实际混合比例: 领域={1-actual_ratio:.1%}  通用={actual_ratio:.1%}")

    # Step 3：混合 + 分割
    print(f"[3/4] 混合打乱，分割训练/验证集（val_size={args.val_size}）")
    train_records, val_records = mix_and_split(
        domain_records=domain_records,
        general_records=general_records,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"      训练集: {len(train_records)}  验证集: {len(val_records)}")

    # Step 4：写入 JSON
    train_path = out_dir / "medmcqa_mix_train.json"
    val_path   = out_dir / "medmcqa_mix_val.json"
    print(f"[4/4] 写入文件")
    train_path.write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n完成")
    print(f"  训练集 → {train_path}  ({len(train_records)} 条)")
    print(f"  验证集 → {val_path}  ({len(val_records)} 条)")
    print(f"\n下一步：将以下内容加入 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps({
        "medmcqa_mix_train": {"file_name": str(train_path), "formatting": "alpaca"},
        "medmcqa_mix_val":   {"file_name": str(val_path),   "formatting": "alpaca"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
