"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-MedMCQA 训练集（格式修正版）

数据集：medmcqa（印度 PGMEE 医学研究生入学考试，单选题）
  - train：182,822 条
  - 仅保留 choice_type='single' + cop != -1
  - 可通过 --subject 参数限定专科（如 "Surgery"）

【关键修复：训练/评测格式完全统一】
  训练 input  = format_prompt(dataset_name="medmcqa")  ← 与 run_baseline.py 完全一致
  训练 output = "Final answer: X"                      ← 极短，永远不会被 max_new_tokens 截断

  LLaMA-Factory alpaca 格式下：
    - instruction + input → 用户消息（不算 loss）
    - output             → 助手消息（算 loss）
  训练时系统/用户提示完全保留，只对 "Final answer: X" 计算梯度。

混合比例：
  75% MedMCQA train MCQ（格式已修正）
  25% tatsu-lab/alpaca 通用指令（格式锚点）

输出文件（远端机器）：
  全科：/data/ocean/decoding/data/medmcqa_mix_train.json
  专科：/data/ocean/decoding/data/medmcqa_surgery_mix_train.json（示例）

用法（远端机器）：
  cd /data/ocean/decoding && conda activate kvner
  export HF_ENDPOINT=https://hf-mirror.com

  # 全科（沿用旧逻辑）
  python prepare_finetune_data_medmcqa.py --out_dir /data/ocean/decoding/data

  # 专项外科
  python prepare_finetune_data_medmcqa.py \\
      --out_dir /data/ocean/decoding/data \\
      --subject Surgery \\
      --domain_limit 20000
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


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_medmcqa_mcq(
    limit: int,
    tokenizer_path: str,
    subject: str = "",
) -> List[Dict[str, Any]]:
    """加载 medmcqa train split，转换为 alpaca 格式。

    参数
    ----
    limit        : 最多保留样本数（0 = 不限）
    tokenizer_path: 仅用于 import 路径校验，不影响格式
    subject      : 若非空，仅保留该 subject_name 的题目（如 "Surgery"）
    """
    from datasets import load_dataset
    from data_loader import SYSTEM_PROMPTS

    system_prompt = SYSTEM_PROMPTS["medmcqa"]

    ds = load_dataset("medmcqa", split="train")

    if subject:
        print(f"  [subject 过滤] 仅保留 subject_name='{subject}' 的题目")

    records: List[Dict[str, Any]] = []
    for item in ds:
        if item.get("choice_type", "single") != "single":
            continue
        cop = item.get("cop", -1)
        if not isinstance(cop, int) or cop not in range(4):
            continue

        # subject 过滤
        if subject and str(item.get("subject_name", "")).strip() != subject:
            continue

        question = str(item.get("question", "")).strip()
        opa = str(item.get("opa", "")).strip()
        opb = str(item.get("opb", "")).strip()
        opc = str(item.get("opc", "")).strip()
        opd = str(item.get("opd", "")).strip()
        if not question or not all([opa, opb, opc, opd]):
            continue

        answer_letter = _LETTER[cop]

        # 与 format_prompt(dataset_name="medmcqa") 的 user_content 完全一致
        opt_lines = [f"A. {opa}", f"B. {opb}", f"C. {opc}", f"D. {opd}"]
        user_content = (
            question.strip()
            + "\n"
            + "\n".join(opt_lines)
            + "\n\nBefore the final answer, repeat the chosen option text exactly once. "
            + "Answer format: after reasoning, output the chosen option text, "
            + "then end with exactly one line in the form 'Final answer: X' "
            + "where X is one of A/B/C/D. No text is allowed after that line."
        )

        records.append({
            "instruction": system_prompt,                       # → system 消息，不计 loss
            "input":       user_content,                        # → user 消息，不计 loss
            "output":      f"Final answer: {answer_letter}",   # → assistant，计 loss
        })
        if limit > 0 and len(records) >= limit:
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
    parser.add_argument("--tokenizer_path", type=str,
                        default="/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct",
                        help="Base 模型路径（仅用于路径校验，不影响权重）")
    parser.add_argument("--subject",       type=str,   default="",
                        help="若指定，仅保留该 subject_name 的题目（如 Surgery、Dental）")
    parser.add_argument("--domain_limit",  type=int,   default=15000,
                        help="MedMCQA 领域数据最大样本数（0=不限，default: 15000）")
    parser.add_argument("--general_ratio", type=float, default=0.25,
                        help="通用数据占总样本比例（default: 0.25）")
    parser.add_argument("--val_size",      type=float, default=0.05,
                        help="验证集比例（default: 0.05）")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 根据 subject 生成输出文件名前缀
    subject_tag = args.subject.lower().replace(" ", "_") if args.subject else "all"
    file_prefix = f"medmcqa_{subject_tag}" if args.subject else "medmcqa"

    # Step 1：加载领域数据（75%）
    print(f"[1/4] 加载 medmcqa train（单选）  limit={args.domain_limit}  subject='{args.subject or '全科'}'")
    domain_records = load_medmcqa_mcq(
        limit=args.domain_limit,
        tokenizer_path=args.tokenizer_path,
        subject=args.subject,
    )
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
    train_path = out_dir / f"{file_prefix}_mix_train.json"
    val_path   = out_dir / f"{file_prefix}_mix_val.json"
    print(f"[4/4] 写入文件")
    train_path.write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_key = f"{file_prefix}_mix_train"
    val_key   = f"{file_prefix}_mix_val"

    print(f"\n完成")
    print(f"  训练集 → {train_path}  ({len(train_records)} 条)")
    print(f"  验证集 → {val_path}  ({len(val_records)} 条)")
    print(f"\n下一步：将以下内容加入 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps({
        train_key: {"file_name": str(train_path), "formatting": "alpaca"},
        val_key:   {"file_name": str(val_path),   "formatting": "alpaca"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
