"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-MedMCQA 训练集（格式修正版）

数据集：medmcqa（印度 PGMEE 医学研究生入学考试，单选题）
  - train：182,822 条
  - 仅保留 choice_type='single' + cop != -1

【关键修复：训练/评测格式完全统一】
  训练 input  = format_prompt(dataset_name="medmcqa")  ← 与 run_baseline.py 完全一致
  训练 output = "Final answer: X"                      ← 极短，永远不会被 max_new_tokens 截断

  之前失败的根因：
    1. 训练用自定义格式，评测用 format_prompt() → prompt 结构从未见过
    2. 训练 output = exp + "Final answer: X"（400+ token），但评测 max_new_tokens=256 截断

  由于 LLaMA-Factory alpaca 格式下：
    - instruction + input → 用户消息（不算 loss）
    - output             → 助手消息（算 loss）
  训练时系统/用户提示完全保留，只对 "Final answer: X" 计算梯度。
  推理时提示不变，模型按正常推理流程生成，最终必然以 "Final answer: X" 结尾。

混合比例：
  75% MedMCQA train MCQ（格式已修正）
  25% tatsu-lab/alpaca 通用指令（格式锚点）

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


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_medmcqa_mcq(limit: int, tokenizer_path: str) -> List[Dict[str, Any]]:
    """加载 medmcqa train split，转换为 alpaca 格式。

    【格式修正核心】：
    训练 input = format_prompt(dataset_name="medmcqa") 的完整输出，
    与 run_baseline.py 的评测 prompt 完全一致，消除格式错位。

    LLaMA-Factory alpaca 格式 + qwen template 的拼接规则：
      - instruction 字段 → <|im_start|>system ... <|im_end|>
      - input 字段       → <|im_start|>user ... <|im_end|>
      - output 字段      → <|im_start|>assistant ... <|im_end|>（计算 loss）

    为统一格式，将 system prompt 放入 instruction，
    将 format_prompt() 输出中的用户消息部分放入 input。
    output 固定为 "Final answer: X"，保证 max_new_tokens=256 永不截断。
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from data_loader import SYSTEM_PROMPTS

    # 加载 tokenizer 仅用于解析 apply_chat_template 产生的用户消息
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 直接从 data_loader 取 system prompt，保证与评测完全一致
    system_prompt = SYSTEM_PROMPTS["medmcqa"]

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

        # 与 format_prompt() 完全一致的选项格式
        opt_lines = [f"A. {opa}", f"B. {opb}", f"C. {opc}", f"D. {opd}"]
        # 与 format_prompt(dataset_name="medmcqa") 的 user_content 完全一致
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
            "instruction": system_prompt,    # → system 消息，不计 loss
            "input":       user_content,     # → user 消息，不计 loss
            "output":      f"Final answer: {answer_letter}",  # → assistant，计 loss
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
    parser.add_argument("--tokenizer_path", type=str,
                        default="/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct",
                        help="Base 模型路径，用于加载 tokenizer（仅用于格式验证，不影响权重）")
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
    print(f"      tokenizer_path={args.tokenizer_path}")
    domain_records = load_medmcqa_mcq(limit=args.domain_limit, tokenizer_path=args.tokenizer_path)
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
