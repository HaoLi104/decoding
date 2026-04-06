"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-CMB 训练集

数据源：FreedomIntelligence/CMB 的 CMB-train-merge.json（本地缓存，269k 条）
过滤后：单选题 + answer in A-D + 恰好4选项，约 100k+ 条可用
训练数据：取其中 20k 条（seed=42 打乱后，跳过前 1000 条 val 数据）

混合比例：
  75% CMB-Exam 中文医学 MCQ（与评测格式完全对齐）
  25% tatsu-lab/alpaca 通用指令（格式锚点）

训练格式（严格对齐 run_baseline.py extract_answer）：
  instruction = SYSTEM_PROMPTS["cmb"]        → system 消息，不计 loss
  input       = format_prompt 用户内容       → user 消息，不计 loss
  output      = "Final answer: X"            → assistant 消息，计 loss

运行（远端机器）：
  cd /data/ocean/decoding && conda activate kvner
  python prepare_finetune_data_cmb.py \\
      --out_dir /data/ocean/decoding/data \\
      --domain_limit 20000 \\
      --general_ratio 0.25 \\
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
_VALID  = set(_LETTER)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_cmb_mcq(limit: int, seed: int, val_size: int = 1000) -> List[Dict[str, Any]]:
    """从本地缓存的 CMB-train-merge.json 加载训练样本。

    过滤规则：单选题 + answer in A-D + 恰好4选项 A/B/C/D。
    跳过前 val_size 条（已用作评测集），从第 val_size+1 条起取 limit 条。

    输出格式（LLaMA-Factory alpaca，严格对齐评测 prompt）：
      instruction = SYSTEM_PROMPTS["cmb"]
      input       = 与 format_prompt(dataset_name="cmb") 完全相同的用户内容
      output      = "Final answer: X"
    """
    from data_loader import SYSTEM_PROMPTS, _find_cmb_train_json

    json_path = _find_cmb_train_json()
    print(f"  加载 JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    system_prompt = SYSTEM_PROMPTS["cmb"]

    filtered = []
    for item in raw:
        if item.get("question_type", "") != "单项选择题":
            continue
        ans = str(item.get("answer", "")).strip().upper()
        if ans not in _VALID:
            continue
        opt = item.get("option", {})
        if not isinstance(opt, dict) or set(opt.keys()) != _VALID:
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        filtered.append((question, opt, ans))

    # Deterministic shuffle（与 load_cmb_exam 相同的 seed）
    rng = random.Random(seed)
    rng.shuffle(filtered)

    # 跳过 val_size 条（这些是评测集），从第 val_size 条起取 limit 条
    train_pool = filtered[val_size:]
    train_pool = train_pool[:limit]

    opt_lines_fn = lambda opt: "\n".join(f"{k}. {v}" for k, v in sorted(opt.items()))

    records: List[Dict[str, Any]] = []
    for question, opt, ans in train_pool:
        user_content = (
            question
            + "\n"
            + opt_lines_fn(opt)
            + "\n\n在最终答案之前，请将所选选项的文字重复一遍。"
            + "格式：推理后输出所选选项的完整文字，"
            + "然后最后一行必须且只能是 'Final answer: X'，X 为 A/B/C/D 之一。"
            + "最后一行之后不要输出任何文字。"
        )
        records.append({
            "instruction": system_prompt,
            "input":       user_content,
            "output":      f"Final answer: {ans}",
        })

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
        records.append({"instruction": instruction, "input": inp, "output": output})
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
    all_records = domain_records + general_records
    rng = random.Random(seed + 1)   # +1 避免与数据集 shuffle 相同 seed
    rng.shuffle(all_records)
    n_val = max(1, int(len(all_records) * val_size))
    return all_records[n_val:], all_records[:n_val]


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 Qwen2.5-3B-Instruct-CMB 微调数据集（LLaMA-Factory alpaca 格式）"
    )
    parser.add_argument("--out_dir",       type=str,   default="/data/ocean/decoding/data")
    parser.add_argument("--domain_limit",  type=int,   default=20000,
                        help="CMB 领域数据最大样本数（default: 20000）")
    parser.add_argument("--general_ratio", type=float, default=0.25)
    parser.add_argument("--val_size",      type=float, default=0.05)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 加载 CMB-Exam train（4选项单选，跳过前1000条val）  limit={args.domain_limit}")
    domain_records = load_cmb_mcq(limit=args.domain_limit, seed=args.seed)
    print(f"      领域 MCQ 样本数: {len(domain_records)}")

    general_limit = int(
        len(domain_records) * args.general_ratio / max(1 - args.general_ratio, 1e-9)
    )
    print(f"[2/4] 加载 tatsu-lab/alpaca（通用格式锚点）  limit={general_limit}")
    general_records = load_general_alpaca(limit=general_limit)
    print(f"      通用样本数: {len(general_records)}")
    actual_ratio = len(general_records) / max(len(domain_records) + len(general_records), 1)
    print(f"      实际混合比例: 领域={1-actual_ratio:.1%}  通用={actual_ratio:.1%}")

    print(f"[3/4] 混合打乱，分割训练/验证集（val_size={args.val_size}）")
    train_records, val_records = mix_and_split(
        domain_records, general_records, args.val_size, args.seed
    )
    print(f"      训练集: {len(train_records)}  验证集: {len(val_records)}")

    train_path = out_dir / "cmb_mix_train.json"
    val_path   = out_dir / "cmb_mix_val.json"
    print(f"[4/4] 写入文件")
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成")
    print(f"  训练集 → {train_path}  ({len(train_records)} 条)")
    print(f"  验证集 → {val_path}  ({len(val_records)} 条)")
    print(f"\n下一步：将以下内容加入 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps({
        "cmb_mix_train": {"file_name": str(train_path), "formatting": "alpaca"},
        "cmb_mix_val":   {"file_name": str(val_path),   "formatting": "alpaca"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
