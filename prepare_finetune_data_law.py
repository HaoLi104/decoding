"""
微调数据准备脚本 — 生成 Qwen2.5-3B-Instruct-Law 训练集

混合比例（实验计划 Section 2）：
  75% ShengbinYue/DISC-Law-SFT — 从中提取：
        · type="司法考试"     (~12K 样本)   — 与测试集同源的法律 MCQ / QA
        · type="法律阅读理解"  (~38K 样本)   — 法律文本理解
        · type="法律问答"     (~15K 样本)   — 通用法律知识 QA
      合并后按 law_limit 截断
  25% tatsu-lab/alpaca — 通用指令数据，充当「格式锚点」防止 Chat Template 坍塌

DISC-Law-SFT 原始字段（对话格式）：
  input    (str):  用户问题 / 题目
  output   (str):  助手回答
  type     (str):  数据类型标签
  （部分记录有 instruction 字段作为 system 角色提示）

关键设计：训练中不强行要求 "Final answer: X" 格式
  — 法律 SFT 数据本身是自由文本回答，强行改格式会损害指令跟随能力
  — 仅 JEC-QA MCQ 类题目（若数据集中存在）才输出 "Final answer: X"
  — 评测时 run_baseline.py 的 extract_answer() 已能兼容自由文本中的 "Final answer: X"

输出格式：LLaMA-Factory alpaca_format（JSON）
  [{"instruction": "...", "input": "...", "output": "..."}]

输出文件（远端机器）：
  /data/ocean/decoding/data/law_mix_train.json
  /data/ocean/decoding/data/law_mix_val.json

用法（远端机器）：
  cd /data/ocean/decoding
  conda activate kvner
  python prepare_finetune_data_law.py \\
    --out_dir /data/ocean/decoding/data \\
    --law_limit 15000 \\
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
# DISC-Law-SFT 目标子集标签
# ---------------------------------------------------------------------------

# 优先抽取的子集（按优先级排列）
_TARGET_TYPES = [
    "司法考试",
    "法律阅读理解",
    "法律问答",
    "法律咨询",
]

# 系统指令前缀
_LAW_SYSTEM_INSTRUCTION = (
    "你是一位专业的中国法律助手，擅长司法考试、法律条文解读与法律问答。"
    "请根据问题，提供准确、简洁的法律解答。"
    "如果题目是选择题，请在最后一行输出 'Final answer: X'，其中 X 为 A/B/C/D 之一。"
)


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

def load_disc_law_sft(limit: int) -> List[Dict[str, Any]]:
    """加载 ShengbinYue/DISC-Law-SFT，提取司法考试相关子集，转换为 alpaca 格式。

    优先加载 _TARGET_TYPES 中的子集；若样本不足 limit，则扩展到其余子集。

    转换逻辑：
      - instruction: 统一用 _LAW_SYSTEM_INSTRUCTION
      - input:       item["input"]（原始用户问题）
      - output:      item["output"]（原始助手回答）
        - 若回答中包含 "(A)"/"(B)" 等选项且有明确选项提及，尝试提取并追加 "Final answer: X"
    """
    from datasets import load_dataset

    print("  正在加载 ShengbinYue/DISC-Law-SFT ...")
    # DISC-Law-SFT 以 JSONL 形式存储，直接加载 train split
    try:
        ds = load_dataset(
            "ShengbinYue/DISC-Law-SFT",
            split="train",
            cache_dir="/data/ocean/decoding/data/disc_law_cache",
        )
    except Exception as exc:
        print(f"  [警告] 无法直接加载 DISC-Law-SFT，尝试 data_files 方式: {exc}")
        ds = load_dataset(
            "ShengbinYue/DISC-Law-SFT",
            data_files="DISC-Law-SFT-Pair.jsonl",
            split="train",
            cache_dir="/data/ocean/decoding/data/disc_law_cache",
        )

    # 按优先类型分桶
    buckets: Dict[str, List[Dict]] = {t: [] for t in _TARGET_TYPES}
    overflow: List[Dict] = []

    for item in ds:
        item_type = str(item.get("type", "")).strip()
        inp    = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()

        if not inp or not output:
            continue

        record = {
            "instruction": _LAW_SYSTEM_INSTRUCTION,
            "input":       inp,
            "output":      output,
        }

        if item_type in buckets:
            buckets[item_type].append(record)
        else:
            overflow.append(record)

    # 按优先级合并，直到满足 limit
    merged: List[Dict] = []
    for t in _TARGET_TYPES:
        merged.extend(buckets[t])
        if len(merged) >= limit:
            break
    if len(merged) < limit:
        merged.extend(overflow[:limit - len(merged)])

    print("  DISC-Law-SFT 各子集样本数：")
    for t, bucket in buckets.items():
        print(f"    {t}: {len(bucket)}")
    print(f"  其他类型 (overflow): {len(overflow)}")
    print(f"  合并后取前 {limit} 条: {min(len(merged), limit)} 条")

    random.seed(0)
    random.shuffle(merged)
    return merged[:limit]


def load_general_alpaca(limit: int) -> List[Dict[str, Any]]:
    """加载 tatsu-lab/alpaca 通用指令数据，作为格式锚点。"""
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
    law_records:     List[Dict],
    general_records: List[Dict],
    val_size:        float,
    seed:            int,
) -> tuple[List[Dict], List[Dict]]:
    """混合打乱，按 val_size 比例分割训练/验证集。"""
    all_records = law_records + general_records
    rng = random.Random(seed)
    rng.shuffle(all_records)

    n_val = max(1, int(len(all_records) * val_size))
    val   = all_records[:n_val]
    train = all_records[n_val:]
    return train, val


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Qwen2.5-3B-Instruct-Law 微调数据集")

    parser.add_argument(
        "--out_dir", type=str,
        default="/data/ocean/decoding/data",
        help="输出目录（会自动创建）",
    )
    parser.add_argument(
        "--law_limit", type=int, default=15000,
        help="法律领域数据最大样本数（含司法考试+法律阅读理解+法律问答）",
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

    # -------------------------------------------------------------------
    # Step 1: 加载法律数据（75%）
    # -------------------------------------------------------------------
    print("[1/4] 加载 ShengbinYue/DISC-Law-SFT  law_limit={lim}".format(lim=args.law_limit))
    law_records = load_disc_law_sft(limit=args.law_limit)
    print("      法律领域样本数: {n}".format(n=len(law_records)))

    # -------------------------------------------------------------------
    # Step 2: 按比例计算通用数据量（25%）
    # general / (law + general) = general_ratio
    # => general = law * general_ratio / (1 - general_ratio)
    # -------------------------------------------------------------------
    general_limit = int(
        len(law_records) * args.general_ratio / max(1 - args.general_ratio, 1e-9)
    )
    print("[2/4] 加载 tatsu-lab/alpaca（通用格式锚点）  limit={lim}".format(lim=general_limit))
    general_records = load_general_alpaca(limit=general_limit)
    print("      通用样本数: {n}".format(n=len(general_records)))

    actual_ratio = len(general_records) / max(len(law_records) + len(general_records), 1)
    print("      实际混合比例: 法律={law:.1%}  通用={gen:.1%}".format(
        law=1 - actual_ratio, gen=actual_ratio
    ))

    # -------------------------------------------------------------------
    # Step 3: 混合 + 分割
    # -------------------------------------------------------------------
    print("[3/4] 混合打乱，分割训练/验证集（val_size={vs}）".format(vs=args.val_size))
    train_records, val_records = mix_and_split(
        law_records=law_records,
        general_records=general_records,
        val_size=args.val_size,
        seed=args.seed,
    )
    print("      训练集: {tr}  验证集: {vl}".format(tr=len(train_records), vl=len(val_records)))

    # -------------------------------------------------------------------
    # Step 4: 写入 JSON
    # -------------------------------------------------------------------
    train_path = out_dir / "law_mix_train.json"
    val_path   = out_dir / "law_mix_val.json"

    print("[4/4] 写入文件")
    train_path.write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n完成！")
    print("  训练集 → {tp}  ({n} 条)".format(tp=train_path, n=len(train_records)))
    print("  验证集 → {vp}  ({n} 条)".format(vp=val_path, n=len(val_records)))
    print("\n下一步：将以下内容加入 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps({
        "law_mix_train": {
            "file_name": str(train_path),
            "formatting": "alpaca",
        },
        "law_mix_val": {
            "file_name": str(val_path),
            "formatting": "alpaca",
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
