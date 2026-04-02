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
import os
import random
from pathlib import Path
from typing import Any, Dict, List

# 国内服务器无法直连 HuggingFace，优先使用镜像站。
# 若用户已在环境中设置 HF_ENDPOINT 则尊重用户设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


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
_MCQ_SYSTEM_INSTRUCTION = (
    "你是一位专业的中国法律助手，擅长司法考试、法律条文解读与法律问答。"
    "请根据问题，提供准确、简洁的中文法律解答。"
    "最后一行必须且只能输出：'Final answer: X'，其中 X 为 A/B/C/D 之一。"
    "最后一行之后不要输出任何文字。"
)

_FREETEXT_SYSTEM_INSTRUCTION = (
    "你是一位专业的中国法律助手，擅长司法考试、法律条文解读与法律问答。"
    "请根据问题，提供准确、简洁的中文法律解答。"
)

# ---------------------------------------------------------------------------
# 辅助函数：从 DISC-Law-SFT 司法考试输出中提取答案字母
# ---------------------------------------------------------------------------
import re as _re

def _extract_answer_letter(text: str):
    """从中文法律推理文本中提取答案字母 A/B/C/D。

    DISC-Law-SFT 司法考试 output 常见格式：
      "...因此答案选A。"  /  "故选B"  /  "正确答案为C"  /  "答案：D"
      "选项A是正确的"    /  末尾孤立字母 "...综上，A"

    Returns:
      str | None — 找到则返回大写字母，否则返回 None
    """
    patterns = [
        r'答案[选为是：:\s]*([ABCD])[^A-Z]',
        r'答案[选为是：:\s]*([ABCD])$',
        r'故选\s*([ABCD])',
        r'应选\s*([ABCD])',
        r'选\s*([ABCD])[。，；\s]',
        r'正确[选项答案]+[选为是：:\s]+([ABCD])',
        r'本题[选答案]+[选为是：:\s]*([ABCD])',
        r'选择\s*([ABCD])',
        r'综上[，,][^。\n]*选\s*([ABCD])',
        r'选项\s*([ABCD])\s*(?:正确|是对的|为正确)',
        r'([ABCD])\s*(?:正确|为正确答案)',
        # 末尾孤立字母（最后防线，仅在 output 非常短时有效）
        r'^\s*([ABCD])\s*$',
        r'([ABCD])[。.）)\s]*$',
    ]
    for pat in patterns:
        m = _re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


# 多行 MCQ 选项检测：匹配 A. / A、/ （A）/ A） 等格式（行首或换行后）
_MCQ_RE = _re.compile(
    r'(?:^|\n)\s*[（(]?[A-D][.、．）)。]\s*\S',
    _re.MULTILINE,
)

def _is_mcq_input(text: str) -> bool:
    """检测输入是否含有 A/B/C/D 四选项（MCQ 格式）。

    匹配以下常见格式（行首或换行后）：
      A. 选项    A、选项    A） 选项    （A）选项    A。选项
    """
    return bool(_MCQ_RE.search(text))


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

def load_jecqa_mcq_for_training(
    eval_limit: int = 200,
    train_limit: int = 1600,
) -> List[Dict[str, Any]]:
    """加载 JEC-QA 非评测样本，转换为 MCQ alpaca 格式训练数据。

    数据划分原则（避免数据泄露）：
      - 评测集：前 eval_limit 条（由 run_baseline.py / run_benchmark.py 使用）
      - 训练集：从第 eval_limit+1 条开始，最多取 train_limit 条

    输出格式：
      instruction: MCQ 系统指令（与评测时 format_prompt 完全一致）
      input:       题目 + A/B/C/D 选项（与评测 prompt 对齐）
      output:      "Final answer: X"（仅格式行，简洁高效）
    """
    import sys
    sys.path.insert(0, "/data/ocean/decoding")
    from data_loader import load_jecqa

    print("  正在加载 JEC-QA 训练集（跳过评测前 {e} 条，取后续 {t} 条）...".format(
        e=eval_limit, t=train_limit
    ))
    ds = load_jecqa(offset=eval_limit, limit=train_limit)
    print("  加载到 {n} 条 JEC-QA 非评测 MCQ 样本".format(n=len(ds)))

    records = []
    for item in ds:
        q      = item["question"]
        opts   = item["options"]   # {"A": "...", "B": "...", ...}
        answer = item["answer_idx"]

        opt_text   = "\n".join("{k}. {v}".format(k=k, v=v) for k, v in sorted(opts.items()))
        input_text = "{q}\n{o}".format(q=q, o=opt_text)

        records.append({
            "instruction": _MCQ_SYSTEM_INSTRUCTION,
            "input":       input_text,
            "output":      "Final answer: " + answer,
        })

    return records


def load_disc_law_sft(limit: int) -> List[Dict[str, Any]]:
    """加载 ShengbinYue/DISC-Law-SFT (仅 Pair 子集)，转换为 alpaca 格式。

    关键设计决策（根据实际数据诊断结果）：
      1. 只加载 DISC-Law-SFT-Pair.jsonl：避免 Triplet 文件的 schema 冲突
         （Triplet 有额外 reference 字段，混合加载会报 DatasetGenerationCastError）
      2. DISC-Law-SFT 实际没有 type 字段，改用内容检测 MCQ 格式
      3. MCQ 样本：提取答案字母，末尾追加 "Final answer: X"（对齐评测格式）
      4. 自由文本样本：保留原始输出（注入法律领域知识）
    """
    from datasets import load_dataset

    print("  正在加载 ShengbinYue/DISC-Law-SFT (仅 DISC-Law-SFT-Pair.jsonl) ...")
    ds = load_dataset(
        "ShengbinYue/DISC-Law-SFT",
        data_files={"train": "DISC-Law-SFT-Pair.jsonl"},
        split="train",
        cache_dir="/data/ocean/decoding/data/disc_law_cache",
        verification_mode="no_checks",
    )

    mcq_ok = 0    # MCQ 样本：成功提取答案并格式化
    mcq_skip = 0  # MCQ 样本：无法提取答案（跳过）
    freetext = 0  # 自由文本样本

    records: List[Dict] = []

    for item in ds:
        inp    = str(item.get("input",  "")).strip()
        output = str(item.get("output", "")).strip()

        if not inp or not output:
            continue

        if _is_mcq_input(inp):
            # MCQ：提取答案字母，末尾追加标准格式行
            letter = _extract_answer_letter(output)
            if letter is None:
                mcq_skip += 1
                # 无法确定答案 → 跳过，避免错误格式污染训练集
                continue
            mcq_ok += 1
            records.append({
                "instruction": _MCQ_SYSTEM_INSTRUCTION,
                "input":       inp,
                "output":      output.rstrip() + "\nFinal answer: " + letter,
            })
        else:
            # 自由文本：法律知识注入，不要求 Final answer 格式
            freetext += 1
            records.append({
                "instruction": _FREETEXT_SYSTEM_INSTRUCTION,
                "input":       inp,
                "output":      output,
            })

        if len(records) >= limit:
            break

    print("  [MCQ] 成功提取答案: {ok}  跳过(无法提取): {skip}".format(
        ok=mcq_ok, skip=mcq_skip
    ))
    print("  [自由文本] 法律知识样本: {n}".format(n=freetext))
    print("  合并后共: {n} 条".format(n=len(records)))

    random.seed(0)
    random.shuffle(records)
    return records[:limit]


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
        "--jecqa_eval_limit", type=int, default=200,
        help="评测集样本数（跳过这部分，从其后抽取训练集）",
    )
    parser.add_argument(
        "--jecqa_train_limit", type=int, default=1600,
        help="从 JEC-QA 非评测样本中最多取多少条作为 MCQ 训练数据",
    )
    parser.add_argument(
        "--law_freetext_limit", type=int, default=2500,
        help="从 DISC-Law-SFT 取多少条自由文本样本作为法律知识注入",
    )
    parser.add_argument(
        "--general_ratio", type=float, default=0.2,
        help="Alpaca 通用数据占总样本比例（0~1），默认 0.2",
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
    # Step 1a: JEC-QA MCQ 训练样本（核心：格式 + 领域对齐）
    # -------------------------------------------------------------------
    print("[1a/4] 加载 JEC-QA MCQ 训练样本（eval 后 {n} 条）".format(n=args.jecqa_train_limit))
    jecqa_records = load_jecqa_mcq_for_training(
        eval_limit=args.jecqa_eval_limit,
        train_limit=args.jecqa_train_limit,
    )
    print("       JEC-QA MCQ 样本数: {n}".format(n=len(jecqa_records)))

    # -------------------------------------------------------------------
    # Step 1b: DISC-Law-SFT 自由文本（法律知识注入）
    # -------------------------------------------------------------------
    print("[1b/4] 加载 DISC-Law-SFT 自由文本  limit={lim}".format(lim=args.law_freetext_limit))
    law_records = load_disc_law_sft(limit=args.law_freetext_limit)
    print("       DISC-Law-SFT 自由文本样本数: {n}".format(n=len(law_records)))

    combined_law = jecqa_records + law_records
    print("       法律数据总计: {n}（MCQ {mcq} + 自由文本 {ft}）".format(
        n=len(combined_law), mcq=len(jecqa_records), ft=len(law_records)
    ))

    # -------------------------------------------------------------------
    # Step 2: Alpaca 通用格式锚点
    # general / (combined_law + general) = general_ratio
    # -------------------------------------------------------------------
    general_limit = int(
        len(combined_law) * args.general_ratio / max(1 - args.general_ratio, 1e-9)
    )
    print("[2/4] 加载 tatsu-lab/alpaca（通用格式锚点）  limit={lim}".format(lim=general_limit))
    general_records = load_general_alpaca(limit=general_limit)
    print("      通用样本数: {n}".format(n=len(general_records)))

    total = len(combined_law) + len(general_records)
    print("      实际混合比例: MCQ={mcq:.1%}  自由文本={ft:.1%}  通用={gen:.1%}".format(
        mcq=len(jecqa_records)/total,
        ft=len(law_records)/total,
        gen=len(general_records)/total,
    ))

    # -------------------------------------------------------------------
    # Step 3: 混合 + 分割
    # -------------------------------------------------------------------
    print("[3/4] 混合打乱，分割训练/验证集（val_size={vs}）".format(vs=args.val_size))
    train_records, val_records = mix_and_split(
        law_records=combined_law,
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
