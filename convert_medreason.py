"""
MedReason 数据集格式转换脚本

功能：将 MedReason 医疗推理数据集转换为 Llama-Factory 所需的 Alpaca 格式
用途：用于微调模型，让模型学习医疗领域的推理能力

Alpaca 格式说明：
- instruction: 问题/指令
- input: 输入（通常为空）
- output: 期望的输出（包含推理过程和最终答案）
"""

import argparse
import json
import pathlib
import re
from typing import List, Dict, Any


def load_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    从文件加载数据记录
    
    支持两种格式：
    1. JSONL 格式：每行一个 JSON 对象
    2. JSON 格式：可以是列表或包含 'data'/'examples' 键的字典
    
    参数:
        path: 输入文件路径
        
    返回:
        记录列表，每个记录是一个字典
    """
    # 处理 JSONL 格式（每行一个 JSON 对象）
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    
    # 处理 JSON 格式
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        
        # 如果直接是列表，直接返回
        if isinstance(data, list):
            return data
        
        # 如果是字典，尝试从常见键中提取数据
        if isinstance(data, dict):
            # 常见的数据结构：{"data": [...]} 或 {"examples": [...]}
            for key in ("data", "examples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        
        raise ValueError(f"Unsupported JSON structure in {path}")
    
    raise ValueError(f"Unsupported file extension: {path}")


def _format_options(options: Any) -> str:
    """
    将 options（候选项）格式化为清晰可读的多选项文本。

    兼容三种常见形态：
    - dict: {"A": "...", "B": "..."} 或 {"opa": "...", ...}（这里仅处理 A-D 键）
    - list/tuple: ["...", "...", "...", "..."]（按顺序映射到 A/B/C/D）
    - str: 例如 "Answer Choices: A. ... B. ... C. ... D. ..."

    返回：
    - 空字符串：表示无法解析或 options 为空
    - 非空字符串：例如：
        "Answer Choices:\nA. ...\nB. ...\nC. ...\nD. ..."
    """
    if options is None:
        return ""

    # 1) dict -> A/B/C/D lines
    if isinstance(options, dict):
        lines = []
        for k in ["A", "B", "C", "D"]:
            if k in options and str(options[k]).strip():
                lines.append(f"{k}. {str(options[k]).strip()}")
        return "Answer Choices:\n" + "\n".join(lines) if lines else ""

    # 2) list/tuple -> map to A/B/C/D
    if isinstance(options, (list, tuple)):
        lines = []
        for i, opt in enumerate(list(options)[:4]):
            if opt is None:
                continue
            text = str(opt).strip()
            if not text:
                continue
            lines.append(f"{chr(65 + i)}. {text}")
        return "Answer Choices:\n" + "\n".join(lines) if lines else ""

    # 3) string -> try parse "Answer Choices: A. ... B. ..."
    if isinstance(options, str):
        s = options.strip()
        if not s:
            return ""

        # 若包含前缀，去掉前缀后解析；否则也尝试直接解析
        s_wo_prefix = re.sub(r"^\s*Answer Choices\s*:\s*", "", s, flags=re.IGNORECASE)

        # 尽量稳健：提取 A/B/C/D 段落，直到下一个字母标记或文本结束
        matches = re.findall(
            r"([A-D])[\.\)]\s*(.*?)(?=(?:\s+[A-D][\.\)]\s)|$)",
            s_wo_prefix,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if matches:
            lines = []
            for k, txt in matches:
                txt = " ".join(str(txt).strip().split())
                if txt:
                    lines.append(f"{k.upper()}. {txt}")
            return "Answer Choices:\n" + "\n".join(lines) if lines else ""

        # 如果原始就是多行，直接原样保留（加一个统一标题帮助模型识别）
        if "\n" in s:
            return "Answer Choices:\n" + s

    return ""


def convert(
    records: List[Dict[str, Any]],
    question_key: str,
    options_key: str,
    answer_key: str,
    cot_key: str = None,
    prefix: str = "",
) -> List[Dict[str, str]]:
    """
    将原始记录转换为 Alpaca 格式
    
    参数:
        records: 原始数据记录列表
        question_key: 问题字段的键名（如 "question"）
        answer_key: 答案字段的键名（如 "answer"）
        cot_key: 链式推理（Chain-of-Thought）字段的键名（如 "reasoning"），可选
        prefix: 在 instruction 前添加的前缀，可选
        
    返回:
        Alpaca 格式的数据列表，每个元素包含：
        - instruction: 问题
        - input: 输入（通常为空字符串）
        - output: 输出（推理过程 + 最终答案）
    """
    out = []
    for r in records:
        # 提取问题、答案和推理链
        q = r.get(question_key, "")  # 问题文本
        opts = r.get(options_key, None)  # 候选项（可能为 str/dict/list/None）
        ans = r.get(answer_key, "")   # 最终答案
        cot = r.get(cot_key, "") if cot_key else ""  # 推理过程（Chain-of-Thought）
        
        # 构建 instruction（问题+选项），可添加前缀
        # - 题干与候选项之间用两个换行符分隔，便于模型清晰区分
        q_text = f"{prefix}{q}".strip()
        opt_text = _format_options(opts)
        instr = (q_text + ("\n\n" + opt_text if opt_text else "")).strip()
        
        # 构建 output（输出）
        if cot:
            # 如果有推理链，将推理过程和答案组合
            # 格式：推理过程 + 空行 + "Final answer: X"
            #
            # 选择英文 "Final answer" 的原因：
            # 1) 你的评测 prompt（data_loader.py 的 SYSTEM_PROMPT）要求 "Final answer: X"
            # 2) evaluator.py 的提取规则已覆盖 "final answer: X"（大小写不敏感）
            # 3) 让数据格式与评测口径一致，避免模型学到不一致的收尾格式
            output = f"{cot}\n\nFinal answer: {ans}"
        else:
            # 如果没有推理链，直接使用答案
            output = ans
        
        # 构建 Alpaca 格式的记录
        out.append(
            {
                "instruction": instr,  # 问题/指令
                "input": "",             # 输入（通常为空）
                "output": output,     # 输出（推理+答案 或 仅答案）
            }
        )
    return out


def main():
    """
    主函数：解析命令行参数并执行转换
    
    使用示例：
        python convert_medreason.py \
            --input /path/to/medreason_train.json \
            --output /path/to/medreason_alpaca_train.json \
            --question-key question \
            --answer-key answer \
            --cot-key reasoning
    """
    ap = argparse.ArgumentParser(
        description="将 MedReason 数据集转换为 Llama-Factory 的 Alpaca 格式"
    )
    
    # 必需参数
    ap.add_argument(
        "--input", 
        required=True, 
        help="原始数据文件路径（支持 .json 或 .jsonl 格式）"
    )
    ap.add_argument(
        "--output", 
        required=True, 
        help="输出 Alpaca 格式 JSON 文件路径"
    )
    
    # 可选参数：字段名映射
    ap.add_argument(
        "--question-key", 
        default="question",
        help="问题字段的键名（默认：question）"
    )
    ap.add_argument(
        "--options-key",
        default="options",
        help="候选项字段的键名（默认：options）。如无候选项字段可忽略"
    )
    ap.add_argument(
        "--answer-key", 
        default="answer",
        help="答案字段的键名（默认：answer）"
    )
    ap.add_argument(
        "--cot-key", 
        default="reasoning",
        help="链式推理字段的键名（默认：reasoning）。如果数据中没有推理链，可以设为空字符串"
    )
    ap.add_argument(
        "--prefix", 
        default="",
        help="在 instruction 前添加的前缀（可选，默认：空）"
    )
    
    args = ap.parse_args()

    # 解析输入输出路径
    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    
    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载原始数据
    print(f"正在加载数据：{in_path}")
    records = load_records(in_path)
    print(f"加载了 {len(records)} 条记录")

    # 转换为 Alpaca 格式
    print("正在转换为 Alpaca 格式...")
    alpaca = convert(
        records,
        question_key=args.question_key,
        options_key=args.options_key,
        answer_key=args.answer_key,
        cot_key=args.cot_key if args.cot_key else None,  # 如果为空字符串，转为 None
        prefix=args.prefix,
    )

    # 保存转换后的数据
    print(f"正在保存到：{out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(alpaca, f, ensure_ascii=False, indent=2)

    print(f"✓ 转换完成！共转换 {len(alpaca)} 条样本 -> {out_path}")


if __name__ == "__main__":
    main()

