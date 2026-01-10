"""
数据加载与提示格式化模块
"""

from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a medical expert. Reason concisely (within 3 sentences) in English. "
    "Always end with a single line: 'Final answer: X' where X is A/B/C/D. "
    "Do not add any text after that line."
)


def load_medqa(split: str = "validation", limit: int = 100):
    """加载 MedQA (USMLE) 数据集

    该数据集只有 train/test 两个 split，无 validation，因此默认使用 test。
    """

    split = "test" if split not in {"train", "test"} else split
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_medmcqa(split: str = "validation", limit: int = 100):
    """加载 MedMCQA，多项选择"""

    split = "validation" if split not in {"train", "validation", "test"} else split
    dataset = load_dataset("medmcqa", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_mmlu(subject: str, split: str = "test", limit: int = 100):
    """加载 MMLU 指定子任务"""

    dataset = load_dataset("cais/mmlu", subject, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_medreason(split: str = "validation", limit: int = 100):
    """加载 MedReason 数据集（医疗推理数据集，与微调数据一致）
    
    split: "train" 或 "validation"，默认使用 "validation" 作为评估集
    """
    split_map = {"train": "train", "validation": "train", "test": "train"}  # MedReason 只有一个 split，我们用 train 作为验证集
    actual_split = split_map.get(split, "train")
    
    # 从 HuggingFace 加载 MedReason 数据集
    dataset = load_dataset("UCSC-VLAA/MedReason", split=actual_split)
    
    # 如果指定了 limit，随机采样（评估时用固定样本）
    if limit and limit < len(dataset):
        # 为了可复现性，固定种子选择前 limit 个样本
        import random
        random.seed(42)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = dataset.select(indices[:limit])
    
    return dataset


def format_prompt(tokenizer: AutoTokenizer, question: str, options) -> str:
    """将题目与选项格式化为 Llama-3 chat 模板字符串

    options 可能是 list 或 dict（如 {'A': 'xxx', 'B': 'yyy'}）。
    """

    # 规范化选项文本
    opt_lines: List[str] = []
    if isinstance(options, dict):
        for key in sorted(options.keys()):
            val = str(options[key]).strip()
            opt_lines.append(f"{key}. {val}")
    else:
        # 默认按顺序映射到 A/B/C/D...
        opt_lines = [f"{chr(65+i)}. {str(opt).strip()}" for i, opt in enumerate(list(options))]

    user_content = (
        question.strip()
        + "\n"
        + "\n".join(opt_lines)
        + "\n\nBefore the final answer, repeat the chosen option text exactly once. "
        + "Answer format: after reasoning, output the chosen option text, "
        + "then end with exactly one line in the form 'Final answer: X' "
        + "where X is one of A/B/C/D. No text is allowed after that line."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_batch_prompts(
    tokenizer: AutoTokenizer, dataset, limit: int = 50
) -> List[Tuple[str, Dict]]:
    """构建 prompts 列表，返回 (prompt, raw_example)"""

    data = dataset.select(range(min(limit, len(dataset))))
    prompts = []
    for item in data:
        # 规范化字段：确保有 question / options / answer
        question = item.get("question", "")
        options = item.get("options", {})
        
        # MedReason 数据集：options 是字符串格式 "Answer Choices: A. ... B. ... C. ... D. ..."
        if isinstance(options, str) and "Answer Choices:" in options:
            # 解析选项字符串
            import re
            opt_text = options.replace("Answer Choices:", "").strip()
            # 匹配 "A. xxx" 或 "A) xxx" 等格式
            opt_matches = re.findall(r'([A-D])[\.\)]\s*([^A-D]+?)(?=[A-D][\.\)]|$)', opt_text, re.IGNORECASE)
            if opt_matches:
                options = {match[0].upper(): match[1].strip() for match in opt_matches}
                item = dict(item)
                item["options"] = options
            else:
                # 如果解析失败，尝试简单的分割
                parts = [p.strip() for p in opt_text.split(re.compile(r'[A-D][\.\)]', re.IGNORECASE))]
                if len(parts) >= 4:
                    options = {chr(65+i): parts[i+1] if i+1 < len(parts) else "" for i in range(4)}
                    item = dict(item)
                    item["options"] = options

        # MMLU: choices + answer(int)
        if "choices" in item and "answer" in item:
            options = item["choices"]
            ans_idx = int(item["answer"])
            item = dict(item)
            item["options"] = options
            if 0 <= ans_idx < len(options):
                item["answer"] = options[ans_idx]

        # MedMCQA: answer/correct option 可能是数字字符串
        # 原始字段为 opa/opb/opc/opd + cop
        if ("cop" in item) and ("answer" not in item):
            item = dict(item)
            ans_raw = item.get("cop")
            opts = item.get("options", [])
            # 若没有 options 字段，尝试由 opa/opb/opc/opd 生成
            if not opts and all(k in item for k in ["opa", "opb", "opc", "opd"]):
                opts = [item["opa"], item["opb"], item["opc"], item["opd"]]
                item["options"] = opts
            options = item.get("options", opts)  # 更新局部变量
            # cop 可能是数字或字母
            idx = None
            if isinstance(ans_raw, str):
                ans_raw = ans_raw.strip()
                if ans_raw.isdigit():
                    idx = int(ans_raw) - 1
                elif ans_raw.lower() in {"a", "b", "c", "d"}:
                    idx = ord(ans_raw.lower()) - ord("a")
            elif isinstance(ans_raw, int):
                idx = ans_raw - 1
            if idx is not None and 0 <= idx < len(opts):
                item["answer"] = opts[idx]

        # MedReason 数据集：answer 可能包含完整解释，需要提取选项字母
        answer = item.get("answer", "")
        if isinstance(answer, str) and len(answer) > 10:
            # 如果 answer 是长文本，尝试提取选项字母（A/B/C/D）
            import re
            # 查找 "The answer is X" 或 "Final answer: X" 等模式
            ans_match = re.search(r'([A-D])[\.\)\s]*$', answer, re.IGNORECASE)
            if ans_match:
                item = dict(item)
                item["answer"] = ans_match.group(1).upper()
            # 如果没找到，尝试从开头或中间提取
            elif re.search(r'^([A-D])[\.\)]', answer, re.IGNORECASE):
                ans_match = re.match(r'^([A-D])[\.\)]', answer, re.IGNORECASE)
                if ans_match:
                    item = dict(item)
                    item["answer"] = ans_match.group(1).upper()

        # 如果缺少答案或选项，跳过该样本，避免 GT 为空
        if not options or not item.get("answer"):
            continue

        prompt = format_prompt(tokenizer, question, options)
        prompts.append((prompt, item))
    return prompts


