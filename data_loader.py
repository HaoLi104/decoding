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

        # MMLU: choices + answer(int)
        if "choices" in item and "answer" in item:
            options = item["choices"]
            ans_idx = int(item["answer"])
            item = dict(item)
            item["options"] = options
            if 0 <= ans_idx < len(options):
                item["answer"] = options[ans_idx]

        # MedMCQA: answer/correct option 可能是数字字符串
        if "cop" in item and "options" in item and "answer" not in item:
            item = dict(item)
            ans_raw = item.get("cop")
            opts = item.get("options", [])
            try:
                ans_idx = int(ans_raw) - 1
                if 0 <= ans_idx < len(opts):
                    item["answer"] = opts[ans_idx]
            except Exception:
                pass

        prompt = format_prompt(tokenizer, question, options)
        prompts.append((prompt, item))
    return prompts


