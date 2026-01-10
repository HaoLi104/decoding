"""
数据加载与提示格式化模块
"""

from typing import Dict, List, Tuple
import re

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


def _parse_medreason_options(options_raw) -> List[str]:
    """
    将 MedReason 的 options 字段解析为按 A/B/C/D 排列的列表。
    options_raw 常见形态：
      - 字符串，形如：
        "Answer Choices:\\nA. ...\\nB. ...\\nC. ...\\nD. ..."
      - 字典或列表（较少见）
    返回长度 4（或 <=4）的列表；若解析失败返回空列表。
    """
    if options_raw is None:
        return []

    # list/tuple
    if isinstance(options_raw, (list, tuple)):
        return [str(x).strip() for x in list(options_raw)]

    # dict，优先按 A-D 顺序
    if isinstance(options_raw, dict):
        vals = []
        for k in ["A", "B", "C", "D"]:
            if k in options_raw:
                vals.append(str(options_raw[k]).strip())
        if vals:
            return vals
        return [str(v).strip() for _, v in sorted(options_raw.items())]

    # str
    if isinstance(options_raw, str):
        s = options_raw.strip()
        if not s:
            return []
        s = re.sub(r"^Answer Choices\\s*:\\s*", "", s, flags=re.IGNORECASE)
        matches = re.findall(
            r"([A-D])[\\.\\)]\\s*(.*?)(?=(?:\\n[A-D][\\.\\)]|\\s+[A-D][\\.\\)]|$))",
            s,
            flags=re.IGNORECASE | re.DOTALL,
        )
        opts = []
        for _, txt in matches:
            txt = " ".join(str(txt).strip().split())
            opts.append(txt)
        if opts:
            return opts
        # 兜底：按行拆
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            return lines
    return []


def _parse_medreason_answer(ans_raw: str, options: List[str]) -> str:
    """
    从 answer 文本中确定正确选项（返回选项文本），策略：
      1) 若答案文本包含某个选项文本（忽略大小写），返回该选项
      2) 若答案文本含有 A/B/C/D 字母，按字母索引返回对应选项
    解析失败返回空字符串。
    """
    if not ans_raw or not options:
        return ""
    ans_low = str(ans_raw).lower()
    # 1) 包含匹配
    for opt in options:
        if opt and opt.lower() in ans_low:
            return opt
    # 2) 字母匹配
    m = re.search(r"\\b([A-D])\\b", ans_raw, flags=re.IGNORECASE)
    if m:
        idx = ord(m.group(1).upper()) - 65
        if 0 <= idx < len(options):
            return options[idx]
    return ""

#load_medreason_mc是用于多选评测（A/B/C/D 准确率）。
def load_medreason_mc(split: str = "train", limit: int = 200):
    """
    加载 MedReason 用于多选评测（A/B/C/D 准确率）。
    - options 解析为列表（A/B/C/D 顺序）
    - answer 解析为选项文本（不含解释）
    """
    ds = load_dataset("UCSC-VLAA/MedReason", split=split)
    rows = []
    for item in ds:
        q = item.get("question", "")
        opts = _parse_medreason_options(item.get("options"))
        ans = _parse_medreason_answer(item.get("answer", ""), opts)
        if not q or not opts or not ans:
            continue
        rows.append({"question": q, "options": opts, "answer": ans})
        if limit and len(rows) >= limit:
            break
    from datasets import Dataset

    return Dataset.from_list(rows)


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

        # 如果缺少答案或选项，跳过该样本，避免 GT 为空
        if not options or not item.get("answer"):
            continue

        prompt = format_prompt(tokenizer, question, options)
        prompts.append((prompt, item))
    return prompts


