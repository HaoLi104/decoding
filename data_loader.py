"""
数据加载与提示格式化模块
"""

import os
from typing import Dict, List, Optional, Tuple
import re

# 国内服务器无法直连 HuggingFace，优先使用镜像站。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from transformers import AutoTokenizer

# 各数据集对应的 system prompt
SYSTEM_PROMPTS: Dict[str, str] = {
    "medqa": (
        "You are a medical expert. Reason concisely (within 3 sentences) in English. "
        "Always end with a single line: 'Final answer: X' where X is A/B/C/D. "
        "Do not add any text after that line."
    ),
    "jecqa": (
        "你是一位中国法律专家。请用中文简洁推理（不超过3句话）。"
        "最后一行必须且只能输出：'Final answer: X'，其中 X 是 A/B/C/D 之一。"
        "最后一行之后不要输出任何文字。"
    ),
}

# 向后兼容：默认 SYSTEM_PROMPT 指向 medqa
SYSTEM_PROMPT = SYSTEM_PROMPTS["medqa"]


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


def load_jecqa(split: str = "test", limit: int = 0, cache_dir: str = "/data/ocean/decoding/data/jecqa_cache"):
    """加载 AGIEval JEC-QA 数据集（KD + CA 合并），仅保留单选题。

    来源：
      hails/agieval-jec-qa-kd  (知识型, 1000 题)
      hails/agieval-jec-qa-ca  (案例型, 999 题)

    原始字段：
      query   (str):        题干（含选项，如 "问题...\\n(A) ... (B) ... "）
      choices (list[str]):  4 个选项，格式为 "(A) ..." "(B) ..." 等
      gold    (list[int]):  正确选项下标列表，单选题 gold=[k]，多选 gold=[i,j,...]

    转换后统一输出 dict，字段与 MedQA pipeline 完全一致：
      question   (str):  题干（不含选项）
      options    (dict): {"A": "...", "B": "...", "C": "...", "D": "..."}
      answer_idx (str):  正确选项字母 A/B/C/D

    过滤规则：
      - 只保留 len(gold) == 1 的单选题（与 MedQA 评测口径一致）
      - 过滤 choices 数量 != 4 的题目
    """
    import os
    from datasets import Dataset, concatenate_datasets

    _KD_ID = "hails/agieval-jec-qa-kd"
    _CA_ID = "hails/agieval-jec-qa-ca"

    # AGIEval 版本只有 test split
    # verification_mode='no_checks'：跳过 split-size 校验，避免镜像站分片下载
    # 不完整时触发 NonMatchingSplitsSizesError 导致崩溃
    kd = load_dataset(_KD_ID, split="test", cache_dir=cache_dir,
                      verification_mode="no_checks")
    ca = load_dataset(_CA_ID, split="test", cache_dir=cache_dir,
                      verification_mode="no_checks")
    raw = concatenate_datasets([kd, ca])

    _letter = ["A", "B", "C", "D"]
    _prefix_re = re.compile(r"^\([A-D]\)\s*")  # 剥离 "(A) " 等前缀

    def _strip_query_options(query: str) -> str:
        """从 query 字段中去掉末尾选项行，只保留题干部分。"""
        lines = query.strip().splitlines()
        # 选项行通常以 (A) / (B) / (C) / (D) 开头
        option_start = None
        for i, line in enumerate(lines):
            if re.match(r"^\([A-D]\)", line.strip()):
                option_start = i
                break
        if option_start is not None:
            return "\n".join(lines[:option_start]).strip()
        return query.strip()

    rows = []
    for item in raw:
        gold = item.get("gold", [])
        choices = item.get("choices", [])

        # 只保留单选且恰好 4 个选项
        if len(gold) != 1 or len(choices) != 4:
            continue

        answer_idx = _letter[gold[0]]
        options = {
            _letter[i]: _prefix_re.sub("", c).strip()
            for i, c in enumerate(choices)
        }
        question = _strip_query_options(item.get("query", ""))
        if not question:
            continue

        rows.append({
            "question":   question,
            "options":    options,
            "answer_idx": answer_idx,
        })
        if limit and len(rows) >= limit:
            break

    return Dataset.from_list(rows)


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
    - 优先使用本地路径，如果不存在则从 Hub 下载
    """
    import os
    import json
    
    # 优先尝试本地路径（检查多个可能的数据文件位置）
    local_base = "/data/ocean/decoding/MedReason"
    ds = None
    
    # 尝试1: 检查是否是 Hugging Face datasets 格式
    if os.path.exists(local_base):
        try:
            ds = load_dataset(local_base, split=split)
        except Exception:
            pass
    
    # 尝试2: 检查 eval_data 或 processed 目录中的 JSON/JSONL 文件
    if ds is None:
        for subdir in ["eval_data", "processed", "raw"]:
            subdir_path = os.path.join(local_base, subdir)
            if os.path.exists(subdir_path):
                # 查找 JSON/JSONL 文件
                for fname in os.listdir(subdir_path):
                    if fname.endswith((".json", ".jsonl")):
                        file_path = os.path.join(subdir_path, fname)
                        try:
                            # 尝试作为 JSONL 加载
                            if fname.endswith(".jsonl"):
                                rows = []
                                with open(file_path, "r", encoding="utf-8") as f:
                                    for line in f:
                                        if line.strip():
                                            rows.append(json.loads(line))
                                if rows:
                                    from datasets import Dataset
                                    ds = Dataset.from_list(rows)
                                    break
                            # 或作为 JSON 数组加载
                            elif fname.endswith(".json"):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        from datasets import Dataset
                                        ds = Dataset.from_list(data)
                                        break
                        except Exception:
                            continue
    
    # 如果本地加载失败，回退到 Hub
    if ds is None:
        try:
            ds = load_dataset("UCSC-VLAA/MedReason", split=split)
        except Exception as e:
            raise RuntimeError(f"无法加载 MedReason 数据集（本地和 Hub 都失败）: {e}")
    
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


def format_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    options,
    dataset_name: str = "medqa",
) -> str:
    """将题目与选项格式化为 Qwen Chat 模板字符串。

    Args:
        tokenizer:    用于 apply_chat_template
        question:     题干文本
        options:      list[str] 或 dict {"A": ..., "B": ...}
        dataset_name: "medqa" 或 "jecqa"，决定 system prompt 与用户引导语

    向后兼容：默认 dataset_name="medqa"，行为与旧版完全一致。
    """
    # 规范化选项文本
    opt_lines: List[str] = []
    if isinstance(options, dict):
        for key in sorted(options.keys()):
            val = str(options[key]).strip()
            opt_lines.append(f"{key}. {val}")
    else:
        opt_lines = [f"{chr(65+i)}. {str(opt).strip()}" for i, opt in enumerate(list(options))]

    sys_prompt = SYSTEM_PROMPTS.get(dataset_name, SYSTEM_PROMPT)

    if dataset_name == "jecqa":
        user_content = (
            question.strip()
            + "\n"
            + "\n".join(opt_lines)
            + "\n\n在最终答案之前，请将所选选项的文字重复一遍。"
            + "格式：推理后输出所选选项的完整文字，"
            + "然后最后一行必须且只能是 'Final answer: X'，X 为 A/B/C/D 之一。"
            + "最后一行之后不要输出任何文字。"
        )
    else:
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
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_batch_prompts(
    tokenizer: AutoTokenizer, dataset, limit: int = 50
) -> List[Tuple[str, Dict]]:
    """构建 prompts 列表，返回 (prompt, raw_example)

    允许 limit 为 None 或 <=0 表示全量。
    """

    if limit is None or (isinstance(limit, int) and limit <= 0):
        data = dataset
    else:
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


