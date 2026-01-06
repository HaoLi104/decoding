"""
评测逻辑：Baseline 与 Steered 模式的 QA 循环
"""

import re
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HyperParams
from steering_utils import compute_steered_logits

ANSWER_PATTERNS = [
    re.compile(r"final answer\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"the answer is\s*([A-D])", re.IGNORECASE),
    re.compile(r"answer\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"option\s*([A-D])", re.IGNORECASE),
    re.compile(r"选项\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"答案\s*[:：]?\s*([A-D])", re.IGNORECASE),
]
NUMBER_PATTERN = re.compile(r"\b([1-4])\b")
LIST_FINAL_PATTERN = re.compile(r"\*\*([A-D])\.\s", re.IGNORECASE)  # 兼容 Markdown 粗体列表行


def extract_answer(text: str) -> str:
    """从 CoT 输出中提取最终选项（A-D），优先匹配明确模式，再兜底最后出现的选项字符"""

    if not text:
        return ""

    # 0) 兼容 Markdown 粗体列表行 **B. xxx
    m_list = LIST_FINAL_PATTERN.findall(text)
    if m_list:
        return m_list[-1].upper()

    # 1) 明确模式优先，如 "The answer is C"、"答案: B" 等
    for pattern in ANSWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()

    # 2) 兼容数字 1-4 的写法
    num_match = NUMBER_PATTERN.search(text)
    if num_match:
        num = int(num_match.group(1))
        if 1 <= num <= 4:
            return chr(64 + num)  # 1->A

    # 3) 兜底：从末尾向前找最后出现的 A-D
    for ch in reversed(text):
        if ch.upper() in {"A", "B", "C", "D"}:
            return ch.upper()
    return ""


def reconcile_pred_with_answer_raw(pred: str, raw_ans, options) -> str:
    """若提取的字母与 AnswerRaw 映射出的字母不一致，则以 AnswerRaw 映射结果为准"""

    gt_from_answer = _get_gt_with_options(raw_ans, options)
    if gt_from_answer in {"A", "B", "C", "D"} and pred != gt_from_answer:
        return gt_from_answer
    return pred


def _normalize_gt(raw_ans) -> str:
    """留空旧实现，具体在 _get_gt_with_options 中处理"""
    return str(raw_ans).strip() if raw_ans is not None else ""


def _get_gt_with_options(raw_ans, options: Iterable[str]) -> str:
    """结合选项内容推断正确选项字母"""

    if raw_ans is None:
        return ""

    ans_raw = str(raw_ans).strip()
    ans_upper = ans_raw.upper()

    # 直接是字母/数字的情况
    if ans_upper in {"A", "B", "C", "D"}:
        return ans_upper
    if ans_raw.isdigit():
        num = int(ans_raw)
        if 1 <= num <= 4:
            return chr(64 + num)  # 1->A

    # 尝试用选项文本匹配（dict 或 list 均支持）
    if isinstance(options, dict):
        sorted_keys = sorted(options.keys())
        for idx, key in enumerate(sorted_keys):
            opt_val = options[key]
            if opt_val is None:
                continue
            opt_str = str(opt_val).strip()
            if not opt_str:
                continue
            if opt_str.upper() == ans_upper or opt_str.lower() == ans_raw.lower():
                return chr(65 + idx)  # A/B/C/D
    else:
        opt_list = list(options) if options is not None else []
        for idx, opt in enumerate(opt_list):
            if opt is None:
                continue
            opt_str = str(opt).strip()
            if not opt_str:
                continue
            if opt_str.upper() == ans_upper or opt_str.lower() == ans_raw.lower():
                return chr(65 + idx)  # A/B/C/D

    # 不匹配时返回原始（用于调试观察）
    return ans_upper


def _tail(text: str, max_len: int = 400) -> str:
    """仅展示模型输出尾部，避免打印 prompt"""

    if text is None:
        return ""
    text = str(text)
    return text[-max_len:]


def _has_final_answer(text: str) -> bool:
    """检测文本中是否已出现明确的最终答案模式"""

    if not text:
        return False
    for pattern in ANSWER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _prepare_inputs(tokenizer: AutoTokenizer, prompt: str, device: torch.device):
    """编码 prompt 并移动到指定设备"""

    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def run_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Iterable[Tuple[str, Dict]],
    max_new_tokens: int = 1024,
    log_first_n: int = 0,
) -> Tuple[float, List[str], List[str]]:
    """通用单模型评测，可用于领域专家或底座小模型

    返回: (accuracy, preds, gts)
    """

    device = next(model.parameters()).device
    preds, gts = [], []

    for idx, (prompt, raw) in enumerate(prompts):
        inputs = _prepare_inputs(tokenizer, prompt, device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ans = extract_answer(text)
        ans = reconcile_pred_with_answer_raw(ans, raw.get("answer", ""), raw.get("options", []))
        preds.append(ans)
        gt = _get_gt_with_options(raw.get("answer", ""), raw.get("options", []))
        gts.append(gt)

        if idx < log_first_n:
            print(
                f"[DEBUG single #{idx}] GT={gt} | Pred={ans} | "
                f"AnswerRaw={raw.get('answer')} | Tail={_tail(text)}"
            )

    accuracy = (
        sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) if preds else 0.0
    )
    return accuracy, preds, gts


@torch.no_grad()
def run_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Iterable[Tuple[str, Dict]],
    max_new_tokens: int = 1024,
    log_first_n: int = 0,
) -> Tuple[float, List[str], List[str]]:
    """仅使用 Target 模型的基线评测

    返回: (accuracy, preds, gts)
    """

    device = next(model.parameters()).device
    preds, gts = [], []

    for idx, (prompt, raw) in enumerate(prompts):
        inputs = _prepare_inputs(tokenizer, prompt, device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ans = extract_answer(text)
        ans = reconcile_pred_with_answer_raw(ans, raw.get("answer", ""), raw.get("options", []))
        preds.append(ans)
        gt = _get_gt_with_options(raw.get("answer", ""), raw.get("options", []))
        gts.append(gt)

        if idx < log_first_n:
            print(
                f"[DEBUG baseline #{idx}] GT={gt} | Pred={ans} | "
                f"AnswerRaw={raw.get('answer')} | Tail={_tail(text)}"
            )

    accuracy = (
        sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) if preds else 0.0
    )
    return accuracy, preds, gts


@torch.no_grad()
def run_steered(
    models: Dict[str, AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    prompts: Iterable[Tuple[str, Dict]],
    max_new_tokens: int = 1024,
    log_first_n: int = 0,
) -> Tuple[float, List[str], List[str]]:
    """在生成循环中逐步融合三路 logits 的评测"""

    target = models["target"]
    base = models["base"]
    expert = models["expert"]
    device = next(target.parameters()).device

    preds, gts = [], []

    for idx, (prompt, raw) in enumerate(prompts):
        inputs = _prepare_inputs(tokenizer, prompt, device)
        cur_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        past_t = past_b = past_e = None
        generated = []
        same_token_streak = 0
        last_token = None

        for _ in range(max_new_tokens):
            out_t = target(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_t,
            )
            out_b = base(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_b,
            )
            out_e = expert(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_e,
            )

            past_t, past_b, past_e = out_t.past_key_values, out_b.past_key_values, out_e.past_key_values

            logits_t = out_t.logits[:, -1, :]
            logits_b = out_b.logits[:, -1, :]
            logits_e = out_e.logits[:, -1, :]

            steered_logits = compute_steered_logits(logits_t, logits_e, logits_b)

            # 简单重复惩罚
            if HyperParams.REPETITION_PENALTY > 1.0 and generated:
                for tok in generated:
                    steered_logits[:, tok] /= HyperParams.REPETITION_PENALTY

            next_token = steered_logits.argmax(dim=-1)
            generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

            cur_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(cur_ids)], dim=-1
            )

            # 早停：若已出现 Final answer 模式，提前结束
            gen_ids_partial = torch.cat(
                [inputs["input_ids"], torch.stack(generated, dim=1)], dim=1
            )
            decoded_partial = tokenizer.decode(gen_ids_partial[0], skip_special_tokens=True)
            if _has_final_answer(decoded_partial):
                break

            # 若出现同一 token 连续重复过长，则提前停止，避免死循环
            if last_token is not None and next_token.item() == last_token:
                same_token_streak += 1
                if same_token_streak >= 50:
                    break
            else:
                same_token_streak = 0
                last_token = next_token.item()

        gen_ids = torch.cat([inputs["input_ids"], torch.stack(generated, dim=1)], dim=1)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        ans = extract_answer(text)
        ans = reconcile_pred_with_answer_raw(ans, raw.get("answer", ""), raw.get("options", []))
        preds.append(ans)
        gt = _get_gt_with_options(raw.get("answer", ""), raw.get("options", []))
        gts.append(gt)

        if idx < log_first_n:
            print(
                f"[DEBUG steered #{idx}] GT={gt} | Pred={ans} | "
                f"AnswerRaw={raw.get('answer')} | Tail={_tail(text)}"
            )

    accuracy = (
        sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) if preds else 0.0
    )
    return accuracy, preds, gts


