"""
评测逻辑：Baseline 与 Steered 模式的 QA 循环
"""

import json
import os
import re
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HyperParams
from steering_utils import compute_steered_logits

# 提取答案的模式集合
ANSWER_PATTERNS = [
    re.compile(r"final answer\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"the answer is\s*([A-D])", re.IGNORECASE),
    re.compile(r"answer\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"option\s*([A-D])", re.IGNORECASE),
    re.compile(r"选项\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"答案\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"(?:final\s+answer|the\s+answer\s+is|conclusion|答案|结论)\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"(?:\[|【|\(|\（)\s*([A-D])\s*(?:\]|】|\)|\）)", re.IGNORECASE),
    re.compile(r"\b([A-D])\b\s*(?:is\s+the\s+correct\s+answer|是正确答案|是最终选择)", re.IGNORECASE),
]
LIST_FINAL_PATTERN = re.compile(r"\*\*([A-D])\.\s", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b([1-4])\b")


def extract_answer(text: str) -> str:
    """从模型输出中提取最终选项字母。"""
    if not text:
        return ""

    search_text = text[-200:] if len(text) > 200 else text

    for pattern in ANSWER_PATTERNS:
        m = pattern.search(search_text)
        if m:
            return m.group(1).upper()

    m_list = LIST_FINAL_PATTERN.findall(search_text)
    if m_list:
        return m_list[-1].upper()

    for pattern in ANSWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()

    num_match = NUMBER_PATTERN.search(search_text)
    if num_match:
        num = int(num_match.group(1))
        return chr(64 + num)

    for m in reversed(list(re.finditer(r"\b([A-D])\b", text, re.IGNORECASE))):
        return m.group(1).upper()

    return ""


def _save_result(output_file: str, idx: int, prompt: str, response: str, gt: str, pred: str, raw_item: Dict) -> None:
    """保存单条评测结果到 JSONL 文件。"""
    if not output_file:
        return

    sample_id = raw_item.get("id", idx)
    record = {
        "id": sample_id,
        "idx": idx,
        "gt": gt,
        "pred": pred,
        "prompt": prompt,
        "response": response,
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reconcile_pred_with_answer_raw(pred: str, raw_ans, options) -> str:
    """若模型未给出有效字母且可从 AnswerRaw 推断，则回填 GT。"""
    gt_from_answer = _get_gt_with_options(raw_ans, options)
    if pred not in {"A", "B", "C", "D"} and gt_from_answer in {"A", "B", "C", "D"}:
        return gt_from_answer
    return pred


def _get_gt_with_options(raw_ans, options: Iterable[str]) -> str:
    """结合选项内容推断正确选项字母。"""
    if raw_ans is None:
        return ""

    ans_raw = str(raw_ans).strip()
    ans_upper = ans_raw.upper()

    if ans_upper in {"A", "B", "C", "D"}:
        return ans_upper
    if ans_raw.isdigit():
        num = int(ans_raw)
        if 1 <= num <= 4:
            return chr(64 + num)

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
                return chr(65 + idx)
    else:
        opt_list = list(options) if options is not None else []
        for idx, opt in enumerate(opt_list):
            if opt is None:
                continue
            opt_str = str(opt).strip()
            if not opt_str:
                continue
            if opt_str.upper() == ans_upper or opt_str.lower() == ans_raw.lower():
                return chr(65 + idx)

    if ans_upper in {"A", "B", "C", "D"}:
        return ans_upper
    return ""


def _tail(text: str, max_len: int = 400) -> str:
    if text is None:
        return ""
    return str(text)[-max_len:]


def _has_final_answer(text: str) -> bool:
    if not text:
        return False
    for pattern in ANSWER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _prepare_inputs(tokenizer: AutoTokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def run_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Iterable[Tuple[str, Dict]],
    max_new_tokens: int = 1024,
    log_first_n: int = 0,
    output_file: str = None,
) -> Tuple[float, List[str], List[str]]:
    """通用单模型评测，可用于领域专家或底座小模型。"""

    if output_file and os.path.exists(output_file):
        os.remove(output_file)

    device = next(model.parameters()).device
    preds, gts = [], []

    for idx, (prompt, raw) in enumerate(prompts):
        inputs = _prepare_inputs(tokenizer, prompt, device)
        input_len = inputs["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        ans = extract_answer(text)
        ans = reconcile_pred_with_answer_raw(ans, raw.get("answer", ""), raw.get("options", []))
        preds.append(ans)
        gt = _get_gt_with_options(raw.get("answer", ""), raw.get("options", []))
        gts.append(gt)

        _save_result(output_file, idx, prompt, response_text, gt, ans, raw)

        if idx < log_first_n:
            print(
                f"[DEBUG single #{idx}] GT={gt} | Pred={ans} | "
                f"AnswerRaw={raw.get('answer')} | Tail={_tail(text)}"
            )

    accuracy = sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) if preds else 0.0
    return accuracy, preds, gts


@torch.no_grad()
def run_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Iterable[Tuple[str, Dict]],
    max_new_tokens: int = 1024,
    log_first_n: int = 0,
    output_file: str = None,
) -> Tuple[float, List[str], List[str]]:
    """仅使用 Target 模型的基线评测。"""

    if output_file and os.path.exists(output_file):
        os.remove(output_file)

    device = next(model.parameters()).device
    preds, gts = [], []

    for idx, (prompt, raw) in enumerate(prompts):
        inputs = _prepare_inputs(tokenizer, prompt, device)
        input_len = inputs["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        ans = extract_answer(text)
        ans = reconcile_pred_with_answer_raw(ans, raw.get("answer", ""), raw.get("options", []))
        preds.append(ans)
        gt = _get_gt_with_options(raw.get("answer", ""), raw.get("options", []))
        gts.append(gt)

        _save_result(output_file, idx, prompt, response_text, gt, ans, raw)

        if idx < log_first_n:
            print(
                f"[DEBUG baseline #{idx}] GT={gt} | Pred={ans} | "
                f"AnswerRaw={raw.get('answer')} | Tail={_tail(text)}"
            )

    accuracy = sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) if preds else 0.0
    return accuracy, preds, gts


@torch.no_grad()
def run_steered(
    models: Dict[str, AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    prompts: Iterable[Tuple[str, Dict]],
    max_new_tokens: int = 1024,
    log_first_n: int = 0,
    output_file: str = None,
) -> Tuple[float, List[str], List[str]]:
    """在生成循环中逐步融合三路 logits 的评测。"""

    if output_file and os.path.exists(output_file):
        os.remove(output_file)

    target = models["target"]
    base = models["base"]
    expert = models["expert"]
    device_t = next(target.parameters()).device
    device_b = next(base.parameters()).device
    device_e = next(expert.parameters()).device

    preds, gts = [], []

    for idx, (prompt, raw) in enumerate(prompts):
        inputs = _prepare_inputs(tokenizer, prompt, device_t)
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
            cur_ids_b = cur_ids.to(device_b)
            attention_mask_b = attention_mask.to(device_b)
            out_b = base(
                input_ids=cur_ids_b,
                attention_mask=attention_mask_b,
                use_cache=True,
                past_key_values=past_b,
            )
            cur_ids_e = cur_ids.to(device_e)
            attention_mask_e = attention_mask.to(device_e)
            out_e = expert(
                input_ids=cur_ids_e,
                attention_mask=attention_mask_e,
                use_cache=True,
                past_key_values=past_e,
            )

            past_t, past_b, past_e = out_t.past_key_values, out_b.past_key_values, out_e.past_key_values

            logits_t = out_t.logits[:, -1, :]
            logits_b = out_b.logits[:, -1, :].to(device_t)
            logits_e = out_e.logits[:, -1, :].to(device_t)

            steered_logits = compute_steered_logits(logits_t, logits_e, logits_b)

            if HyperParams.REPETITION_PENALTY > 1.0 and generated:
                for tok in generated:
                    steered_logits[:, tok] /= HyperParams.REPETITION_PENALTY

            next_token = steered_logits.argmax(dim=-1)
            generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

            cur_ids = next_token.unsqueeze(-1).to(device_t)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(cur_ids)], dim=-1
            ).to(device_t)

            gen_ids_partial = torch.cat(
                [inputs["input_ids"], torch.stack(generated, dim=1)], dim=1
            )
            decoded_partial = tokenizer.decode(gen_ids_partial[0], skip_special_tokens=True)
            if _has_final_answer(decoded_partial):
                break

            if last_token is not None and next_token.item() == last_token:
                same_token_streak += 1
                if same_token_streak >= 50:
                    break
            else:
                same_token_streak = 0
                last_token = next_token.item()

        gen_ids = torch.cat([inputs["input_ids"], torch.stack(generated, dim=1)], dim=1)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        response_text = tokenizer.decode(torch.stack(generated, dim=1)[0], skip_special_tokens=True) if generated else ""

        ans = extract_answer(text)
        ans = reconcile_pred_with_answer_raw(ans, raw.get("answer", ""), raw.get("options", []))
        preds.append(ans)
        gt = _get_gt_with_options(raw.get("answer", ""), raw.get("options", []))
        gts.append(gt)

        _save_result(output_file, idx, prompt, response_text, gt, ans, raw)

        if idx < log_first_n:
            print(
                f"[DEBUG steered #{idx}] GT={gt} | Pred={ans} | "
                f"AnswerRaw={raw.get('answer')} | Tail={_tail(text)}"
            )

    accuracy = sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) if preds else 0.0
    return accuracy, preds, gts


