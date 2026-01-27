import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def format_prompt(item: Dict[str, Any]) -> str:
    q = item.get("question", "")
    opts = item.get("options", {})
    gt = item.get("ground_truth", "")
    big_resp = item.get("big_model_response", "")
    big_pred = item.get("big_model_pred", "")
    opts_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    prompt = (
        "You are a careful grader. Given a medical MCQ, the correct answer, and a student's response, "
        "locate the first character position (1-based index) in the student's response where it becomes incorrect. "
        "Rules: if it is incorrect from the very first character, output 1; if unsure, output -1. "
        "Only return strict JSON, no prose.\n\n"
        f"Question:\n{q}\n\nOptions:\n{opts_text}\n\nCorrect answer: {gt}\n\n"
        f"Student final choice (extracted): {big_pred}\n"
        f"Student response (full):\n{big_resp}\n\n"
        "Respond ONLY this JSON object and nothing else: {\"first_error_pos\": <int>, \"explanation\": <short text>}"
    )
    return prompt


def extract_first_int(text: str) -> int:
    # Try to extract JSON object first
    json_match = re.search(r"\{[^{}]*first_error_pos[^{}]*\}", text, flags=re.IGNORECASE | re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if "first_error_pos" in obj:
                return int(obj["first_error_pos"])
        except Exception:
            pass
    m = re.search(r"first_error_pos\"?\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return -1


def judge_case(item: Dict[str, Any], tokenizer, model, max_new_tokens: int = 128) -> Tuple[int, str]:
    prompt = format_prompt(item)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Keep only generated part
    gen_text = text[len(prompt):].strip()
    pos = extract_first_int(gen_text)
    return pos, gen_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate first error position using a strong reviewer model.")
    parser.add_argument("--cases", required=True, help="Path to filtered cases JSON (list of dicts)")
    parser.add_argument(
        "--model",
        default="/data/ocean/decoding/model/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
        help="Reviewer model path",
    )
    parser.add_argument("--output", default="logs/medqa_big_wrong_small_right_annotated.json", help="Output JSON path")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    tokenizer, model = load_model(args.model)

    annotated = []
    for idx, item in enumerate(cases):
        try:
            pos, review_text = judge_case(item, tokenizer, model)
        except Exception as e:
            pos, review_text = -1, f"error: {e}"
        enriched = dict(item)
        enriched["review_first_error_pos"] = pos
        enriched["review_comment"] = review_text
        annotated.append(enriched)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(cases)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(annotated, ensure_ascii=False, indent=2))
    print(f"Saved {len(annotated)} annotated cases to {out_path}")


if __name__ == "__main__":
    main()
