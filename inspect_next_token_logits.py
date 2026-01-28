import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import format_prompt


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


def truncate_prefix(response: str, error_pos: int) -> str:
    if not response:
        return ""
    if error_pos is None or error_pos <= 1:
        return ""
    idx = min(max(error_pos - 1, 0), len(response))
    return response[:idx]


def build_input_text(tokenizer, item: Dict[str, Any], prefix: str) -> str:
    question = item.get("question", "")
    options = item.get("options", {})
    prompt = format_prompt(tokenizer, question, options)
    return prompt + prefix


def topk_next_token(model, tokenizer, text: str, top_k: int) -> List[Dict[str, Any]]:
    if not text:
        return []
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
    logits = out.logits[:, -1, :]
    top = torch.topk(logits, k=top_k)
    probs = torch.softmax(top.values, dim=-1)
    results: List[Dict[str, Any]] = []
    for idx, logit, prob in zip(top.indices[0].tolist(), top.values[0].tolist(), probs[0].tolist()):
        results.append(
            {
                "token_id": int(idx),
                "token_piece": tokenizer.convert_ids_to_tokens(idx),
                "token_text": tokenizer.decode([idx], skip_special_tokens=True),
                "logit": float(logit),
                "prob": float(prob),
            }
        )
    return results


def process_cases(
    cases: List[Dict[str, Any]],
    small_model_path: str,
    big_model_path: str,
    top_k: int,
    limit: int = None,
) -> List[Dict[str, Any]]:
    small_tok, small_model = load_model(small_model_path)
    big_tok, big_model = load_model(big_model_path)

    results = []
    total = len(cases) if limit is None else min(limit, len(cases))
    for idx, item in enumerate(cases[:total]):
        resp = str(item.get("big_model_response", ""))
        err_raw = item.get("review_first_error_pos")
        try:
            err_pos = int(err_raw) if err_raw is not None else None
        except Exception:
            err_pos = None
        prefix = truncate_prefix(resp, err_pos)
        inp_small = build_input_text(small_tok, item, prefix)
        inp_big = build_input_text(big_tok, item, prefix)

        top_small = topk_next_token(small_model, small_tok, inp_small, top_k)
        top_big = topk_next_token(big_model, big_tok, inp_big, top_k)

        results.append(
            {
                "id": item.get("id", idx),
                "review_first_error_pos": err_pos,
                "prefix_text": prefix,
                "prefix_char_len": len(prefix),
                "big_model_pred": item.get("big_model_pred"),
                "ground_truth": item.get("ground_truth"),
                "next_token_topk": {
                    "small": top_small,
                    "big": top_big,
                },
            }
        )
        if (idx + 1) % 5 == 0:
            print(f"[{idx + 1}/{total}] done")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect next-token logits before the first error")
    parser.add_argument("--annotated", required=True, help="Annotated JSON with review_first_error_pos")
    parser.add_argument("--small_model", default="/data/ocean/decoding/model/II-Medical-8B", help="Path to small model")
    parser.add_argument("--big_model", default="/data/ocean/decoding/model/Qwen/Qwen3-14B", help="Path to big model")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k logits to keep")
    parser.add_argument("--limit", type=int, default=None, help="Process first N cases (debug)")
    parser.add_argument("--output", default="logs/next_token_logits.json", help="Output path")
    args = parser.parse_args()

    cases = json.loads(Path(args.annotated).read_text())
    results = process_cases(cases, args.small_model, args.big_model, args.top_k, args.limit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Completed {len(results)} cases, saved to {out_path}")


if __name__ == "__main__":
    main()
