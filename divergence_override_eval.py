#!/usr/bin/env python3
"""Evaluate divergence-override speculative decoding on MedQA.

Modes:
- baseline: target-only greedy decoding
- strict: target+draft one-step compare, mismatch always falls back to target token
- divergence_v0: on mismatch, override-accept draft token if draft_top1 != base_top1
- divergence_v1: v0 + delta_logp(draft, base; draft_top1) > tau_delta
- divergence_v2: v1 + target_opposition_score < tau_target_opp

This script is standalone and does not modify existing lossy gate code.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import format_prompt, load_medqa
from evaluator import extract_answer


VALID_MODES = {"baseline", "strict", "divergence_v0", "divergence_v1", "divergence_v2"}


@dataclass
class OverrideEvent:
    step: int
    draft_token_id: int
    target_token_id: int
    base_token_id: int
    draft_token_text: str
    target_token_text: str
    base_token_text: str
    logp_draft_on_draft: float
    logp_base_on_draft: float
    delta_logp: float
    target_opposition_score: float
    override_mode: str
    override_reason: str
    override_triggered: bool


@dataclass
class SampleResult:
    id: str
    gt: str
    pred: str
    correct: bool
    steps: int
    generated_tokens: int
    accepted_match: int
    accepted_override: int
    rejected_mismatch: int
    override_calls: int
    override_events: List[OverrideEvent]
    duration_sec: float
    response: str


def _device_of(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


def _load_model(model_path: str, device_map: str = "auto") -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
    )
    model.eval()
    return model


def _tokenize_prompt(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.inference_mode()
def _forward_last_hidden_and_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values=None,
):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1][:, -1, :]
    next_logits = out.logits[:, -1, :]
    return last_hidden, next_logits, out.past_key_values


def _argmax_id(logits: torch.Tensor) -> int:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return int(logits.argmax(dim=-1).item())


def _answer_letter(item: Dict[str, Any]) -> str:
    gt = str(item.get("answer_idx", "")).strip().upper()
    if gt in {"A", "B", "C", "D"}:
        return gt
    return ""


def _token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _logp_of(logits: torch.Tensor, token_id: int) -> float:
    logp = torch.log_softmax(logits, dim=-1)[0, int(token_id)]
    return float(logp.item())


def _compute_scores(
    logits_target: torch.Tensor,
    logits_draft: torch.Tensor,
    logits_base: torch.Tensor,
    draft_token_id: int,
    target_token_id: int,
) -> Tuple[float, float, float, float]:
    logp_draft_on_draft = _logp_of(logits_draft, draft_token_id)
    logp_base_on_draft = _logp_of(logits_base, draft_token_id)
    delta_logp = logp_draft_on_draft - logp_base_on_draft

    logp_target_on_target = _logp_of(logits_target, target_token_id)
    logp_target_on_draft = _logp_of(logits_target, draft_token_id)
    target_opp = logp_target_on_target - logp_target_on_draft

    return logp_draft_on_draft, logp_base_on_draft, delta_logp, target_opp


def _should_override(
    mode: str,
    target_reject: bool,
    draft_token_id: int,
    base_token_id: int,
    delta_logp: float,
    target_opp: float,
    tau_delta: float,
    tau_target_opp: float,
) -> Tuple[bool, str]:
    if not target_reject:
        return False, "no_target_reject"

    if draft_token_id == base_token_id:
        return False, "no_draft_base_divergence"

    if mode == "divergence_v0":
        return True, "v0_divergence"

    if mode == "divergence_v1":
        if delta_logp > tau_delta:
            return True, "v1_delta_pass"
        return False, "v1_delta_fail"

    if mode == "divergence_v2":
        if delta_logp <= tau_delta:
            return False, "v2_delta_fail"
        if target_opp >= tau_target_opp:
            return False, "v2_target_opp_fail"
        return True, "v2_pass"

    return False, "mode_no_override"


@torch.inference_mode()
def run_case(
    item: Dict[str, Any],
    target: AutoModelForCausalLM,
    draft: Optional[AutoModelForCausalLM],
    small_base: Optional[AutoModelForCausalLM],
    tokenizer,
    mode: str,
    max_new_tokens: int,
    tau_delta: float,
    tau_target_opp: float,
    stop_on_eos: bool,
) -> SampleResult:
    question = item.get("question", "")
    options = item.get("options", {})
    gt = _answer_letter(item)
    case_id = str(item.get("id", ""))

    prompt = format_prompt(tokenizer, question, options)
    encoded = _tokenize_prompt(tokenizer, prompt)
    prompt_ids_cpu = encoded["input_ids"]
    prompt_mask_cpu = encoded["attention_mask"]

    target_dev = _device_of(target)
    draft_dev = _device_of(draft) if draft is not None else target_dev
    base_dev = _device_of(small_base) if small_base is not None else target_dev

    past_t = None
    past_d = None
    past_b = None
    cur_ids_cpu = prompt_ids_cpu
    cur_mask_cpu = prompt_mask_cpu
    generated: List[int] = []

    accepted_match = 0
    accepted_override = 0
    rejected_mismatch = 0
    override_calls = 0
    override_events: List[OverrideEvent] = []
    steps = 0

    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        steps += 1

        _, logits_t, past_t = _forward_last_hidden_and_logits(
            model=target,
            input_ids=cur_ids_cpu.to(target_dev),
            attention_mask=cur_mask_cpu.to(target_dev),
            past_key_values=past_t,
        )
        next_t = _argmax_id(logits_t)

        if mode == "baseline":
            accepted = next_t
        else:
            if draft is None:
                raise ValueError("draft model is required for non-baseline modes")

            _, logits_d, past_d = _forward_last_hidden_and_logits(
                model=draft,
                input_ids=cur_ids_cpu.to(draft_dev),
                attention_mask=cur_mask_cpu.to(draft_dev),
                past_key_values=past_d,
            )
            next_d = _argmax_id(logits_d)

            if next_d == next_t:
                accepted = next_t
                accepted_match += 1
            else:
                if mode == "strict":
                    accepted = next_t
                    rejected_mismatch += 1
                else:
                    if small_base is None:
                        raise ValueError("small_base model is required for divergence modes")

                    _, logits_b, past_b = _forward_last_hidden_and_logits(
                        model=small_base,
                        input_ids=cur_ids_cpu.to(base_dev),
                        attention_mask=cur_mask_cpu.to(base_dev),
                        past_key_values=past_b,
                    )
                    next_b = _argmax_id(logits_b)
                    override_calls += 1

                    (
                        logp_draft_on_draft,
                        logp_base_on_draft,
                        delta_logp,
                        target_opp,
                    ) = _compute_scores(
                        logits_target=logits_t,
                        logits_draft=logits_d,
                        logits_base=logits_b,
                        draft_token_id=next_d,
                        target_token_id=next_t,
                    )

                    do_override, reason = _should_override(
                        mode=mode,
                        target_reject=True,
                        draft_token_id=next_d,
                        base_token_id=next_b,
                        delta_logp=delta_logp,
                        target_opp=target_opp,
                        tau_delta=tau_delta,
                        tau_target_opp=tau_target_opp,
                    )

                    if do_override:
                        accepted = next_d
                        accepted_override += 1
                    else:
                        accepted = next_t
                        rejected_mismatch += 1

                    override_events.append(
                        OverrideEvent(
                            step=int(steps),
                            draft_token_id=int(next_d),
                            target_token_id=int(next_t),
                            base_token_id=int(next_b),
                            draft_token_text=_token_text(tokenizer, next_d),
                            target_token_text=_token_text(tokenizer, next_t),
                            base_token_text=_token_text(tokenizer, next_b),
                            logp_draft_on_draft=float(logp_draft_on_draft),
                            logp_base_on_draft=float(logp_base_on_draft),
                            delta_logp=float(delta_logp),
                            target_opposition_score=float(target_opp),
                            override_mode=mode,
                            override_reason=reason,
                            override_triggered=bool(do_override),
                        )
                    )

        generated.append(int(accepted))
        if stop_on_eos and tokenizer.eos_token_id is not None and accepted == tokenizer.eos_token_id:
            break

        cur_ids_cpu = torch.tensor([[accepted]], dtype=torch.long)
        cur_mask_cpu = torch.cat([cur_mask_cpu, torch.ones_like(cur_ids_cpu)], dim=1)

    t1 = time.perf_counter()

    full_ids = torch.cat(
        [
            prompt_ids_cpu,
            torch.tensor([generated], dtype=torch.long) if generated else torch.empty((1, 0), dtype=torch.long),
        ],
        dim=1,
    )
    decoded = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    response = tokenizer.decode(full_ids[0][prompt_ids_cpu.shape[1] :], skip_special_tokens=True)
    pred = extract_answer(decoded)
    pred = pred if pred in {"A", "B", "C", "D"} else ""
    correct = bool(pred) and (pred == gt)

    return SampleResult(
        id=case_id,
        gt=gt,
        pred=pred,
        correct=bool(correct),
        steps=int(steps),
        generated_tokens=int(len(generated)),
        accepted_match=int(accepted_match),
        accepted_override=int(accepted_override),
        rejected_mismatch=int(rejected_mismatch),
        override_calls=int(override_calls),
        override_events=override_events,
        duration_sec=float(t1 - t0),
        response=response,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Divergence-override speculative decoding evaluator")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), required=True)
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--draft_model", default=None)
    parser.add_argument("--small_base_model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device_map", default="auto")

    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--stop_on_eos", action="store_true")

    parser.add_argument("--tau_delta", type=float, default=0.0)
    parser.add_argument("--tau_target_opp", type=float, default=1.0)

    parser.add_argument("--save_event_level", action="store_true", help="Store per-step override events")
    parser.add_argument("--out", required=True, help="Output json file path")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode != "baseline" and not args.draft_model:
        raise ValueError("--draft_model is required for non-baseline modes")
    if args.mode.startswith("divergence") and not args.small_base_model:
        raise ValueError("--small_base_model is required for divergence modes")

    tok_path = args.tokenizer or args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = _load_model(args.target_model, device_map=args.device_map)
    draft = _load_model(args.draft_model, device_map=args.device_map) if args.draft_model else None
    small_base = _load_model(args.small_base_model, device_map=args.device_map) if args.small_base_model else None

    ds = load_medqa(split=args.split, limit=args.limit)
    cases: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        q = item.get("question", "")
        opts = item.get("options", {})
        gt = _answer_letter(item)
        if not q or not opts or gt not in {"A", "B", "C", "D"}:
            continue
        cases.append(
            {
                "id": str(item.get("id", idx)),
                "question": q,
                "options": opts,
                "answer_idx": gt,
            }
        )

    results: List[SampleResult] = []
    t_start = time.perf_counter()
    for i, item in enumerate(cases):
        res = run_case(
            item=item,
            target=target,
            draft=draft,
            small_base=small_base,
            tokenizer=tokenizer,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            tau_delta=args.tau_delta,
            tau_target_opp=args.tau_target_opp,
            stop_on_eos=args.stop_on_eos,
        )
        results.append(res)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(cases)}] processed")
    t_end = time.perf_counter()

    n = len(results)
    correct = sum(1 for r in results if r.correct)
    total_tokens = sum(r.generated_tokens for r in results)
    total_steps = sum(r.steps for r in results)
    total_override_calls = sum(r.override_calls for r in results)
    total_override = sum(r.accepted_override for r in results)
    total_match = sum(r.accepted_match for r in results)
    total_reject = sum(r.rejected_mismatch for r in results)
    total_gen_time = sum(r.duration_sec for r in results)

    all_events = [e for r in results for e in r.override_events]
    reason_counter = Counter(e.override_reason for e in all_events)

    summary = {
        "mode": args.mode,
        "split": args.split,
        "n_cases": n,
        "accuracy": (correct / n) if n else 0.0,
        "correct": int(correct),
        "total_tokens": int(total_tokens),
        "total_steps": int(total_steps),
        "accepted_match": int(total_match),
        "accepted_override": int(total_override),
        "rejected_mismatch": int(total_reject),
        "override_calls": int(total_override_calls),
        "override_rate": (total_override / total_override_calls) if total_override_calls else 0.0,
        "accepted_tokens_per_step": ((total_match + total_override) / total_steps) if total_steps else 0.0,
        "tokens_per_sec_gen_only": (total_tokens / total_gen_time) if total_gen_time > 0 else 0.0,
        "tokens_per_sec_end_to_end": (total_tokens / (t_end - t_start)) if (t_end - t_start) > 0 else 0.0,
        "avg_latency_per_case_sec": ((t_end - t_start) / n) if n else 0.0,
        "tau_delta": float(args.tau_delta),
        "tau_target_opp": float(args.tau_target_opp),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "small_base_model": args.small_base_model,
        "override_reason_counts": dict(reason_counter),
    }

    if args.save_event_level:
        sample_payload = [asdict(r) for r in results]
    else:
        sample_payload = []
        for r in results:
            row = asdict(r)
            row["override_events"] = []
            sample_payload.append(row)

    payload = {
        "summary": summary,
        "samples": sample_payload,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
