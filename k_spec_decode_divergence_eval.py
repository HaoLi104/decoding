#!/usr/bin/env python3
"""K-token speculative verify evaluator with divergence override.

This script implements a practical K-token speculative decoding loop:
- draft proposes K tokens
- target verifies proposed tokens and finds first reject position
- on reject, optional divergence override with small_base (v0/v1/v2)

Modes:
- baseline: target-only greedy decoding
- standard_speculative: K-token verify, reject falls back to target token
- divergence_v0: standard_speculative + reject override if draft != small_base
- divergence_v1: v0 + delta_logp > tau_delta
- divergence_v2: v1 + target_opp < tau_target_opp
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import format_prompt, load_medqa
from evaluator import extract_answer
from k_spec_kernels import (
    ModelState,
    advance_state_with_tokens,
    argmax_id,
    find_first_reject_pos,
    get_target_verify_logits,
    init_state_from_prompt,
    logp_of,
    propose_k_tokens_with_temp_state,
    should_override,
    step_state_with_token,
)


VALID_MODES = {
    "baseline",
    "standard_speculative",
    "strict",  # backward compatibility alias
    "divergence_v0",
    "divergence_v1",
    "divergence_v2",
}


@dataclass
class RoundEvent:
    # 每一轮（一次 K-token proposal + verify）的关键调试信息
    round_idx: int
    proposed_len: int
    first_reject_pos: int
    verified_prefix_len: int
    reject_recheck_called: bool
    reject_recheck_override: bool
    accepted_tokens_this_round: int


@dataclass
class SampleResult:
    # 单题结果：既包含准确性信息，也包含吞吐/行为统计。
    id: str
    gt: str
    pred: str
    correct: bool
    rounds: int
    generated_tokens: int
    speculative_tokens: int
    proposed_tokens_total: int
    accepted_match: int
    accepted_override: int
    rejected_mismatch: int
    override_calls: int
    small_base_calls: int
    v2_precheck_skips: int
    duration_sec: float
    response: str
    round_events: List[RoundEvent]


@dataclass
class BaseLazyState:
    # small_base 的懒同步状态：
    # 仅在“出现拒绝点且需要复判”时，才把它追到当前生成位置。
    state: ModelState
    synced_generated_len: int


def _device_of(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


def _resolve_dtype(dtype_name: str):
    name = str(dtype_name).strip().lower()
    if name == "auto":
        return "auto"
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}. Use one of: auto, fp16, bf16, fp32")


def _load_model(model_path: str, device_map: str = "auto", dtype_name: str = "auto") -> AutoModelForCausalLM:
    dtype = _resolve_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
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


def _answer_letter(item: Dict[str, Any]) -> str:
    gt = str(item.get("answer_idx", "")).strip().upper()
    if gt in {"A", "B", "C", "D"}:
        return gt
    return ""


def _sync_base_to_generated_len(
    base_model: AutoModelForCausalLM,
    base_lazy: BaseLazyState,
    accepted_tokens_all: List[int],
    target_generated_len: int,
    base_device: torch.device,
) -> BaseLazyState:
    # 将 small_base 从“上次同步位置”补齐到目标生成长度。
    # 这样 small_base 不用每轮都跟着走，降低额外开销。
    if target_generated_len < base_lazy.synced_generated_len:
        raise ValueError("target_generated_len is smaller than base synced length")

    pending = accepted_tokens_all[base_lazy.synced_generated_len : target_generated_len]
    cur_state = advance_state_with_tokens(
        base_model,
        base_lazy.state,
        token_ids=pending,
        device=base_device,
    )

    return BaseLazyState(state=cur_state, synced_generated_len=target_generated_len)


@torch.inference_mode()
def run_case(
    item: Dict[str, Any],
    target: AutoModelForCausalLM,
    draft: Optional[AutoModelForCausalLM],
    small_base: Optional[AutoModelForCausalLM],
    tokenizer,
    mode: str,
    speculative_tokens: int,
    max_new_tokens: int,
    tau_delta: float,
    tau_target_opp: float,
    stop_on_eos: bool,
    save_round_level: bool,
    enable_v2_target_opp_precheck: bool,
) -> SampleResult:
    # 核心解码循环：
    # 1) draft 提 K 个 token
    # 2) target 一次 verify K 个位置
    # 3) 首拒绝点触发 strict 或 divergence 复判
    # 4) 提交本轮接受的 token，推进 target/draft 状态
    question = item.get("question", "")
    options = item.get("options", {})
    gt = _answer_letter(item)
    case_id = str(item.get("id", ""))

    prompt = format_prompt(tokenizer, question, options)
    encoded = _tokenize_prompt(tokenizer, prompt)
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded["attention_mask"]

    target_dev = _device_of(target)
    draft_dev = _device_of(draft) if draft is not None else target_dev
    base_dev = _device_of(small_base) if small_base is not None else target_dev

    target_state = init_state_from_prompt(target, prompt_ids, prompt_mask, target_dev)
    draft_state = init_state_from_prompt(draft, prompt_ids, prompt_mask, draft_dev) if draft is not None else None

    base_lazy: Optional[BaseLazyState] = None
    if small_base is not None:
        base_state = init_state_from_prompt(small_base, prompt_ids, prompt_mask, base_dev)
        base_lazy = BaseLazyState(state=base_state, synced_generated_len=0)

    generated: List[int] = []

    accepted_match = 0
    accepted_override = 0
    rejected_mismatch = 0
    override_calls = 0
    small_base_calls = 0
    v2_precheck_skips = 0
    proposed_tokens_total = 0
    rounds = 0
    round_events: List[RoundEvent] = []

    t0 = time.perf_counter()

    normalized_mode = "standard_speculative" if mode == "strict" else mode

    while len(generated) < max_new_tokens:
        rounds += 1
        k_now = min(speculative_tokens, max_new_tokens - len(generated))

        if normalized_mode == "baseline":
            # baseline 不走 speculative 流程，直接 target greedy。
            accepted_token = argmax_id(target_state.next_logits)
            target_state = step_state_with_token(target, target_state, accepted_token, target_dev)
            generated.append(int(accepted_token))

            if save_round_level:
                round_events.append(
                    RoundEvent(
                        round_idx=int(rounds),
                        proposed_len=1,
                        first_reject_pos=-1,
                        verified_prefix_len=1,
                        reject_recheck_called=False,
                        reject_recheck_override=False,
                        accepted_tokens_this_round=1,
                    )
                )

            if stop_on_eos and tokenizer.eos_token_id is not None and accepted_token == tokenizer.eos_token_id:
                break
            continue

        if draft_state is None:
            raise ValueError("draft_state is required for non-baseline modes")

        proposed_tokens, draft_logits_per_pos = propose_k_tokens_with_temp_state(
            model=draft,
            state=draft_state,
            k=k_now,
            device=draft_dev,
        )
        proposed_tokens_total += len(proposed_tokens)

        target_logits_per_pos = get_target_verify_logits(
            model=target,
            state=target_state,
            proposed_tokens=proposed_tokens,
            device=target_dev,
        )

        reject_pos = find_first_reject_pos(proposed_tokens, target_logits_per_pos)

        reject_recheck_called = False
        reject_recheck_override = False

        if reject_pos < 0:
            # K 个候选全部通过 target 验证
            accepted_round = [int(t) for t in proposed_tokens]
            accepted_match += len(accepted_round)
        else:
            # 出现首拒绝点：先接受前缀通过部分，再对拒绝位做决策
            accepted_round = [int(t) for t in proposed_tokens[:reject_pos]]
            accepted_match += reject_pos

            draft_token = int(proposed_tokens[reject_pos])
            target_logits_reject = target_logits_per_pos[reject_pos]
            target_token = int(argmax_id(target_logits_reject))

            if normalized_mode == "standard_speculative":
                # 标准投机解码：拒绝即回退到 target token
                chosen = target_token
                rejected_mismatch += 1
            else:
                # divergence：拒绝点触发 small_base 复判（v0/v1/v2）
                if small_base is None or base_lazy is None:
                    raise ValueError("small_base is required for divergence modes")

                reject_recheck_called = True
                override_calls += 1

                logp_target_on_target = logp_of(target_logits_reject, target_token)
                logp_target_on_draft = logp_of(target_logits_reject, draft_token)
                target_opp = float(logp_target_on_target - logp_target_on_draft)

                # v2 下先做 target 反对度预筛：若 target 明确反对，就不必调用 small_base。
                if (
                    normalized_mode == "divergence_v2"
                    and enable_v2_target_opp_precheck
                    and target_opp >= tau_target_opp
                ):
                    do_override = False
                    reason = "v2_target_opp_precheck_fail"
                    v2_precheck_skips += 1
                else:
                    needed_generated_len = len(generated) + reject_pos
                    base_lazy = _sync_base_to_generated_len(
                        base_model=small_base,
                        base_lazy=base_lazy,
                        accepted_tokens_all=generated,
                        target_generated_len=needed_generated_len,
                        base_device=base_dev,
                    )
                    small_base_calls += 1

                    base_logits_reject = base_lazy.state.next_logits
                    base_token = int(argmax_id(base_logits_reject))

                    logp_draft_on_draft = logp_of(draft_logits_per_pos[reject_pos], draft_token)
                    logp_base_on_draft = logp_of(base_logits_reject, draft_token)
                    delta_logp = float(logp_draft_on_draft - logp_base_on_draft)

                    do_override, reason = should_override(
                        mode=normalized_mode,
                        draft_token_id=draft_token,
                        base_token_id=base_token,
                        delta_logp=delta_logp,
                        target_opp=target_opp,
                        tau_delta=tau_delta,
                        tau_target_opp=tau_target_opp,
                    )

                if do_override:
                    # 放行 draft token
                    chosen = draft_token
                    accepted_override += 1
                    reject_recheck_override = True
                else:
                    # 不放行，回退 target token
                    chosen = target_token
                    rejected_mismatch += 1

            accepted_round.append(int(chosen))

        target_state = advance_state_with_tokens(target, target_state, token_ids=accepted_round, device=target_dev)
        draft_state = advance_state_with_tokens(draft, draft_state, token_ids=accepted_round, device=draft_dev)

        generated.extend(int(t) for t in accepted_round)

        if save_round_level:
            round_events.append(
                RoundEvent(
                    round_idx=int(rounds),
                    proposed_len=int(len(proposed_tokens)),
                    first_reject_pos=int(reject_pos),
                    verified_prefix_len=int(len(accepted_round) if reject_pos < 0 else reject_pos),
                    reject_recheck_called=bool(reject_recheck_called),
                    reject_recheck_override=bool(reject_recheck_override),
                    accepted_tokens_this_round=int(len(accepted_round)),
                )
            )

        if stop_on_eos and tokenizer.eos_token_id is not None and generated and generated[-1] == tokenizer.eos_token_id:
            break

    t1 = time.perf_counter()

    full_ids = torch.cat(
        [
            prompt_ids,
            torch.tensor([generated], dtype=torch.long) if generated else torch.empty((1, 0), dtype=torch.long),
        ],
        dim=1,
    )

    decoded = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    response = tokenizer.decode(full_ids[0][prompt_ids.shape[1] :], skip_special_tokens=True)
    pred = extract_answer(decoded)
    pred = pred if pred in {"A", "B", "C", "D"} else ""
    correct = bool(pred) and (pred == gt)

    return SampleResult(
        id=case_id,
        gt=gt,
        pred=pred,
        correct=bool(correct),
        rounds=int(rounds),
        generated_tokens=int(len(generated)),
        speculative_tokens=int(speculative_tokens),
        proposed_tokens_total=int(proposed_tokens_total),
        accepted_match=int(accepted_match),
        accepted_override=int(accepted_override),
        rejected_mismatch=int(rejected_mismatch),
        override_calls=int(override_calls),
        small_base_calls=int(small_base_calls),
        v2_precheck_skips=int(v2_precheck_skips),
        duration_sec=float(t1 - t0),
        response=response,
        round_events=round_events,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    # 参数设计原则：
    # - 模型路径与 device_map 可独立设置（便于多卡/CPU兜底）
    # - speculative_tokens 控制 K
    # - tau_* 控制 v1/v2 阈值
    parser = argparse.ArgumentParser(description="K-token speculative verify evaluator with divergence override")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), required=True)
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--draft_model", default=None)
    parser.add_argument("--small_base_model", default=None)
    parser.add_argument("--tokenizer", default=None)

    parser.add_argument("--device_map", default="auto", help="Global fallback device_map")
    parser.add_argument("--target_device_map", default=None)
    parser.add_argument("--draft_device_map", default=None)
    parser.add_argument("--small_base_device_map", default=None)
    parser.add_argument("--dtype", default="auto", help="auto|fp16|bf16|fp32")

    parser.add_argument("--speculative_tokens", type=int, default=4)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--stop_on_eos", action="store_true")

    parser.add_argument("--tau_delta", type=float, default=0.0)
    parser.add_argument("--tau_target_opp", type=float, default=1.0)

    parser.add_argument("--save_round_level", action="store_true")
    parser.add_argument(
        "--disable_v2_target_opp_precheck",
        action="store_true",
        help="Disable v2 target opposition precheck before small_base recheck",
    )
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.speculative_tokens < 1:
        raise ValueError("--speculative_tokens must be >= 1")

    normalized_mode = "standard_speculative" if args.mode == "strict" else args.mode

    if normalized_mode != "baseline" and not args.draft_model:
        raise ValueError("--draft_model is required for non-baseline modes")
    if normalized_mode.startswith("divergence") and not args.small_base_model:
        raise ValueError("--small_base_model is required for divergence modes")

    tok_path = args.tokenizer or args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_device_map = args.target_device_map or args.device_map
    draft_device_map = args.draft_device_map or args.device_map
    base_device_map = args.small_base_device_map or args.device_map

    target = _load_model(args.target_model, device_map=target_device_map, dtype_name=args.dtype)
    draft = (
        _load_model(args.draft_model, device_map=draft_device_map, dtype_name=args.dtype)
        if args.draft_model
        else None
    )
    small_base = (
        _load_model(args.small_base_model, device_map=base_device_map, dtype_name=args.dtype)
        if args.small_base_model
        else None
    )

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
            speculative_tokens=args.speculative_tokens,
            max_new_tokens=args.max_new_tokens,
            tau_delta=args.tau_delta,
            tau_target_opp=args.tau_target_opp,
            stop_on_eos=args.stop_on_eos,
            save_round_level=args.save_round_level,
            enable_v2_target_opp_precheck=not args.disable_v2_target_opp_precheck,
        )
        results.append(res)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(cases)}] processed")
    t_end = time.perf_counter()

    n = len(results)
    correct = sum(1 for r in results if r.correct)
    total_tokens = sum(r.generated_tokens for r in results)
    total_rounds = sum(r.rounds for r in results)
    total_proposed = sum(r.proposed_tokens_total for r in results)
    total_override_calls = sum(r.override_calls for r in results)
    total_small_base_calls = sum(r.small_base_calls for r in results)
    total_v2_precheck_skips = sum(r.v2_precheck_skips for r in results)
    total_override = sum(r.accepted_override for r in results)
    total_match = sum(r.accepted_match for r in results)
    total_reject = sum(r.rejected_mismatch for r in results)
    total_gen_time = sum(r.duration_sec for r in results)

    all_rounds = [evt for r in results for evt in r.round_events]
    reject_rechecks = sum(1 for evt in all_rounds if evt.reject_recheck_called)
    reject_recheck_overrides = sum(1 for evt in all_rounds if evt.reject_recheck_override)

    summary = {
        "mode": normalized_mode,
        "mode_input": args.mode,
        "split": args.split,
        "n_cases": n,
        "accuracy": (correct / n) if n else 0.0,
        "correct": int(correct),
        "speculative_tokens": int(args.speculative_tokens),
        "total_tokens": int(total_tokens),
        "total_rounds": int(total_rounds),
        "total_proposed_tokens": int(total_proposed),
        "accepted_match": int(total_match),
        "accepted_override": int(total_override),
        "rejected_mismatch": int(total_reject),
        "override_calls": int(total_override_calls),
        "small_base_calls": int(total_small_base_calls),
        "v2_precheck_skips": int(total_v2_precheck_skips),
        "small_base_call_rate_per_override_call": (
            total_small_base_calls / total_override_calls
        ) if total_override_calls else 0.0,
        "override_rate": (total_override / total_override_calls) if total_override_calls else 0.0,
        "accepted_tokens_per_round": (total_tokens / total_rounds) if total_rounds else 0.0,
        # proposal_efficiency 越接近 1 越好：
        # 表示提案出来的 token 绝大部分最终都被接受。
        "proposal_efficiency": (total_tokens / total_proposed) if total_proposed else 0.0,
        "reject_rechecks": int(reject_rechecks),
        "reject_recheck_overrides": int(reject_recheck_overrides),
        "tokens_per_sec_gen_only": (total_tokens / total_gen_time) if total_gen_time > 0 else 0.0,
        "tokens_per_sec_end_to_end": (total_tokens / (t_end - t_start)) if (t_end - t_start) > 0 else 0.0,
        "avg_latency_per_case_sec": ((t_end - t_start) / n) if n else 0.0,
        "tau_delta": float(args.tau_delta),
        "tau_target_opp": float(args.tau_target_opp),
        "dtype": args.dtype,
        "target_device_map": target_device_map,
        "draft_device_map": draft_device_map,
        "small_base_device_map": base_device_map,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "small_base_model": args.small_base_model,
    }

    payload = {
        "summary": summary,
        "samples": [asdict(r) for r in results],
    }

    if not args.save_round_level:
        for sample in payload["samples"]:
            sample["round_events"] = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
