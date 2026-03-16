#!/usr/bin/env python3
"""Evaluate baseline/strict/gate decoding on MedQA.

Modes:
- baseline: target-only greedy decoding
- strict: target+draft one-step compare, mismatch always falls back to target token
- gate: target+draft one-step compare, mismatch can accept draft token by gate score

This is the phase-A minimal runnable implementation for classifier-gated lossy
speculative decoding.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import load_medqa, format_prompt
from evaluator import extract_answer
from lossy_gate_model import GateRuntime


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
	gate_calls: int
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



def _build_feature_student(last_h_target: torch.Tensor, last_h_draft: torch.Tensor) -> np.ndarray:
	t = last_h_target.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
	d = last_h_draft.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
	return np.concatenate([t, d], axis=0)


@torch.inference_mode()
def run_case(
	item: Dict[str, Any],
	target: AutoModelForCausalLM,
	draft: Optional[AutoModelForCausalLM],
	tokenizer,
	mode: str,
	max_new_tokens: int,
	gate: Optional[GateRuntime],
	tau: float,
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

	past_t = None
	past_d = None
	cur_ids_cpu = prompt_ids_cpu
	cur_mask_cpu = prompt_mask_cpu
	generated: List[int] = []

	accepted_match = 0
	accepted_override = 0
	rejected_mismatch = 0
	gate_calls = 0
	steps = 0

	t0 = time.perf_counter()

	for _ in range(max_new_tokens):
		steps += 1
		last_h_t, logits_t, past_t = _forward_last_hidden_and_logits(
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
				raise ValueError("draft model is required for strict/gate modes")
			last_h_d, logits_d, past_d = _forward_last_hidden_and_logits(
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
					if gate is None:
						raise ValueError("gate runtime is required for gate mode")
					gate_calls += 1
					feat = _build_feature_student(last_h_t, last_h_d)
					p_accept = gate.predict_proba(feat)
					if p_accept >= tau:
						accepted = next_d
						accepted_override += 1
					else:
						accepted = next_t
						rejected_mismatch += 1

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
		gate_calls=int(gate_calls),
		duration_sec=float(t1 - t0),
		response=response,
	)



def main() -> None:
	parser = argparse.ArgumentParser(description="Phase-A lossy speculative decoding evaluator")
	parser.add_argument("--mode", choices=["baseline", "strict", "gate"], required=True)
	parser.add_argument("--target_model", required=True)
	parser.add_argument("--draft_model", default=None)
	parser.add_argument("--tokenizer", default=None)
	parser.add_argument("--gate_ckpt", default=None)
	parser.add_argument("--tau", type=float, default=0.5)
	parser.add_argument("--device_map", default="auto")
	parser.add_argument("--split", default="test", choices=["train", "test"])
	parser.add_argument("--limit", type=int, default=300)
	parser.add_argument("--max_new_tokens", type=int, default=256)
	parser.add_argument("--stop_on_eos", action="store_true")
	parser.add_argument("--out", required=True, help="Output json file path")
	args = parser.parse_args()

	if args.mode in {"strict", "gate"} and not args.draft_model:
		raise ValueError("--draft_model is required for strict/gate modes")
	if args.mode == "gate" and not args.gate_ckpt:
		raise ValueError("--gate_ckpt is required for gate mode")

	tok_path = args.tokenizer or args.target_model
	tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	target = _load_model(args.target_model, device_map=args.device_map)
	draft = _load_model(args.draft_model, device_map=args.device_map) if args.draft_model else None
	gate = GateRuntime.from_checkpoint(args.gate_ckpt, device="cpu") if args.gate_ckpt else None

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
			tokenizer=tokenizer,
			mode=args.mode,
			max_new_tokens=args.max_new_tokens,
			gate=gate,
			tau=args.tau,
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
	total_gate_calls = sum(r.gate_calls for r in results)
	total_override = sum(r.accepted_override for r in results)
	total_match = sum(r.accepted_match for r in results)
	total_reject = sum(r.rejected_mismatch for r in results)
	total_gen_time = sum(r.duration_sec for r in results)

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
		"gate_calls": int(total_gate_calls),
		"gate_override_rate": (total_override / total_gate_calls) if total_gate_calls else 0.0,
		"accepted_tokens_per_step": ((total_match + total_override) / total_steps) if total_steps else 0.0,
		"tokens_per_sec_gen_only": (total_tokens / total_gen_time) if total_gen_time > 0 else 0.0,
		"tokens_per_sec_end_to_end": (total_tokens / (t_end - t_start)) if (t_end - t_start) > 0 else 0.0,
		"avg_latency_per_case_sec": ((t_end - t_start) / n) if n else 0.0,
		"tau": float(args.tau),
		"target_model": args.target_model,
		"draft_model": args.draft_model,
		"gate_ckpt": args.gate_ckpt,
	}

	payload = {
		"summary": summary,
		"samples": [asdict(r) for r in results],
	}
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

	print("Done.")
	print(json.dumps(summary, ensure_ascii=False, indent=2))
	print(f"Saved: {out_path}")


if __name__ == "__main__":
	main()
