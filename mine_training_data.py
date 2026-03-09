"""Automatic mining of training data for the accept/reject token classifier.

Implements the workflow described in project.md:
1) Start from cases where target is wrong and small is correct.
2) Walk along the target generation prefix; at each step, ask small model for the next token.
   If small's greedy next token differs from target's greedy next token, we found a divergence point.
3) Force the divergence token (small token) into target context, let target continue generation.
   If target can recover and end with the correct answer, label this token as IMPORTANT (True), else False.
4) For each divergence point, extract and concatenate hidden states from (target, draft, small) models.

Outputs:
- <out_prefix>.meta.jsonl: one record per mined divergence point
- <out_prefix>.features.npz: numpy arrays {X, y}
- <out_prefix>.info.json: run configuration and feature dimensions

Notes:
- This script assumes the three models share a compatible tokenizer (recommended).
  If tokenizers differ, the token-by-token scan becomes less meaningful.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import format_prompt
from evaluator import extract_answer


@dataclass
class ModelSpec:
	name: str
	path: str


def _load_json(path: str) -> Any:
	return json.loads(Path(path).read_text())


def _maybe_bool(v: Any) -> Optional[bool]:
	if isinstance(v, bool):
		return v
	if v is None:
		return None
	if isinstance(v, (int, float)):
		return bool(v)
	s = str(v).strip().lower()
	if s in {"true", "1", "yes", "y"}:
		return True
	if s in {"false", "0", "no", "n"}:
		return False
	return None


def load_hf_model(model_path: str, device_map: str = "auto"):
	# Tokenizer is loaded separately to ensure all models share the same token IDs.
	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		trust_remote_code=True,
		torch_dtype="auto",
		device_map=device_map,
	)
	model.eval()
	return model


def load_tokenizer(tokenizer_path: str):
	tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	return tok


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


@torch.inference_mode()
def _generate_from_input_ids(
	model: AutoModelForCausalLM,
	tokenizer,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	max_new_tokens: int,
):
	out_ids = model.generate(
		input_ids=input_ids,
		attention_mask=attention_mask,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		pad_token_id=tokenizer.eos_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)
	return out_ids


def _device_of(model: AutoModelForCausalLM) -> torch.device:
	return next(model.parameters()).device


def _as_1x1(token_id: int) -> torch.Tensor:
	return torch.tensor([[token_id]], dtype=torch.long)


def _concat_features(vs: List[torch.Tensor]) -> np.ndarray:
	parts = [v.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1) for v in vs]
	return np.concatenate(parts, axis=0)


def _iter_filtered_cases(
	cases: List[Dict[str, Any]],
	use_input_correctness: bool,
) -> Iterable[Dict[str, Any]]:
	if not use_input_correctness:
		yield from cases
		return

	for item in cases:
		big_ok = _maybe_bool(item.get("big_model_is_correct"))
		small_ok = _maybe_bool(item.get("small_model_is_correct"))
		if big_ok is False and small_ok is True:
			yield item


def _assert_vocab_compatible(tokenizer, models: Dict[str, AutoModelForCausalLM], allow_mismatch: bool) -> None:
	tok_vs = int(getattr(tokenizer, "vocab_size", 0) or 0)
	bad: List[str] = []
	for name, model in models.items():
		m_vs = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
		if tok_vs and m_vs and tok_vs != m_vs:
			bad.append(f"{name}: model_vocab_size={m_vs} tokenizer_vocab_size={tok_vs}")
	if bad and not allow_mismatch:
		raise ValueError(
			"Tokenizer/model vocab_size mismatch (unsafe to feed the same input_ids). "
			"Pass --allow_tokenizer_mismatch to override (not recommended). Details: "
			+ "; ".join(bad)
		)


@torch.inference_mode()
def _judge_correctness(
	item: Dict[str, Any],
	tokenizer,
	target: AutoModelForCausalLM,
	small: AutoModelForCausalLM,
	max_new_tokens: int,
) -> Dict[str, Any]:
	question = item.get("question", "")
	options = item.get("options", {})
	gt = str(item.get("ground_truth", "")).strip().upper()

	prompt = format_prompt(tokenizer, question, options)
	encoded = tokenizer(prompt, return_tensors="pt", return_dict=True)
	in_ids = encoded["input_ids"]
	in_mask = encoded["attention_mask"]

	dev_t = _device_of(target)
	dev_s = _device_of(small)

	out_t = _generate_from_input_ids(
		model=target,
		tokenizer=tokenizer,
		input_ids=in_ids.to(dev_t),
		attention_mask=in_mask.to(dev_t),
		max_new_tokens=max_new_tokens,
	)
	out_s = _generate_from_input_ids(
		model=small,
		tokenizer=tokenizer,
		input_ids=in_ids.to(dev_s),
		attention_mask=in_mask.to(dev_s),
		max_new_tokens=max_new_tokens,
	)

	dec_t = tokenizer.decode(out_t[0], skip_special_tokens=True)
	dec_s = tokenizer.decode(out_s[0], skip_special_tokens=True)
	pred_t = extract_answer(dec_t)
	pred_s = extract_answer(dec_s)
	ok_t = bool(pred_t) and pred_t == gt
	ok_s = bool(pred_s) and pred_s == gt
	return {
		"gt": gt,
		"target_pred": pred_t,
		"small_pred": pred_s,
		"target_correct": bool(ok_t),
		"small_correct": bool(ok_s),
	}


def mine_case(
	item: Dict[str, Any],
	tokenizer,
	target: AutoModelForCausalLM,
	draft: AutoModelForCausalLM,
	small: AutoModelForCausalLM,
	max_prefix_steps: int,
	max_new_tokens_after: int,
	max_points_per_case: int,
	stop_on_eos: bool,
) -> List[Dict[str, Any]]:
	question = item.get("question", "")
	options = item.get("options", {})
	gt = str(item.get("ground_truth", "")).strip().upper()

	prompt = format_prompt(tokenizer, question, options)
	encoded = tokenizer(prompt, return_tensors="pt", return_dict=True)
	prompt_ids_cpu = encoded["input_ids"]
	prompt_mask_cpu = encoded["attention_mask"]

	models = {"target": target, "draft": draft, "small": small}
	devices = {k: _device_of(m) for k, m in models.items()}
	past = {k: None for k in models}

	cur_ids_cpu = prompt_ids_cpu
	cur_mask_cpu = prompt_mask_cpu

	generated_target: List[int] = []
	mined: List[Dict[str, Any]] = []

	for step in range(max_prefix_steps):
		last_h = {}
		next_logits = {}

		for name, model in models.items():
			dev = devices[name]
			in_ids = cur_ids_cpu.to(dev)
			in_mask = cur_mask_cpu.to(dev)
			h, logits, pkv = _forward_last_hidden_and_logits(
				model=model,
				input_ids=in_ids,
				attention_mask=in_mask,
				past_key_values=past[name],
			)
			past[name] = pkv
			last_h[name] = h
			next_logits[name] = logits

		next_t = int(next_logits["target"].argmax(dim=-1).item())
		next_s = int(next_logits["small"].argmax(dim=-1).item())

		if next_s != next_t:
			feat = _concat_features([last_h["target"], last_h["draft"], last_h["small"]])
			token_text = tokenizer.decode([next_s], skip_special_tokens=True)

			prefix_ids = (
				torch.tensor([generated_target], dtype=torch.long)
				if generated_target
				else torch.empty((1, 0), dtype=torch.long)
			)
			forced_input_ids_cpu = torch.cat([prompt_ids_cpu, prefix_ids, _as_1x1(next_s)], dim=1)
			forced_mask_cpu = torch.ones_like(forced_input_ids_cpu)

			out_ids = _generate_from_input_ids(
				model=target,
				tokenizer=tokenizer,
				input_ids=forced_input_ids_cpu.to(devices["target"]),
				attention_mask=forced_mask_cpu.to(devices["target"]),
				max_new_tokens=max_new_tokens_after,
			)
			decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
			pred = extract_answer(decoded)
			is_correct = bool(pred) and (pred == gt)

			mined.append(
				{
					"case_id": item.get("id"),
					"step": step,
					"gt": gt,
					"target_next_token_id": next_t,
					"small_next_token_id": next_s,
					"small_next_token_text": token_text,
					"label_important": bool(is_correct),
					"forced_pred": pred,
					"feature": feat,
				}
			)

			if len(mined) >= max_points_per_case:
				break

		generated_target.append(next_t)
		if stop_on_eos and next_t == tokenizer.eos_token_id:
			break

		cur_ids_cpu = _as_1x1(next_t)
		cur_mask_cpu = torch.cat([cur_mask_cpu, torch.ones_like(cur_ids_cpu)], dim=1)

	return mined


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Mine training data (hidden states + labels) for token accept/reject classifier"
	)
	parser.add_argument(
		"--cases",
		default="logs/medqa_big_wrong_small_right_annotated.json",
		help="Input JSON (list of dicts) with at least question/options/ground_truth",
	)
	parser.add_argument(
		"--out_prefix",
		default="logs/mined_train",
		help="Output prefix; writes .meta.jsonl/.features.npz/.info.json",
	)
	parser.add_argument("--target_model", default=None, help="HF path/id for target model")
	parser.add_argument("--draft_model", default=None, help="HF path/id for draft model")
	parser.add_argument("--small_model", default=None, help="HF path/id for small model")
	parser.add_argument(
		"--use_model_loader",
		action="store_true",
		help="Load models via model_loader.get_model_and_tokenizer() (uses config.py)",
	)
	parser.add_argument(
		"--device_map",
		default="auto",
		help="device_map for transformers.from_pretrained when not using model_loader",
	)
	parser.add_argument(
		"--tokenizer",
		default=None,
		help="Tokenizer path/id to build prompts and input_ids (recommended: same family as all models)",
	)
	parser.add_argument(
		"--allow_tokenizer_mismatch",
		action="store_true",
		help="Allow tokenizer/model vocab mismatch (NOT recommended; may corrupt mining)",
	)
	parser.add_argument("--limit", type=int, default=None, help="Only process first N cases (debug)")
	parser.add_argument(
		"--use_input_correctness",
		action="store_true",
		help="If set, only keep cases where (big wrong, small right) according to input file flags",
	)
	parser.add_argument(
		"--filter_with_models",
		action="store_true",
		help="Filter cases by running target/small generation with current models (target wrong, small right)",
	)
	parser.add_argument(
		"--eval_max_new_tokens",
		type=int,
		default=512,
		help="Max new tokens used when --filter_with_models is enabled",
	)
	parser.add_argument("--max_prefix_steps", type=int, default=256, help="Max tokens to scan along target prefix")
	parser.add_argument(
		"--max_new_tokens_after",
		type=int,
		default=256,
		help="Max new tokens for target continuation after forcing the divergence token",
	)
	parser.add_argument(
		"--max_points_per_case",
		type=int,
		default=1,
		help="Max divergence points to mine per case (1 = first divergence only)",
	)
	parser.add_argument("--stop_on_eos", action="store_true", help="Stop prefix scan when target outputs EOS")

	args = parser.parse_args()

	cases = _load_json(args.cases)
	if not isinstance(cases, list):
		raise ValueError("--cases must be a JSON list")
	if args.limit is not None:
		cases = cases[: args.limit]

	if args.use_model_loader:
		from model_loader import get_model_and_tokenizer

		models, tokenizer = get_model_and_tokenizer()
		target = models["target"]
		draft = models["base"]
		small = models["expert"]
		_assert_vocab_compatible(
			tokenizer,
			{"target": target, "draft": draft, "small": small},
			allow_mismatch=args.allow_tokenizer_mismatch,
		)
		model_specs = [
			ModelSpec("target", "config:ModelIDs.TARGET"),
			ModelSpec("draft", "config:ModelIDs.DRAFT_BASE"),
			ModelSpec("small", "config:ModelIDs.DRAFT_EXPERT"),
		]
	else:
		if not (args.target_model and args.draft_model and args.small_model):
			raise ValueError(
				"Provide --target_model/--draft_model/--small_model, or pass --use_model_loader"
			)

		tokenizer_path = args.tokenizer or args.target_model
		tokenizer = load_tokenizer(tokenizer_path)
		target = load_hf_model(args.target_model, device_map=args.device_map)
		draft = load_hf_model(args.draft_model, device_map=args.device_map)
		small = load_hf_model(args.small_model, device_map=args.device_map)
		_assert_vocab_compatible(
			tokenizer,
			{"target": target, "draft": draft, "small": small},
			allow_mismatch=args.allow_tokenizer_mismatch,
		)
		model_specs = [
			ModelSpec("target", args.target_model),
			ModelSpec("draft", args.draft_model),
			ModelSpec("small", args.small_model),
		]

	out_prefix = Path(args.out_prefix)
	out_prefix.parent.mkdir(parents=True, exist_ok=True)

	meta_path = out_prefix.with_suffix(".meta.jsonl")
	info_path = out_prefix.with_suffix(".info.json")
	feats_path = out_prefix.with_suffix(".features.npz")

	if meta_path.exists():
		meta_path.unlink()

	X_list: List[np.ndarray] = []
	y_list: List[int] = []

	if args.filter_with_models:
		kept_cases = []
		for idx, item in enumerate(cases):
			res = _judge_correctness(
				item=item,
				tokenizer=tokenizer,
				target=target,
				small=small,
				max_new_tokens=args.eval_max_new_tokens,
			)
			if (not res["target_correct"]) and res["small_correct"]:
				kept_cases.append(item)
			if (idx + 1) % 10 == 0:
				print(f"[filter] {idx + 1}/{len(cases)} checked, kept={len(kept_cases)}")
	else:
		kept_cases = list(_iter_filtered_cases(cases, use_input_correctness=args.use_input_correctness))

	total_points = 0
	for idx, item in enumerate(kept_cases):
		mined = mine_case(
			item=item,
			tokenizer=tokenizer,
			target=target,
			draft=draft,
			small=small,
			max_prefix_steps=args.max_prefix_steps,
			max_new_tokens_after=args.max_new_tokens_after,
			max_points_per_case=args.max_points_per_case,
			stop_on_eos=args.stop_on_eos,
		)

		for rec in mined:
			feat = rec.pop("feature")
			X_list.append(feat)
			y_list.append(1 if rec["label_important"] else 0)

			rec["feature_idx"] = len(X_list) - 1
			with open(meta_path, "a", encoding="utf-8") as f:
				f.write(json.dumps(rec, ensure_ascii=False) + "\n")

		total_points += len(mined)
		if (idx + 1) % 5 == 0:
			print(f"[{idx + 1}/{len(kept_cases)}] cases processed, {total_points} points mined")

	X = np.stack(X_list, axis=0) if X_list else np.zeros((0, 0), dtype=np.float32)
	y = np.array(y_list, dtype=np.int64)
	np.savez_compressed(feats_path, X=X, y=y)

	info = {
		"n_cases_input": len(cases),
		"n_cases_used": len(kept_cases),
		"n_points": int(total_points),
		"max_prefix_steps": int(args.max_prefix_steps),
		"max_new_tokens_after": int(args.max_new_tokens_after),
		"max_points_per_case": int(args.max_points_per_case),
		"use_input_correctness": bool(args.use_input_correctness),
		"tokenizer": {
			"name_or_path": getattr(tokenizer, "name_or_path", ""),
			"vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
		},
		"models": [asdict(s) for s in model_specs],
		"feature_dim": int(X.shape[1]) if X.ndim == 2 and X.size else 0,
	}
	info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))

	print(f"Done. meta={meta_path} feats={feats_path} info={info_path}")


if __name__ == "__main__":
	main()