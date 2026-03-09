"""Automatic mining of training data for the accept/reject token classifier.

Implements the workflow described in project.md:
This script follows the 3-model setup described in project.md:
- target: a general big model (e.g., Qwen-14B)
- draft: a domain expert small model (e.g., II-Medical-8B)
- small_base: the base model corresponding to the expert (e.g., Qwen-8B-Base)

Mining workflow:
1) Start from cases where target is wrong and draft is correct.
2) Walk along the target generation prefix; at each step, ask draft for the next token.
	If draft's greedy next token differs from target's greedy next token, we found a divergence point.
3) Force the divergence token (draft token) into target context, let target continue generation.
   If target can recover and end with the correct answer, label this token as IMPORTANT (True), else False.
4) For each divergence point, extract and concatenate hidden states from (target, draft, small_base) models.

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


def _restricted_argmax_token_id(next_logits: torch.Tensor, tok_len: int) -> int:
	"""Argmax next-token id, optionally restricting to ids < tok_len.

	This is important when a model's output vocab is larger than the tokenizer length.
	Without this restriction, a model may select token ids that the shared tokenizer cannot
	decode or that other models cannot safely consume.
	"""
	if next_logits.dim() == 1:
		next_logits = next_logits.unsqueeze(0)
	if tok_len and next_logits.size(-1) > tok_len:
		# Mask out logits for token ids that this tokenizer can't represent.
		masked = next_logits[..., :tok_len]
		return int(masked.argmax(dim=-1).item())
	return int(next_logits.argmax(dim=-1).item())


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

	# Backward compatibility for legacy JSONs that already store correctness flags.
	# We interpret them as: big_model_is_correct -> target correctness; small_model_is_correct -> draft correctness.
	for item in cases:
		target_ok = _maybe_bool(item.get("big_model_is_correct"))
		draft_ok = _maybe_bool(item.get("small_model_is_correct"))
		if target_ok is False and draft_ok is True:
			yield item


def _assert_vocab_compatible(tokenizer, models: Dict[str, AutoModelForCausalLM], allow_mismatch: bool) -> None:
	# NOTE: For many HF tokenizers, `tokenizer.vocab_size` excludes added tokens.
	# What actually matters for safety is: max possible token id produced by this tokenizer
	# must be < model input embedding size for every model we feed these ids into.
	try:
		tok_len = int(len(tokenizer))
	except Exception:
		tok_len = int(getattr(tokenizer, "vocab_size", 0) or 0)

	bad: List[str] = []
	for name, model in models.items():
		m_vs_cfg = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
		try:
			emb = model.get_input_embeddings()
			m_vs_emb = int(getattr(emb, "num_embeddings", 0) or 0) if emb is not None else 0
		except Exception:
			m_vs_emb = 0

		m_vs = m_vs_emb or m_vs_cfg
		if tok_len and m_vs and tok_len > m_vs:
			bad.append(
				f"{name}: model_vocab_size={m_vs} tokenizer_len={tok_len} "
				f"(config_vocab_size={m_vs_cfg}, embedding_vocab_size={m_vs_emb})"
			)

	if bad and not allow_mismatch:
		raise ValueError(
			"Tokenizer/model vocab mismatch: tokenizer can produce token ids that exceed at least one model's "
			"input embedding size (unsafe to feed the same input_ids). "
			"Pass --allow_tokenizer_mismatch to override (not recommended). Details: "
			+ "; ".join(bad)
		)


@torch.inference_mode()
def _judge_correctness(
	item: Dict[str, Any],
	tokenizer,
	target: AutoModelForCausalLM,
	draft: AutoModelForCausalLM,
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
	dev_d = _device_of(draft)

	out_t = _generate_from_input_ids(
		model=target,
		tokenizer=tokenizer,
		input_ids=in_ids.to(dev_t),
		attention_mask=in_mask.to(dev_t),
		max_new_tokens=max_new_tokens,
	)
	out_d = _generate_from_input_ids(
		model=draft,
		tokenizer=tokenizer,
		input_ids=in_ids.to(dev_d),
		attention_mask=in_mask.to(dev_d),
		max_new_tokens=max_new_tokens,
	)

	dec_t = tokenizer.decode(out_t[0], skip_special_tokens=True)
	dec_d = tokenizer.decode(out_d[0], skip_special_tokens=True)
	pred_t = extract_answer(dec_t)
	pred_d = extract_answer(dec_d)
	ok_t = bool(pred_t) and pred_t == gt
	ok_d = bool(pred_d) and pred_d == gt
	return {
		"gt": gt,
		"target_pred": pred_t,
		"draft_pred": pred_d,
		"target_correct": bool(ok_t),
		"draft_correct": bool(ok_d),
	}


def mine_case(
	item: Dict[str, Any],
	tokenizer,
	target: AutoModelForCausalLM,
	draft: AutoModelForCausalLM,
	small_base: AutoModelForCausalLM,
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

	models = {"target": target, "draft": draft, "small_base": small_base}
	devices = {k: _device_of(m) for k, m in models.items()}
	past = {k: None for k in models}

	cur_ids_cpu = prompt_ids_cpu
	cur_mask_cpu = prompt_mask_cpu

	try:
		tok_len = int(len(tokenizer))
	except Exception:
		tok_len = int(getattr(tokenizer, "vocab_size", 0) or 0)

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

		next_t = _restricted_argmax_token_id(next_logits["target"], tok_len)
		next_d = _restricted_argmax_token_id(next_logits["draft"], tok_len)

		# Divergence point: draft(expert) disagrees with target.
		if next_d != next_t:
			feat = _concat_features([last_h["target"], last_h["draft"], last_h["small_base"]])
			token_text = tokenizer.decode([next_d], skip_special_tokens=True)

			prefix_ids = (
				torch.tensor([generated_target], dtype=torch.long)
				if generated_target
				else torch.empty((1, 0), dtype=torch.long)
			)
			forced_input_ids_cpu = torch.cat([prompt_ids_cpu, prefix_ids, _as_1x1(next_d)], dim=1)
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
					"draft_next_token_id": next_d,
					"draft_next_token_text": token_text,
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
		"--source",
		choices=["json", "medqa"],
		default="json",
		help=(
			"Where to load candidate questions from. "
			"IMPORTANT: if you use a JSON produced by other models (e.g., Qwen), it biases the pool; "
			"prefer --source medqa + --filter_with_models for a clean pipeline."
		),
	)
	parser.add_argument(
		"--medqa_split",
		default="test",
		help="MedQA split when --source medqa is used (train/test)",
	)
	parser.add_argument(
		"--dataset_limit",
		type=int,
		default=0,
		help="Limit dataset size when --source medqa is used (0 = full)",
	)
	parser.add_argument(
		"--cases",
		default="logs/medqa_big_wrong_small_right_annotated.json",
		help="Input JSON (list of dicts) with at least question/options/ground_truth (only used when --source json)",
	)
	parser.add_argument(
		"--out_prefix",
		default="logs/mined_train",
		help="Output prefix; writes .meta.jsonl/.features.npz/.info.json",
	)
	parser.add_argument("--target_model", default=None, help="HF path/id for target model")
	parser.add_argument("--draft_model", default=None, help="HF path/id for draft model (domain expert)")
	parser.add_argument("--small_base_model", default=None, help="HF path/id for small_base model")
	parser.add_argument(
		"--small_model",
		default=None,
		help="Alias of --small_base_model (kept for backward compatibility)",
	)
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
		help="Filter cases by running target/draft generation with current models (target wrong, draft right)",
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

	if args.source == "medqa":
		from data_loader import load_medqa

		# Full MedQA candidate pool (no pre-filtering by other models)
		ds = load_medqa(split=args.medqa_split, limit=args.dataset_limit or 0)
		cases = []
		for idx, item in enumerate(ds):
			q = item.get("question", "")
			opts = item.get("options", {})
			gt = item.get("answer_idx", "")
			gt = str(gt).strip().upper() if gt is not None else ""
			if not q or not opts or gt not in {"A", "B", "C", "D"}:
				continue
			cases.append(
				{
					"id": str(item.get("id", idx)),
					"question": q,
					"options": opts,
					"ground_truth": gt,
				}
			)
	else:
		cases = _load_json(args.cases)
		if not isinstance(cases, list):
			raise ValueError("--cases must be a JSON list")

	if args.limit is not None:
		cases = cases[: args.limit]

	if args.use_model_loader:
		from model_loader import get_model_and_tokenizer

		models, tokenizer = get_model_and_tokenizer()
		target = models["target"]
		# NOTE: model_loader returns keys {target, base, expert}
		# In project.md definitions: draft=expert, small_base=base.
		small_base = models["base"]
		draft = models["expert"]
		_assert_vocab_compatible(
			tokenizer,
			{"target": target, "draft": draft, "small_base": small_base},
			allow_mismatch=args.allow_tokenizer_mismatch,
		)
		model_specs = [
			ModelSpec("target", "config:ModelIDs.TARGET"),
			ModelSpec("draft", "config:ModelIDs.DRAFT_EXPERT"),
			ModelSpec("small_base", "config:ModelIDs.DRAFT_BASE"),
		]
	else:
		small_base_path = args.small_base_model or args.small_model
		if not (args.target_model and args.draft_model and small_base_path):
			raise ValueError(
				"Provide --target_model/--draft_model/--small_base_model (or --small_model), or pass --use_model_loader"
			)

		tokenizer_path = args.tokenizer or args.target_model
		tokenizer = load_tokenizer(tokenizer_path)
		target = load_hf_model(args.target_model, device_map=args.device_map)
		draft = load_hf_model(args.draft_model, device_map=args.device_map)
		small_base = load_hf_model(small_base_path, device_map=args.device_map)
		_assert_vocab_compatible(
			tokenizer,
			{"target": target, "draft": draft, "small_base": small_base},
			allow_mismatch=args.allow_tokenizer_mismatch,
		)
		model_specs = [
			ModelSpec("target", args.target_model),
			ModelSpec("draft", args.draft_model),
			ModelSpec("small_base", small_base_path),
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
				draft=draft,
				max_new_tokens=args.eval_max_new_tokens,
			)
			if (not res["target_correct"]) and res["draft_correct"]:
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
			small_base=small_base,
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