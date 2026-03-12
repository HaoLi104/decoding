#!/usr/bin/env python3
"""Analyze mined divergence-point dataset.

Reads:
- <prefix>.meta.jsonl
- <prefix>.features.npz
- <prefix>.info.json

Prints:
- integrity checks (row alignment)
- label distribution
- points-per-case statistics
- step statistics
- delta_p / p_draft / p_base distribution (overall + by label)
- simple score baselines (AUC for delta_p, p_draft, -p_base)

No third-party deps beyond numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def _percentiles(x: np.ndarray, ps: List[float]) -> Dict[str, float]:
	if x.size == 0:
		return {str(p): float("nan") for p in ps}
	vals = np.percentile(x, ps)
	return {str(p): float(v) for p, v in zip(ps, vals)}


def _safe_float(v: Any, default: float = 0.0) -> float:
	try:
		if v is None:
			return default
		return float(v)
	except Exception:
		return default


def _is_punct_like(tok_text: str) -> bool:
	# heuristic: whitespace / short punctuation / quotes / brackets / dot/comma etc
	s = tok_text
	if s is None:
		return False
	s = str(s)
	if s.strip() == "":
		return True
	# if it's mostly non-alnum characters
	non_alnum = sum(1 for ch in s if not ch.isalnum())
	return non_alnum >= max(1, len(s) - 1)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
	"""Compute ROC AUC (binary) without sklearn.

	If all labels are the same, returns NaN.
	"""
	if y_true.size == 0:
		return float("nan")
	pos = int(np.sum(y_true == 1))
	neg = int(np.sum(y_true == 0))
	if pos == 0 or neg == 0:
		return float("nan")

	order = np.argsort(y_score)
	ranks = np.empty_like(order, dtype=np.float64)
	ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)

	# handle ties by assigning average ranks
	sorted_scores = y_score[order]
	start = 0
	while start < sorted_scores.size:
		end = start + 1
		while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
			end += 1
		if end - start > 1:
			avg = (start + 1 + end) / 2.0
			ranks[order[start:end]] = avg
		start = end

	sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
	auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
	return float(auc)


def analyze(prefix: str, topk: int) -> Dict[str, Any]:
	out_prefix = Path(prefix)
	meta_path = out_prefix.with_suffix(".meta.jsonl")
	info_path = out_prefix.with_suffix(".info.json")
	feats_path = out_prefix.with_suffix(".features.npz")

	info = _read_json(info_path) if info_path.exists() else {}
	z = np.load(feats_path)
	X = z.get("X")
	y = z.get("y")
	delta_p = z.get("delta_p")
	p_draft = z.get("p_draft")
	p_base = z.get("p_base")

	meta_n = 0
	case_counter: Counter[str] = Counter()
	step_counter: Counter[int] = Counter()
	punct_like = 0
	missing_delta = 0
	max_step = -1

	# for sampling examples
	examples_pos: List[Tuple[float, Dict[str, Any]]] = []
	examples_neg: List[Tuple[float, Dict[str, Any]]] = []

	for rec in _iter_jsonl(meta_path):
		meta_n += 1
		cid = str(rec.get("case_id"))
		case_counter[cid] += 1
		step = int(rec.get("step", -1))
		step_counter[step] += 1
		max_step = max(max_step, step)

		txt = str(rec.get("draft_next_token_text", ""))
		if _is_punct_like(txt):
			punct_like += 1

		dp = rec.get("delta_p", None)
		if dp is None:
			missing_delta += 1
			dp_f = float("nan")
		else:
			dp_f = _safe_float(dp, default=float("nan"))

		lbl = bool(rec.get("label_important", False))
		bucket = examples_pos if lbl else examples_neg
		bucket.append((dp_f, rec))

	# pick topk by delta_p (descending)
	examples_pos = sorted(examples_pos, key=lambda t: (-(t[0] if not math.isnan(t[0]) else -1e30)))[:topk]
	examples_neg = sorted(examples_neg, key=lambda t: (-(t[0] if not math.isnan(t[0]) else -1e30)))[:topk]

	# integrity checks
	checks = {
		"meta_lines": int(meta_n),
		"X_shape": list(X.shape) if X is not None else None,
		"y_shape": list(y.shape) if y is not None else None,
		"delta_p_shape": list(delta_p.shape) if delta_p is not None else None,
		"aligned": bool(
			X is not None
			and y is not None
			and int(meta_n) == int(X.shape[0]) == int(y.shape[0])
			and (delta_p is None or int(delta_p.shape[0]) == int(y.shape[0]))
		),
	}

	# label stats
	if y is None:
		y_arr = np.zeros((0,), dtype=np.int64)
	else:
		y_arr = np.asarray(y).astype(np.int64)

	pos = int(np.sum(y_arr == 1))
	neg = int(np.sum(y_arr == 0))
	label_stats = {
		"n": int(y_arr.size),
		"pos": pos,
		"neg": neg,
		"pos_rate": float(pos / y_arr.size) if y_arr.size else float("nan"),
	}

	# delta stats
	def _dist(name: str, arr: Optional[np.ndarray]) -> Dict[str, Any]:
		if arr is None:
			return {"present": False}
		a = np.asarray(arr).astype(np.float32)
		out = {
			"present": True,
			"min": float(np.min(a)) if a.size else float("nan"),
			"max": float(np.max(a)) if a.size else float("nan"),
			"mean": float(np.mean(a)) if a.size else float("nan"),
			"std": float(np.std(a)) if a.size else float("nan"),
			"percentiles": _percentiles(a, [1, 5, 25, 50, 75, 95, 99]),
			"frac_gt_0": float(np.mean(a > 0)) if a.size else float("nan"),
			"frac_gt_0_01": float(np.mean(a > 0.01)) if a.size else float("nan"),
			"frac_gt_0_05": float(np.mean(a > 0.05)) if a.size else float("nan"),
			"frac_gt_0_10": float(np.mean(a > 0.10)) if a.size else float("nan"),
		}
		return out

	def _dist_by_label(arr: Optional[np.ndarray]) -> Dict[str, Any]:
		if arr is None or y_arr.size == 0:
			return {}
		a = np.asarray(arr)
		return {
			"pos": _dist("pos", a[y_arr == 1]),
			"neg": _dist("neg", a[y_arr == 0]),
		}

	delta_stats = _dist("delta_p", delta_p)
	p_draft_stats = _dist("p_draft", p_draft)
	p_base_stats = _dist("p_base", p_base)

	# simple AUC baselines
	auc = {}
	if y_arr.size and delta_p is not None:
		auc["delta_p"] = _roc_auc(y_arr, np.asarray(delta_p))
	if y_arr.size and p_draft is not None:
		auc["p_draft"] = _roc_auc(y_arr, np.asarray(p_draft))
	if y_arr.size and p_base is not None:
		auc["neg_p_base"] = _roc_auc(y_arr, -np.asarray(p_base))

	# per-case stats
	points_per_case = np.array(list(case_counter.values()), dtype=np.int64)
	case_stats = {
		"n_cases": int(len(case_counter)),
		"mean_points": float(np.mean(points_per_case)) if points_per_case.size else float("nan"),
		"median_points": float(np.median(points_per_case)) if points_per_case.size else float("nan"),
		"max_points": int(np.max(points_per_case)) if points_per_case.size else 0,
		"min_points": int(np.min(points_per_case)) if points_per_case.size else 0,
		"percentiles": _percentiles(points_per_case.astype(np.float32), [50, 75, 90, 95, 99]),
		"top_cases": case_counter.most_common(min(10, len(case_counter))),
	}

	# step stats
	steps = np.array(list(step_counter.elements()), dtype=np.int64)
	step_stats = {
		"max_step_seen": int(max_step),
		"mean": float(np.mean(steps)) if steps.size else float("nan"),
		"median": float(np.median(steps)) if steps.size else float("nan"),
		"percentiles": _percentiles(steps.astype(np.float32), [50, 75, 90, 95, 99]),
		"top_steps": step_counter.most_common(min(10, len(step_counter))),
	}

	text_stats = {
		"punct_like_frac": float(punct_like / meta_n) if meta_n else float("nan"),
		"missing_delta_p_in_meta": int(missing_delta),
	}

	# feature slicing info for downstream teacher/student
	feature_info = {
		"feature_dim": int(X.shape[1]) if X is not None and X.ndim == 2 else 0,
		"hidden_sizes": info.get("hidden_sizes", {}),
		"feature_slices": info.get("feature_slices", {}),
		"npz_arrays": info.get("npz_arrays", list(z.keys())),
	}

	summary = {
		"prefix": str(out_prefix),
		"checks": checks,
		"info_json": {
			"n_cases_input": info.get("n_cases_input"),
			"n_cases_used": info.get("n_cases_used"),
			"n_points": info.get("n_points"),
			"max_prefix_steps": info.get("max_prefix_steps"),
			"max_new_tokens_after": info.get("max_new_tokens_after"),
			"max_points_per_case": info.get("max_points_per_case"),
			"max_points_per_case_semantics": info.get("max_points_per_case_semantics"),
		},
		"labels": label_stats,
		"per_case": case_stats,
		"per_step": step_stats,
		"text": text_stats,
		"delta_p": delta_stats,
		"delta_p_by_label": _dist_by_label(delta_p),
		"p_draft": p_draft_stats,
		"p_base": p_base_stats,
		"auc": auc,
		"examples": {
			"top_delta_p_pos": [r for _, r in examples_pos],
			"top_delta_p_neg": [r for _, r in examples_neg],
		},
		"notes": {
			"student_features": "Use feature_slices.student_target_draft to slice X[:, a0:a2] as [H_target,H_draft]",
			"teacher_features": "Use full X as [H_target,H_draft,H_small_base] or slice with feature_slices",
			"sample_weight": "Use delta_p (or a mapping of it) as sample weights for training",
		},
	}
	return summary


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--prefix",
		required=True,
		help="Output prefix, e.g. logs/mined_full_test_qwen_dp",
	)
	ap.add_argument(
		"--topk_examples",
		type=int,
		default=5,
		help="How many examples to include for top delta_p positives/negatives",
	)
	ap.add_argument(
		"--save_json",
		default=None,
		help="Optional path to save the JSON summary",
	)
	args = ap.parse_args()

	summary = analyze(args.prefix, topk=args.topk_examples)

	# Human-readable headline
	labels = summary["labels"]
	print(
		"headline: "
		+ f"n_points={labels['n']} pos={labels['pos']} neg={labels['neg']} pos_rate={labels['pos_rate']:.4f}; "
		+ f"aligned={summary['checks']['aligned']}; "
		+ f"delta_p_auc={summary['auc'].get('delta_p', float('nan')):.4f}"
	)

	print("\n===== JSON SUMMARY =====")
	print(json.dumps(summary, ensure_ascii=False, indent=2))

	if args.save_json:
		Path(args.save_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"\nSaved: {args.save_json}")


if __name__ == "__main__":
	main()
