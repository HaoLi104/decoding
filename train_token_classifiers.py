#!/usr/bin/env python3
"""Train teacher/student token classifiers on mined divergence-point data.

Supported phases:
- teacher: train on full teacher features [H_target, H_draft, H_small_base] with hard labels.
- student: train on student features [H_target, H_draft] with teacher distillation.

Supported architectures:
- logistic: single linear layer
- mlp: one hidden layer MLP

Inputs come from a single mined dataset prefix:
- <prefix>.features.npz
- <prefix>.info.json

The script supports optional delta_p-based sample weighting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)



def load_json(path: Path) -> Dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))



def save_json(path: Path, obj: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")



def parse_c_values(text: str) -> List[float]:
	vals = []
	for s in text.split(","):
		s = s.strip()
		if not s:
			continue
		vals.append(float(s))
	if not vals:
		raise ValueError("No valid C values parsed")
	return vals



def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
	if y_true.size == 0:
		return float("nan")
	pos = int(np.sum(y_true == 1))
	neg = int(np.sum(y_true == 0))
	if pos == 0 or neg == 0:
		return float("nan")

	order = np.argsort(y_score)
	ranks = np.empty_like(order, dtype=np.float64)
	ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)

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



def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
	pred = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.int64)
	return float(np.mean(pred == y_true)) if y_true.size else float("nan")



def stratified_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	idx_pos = np.where(y == 1)[0]
	idx_neg = np.where(y == 0)[0]
	rng.shuffle(idx_pos)
	rng.shuffle(idx_neg)

	def _split(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		n_val = max(1, int(round(arr.size * val_ratio))) if arr.size > 1 else arr.size
		val = arr[:n_val]
		train = arr[n_val:]
		if train.size == 0 and val.size > 0:
			train = val[:1]
			val = val[1:]
		return train, val

	train_pos, val_pos = _split(idx_pos)
	train_neg, val_neg = _split(idx_neg)
	train_idx = np.concatenate([train_pos, train_neg])
	val_idx = np.concatenate([val_pos, val_neg])
	rng.shuffle(train_idx)
	rng.shuffle(val_idx)
	return train_idx, val_idx



def compute_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	mean = X.mean(axis=0, keepdims=True)
	std = X.std(axis=0, keepdims=True)
	std = np.where(std < 1e-6, 1.0, std)
	return mean.astype(np.float32), std.astype(np.float32)



def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	return ((X - mean) / std).astype(np.float32)



def bernoulli_kl_from_probs(student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
	# teacher_probs expected already temperature-scaled if desired.
	student_probs = torch.sigmoid(student_logits / temperature)
	eps = 1e-6
	p = teacher_probs.clamp(eps, 1 - eps)
	q = student_probs.clamp(eps, 1 - eps)
	kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
	return kl * (temperature ** 2)



def build_sample_weights(
	delta_p: np.ndarray,
	mode: str,
	gamma: float,
	cap_percentile: float,
	normalize_mean_one: bool,
) -> np.ndarray:
	d = np.asarray(delta_p, dtype=np.float32)
	if mode == "none":
		w = np.ones_like(d, dtype=np.float32)
	elif mode == "relu":
		w = np.maximum(d, 0.0)
	elif mode == "one_plus_relu":
		w = 1.0 + gamma * np.maximum(d, 0.0)
	elif mode == "sigmoid":
		w = 1.0 + gamma * (1.0 / (1.0 + np.exp(-d)))
	else:
		raise ValueError(f"Unknown weight mode: {mode}")

	if cap_percentile > 0:
		cap = np.percentile(w, cap_percentile)
		w = np.minimum(w, cap)
	w = np.maximum(w, 1e-6)
	if normalize_mean_one and w.size:
		w = w / float(np.mean(w))
	return w.astype(np.float32)



def select_features(mode: str, X: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
	feature_slices = info.get("feature_slices", {})
	if mode == "teacher":
		return X.astype(np.float32), {"feature_mode": "teacher_full"}
	if mode != "student":
		raise ValueError(f"Unsupported mode: {mode}")

	student_slice = feature_slices.get("student_target_draft")
	if student_slice and len(student_slice) == 2:
		a0, a1 = int(student_slice[0]), int(student_slice[1])
		return X[:, a0:a1].astype(np.float32), {
			"feature_mode": "student_target_draft",
			"slice": [a0, a1],
		}

	# fallback: infer from hidden sizes
	hs = info.get("hidden_sizes", {})
	h_target = int(hs.get("target", 0) or 0)
	h_draft = int(hs.get("draft", 0) or 0)
	if h_target > 0 and h_draft > 0:
		return X[:, : h_target + h_draft].astype(np.float32), {
			"feature_mode": "student_target_draft_inferred",
			"slice": [0, h_target + h_draft],
		}

	raise ValueError("Could not determine student feature slice from info.json")


# -----------------------------
# Dataset / Models
# -----------------------------


class NumpyDataset(Dataset):
	def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
		self.X = torch.from_numpy(X).float()
		self.y = torch.from_numpy(y).float().view(-1, 1)
		self.w = torch.from_numpy(w).float().view(-1, 1)

	def __len__(self) -> int:
		return int(self.X.shape[0])

	def __getitem__(self, idx: int):
		return self.X[idx], self.y[idx], self.w[idx]


class LogisticHead(nn.Module):
	def __init__(self, input_dim: int):
		super().__init__()
		self.linear = nn.Linear(input_dim, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.linear(x)


class MLPHead(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)



def build_model(model_type: str, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
	if model_type == "logistic":
		return LogisticHead(input_dim)
	if model_type == "mlp":
		return MLPHead(input_dim, hidden_dim=hidden_dim, dropout=dropout)
	raise ValueError(f"Unsupported model_type: {model_type}")


@dataclass
class TrainMetrics:
	train_loss: float
	val_loss: float
	val_auc: float
	val_acc: float
	best_epoch: int
	C: float


# -----------------------------
# Training loops
# -----------------------------


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
	model.eval()
	outs: List[np.ndarray] = []
	for start in range(0, X.shape[0], batch_size):
		xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
		logits = model(xb).squeeze(-1).detach().cpu().numpy()
		outs.append(logits)
	return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)



def train_teacher_once(
	X_train: np.ndarray,
	y_train: np.ndarray,
	w_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	w_val: np.ndarray,
	model_type: str,
	input_dim: int,
	hidden_dim: int,
	dropout: float,
	lr: float,
	C: float,
	epochs: int,
	batch_size: int,
	device: torch.device,
) -> Tuple[nn.Module, TrainMetrics]:
	model = build_model(model_type, input_dim, hidden_dim, dropout).to(device)
	weight_decay = 0.0 if C <= 0 else (1.0 / C)
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	train_loader = DataLoader(NumpyDataset(X_train, y_train, w_train), batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(NumpyDataset(X_val, y_val, w_val), batch_size=batch_size, shuffle=False)

	best_state = None
	best = TrainMetrics(train_loss=float("inf"), val_loss=float("inf"), val_auc=float("-inf"), val_acc=float("nan"), best_epoch=-1, C=C)

	for epoch in range(1, epochs + 1):
		model.train()
		train_losses = []
		for xb, yb, wb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			wb = wb.to(device)
			logits = model(xb)
			loss_raw = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
			loss = (loss_raw * wb).mean()
			opt.zero_grad()
			loss.backward()
			opt.step()
			train_losses.append(float(loss.detach().cpu().item()))

		model.eval()
		val_losses = []
		val_logits_all = []
		val_y_all = []
		for xb, yb, wb in val_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			wb = wb.to(device)
			logits = model(xb)
			loss_raw = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
			loss = (loss_raw * wb).mean()
			val_losses.append(float(loss.detach().cpu().item()))
			val_logits_all.append(logits.squeeze(-1).detach().cpu().numpy())
			val_y_all.append(yb.squeeze(-1).detach().cpu().numpy())

		val_logits = np.concatenate(val_logits_all, axis=0) if val_logits_all else np.zeros((0,), dtype=np.float32)
		val_y = np.concatenate(val_y_all, axis=0).astype(np.int64) if val_y_all else np.zeros((0,), dtype=np.int64)
		val_auc = roc_auc_score_np(val_y, val_logits)
		val_acc = accuracy_from_logits(val_logits, val_y)
		metrics = TrainMetrics(
			train_loss=float(np.mean(train_losses)) if train_losses else float("nan"),
			val_loss=float(np.mean(val_losses)) if val_losses else float("nan"),
			val_auc=val_auc,
			val_acc=val_acc,
			best_epoch=epoch,
			C=C,
		)
		if math.isnan(best.val_auc) or (not math.isnan(val_auc) and val_auc > best.val_auc):
			best = metrics
			best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

	if best_state is not None:
		model.load_state_dict(best_state)
	return model, best



def train_student_once(
	X_train_student: np.ndarray,
	y_train: np.ndarray,
	w_train: np.ndarray,
	X_val_student: np.ndarray,
	y_val: np.ndarray,
	w_val: np.ndarray,
	teacher_probs_train: np.ndarray,
	teacher_probs_val: np.ndarray,
	model_type: str,
	input_dim: int,
	hidden_dim: int,
	dropout: float,
	lr: float,
	C: float,
	epochs: int,
	batch_size: int,
	temperature: float,
	distill_alpha: float,
	device: torch.device,
) -> Tuple[nn.Module, TrainMetrics]:
	model = build_model(model_type, input_dim, hidden_dim, dropout).to(device)
	weight_decay = 0.0 if C <= 0 else (1.0 / C)
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	class _StudentDataset(Dataset):
		def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, tprob: np.ndarray):
			self.X = torch.from_numpy(X).float()
			self.y = torch.from_numpy(y).float().view(-1, 1)
			self.w = torch.from_numpy(w).float().view(-1, 1)
			self.t = torch.from_numpy(tprob).float().view(-1, 1)

		def __len__(self) -> int:
			return int(self.X.shape[0])

		def __getitem__(self, idx: int):
			return self.X[idx], self.y[idx], self.w[idx], self.t[idx]

	train_loader = DataLoader(_StudentDataset(X_train_student, y_train, w_train, teacher_probs_train), batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(_StudentDataset(X_val_student, y_val, w_val, teacher_probs_val), batch_size=batch_size, shuffle=False)

	best_state = None
	best = TrainMetrics(train_loss=float("inf"), val_loss=float("inf"), val_auc=float("-inf"), val_acc=float("nan"), best_epoch=-1, C=C)

	for epoch in range(1, epochs + 1):
		model.train()
		train_losses = []
		for xb, yb, wb, tb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			wb = wb.to(device)
			tb = tb.to(device)
			logits = model(xb)
			hard_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
			soft_loss = bernoulli_kl_from_probs(logits, tb, temperature=temperature)
			loss = ((distill_alpha * hard_loss + (1.0 - distill_alpha) * soft_loss) * wb).mean()
			opt.zero_grad()
			loss.backward()
			opt.step()
			train_losses.append(float(loss.detach().cpu().item()))

		model.eval()
		val_losses = []
		val_logits_all = []
		val_y_all = []
		for xb, yb, wb, tb in val_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			wb = wb.to(device)
			tb = tb.to(device)
			logits = model(xb)
			hard_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
			soft_loss = bernoulli_kl_from_probs(logits, tb, temperature=temperature)
			loss = ((distill_alpha * hard_loss + (1.0 - distill_alpha) * soft_loss) * wb).mean()
			val_losses.append(float(loss.detach().cpu().item()))
			val_logits_all.append(logits.squeeze(-1).detach().cpu().numpy())
			val_y_all.append(yb.squeeze(-1).detach().cpu().numpy())

		val_logits = np.concatenate(val_logits_all, axis=0) if val_logits_all else np.zeros((0,), dtype=np.float32)
		val_y = np.concatenate(val_y_all, axis=0).astype(np.int64) if val_y_all else np.zeros((0,), dtype=np.int64)
		val_auc = roc_auc_score_np(val_y, val_logits)
		val_acc = accuracy_from_logits(val_logits, val_y)
		metrics = TrainMetrics(
			train_loss=float(np.mean(train_losses)) if train_losses else float("nan"),
			val_loss=float(np.mean(val_losses)) if val_losses else float("nan"),
			val_auc=val_auc,
			val_acc=val_acc,
			best_epoch=epoch,
			C=C,
		)
		if math.isnan(best.val_auc) or (not math.isnan(val_auc) and val_auc > best.val_auc):
			best = metrics
			best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

	if best_state is not None:
		model.load_state_dict(best_state)
	return model, best


# -----------------------------
# Main
# -----------------------------


def main() -> None:
	ap = argparse.ArgumentParser(description="Train teacher/student token classifiers on mined divergence data")
	ap.add_argument("--prefix", required=True, help="Mined data prefix, e.g. logs/mined_full_test_qwen_dp")
	ap.add_argument("--mode", choices=["teacher", "student"], required=True)
	ap.add_argument("--model_type", choices=["logistic", "mlp"], default="logistic")
	ap.add_argument("--teacher_ckpt", default=None, help="Required for --mode student")
	ap.add_argument("--save_dir", default="checkpoints/token_classifier", help="Directory to save checkpoints/metrics")
	ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--val_ratio", type=float, default=0.2)
	ap.add_argument("--batch_size", type=int, default=256)
	ap.add_argument("--epochs", type=int, default=20)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--hidden_dim", type=int, default=512)
	ap.add_argument("--dropout", type=float, default=0.1)
	ap.add_argument("--c_grid", default="1,0.3,0.1,0.03,0.01,0.003,0.001,0.0001")
	ap.add_argument("--weight_mode", choices=["none", "relu", "one_plus_relu", "sigmoid"], default="one_plus_relu")
	ap.add_argument("--weight_gamma", type=float, default=1.0)
	ap.add_argument("--weight_cap_percentile", type=float, default=99.0)
	ap.add_argument("--weight_normalize_mean_one", action="store_true")
	ap.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature for student")
	ap.add_argument("--distill_alpha", type=float, default=0.2, help="Blend hard-label BCE and soft-label KL in student")
	args = ap.parse_args()

	set_seed(args.seed)
	device = torch.device(args.device)

	prefix = Path(args.prefix)
	z = np.load(prefix.with_suffix(".features.npz"))
	info = load_json(prefix.with_suffix(".info.json"))
	X_full = np.asarray(z["X"], dtype=np.float32)
	y = np.asarray(z["y"], dtype=np.int64)
	delta_p = np.asarray(z["delta_p"], dtype=np.float32) if "delta_p" in z else np.zeros((y.shape[0],), dtype=np.float32)

	weights = build_sample_weights(
		delta_p=delta_p,
		mode=args.weight_mode,
		gamma=args.weight_gamma,
		cap_percentile=args.weight_cap_percentile,
		normalize_mean_one=args.weight_normalize_mean_one,
	)

	train_idx, val_idx = stratified_split_indices(y, val_ratio=args.val_ratio, seed=args.seed)

	X_selected, feature_meta = select_features(args.mode, X_full, info)
	X_train = X_selected[train_idx]
	X_val = X_selected[val_idx]
	y_train = y[train_idx]
	y_val = y[val_idx]
	w_train = weights[train_idx]
	w_val = weights[val_idx]

	mean, std = compute_standardizer(X_train)
	X_train_std = apply_standardizer(X_train, mean, std)
	X_val_std = apply_standardizer(X_val, mean, std)

	c_grid = parse_c_values(args.c_grid)
	save_dir = Path(args.save_dir)
	save_dir.mkdir(parents=True, exist_ok=True)

	results: List[Dict[str, Any]] = []
	best_model = None
	best_metrics = None
	best_c = None

	if args.mode == "teacher":
		for C in c_grid:
			model, metrics = train_teacher_once(
				X_train=X_train_std,
				y_train=y_train,
				w_train=w_train,
				X_val=X_val_std,
				y_val=y_val,
				w_val=w_val,
				model_type=args.model_type,
				input_dim=X_train_std.shape[1],
				hidden_dim=args.hidden_dim,
				dropout=args.dropout,
				lr=args.lr,
				C=C,
				epochs=args.epochs,
				batch_size=args.batch_size,
				device=device,
			)
			results.append(asdict(metrics))
			if best_metrics is None or (not math.isnan(metrics.val_auc) and metrics.val_auc > best_metrics.val_auc):
				best_metrics = metrics
				best_model = model
				best_c = C
	else:
		if not args.teacher_ckpt:
			raise ValueError("--teacher_ckpt is required for --mode student")
		teacher_blob = torch.load(args.teacher_ckpt, map_location=device)
		teacher_cfg = teacher_blob["config"]
		teacher_model = build_model(
			teacher_cfg["model_type"],
			input_dim=int(teacher_cfg["input_dim"]),
			hidden_dim=int(teacher_cfg.get("hidden_dim", 512)),
			dropout=float(teacher_cfg.get("dropout", 0.1)),
		).to(device)
		teacher_model.load_state_dict(teacher_blob["model_state"])
		teacher_model.eval()

		teacher_mean = np.asarray(teacher_blob["standardizer"]["mean"], dtype=np.float32)
		teacher_std = np.asarray(teacher_blob["standardizer"]["std"], dtype=np.float32)
		X_train_teacher = apply_standardizer(X_full[train_idx], teacher_mean, teacher_std)
		X_val_teacher = apply_standardizer(X_full[val_idx], teacher_mean, teacher_std)
		teacher_probs_train = 1.0 / (1.0 + np.exp(-predict_logits(teacher_model, X_train_teacher, args.batch_size, device) / args.temperature))
		teacher_probs_val = 1.0 / (1.0 + np.exp(-predict_logits(teacher_model, X_val_teacher, args.batch_size, device) / args.temperature))

		for C in c_grid:
			model, metrics = train_student_once(
				X_train_student=X_train_std,
				y_train=y_train,
				w_train=w_train,
				X_val_student=X_val_std,
				y_val=y_val,
				w_val=w_val,
				teacher_probs_train=teacher_probs_train.astype(np.float32),
				teacher_probs_val=teacher_probs_val.astype(np.float32),
				model_type=args.model_type,
				input_dim=X_train_std.shape[1],
				hidden_dim=args.hidden_dim,
				dropout=args.dropout,
				lr=args.lr,
				C=C,
				epochs=args.epochs,
				batch_size=args.batch_size,
				temperature=args.temperature,
				distill_alpha=args.distill_alpha,
				device=device,
			)
			results.append(asdict(metrics))
			if best_metrics is None or (not math.isnan(metrics.val_auc) and metrics.val_auc > best_metrics.val_auc):
				best_metrics = metrics
				best_model = model
				best_c = C

	if best_model is None or best_metrics is None:
		raise RuntimeError("Training failed to produce a best model")

	final_logits_val = predict_logits(best_model, X_val_std, args.batch_size, device)
	final_probs_val = 1.0 / (1.0 + np.exp(-final_logits_val))

	ckpt_name = f"{args.mode}_{args.model_type}_{prefix.name}.pt"
	ckpt_path = save_dir / ckpt_name
	metrics_path = save_dir / f"{args.mode}_{args.model_type}_{prefix.name}.metrics.json"

	torch.save(
		{
			"config": {
				"prefix": str(prefix),
				"mode": args.mode,
				"model_type": args.model_type,
				"input_dim": int(X_train_std.shape[1]),
				"hidden_dim": int(args.hidden_dim),
				"dropout": float(args.dropout),
				"best_C": float(best_c),
				"feature_meta": feature_meta,
				"weight_mode": args.weight_mode,
				"weight_gamma": float(args.weight_gamma),
				"weight_cap_percentile": float(args.weight_cap_percentile),
				"weight_normalize_mean_one": bool(args.weight_normalize_mean_one),
				"temperature": float(args.temperature),
				"distill_alpha": float(args.distill_alpha),
			},
			"model_state": best_model.state_dict(),
			"standardizer": {
				"mean": mean,
				"std": std,
			},
			"split": {
				"train_idx": train_idx,
				"val_idx": val_idx,
			},
		},
		ckpt_path,
	)

	metrics_obj = {
		"best": asdict(best_metrics),
		"all_grid_results": results,
		"val_summary": {
			"auc": roc_auc_score_np(y_val, final_logits_val),
			"acc": accuracy_from_logits(final_logits_val, y_val),
			"pos_rate": float(np.mean(y_val == 1)),
			"mean_prob": float(np.mean(final_probs_val)) if final_probs_val.size else float("nan"),
		},
		"config": {
			"prefix": str(prefix),
			"mode": args.mode,
			"model_type": args.model_type,
			"save_ckpt": str(ckpt_path),
			"val_ratio": float(args.val_ratio),
			"epochs": int(args.epochs),
			"lr": float(args.lr),
			"batch_size": int(args.batch_size),
			"c_grid": c_grid,
			"feature_meta": feature_meta,
			"weight_mode": args.weight_mode,
			"weight_gamma": float(args.weight_gamma),
			"weight_cap_percentile": float(args.weight_cap_percentile),
			"weight_normalize_mean_one": bool(args.weight_normalize_mean_one),
			"temperature": float(args.temperature),
			"distill_alpha": float(args.distill_alpha),
		},
	}
	save_json(metrics_path, metrics_obj)

	print(
		f"Done. mode={args.mode} model_type={args.model_type} "
		f"best_C={best_c} val_auc={metrics_obj['val_summary']['auc']:.4f} "
		f"val_acc={metrics_obj['val_summary']['acc']:.4f} ckpt={ckpt_path} metrics={metrics_path}"
	)


if __name__ == "__main__":
	main()
