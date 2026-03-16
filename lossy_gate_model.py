#!/usr/bin/env python3
"""Runtime gate scorer for lossy speculative decoding.

Loads checkpoints produced by train_token_classifiers.py and returns accept
probability p_accept from online features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


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



def _build_model(model_type: str, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
	if model_type == "logistic":
		return LogisticHead(input_dim)
	if model_type == "mlp":
		return MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
	raise ValueError(f"Unsupported gate model_type: {model_type}")


@dataclass
class GateRuntime:
	model: nn.Module
	mean: np.ndarray
	std: np.ndarray
	input_dim: int
	device: torch.device
	config: Dict[str, Any]

	@classmethod
	def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "GateRuntime":
		blob = torch.load(ckpt_path, map_location=device)
		config = dict(blob.get("config", {}))
		model_type = str(config.get("model_type", "logistic"))
		input_dim = int(config.get("input_dim", 0) or 0)
		hidden_dim = int(config.get("hidden_dim", 512) or 512)
		dropout = float(config.get("dropout", 0.0) or 0.0)

		if input_dim <= 0:
			raise ValueError(f"Invalid input_dim in checkpoint: {input_dim}")

		model = _build_model(model_type, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
		state = blob.get("model_state")
		if state is None:
			raise ValueError("Checkpoint missing 'model_state'")
		model.load_state_dict(state)
		model.eval()
		model.to(device)

		std_obj = blob.get("standardizer", {})
		mean = np.asarray(std_obj.get("mean"), dtype=np.float32)
		std = np.asarray(std_obj.get("std"), dtype=np.float32)
		if mean.size == 0 or std.size == 0:
			raise ValueError("Checkpoint missing standardizer mean/std")

		mean = mean.reshape(1, -1)
		std = std.reshape(1, -1)
		if mean.shape[1] != input_dim:
			raise ValueError(
				f"standardizer dim mismatch: mean dim={mean.shape[1]} input_dim={input_dim}"
			)

		return cls(
			model=model,
			mean=mean,
			std=np.where(std < 1e-6, 1.0, std),
			input_dim=input_dim,
			device=torch.device(device),
			config=config,
		)

	def _transform(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float32).reshape(1, -1)
		if x.shape[1] != self.input_dim:
			raise ValueError(f"Gate feature dim mismatch: got {x.shape[1]}, expected {self.input_dim}")
		return ((x - self.mean) / self.std).astype(np.float32)

	@torch.no_grad()
	def predict_proba(self, x: np.ndarray) -> float:
		x_std = self._transform(x)
		x_t = torch.from_numpy(x_std).float().to(self.device)
		logit = self.model(x_t).squeeze(-1)
		prob = torch.sigmoid(logit).item()
		return float(prob)
