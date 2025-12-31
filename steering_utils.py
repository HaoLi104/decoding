"""
核心引导算法：自适应 alpha 计算与 logits 融合
"""

from typing import Literal

import torch
import torch.nn.functional as F

from config import HyperParams


def _compute_distance(
    probs_t: torch.Tensor,
    probs_e: torch.Tensor,
    metric: Literal["l2", "jsd"] = "l2",
) -> torch.Tensor:
    """计算分布距离，用于调节 alpha"""

    if metric == "jsd":
        m = 0.5 * (probs_t + probs_e)
        kl_te = F.kl_div(probs_t.log(), m, reduction="batchmean")
        kl_et = F.kl_div(probs_e.log(), m, reduction="batchmean")
        return 0.5 * (kl_te + kl_et)

    # 默认 L2
    return torch.norm(probs_t - probs_e, p=2, dim=-1)


def calculate_adaptive_alpha(
    logits_target: torch.Tensor,
    logits_expert: torch.Tensor,
) -> torch.Tensor:
    """根据 Target 与 Expert 的分歧动态计算引导强度 alpha"""

    probs_t = logits_target.softmax(dim=-1)
    probs_e = logits_expert.softmax(dim=-1)
    distance = _compute_distance(probs_t, probs_e, HyperParams.DISTANCE_METRIC)
    adaptive = torch.tanh(distance * HyperParams.ADAPTIVE_SCALE) * HyperParams.MAX_BOOST
    alpha = HyperParams.BASE_ALPHA + adaptive
    return alpha.detach()  # 防止梯度回传


def compute_steered_logits(
    logits_t: torch.Tensor,
    logits_e: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """融合三路 logits，生成最终引导后的 logits"""

    alpha = calculate_adaptive_alpha(logits_t, logits_e)
    steered = logits_t + alpha.unsqueeze(-1) * (logits_e - logits_b)
    return steered


