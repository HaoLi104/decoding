"""
领域知识探针信号模块

实现实验计划 Section 4 中的数学定义：
  Step 1：固定温度去敏 → 计算 ΔP
  Step 2：Condition_Domain 过滤伪信号
  辅助：Z-score 标准化 ΔLogit（策略 C2 专用）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1：ΔP 计算
# ---------------------------------------------------------------------------

def compute_delta_p(
    logit_draft: torch.Tensor,   # shape: [1, vocab_size]
    logit_base:  torch.Tensor,   # shape: [1, vocab_size]
    token_id:    int,
    t_fixed:     float = 1.0,
) -> tuple[float, float, float]:
    """计算候选 token x 的领域置信度差值 ΔP。

    数学定义（论文 Section 4 Step 1）：
        ΔP = Softmax(logit_draft / T_fixed)[x]
           - Softmax(logit_base  / T_fixed)[x]

    引入独立的固定温度 T_fixed（通常 = 1.0），与全局采样温度 T_sample 解耦，
    避免采样温度的平滑效果干扰领域差异信号的提取。

    Args:
        logit_draft: Draft 模型输出的原始 logit，shape [1, vocab_size]
        logit_base:  Base 模型输出的原始 logit，shape [1, vocab_size]
        token_id:    候选 token x 的 id
        t_fixed:     固定锐化温度（论文默认 1.0）

    Returns:
        (delta_p, p_draft_x, p_base_x)
          delta_p:   ΔP = p_draft_x - p_base_x，范围 (-1, 1)
          p_draft_x: Draft 在 T_fixed 温度下对 x 的概率
          p_base_x:  Base  在 T_fixed 温度下对 x 的概率
    """
    # 防御性检查：t_fixed 必须 > 0
    if t_fixed <= 0.0:
        raise ValueError(f"t_fixed 必须 > 0，当前值: {t_fixed}")

    # 计算固定温度下的概率分布，shape: [1, vocab_size]
    p_draft = F.softmax(logit_draft / t_fixed, dim=-1)  # shape: [1, vocab_size]
    p_base  = F.softmax(logit_base  / t_fixed, dim=-1)  # shape: [1, vocab_size]

    p_draft_x = float(p_draft[0, token_id].item())
    p_base_x  = float(p_base[0,  token_id].item())
    delta_p   = p_draft_x - p_base_x

    return delta_p, p_draft_x, p_base_x


# ---------------------------------------------------------------------------
# Step 2：Condition_Domain 过滤
# ---------------------------------------------------------------------------

def check_domain_condition(
    p_draft:    float,
    delta_p:    float,
    theta_high: float,
    tau:        float,
) -> bool:
    """判断候选 token x 是否满足领域知识触发条件（Condition_Domain）。

    触发条件（论文 Section 4 Step 2）：
        P_draft(x) > θ_high  AND  ΔP > τ

    双重门控：
      - P_draft(x) > θ_high：Draft 对该 token 具有高置信度（非低概率猜测）
      - ΔP > τ：Draft 相比 Base 具有显著的领域偏置（信号非伪）

    Args:
        p_draft:    Draft 在 T_fixed 温度下对候选 token 的概率（来自 compute_delta_p）
        delta_p:    ΔP = p_draft - p_base（来自 compute_delta_p）
        theta_high: Draft 置信度阈值（论文默认 0.6）
        tau:        ΔP 最小触发阈值（论文默认 0.1）

    Returns:
        True 表示该 token 被认定为「领域知识 token」，触发后续策略分支。
    """
    return (p_draft > theta_high) and (delta_p > tau)


# ---------------------------------------------------------------------------
# 辅助：Z-score 标准化 ΔLogit（策略 C2 专用）
# ---------------------------------------------------------------------------

def compute_delta_logit_normalized(
    logit_draft: torch.Tensor,   # shape: [1, vocab_size]
    logit_base:  torch.Tensor,   # shape: [1, vocab_size]
    eps:         float = 1e-8,
) -> torch.Tensor:
    """计算词表级别 Z-score 标准化的 Δlogit，用于策略 C2 的残差注入。

    数学定义（论文 Section 4 Step 3 变体 C2）：
        ΔLogit      = logit_draft - logit_base
        ΔLogit_norm = (ΔLogit - μ(ΔLogit)) / σ(ΔLogit)

    Z-score 标准化的目的：消除 32B Target 与 3B Draft/Base 在 logit 尺度上的
    天然量级差异（规模不同导致 logit 幅度不可直接比较），使残差注入的幅度
    与 α 超参数解耦，让 α 具有跨模型的可迁移语义。

    Args:
        logit_draft: Draft 输出 logit，shape [1, vocab_size]
        logit_base:  Base  输出 logit，shape [1, vocab_size]
        eps:         防止除零的小量（σ 极小时保护数值稳定性）

    Returns:
        delta_logit_norm: Z-score 标准化后的 ΔLogit，shape [1, vocab_size]
    """
    # ΔLogit，shape: [1, vocab_size]
    delta_logit = logit_draft - logit_base

    # 词表级别的均值和标准差，沿 vocab 维度计算
    mu    = delta_logit.mean(dim=-1, keepdim=True)   # shape: [1, 1]
    sigma = delta_logit.std(dim=-1,  keepdim=True)   # shape: [1, 1]

    # Z-score 标准化
    delta_logit_norm = (delta_logit - mu) / (sigma + eps)  # shape: [1, vocab_size]

    return delta_logit_norm


# ---------------------------------------------------------------------------
# 辅助：提取全分布概率（不针对特定 token）
# ---------------------------------------------------------------------------

def softmax_with_fixed_temp(
    logit: torch.Tensor,   # shape: [1, vocab_size]
    t_fixed: float = 1.0,
) -> torch.Tensor:
    """以固定温度对 logit 做 Softmax，返回完整概率分布。

    Args:
        logit:   原始 logit，shape [1, vocab_size]
        t_fixed: 固定锐化温度

    Returns:
        probs: shape [1, vocab_size]
    """
    return F.softmax(logit / t_fixed, dim=-1)
