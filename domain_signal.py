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


def compute_delta_logit_masked(
    logit_draft:    torch.Tensor,   # shape: [1, vocab_size]
    logit_base:     torch.Tensor,   # shape: [1, vocab_size]
    draft_token_id: int,
    c2_variant:     str   = "full", # "full" | "onehot" | "topk"
    topk:           int   = 5,
    eps:            float = 1e-8,
) -> torch.Tensor:
    """计算带掩码的 Z-score ΔLogit_norm，策略 C2 三种注入变体的统一入口。

    三种变体的数学定义：

    ① full（全词表注入，原始 C2）：
        ΔLogit_masked = ΔLogit_norm           （无掩码，全词表）

    ② onehot（单 Token 定向提升）：
        M[i] = 1 if i == draft_token_id else 0
        ΔLogit_masked = ΔLogit_norm ⊙ M       （仅提升 draft 提案 Token）
        优点：消除全词表竞争，只定向抬高目标 Token 的 logit

    ③ topk（专家核心意见注入）：
        V_K = {top-K indices of logit_draft}
        M[i] = 1 if i ∈ V_K else 0
        ΔLogit_masked = ΔLogit_norm ⊙ M       （仅注入 Draft Top-K 的领域信号）
        优点：保留 Draft 的高置信 Expert Token 集合，过滤长尾噪音

    Args:
        logit_draft:    Draft 输出 logit，shape [1, vocab_size]
        logit_base:     Base  输出 logit，shape [1, vocab_size]
        draft_token_id: Draft 本步提案 token id（onehot 时使用）
        c2_variant:     注入模式：full | onehot | topk
        topk:           topk 模式下的 K 值（默认 5）
        eps:            Z-score 防零小量

    Returns:
        delta_logit_masked: 掩码后的 ΔLogit_norm，shape [1, vocab_size]
        （full 模式下与 compute_delta_logit_normalized 等价）
    """
    # 先计算 Z-score ΔLogit_norm，shape: [1, V]
    delta_logit      = logit_draft - logit_base
    mu               = delta_logit.mean(dim=-1, keepdim=True)   # [1, 1]
    sigma            = delta_logit.std(dim=-1,  keepdim=True)   # [1, 1]
    delta_logit_norm = (delta_logit - mu) / (sigma + eps)       # [1, V]

    if c2_variant == "full":
        return delta_logit_norm

    V    = delta_logit_norm.shape[-1]
    mask = torch.zeros_like(delta_logit_norm)   # [1, V]，全零掩码

    if c2_variant == "onehot":
        # 仅在 draft_token_id 位置置 1
        if 0 <= draft_token_id < V:
            mask[..., draft_token_id] = 1.0
        # draft_token_id 超出 vocab 范围（理论上不应出现）时 mask 全零，退化为零注入

    elif c2_variant == "topk":
        # Top-K Draft Token 的索引集合 V_K
        k_actual = min(topk, V)
        # logit_draft 可能与 delta_logit_norm vocab 大小相同
        _, top_indices = logit_draft[..., :V].topk(k_actual, dim=-1)  # [1, K]
        mask.scatter_(-1, top_indices, 1.0)                           # 置对应位置为 1

    else:
        raise ValueError(f"未知 c2_variant: {c2_variant}，可选: full | onehot | topk")

    return delta_logit_norm * mask   # [1, V]，非掩码位置为 0


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
