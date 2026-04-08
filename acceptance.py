"""
验收策略层 — 5 种策略的解耦实现 + 工厂函数

对应实验计划 Section 4 Step 3 的 Strategy Routing：
  StandardSD          策略 A：标准投机解码，概率比值验收
  OriginalHardOverride 策略 B0：离散 argmax 对比强制放行
  ThresholdHardOverride 策略 B：连续 ΔP 阈值硬覆盖
  SoftGuidanceC1      策略 C1：概率层线性补偿
  SoftGuidanceC2      策略 C2：Logit 层 Z-score 残差注入

设计原则：
  - 所有策略通过 VerifyContext 统一接受输入，通过 AcceptResult 统一输出
  - evaluate() 不依赖任何全局状态，可无副作用地并发调用
  - 拒绝时的 chosen_token_id 由策略内部决定（Standard/B0 选 target argmax；
    B/C1/C2 视验收结果而定）
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from config_v2 import DomainSignalParams, StrategyType
from domain_signal import (
    check_domain_condition,
    compute_delta_logit_masked,
    compute_delta_logit_normalized,
    compute_delta_p,
)


# ---------------------------------------------------------------------------
# 数据类：验证上下文 & 验收结果
# ---------------------------------------------------------------------------

@dataclass
class VerifyContext:
    """单个候选位置的完整验证上下文。

    由解码循环在每个 token 位置构造，传入验收策略。

    Attributes:
        draft_token_id: Draft 在该位置提案的 token id
        logit_target:   Target 模型对该位置的 next-token logits，shape [1, vocab_size]
        logit_draft:    Draft  模型对该位置的 next-token logits，shape [1, vocab_size]
        logit_base:     Base   模型对该位置的 next-token logits，shape [1, vocab_size]
        t_sample:       全局采样温度（0=贪婪；0.6=随机采样）
    """
    draft_token_id: int
    logit_target:   torch.Tensor   # shape: [1, vocab_size]
    logit_draft:    torch.Tensor   # shape: [1, vocab_size]
    logit_base:     torch.Tensor   # shape: [1, vocab_size]
    t_sample:       float


@dataclass
class AcceptResult:
    """验收策略的决策结果。

    Attributes:
        accepted:          是否接受 draft_token_id
        chosen_token_id:   最终写入序列的 token（接受时 = draft_token_id；
                           拒绝时 = target resample 结果）
        reason:            决策原因字符串（用于遥测日志）
        delta_p:           本次计算的 ΔP 值（无需计算时为 0.0）
        p_draft:           Draft 对 draft_token_id 的概率（T_fixed 温度下）
        p_target:          Target 对 draft_token_id 的概率（T_fixed 温度下）
    """
    accepted:        bool
    chosen_token_id: int
    reason:          str
    delta_p:         float = 0.0
    p_draft:         float = 0.0
    p_target:        float = 0.0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _argmax_token(logits: torch.Tensor) -> int:
    """贪婪取 argmax token id。"""
    return int(logits.argmax(dim=-1).item())


def _sample_token(logits: torch.Tensor, temperature: float) -> int:
    """从 logits 采样 token（temperature=0 退化为贪婪）。"""
    if temperature == 0.0:
        return _argmax_token(logits)
    probs = F.softmax(logits / temperature, dim=-1)  # shape: [1, vocab_size]
    return int(torch.multinomial(probs, num_samples=1).item())


def _prob_at(logits: torch.Tensor, token_id: int, temperature: float = 1.0) -> float:
    """计算 token_id 在给定温度下的归一化概率。"""
    probs = F.softmax(logits / max(temperature, 1e-9), dim=-1)
    return float(probs[0, token_id].item())


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class AcceptanceStrategy(ABC):
    """验收策略抽象基类。所有策略必须实现 evaluate()。"""

    @abstractmethod
    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        """对单个候选位置做验收决策。

        Args:
            ctx: 该位置的完整验证上下文

        Returns:
            AcceptResult（包含接受/拒绝决定和最终选定 token）
        """
        ...


# ---------------------------------------------------------------------------
# 策略 A：Standard SD
# ---------------------------------------------------------------------------

class StandardSD(AcceptanceStrategy):
    """策略 A：标准投机解码。

    验收概率：P_accept = min(1, P_target(x) / P_draft(x))

    在随机采样模式（t_sample > 0）下使用真正的概率比值验收；
    在贪婪模式（t_sample = 0）下退化为 argmax 匹配验证。

    若拒绝：以 t_sample 温度从修正分布 (P_target - P_draft)+ 中重采样矫正 token。
    """

    def __init__(self, t_fixed: float = 1.0) -> None:
        self._t_fixed = t_fixed

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id

        # 使用 t_fixed 计算 p_draft 和 p_target（与 ΔP 探针保持温度一致）
        p_target = _prob_at(ctx.logit_target, x, self._t_fixed)
        p_draft  = _prob_at(ctx.logit_draft,  x, self._t_fixed)

        if p_draft < 1e-12:
            # Draft 概率极小，直接拒绝
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="draft_prob_zero",
                p_draft=p_draft,
                p_target=p_target,
            )

        # greedy 模式（t_sample=0）：直接对比 argmax，保证与 target-only greedy 等价
        if ctx.t_sample == 0.0:
            target_top1 = _argmax_token(ctx.logit_target)
            if target_top1 == x:
                return AcceptResult(
                    accepted=True,
                    chosen_token_id=x,
                    reason="standard_greedy_accepted",
                    p_draft=p_draft,
                    p_target=p_target,
                )
            else:
                return AcceptResult(
                    accepted=False,
                    chosen_token_id=target_top1,
                    reason="standard_greedy_rejected",
                    p_draft=p_draft,
                    p_target=p_target,
                )

        p_accept = min(1.0, p_target / p_draft)

        # 随机验收
        if torch.rand(1).item() < p_accept:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="standard_accepted",
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            # 从修正分布采样矫正 token：max(0, P_target - P_draft)
            # greedy 模式（t_sample=0）直接取 target argmax，避免引入随机性
            if ctx.t_sample == 0.0:
                chosen = _argmax_token(ctx.logit_target)
            else:
                p_t_full = F.softmax(ctx.logit_target / max(self._t_fixed, 1e-9), dim=-1)  # [1, V_target]
                p_d_full = F.softmax(ctx.logit_draft  / max(self._t_fixed, 1e-9), dim=-1)  # [1, V_draft]
                # Target(32B) vocab=152064, Draft(3B) vocab=151936，取最小公共词表避免广播失败
                _v = min(p_t_full.shape[-1], p_d_full.shape[-1])
                corrected = F.relu(p_t_full[..., :_v] - p_d_full[..., :_v])
                corrected_sum = corrected.sum()
                if corrected_sum < 1e-12:
                    chosen = _argmax_token(ctx.logit_target)
                else:
                    corrected = corrected / corrected_sum
                    chosen = int(torch.multinomial(corrected, num_samples=1).item())

            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="standard_rejected_correction",
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 B0：Original Hard Override
# ---------------------------------------------------------------------------

class OriginalHardOverride(AcceptanceStrategy):
    """策略 B0：原设想硬覆盖（离散对比基线）。

    若 Target 拒绝（argmax 不匹配），且 argmax(draft) ≠ argmax(base)，
    则强制接受 draft_token_id（无视 ΔP 连续信号）。

    这是最极端的基线，探究"仅凭 Draft-Base Top-1 分歧"能否有效识别领域 token。
    """

    def __init__(self, t_fixed: float = 1.0) -> None:
        self._t_fixed = t_fixed

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        p_target = _prob_at(ctx.logit_target, x, self._t_fixed)
        p_draft  = _prob_at(ctx.logit_draft,  x, self._t_fixed)

        target_top1 = _argmax_token(ctx.logit_target)
        draft_top1  = _argmax_token(ctx.logit_draft)
        base_top1   = _argmax_token(ctx.logit_base)

        # 首先检查 Target 是否接受（argmax 匹配）
        if target_top1 == x:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="b0_target_match",
                p_draft=p_draft,
                p_target=p_target,
            )

        # Target 拒绝：检查 Draft-Base Top-1 分歧
        if draft_top1 != base_top1:
            # 强制放行：Draft 与 Base 分歧，认为该 token 携带领域知识
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="b0_draft_base_divergence_override",
                p_draft=p_draft,
                p_target=p_target,
            )

        # 不满足分歧条件，回退 target argmax
        chosen = _sample_token(ctx.logit_target, ctx.t_sample)
        return AcceptResult(
            accepted=False,
            chosen_token_id=chosen,
            reason="b0_no_divergence_rejected",
            p_draft=p_draft,
            p_target=p_target,
        )


# ---------------------------------------------------------------------------
# 策略 B：Threshold Hard Override
# ---------------------------------------------------------------------------

class ThresholdHardOverride(AcceptanceStrategy):
    """策略 B：连续阈值硬覆盖。

    若 Target 拒绝（argmax 不匹配），且满足 Condition_Domain：
        P_draft(x) > θ_high  AND  ΔP > τ
    则强制接受 draft_token_id。

    相比 B0，使用连续 ΔP 阈值，过滤掉 Draft 虽与 Base 分歧但置信度低的伪信号。
    """

    def __init__(self, signal_params: DomainSignalParams) -> None:
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id

        delta_p, p_draft, p_base = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        target_top1 = _argmax_token(ctx.logit_target)

        # Target 接受（argmax 匹配）
        if target_top1 == x:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="b_target_match",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # Target 拒绝 → 检查 Condition_Domain
        domain_triggered = check_domain_condition(
            p_draft=p_draft,
            delta_p=delta_p,
            theta_high=self._params.theta_high,
            tau=self._params.tau,
        )

        if domain_triggered:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="b_domain_threshold_override",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        chosen = _sample_token(ctx.logit_target, ctx.t_sample)
        return AcceptResult(
            accepted=False,
            chosen_token_id=chosen,
            reason="b_domain_threshold_rejected",
            delta_p=delta_p,
            p_draft=p_draft,
            p_target=p_target,
        )


# ---------------------------------------------------------------------------
# 策略 C1：Soft Guidance（概率层线性补偿）
# ---------------------------------------------------------------------------

class SoftGuidanceC1(AcceptanceStrategy):
    """策略 C1：概率层线性补偿。

    数学定义：
        P'_accept = min(1, P_target(x) / P_draft(x) + α · ΔP)

    将 ΔP 作为软引导信号线性叠加到标准验收概率上，
    使领域知识 token 获得更高的验收概率，而非硬性覆盖。
    """

    def __init__(self, alpha: float, signal_params: DomainSignalParams) -> None:
        self._alpha  = alpha
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id

        delta_p, p_draft, p_base = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c1_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # P'_accept = min(1, P_target / P_draft + α · ΔP)
        p_accept_prime = min(1.0, (p_target / p_draft) + self._alpha * delta_p)

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="c1_soft_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c1_soft_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 C2：Soft Guidance（Logit 层 Z-score 残差注入）
# ---------------------------------------------------------------------------

class SoftGuidanceC2(AcceptanceStrategy):
    """策略 C2：Logit 层标准化残差注入（支持三种注入变体）。

    数学定义：
        ΔLogit      = logit_draft - logit_base
        ΔLogit_norm = (ΔLogit - μ) / σ           ← Z-score 标准化
        ΔLogit_masked = ΔLogit_norm ⊙ M(variant)  ← 按变体施加掩码
        Logit'_target = logit_target + α · ΔLogit_masked
        P'_target(x) = Softmax(Logit'_target)[x]
        P_accept     = min(1, P'_target(x) / P_draft(x))

    variant 控制注入范围（通过 --c2_variant 命令行参数设置）：
        "full"   → 全词表注入（原始 C2，无掩码）
        "onehot" → 仅对 draft 提案 token x 定向注入（one-hot 掩码）
        "topk"   → 仅注入 Draft Top-K Token 的领域信号，过滤长尾噪音

    Z-score 消除不同规模模型 logit 尺度差异，使 α 具有跨模型可迁移语义。
    """

    def __init__(
        self,
        alpha:      float,
        t_fixed:    float = 1.0,
        c2_variant: str   = "full",  # "full" | "onehot" | "topk"
        c2_topk:    int   = 5,
    ) -> None:
        self._alpha      = alpha
        self._t_fixed    = t_fixed
        self._c2_variant = c2_variant
        self._c2_topk    = c2_topk

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id

        # 计算带掩码的 Z-score ΔLogit_norm，shape: [1, vocab_size]
        # variant 决定哪些位置非零：full=全词表, onehot=仅x, topk=Draft Top-K
        delta_logit_norm = compute_delta_logit_masked(
            ctx.logit_draft, ctx.logit_base,
            draft_token_id=x,
            c2_variant=self._c2_variant,
            topk=self._c2_topk,
        )

        # 残差注入：Logit'_target = logit_target + α · ΔLogit_norm
        # Target(32B) vocab 与 draft/base(3B) vocab 可能不同，截断对齐后再相加
        _v2 = min(ctx.logit_target.shape[-1], delta_logit_norm.shape[-1])
        logit_target_prime = ctx.logit_target[..., :_v2] + self._alpha * delta_logit_norm[..., :_v2]  # [1, _v2]

        # 基于修正后的 Target logit 计算概率
        p_target_prime = _prob_at(logit_target_prime, x, self._t_fixed)
        p_draft        = _prob_at(ctx.logit_draft,    x, self._t_fixed)

        # 用于遥测的原始 delta_p
        delta_p, _, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._t_fixed
        )

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c2_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )

        # 贪婪模式（t_sample=0）：检查修正后 logit 的 argmax 是否等于 draft token
        # 这是 C2 的核心：若 alpha 足够大翻转了 Target 的 argmax，则接受该 token
        if ctx.t_sample == 0.0:
            target_prime_top1 = _argmax_token(logit_target_prime)
            if target_prime_top1 == x:
                return AcceptResult(
                    accepted=True,
                    chosen_token_id=x,
                    reason="c2_greedy_argmax_flipped_accepted",
                    delta_p=delta_p,
                    p_draft=p_draft,
                    p_target=p_target_prime,
                )
            else:
                return AcceptResult(
                    accepted=False,
                    chosen_token_id=target_prime_top1,
                    reason="c2_greedy_argmax_not_flipped_rejected",
                    delta_p=delta_p,
                    p_draft=p_draft,
                    p_target=p_target_prime,
                )

        # 随机采样模式（t_sample > 0）：标准概率比值验收
        p_accept = min(1.0, p_target_prime / p_draft)

        if torch.rand(1).item() < p_accept:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="c2_logit_injection_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )
        else:
            # 从修正 logit 的概率分布中采样矫正 token
            p_tp_full = F.softmax(logit_target_prime / max(self._t_fixed, 1e-9), dim=-1)  # [1, _v2]
            p_d_full  = F.softmax(ctx.logit_draft    / max(self._t_fixed, 1e-9), dim=-1)  # [1, V_draft]
            # 已在 logit_target_prime 截断为 _v2，再对 p_d_full 对齐
            _v3 = min(p_tp_full.shape[-1], p_d_full.shape[-1])
            corrected = F.relu(p_tp_full[..., :_v3] - p_d_full[..., :_v3])
            corrected_sum = corrected.sum()
            if corrected_sum < 1e-12:
                chosen = _argmax_token(logit_target_prime)
            else:
                corrected = corrected / corrected_sum
                chosen = int(torch.multinomial(corrected, num_samples=1).item())

            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c2_logit_injection_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_strategy(
    strategy_type:  StrategyType,
    signal_params:  DomainSignalParams = DomainSignalParams(),
    alpha:          float = 1.0,
    **kwargs,
) -> AcceptanceStrategy:
    """根据 StrategyType 枚举实例化对应的验收策略。

    Args:
        strategy_type: 策略枚举值
        signal_params: 领域信号超参数（θ_high, τ, T_fixed）
        alpha:         C1/C2 软引导强度
        **kwargs:      C2 专用扩展参数：
                         c2_variant (str):  注入变体 full | onehot | topk（默认 full）
                         c2_topk    (int):  topk 变体的 K 值（默认 5）

    Returns:
        AcceptanceStrategy 子类实例
    """
    t_fixed = signal_params.t_fixed

    if strategy_type == StrategyType.STANDARD_SD:
        return StandardSD(t_fixed=t_fixed)

    elif strategy_type == StrategyType.HARD_OVERRIDE_B0:
        return OriginalHardOverride(t_fixed=t_fixed)

    elif strategy_type == StrategyType.HARD_OVERRIDE_B:
        return ThresholdHardOverride(signal_params=signal_params)

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C1:
        return SoftGuidanceC1(alpha=alpha, signal_params=signal_params)

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C2:
        return SoftGuidanceC2(
            alpha=alpha,
            t_fixed=t_fixed,
            c2_variant=kwargs.get("c2_variant", "full"),
            c2_topk=int(kwargs.get("c2_topk", 5)),
        )

    else:
        raise ValueError(f"未知策略类型: {strategy_type}")
