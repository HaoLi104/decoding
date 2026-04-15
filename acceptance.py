"""
验收策略层 — 10 种策略的解耦实现 + 工厂函数

对应实验计划 Section 4 Step 3 的 Strategy Routing：
  StandardSD            策略 A：标准投机解码，概率比值验收
  OriginalHardOverride  策略 B0：离散 argmax 对比强制放行
  ThresholdHardOverride 策略 B：连续 ΔP 阈值硬覆盖
  SoftGuidanceC1        策略 C1：概率层加法（比值域补偿，固定 α）
  SoftGuidanceC2        策略 C2：Logit 层 Z-score 残差注入
  SoftGuidanceC3        策略 C3：Target 概率局部校准（概率域直接补贴）
  SoftGuidanceC4        策略 C4：Draft 领域自信度动态门控（Confidence-Gated α）
  SoftGuidanceC5        策略 C5：Target 认知不确定性驱动路由（Entropy-Aware α）
  SoftGuidanceC6        策略 C6：双信号联合门控（C4 × C5，Draft 自信 AND Target 懵逼）
  SoftGuidanceC7        策略 C7：C3 框架 + C6 双信号动态 α（概率域加法 + 联合门控，无 P_d 隐式乘积）

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

from config_v2 import DomainSignalParams, StrategyType  # noqa: F401 (StrategyType re-exported)
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
# 策略 C3：Soft Guidance（Target 概率局部校准）
# ---------------------------------------------------------------------------

class SoftGuidanceC3(AcceptanceStrategy):
    """策略 C3：Target 概率局部校准（Local Probability Calibration）。

    两步分解，量纲完全一致：

      Step 1 — 局部校准（在概率域注入领域补贴）：
          P'_target(x) = P_target(x) + α · ΔP
          物理意义：ΔP 作为"概率补贴（Probability Subsidy）"直接叠加到 Target 的原始概率上，
          形成融合了领域知识的虚拟 Target 概率 P'_target。

      Step 2 — 标准投机解码验收（回归经典框架）：
          P'_accept = min(1, P'_target(x) / P_draft(x))
          完全等价于用 P'_target 代替 P_target 走一遍 Standard SD。

    与 C1 的区别（量纲一致性）：
      C1: P'_accept = min(1, P_target/P_draft  +  α·ΔP)
          ← 在"比值"维度相加，α·ΔP 与 P_target/P_draft（无量纲比值）量纲不同
      C3: P'_accept = min(1, (P_target + α·ΔP) / P_draft)
          ← 在"概率"维度相加，分子分母均为概率 [0,1]，α·ΔP 量纲与 P_target 一致

    贪婪模式（t_sample=0）的确定性判据：
      接受当且仅当 P'_target(x) ≥ P_draft(x)，即 p_accept_prime ≥ 1.0。
      拒绝时回退到 argmax(logit_target)（与 StandardSD 贪婪一致）。
    """

    def __init__(self, alpha: float, signal_params: DomainSignalParams) -> None:
        self._alpha  = alpha
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id

        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c3_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # Step 1: 概率域局部校准 — P'_target(x) = P_target(x) + α · ΔP
        # p_target_prime 可能超过 1；min(1, ...) 在 Step 2 中由 p_accept_prime 完成
        p_target_prime = p_target + self._alpha * delta_p  # float，概率域线性叠加

        # Step 2: 标准 SD 验收 — P'_accept = min(1, P'_target / P_draft)
        p_accept_prime = min(1.0, p_target_prime / p_draft)

        # 贪婪模式（t_sample=0）：确定性接受判据
        if ctx.t_sample == 0.0:
            if p_accept_prime >= 1.0:
                return AcceptResult(
                    accepted=True,
                    chosen_token_id=x,
                    reason="c3_greedy_subsidy_accepted",
                    delta_p=delta_p,
                    p_draft=p_draft,
                    p_target=p_target_prime,
                )
            else:
                chosen = _argmax_token(ctx.logit_target)
                return AcceptResult(
                    accepted=False,
                    chosen_token_id=chosen,
                    reason="c3_greedy_subsidy_rejected",
                    delta_p=delta_p,
                    p_draft=p_draft,
                    p_target=p_target_prime,
                )

        # 随机采样模式（t_sample > 0）：标准概率比值随机验收
        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="c3_subsidy_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c3_subsidy_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )


# ---------------------------------------------------------------------------
# 策略 C4：Soft Guidance（Draft 领域自信度动态门控）
# ---------------------------------------------------------------------------

class SoftGuidanceC4(AcceptanceStrategy):
    """策略 C4：基于 Draft 领域自信度的动态门控（Confidence-Gated α）。

    核心直觉：只有当 Draft"大幅领先"Base 时（专家极度自信且分歧显著），
    才激活领域注入；否则退化为 StandardSD，保护 Target 的通用流利度。

    信号强度定义：
        S_t = max(P_draft) - max(P_base)

    动态 α 计算（稀疏激活）：
        α_t = α_base · I(S_t > τ) · S_t

    论文故事（Sparse Activation）：
        - 当 Draft 与 Base 的最高置信预测差值 S_t 超过阈值 τ 时，
          才触发领域辅助（稀疏）；α_t 与 S_t 成正比，信号越强注入越大。
        - S_t ≤ τ 时 α_t = 0，策略完全退化为 StandardSD，
          证明本框架不是"全程劫持"，而是精准的"领域急转弯响应"机制。

    完整验收公式（沿用 C1 的比值域加法框架）：
        P'_accept = min(1, P_target(x)/P_draft(x) + α_t · ΔP)

    超参数：
        alpha_base (--alpha): 信号强度放大系数 α_base（论文 λ 参数）
        c4_tau (--c4_tau):    激活阈值 τ（默认 0.1）
    """

    def __init__(
        self,
        alpha_base:    float,
        c4_tau:        float,
        signal_params: DomainSignalParams,
    ) -> None:
        self._alpha_base   = alpha_base
        self._tau          = c4_tau
        self._params       = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── 计算信号强度 S_t = max(P_draft) - max(P_base) ──────────────
        p_draft_full = F.softmax(ctx.logit_draft / t, dim=-1)  # shape: [1, V_draft]
        p_base_full  = F.softmax(ctx.logit_base  / t, dim=-1)  # shape: [1, V_base]
        s_t = float(p_draft_full.max().item()) - float(p_base_full.max().item())

        # ── 稀疏激活：动态 α ─────────────────────────────────────────────
        # α_t = α_base · I(S_t > τ) · S_t
        alpha_t = self._alpha_base * s_t if s_t > self._tau else 0.0

        # ── 计算 ΔP 与各模型对 draft token 的概率 ───────────────────────
        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c4_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # ── C1 公式 + 动态 α ─────────────────────────────────────────────
        # P'_accept = min(1, P_target/P_draft + α_t · ΔP)
        p_accept_prime = min(1.0, (p_target / p_draft) + alpha_t * delta_p)

        reason_tag = "gated" if alpha_t > 0.0 else "passthrough"

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason=f"c4_{reason_tag}_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason=f"c4_{reason_tag}_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 C5：Soft Guidance（Target 认知不确定性驱动路由）
# ---------------------------------------------------------------------------

class SoftGuidanceC5(AcceptanceStrategy):
    """策略 C5：基于 Target 认知不确定性的动态门控（Entropy-Aware α）。

    核心直觉：Target 的输出熵是其"自信程度"的天然量化器。
    熵低（尖锐分布）→ Target 确定，不需要外援；
    熵高（平坦分布）→ Target 面对知识盲区，允许 Draft 趁虚而入。

    香农熵计算：
        H_t = -∑ P_target(x) · log P_target(x)

    最大熵归一化（词表均匀分布时熵最大）：
        H_max = log(V)   （V = Target 词表大小）

    动态 α 计算：
        α_t = λ · H_t / H_max

    论文故事（Uncertainty-Driven Routing）：
        - Target 遇到通用语法/事实，熵低（确定），α_t → 0，Draft 闲置；
        - Target 遇到医学专有名词/临床推理，熵高（懵逼），α_t → λ，
          Draft 的领域先验被最大程度引入，实现"按需路由"。
        - λ（即 --alpha）控制最大注入强度上限。

    完整验收公式（沿用 C1 的比值域加法框架）：
        P'_accept = min(1, P_target(x)/P_draft(x) + α_t · ΔP)

    超参数：
        lambda (--alpha): 最大注入强度 λ（H_t/H_max=1 时退化为 C1 的 α）
    """

    def __init__(
        self,
        lam:           float,          # λ，对应 CLI --alpha
        signal_params: DomainSignalParams,
    ) -> None:
        self._lam    = lam
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── 计算 Target 输出分布的香农熵 H_t ────────────────────────────
        p_target_full = F.softmax(ctx.logit_target / t, dim=-1)  # shape: [1, V_target]
        # 数值稳定：log(p + ε) 避免 log(0)
        h_t = float(
            -torch.sum(p_target_full * torch.log(p_target_full + 1e-12)).item()
        )
        # H_max = log(V)，词表均匀时的理论最大熵
        h_max = math.log(p_target_full.shape[-1])  # ≈ log(152064) ≈ 11.93

        # ── 动态 α：α_t = λ · H_t / H_max ──────────────────────────────
        # H_t / H_max ∈ [0, 1]，α_t ∈ [0, λ]
        alpha_t = self._lam * (h_t / h_max)

        # ── 计算 ΔP 与各模型对 draft token 的概率 ───────────────────────
        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c5_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # ── C1 公式 + 熵驱动 α ──────────────────────────────────────────
        # P'_accept = min(1, P_target/P_draft + α_t · ΔP)
        p_accept_prime = min(1.0, (p_target / p_draft) + alpha_t * delta_p)

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="c5_entropy_routed_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c5_entropy_routed_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 C6：Soft Guidance（双信号联合门控：C4 × C5）
# ---------------------------------------------------------------------------

class SoftGuidanceC6(AcceptanceStrategy):
    """策略 C6：双信号联合门控（Dual-Signal Gated α）。

    C4 和 C5 的信号正交（一个来自 Draft-Base 外部对比，一个来自 Target 内部不确定性），
    将两者相乘得到联合动态 α：

        α_t = λ · I(S_t > τ) · S_t · (H_t / H_max)

    物理解读（双重 AND 门）：
      - S_t ≤ τ → α_t = 0：Draft 无显著领域优势，完全不注入（C4 门关闭）
      - H_t ≈ 0 → α_t ≈ 0：Target 对该 token 极度自信，无需外援（C5 权重接近 0）
      - S_t > τ AND H_t 高 → α_t = λ·S_t·(H_t/H_max)：Draft 自信 + Target 懵逼，
        "两个条件同时满足"才强力注入，最大程度减少误触发，精准攻击领域盲区。

    论文故事（Dual-Signal Routing）：
        C4 是"专家主动举手"（Draft 自信超过 Base），
        C5 是"学生主动求助"（Target 遇到知识盲区）。
        C6 只在两者同时成立时才触发领域引导，实现最精准的"按需注入"。

    完整验收公式（沿用 C1 的比值域加法框架）：
        P'_accept = min(1, P_target(x)/P_draft(x) + α_t · ΔP)

    超参数：
        lambda  (--alpha): 联合信号的最大注入强度 λ
        c4_tau  (--c4_tau): C4 门的激活阈值 τ（默认 0.1）
    """

    def __init__(
        self,
        lam:           float,          # λ，对应 CLI --alpha
        c4_tau:        float,          # τ，C4 稀疏激活阈值
        signal_params: DomainSignalParams,
    ) -> None:
        self._lam    = lam
        self._tau    = c4_tau
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── C4 信号：S_t = max(P_draft) - max(P_base) ────────────────────
        p_draft_full = F.softmax(ctx.logit_draft / t, dim=-1)  # [1, V_draft]
        p_base_full  = F.softmax(ctx.logit_base  / t, dim=-1)  # [1, V_base]
        s_t = float(p_draft_full.max().item()) - float(p_base_full.max().item())

        # C4 稀疏激活：S_t ≤ τ 时直接置零（Draft 无显著优势，关闭注入）
        c4_factor = s_t if s_t > self._tau else 0.0

        # ── C5 信号：H_t / H_max（Target 输出熵归一化） ───────────────────
        p_target_full = F.softmax(ctx.logit_target / t, dim=-1)  # [1, V_target]
        h_t = float(
            -torch.sum(p_target_full * torch.log(p_target_full + 1e-12)).item()
        )
        h_max = math.log(p_target_full.shape[-1])  # log(V) ≈ 11.93
        c5_factor = h_t / h_max  # ∈ [0, 1]

        # ── 联合动态 α：两信号相乘 ────────────────────────────────────────
        # α_t = λ · I(S_t > τ) · S_t · (H_t / H_max)
        alpha_t = self._lam * c4_factor * c5_factor

        # ── 计算 ΔP 与各模型对 draft token 的概率 ───────────────────────
        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c6_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # ── C1 框架：比值域加法 + 联合动态 α ─────────────────────────────
        # P'_accept = min(1, P_target/P_draft + α_t · ΔP)
        p_accept_prime = min(1.0, (p_target / p_draft) + alpha_t * delta_p)

        # 判断是否触发（用于 reason tag，方便遥测分析）
        dual_triggered = c4_factor > 0.0  # C4 门已开，C5 连续权重始终参与
        reason_tag = "dual_gated" if dual_triggered else "passthrough"

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason=f"c6_{reason_tag}_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason=f"c6_{reason_tag}_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 C7：C3 框架 + C6 双信号动态 α（概率域加法 + 联合门控）
# ---------------------------------------------------------------------------

class SoftGuidanceC7(AcceptanceStrategy):
    """策略 C7：C3 框架 + C6 双信号动态 α。

    动机：
        C6 在 C1 框架 (P_t/P_d + α_t·ΔP) 下等价于分子 P_t + α_t·ΔP·P_d，
        有效 bonus 是 α_t·ΔP·P_d，即在 C6 的双信号门控基础上又隐式地乘了 P_d(x)。
        由于 C4/C6 的 S_t 已经包含了对 Draft 置信度的全局度量，P_d(x) 再次参与
        构成了对 Draft 置信度的"双重加权"，不一定是最优设计。

        C7 将 C6 的双信号动态 α 迁移到 C3 框架（概率域直接加法），
        消除隐式 P_d 乘积，让 bonus 保持量纲一致的纯概率补贴：

            Step 1（局部校准）：P'_target(x) = P_target(x) + α_t · ΔP
            Step 2（标准验收）：P'_accept = min(1, P'_target(x) / P_draft(x))

        其中双信号动态 α_t = λ · I(S_t>τ) · S_t · H_t/H_max（与 C6 完全相同）。

    C7 vs C6 的唯一区别：
        C6: bonus = α_t · ΔP · P_d(x)   ← C1 框架，隐式 P_d 乘积
        C7: bonus = α_t · ΔP             ← C3 框架，纯概率域加法，无 P_d 乘积

    这使得 C7 是**消融 P_d 隐式乘积影响**的受控实验对照组。

    贪婪模式（t_sample=0）的确定性判据（继承 C3）：
        接受当且仅当 p_accept_prime >= 1.0，即 P'_target(x) >= P_draft(x)。
        拒绝时回退到 argmax(logit_target)。

    超参数：
        lambda  (--alpha): 联合信号最大注入强度 λ
        c4_tau  (--c4_tau): C4 稀疏激活阈值 τ（默认 0.1）
    """

    def __init__(
        self,
        lam:           float,
        c4_tau:        float,
        signal_params: DomainSignalParams,
    ) -> None:
        self._lam    = lam
        self._tau    = c4_tau
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── C4 信号：S_t = max(P_draft) - max(P_base)（全局 Draft 自信门）──
        p_draft_full = F.softmax(ctx.logit_draft / t, dim=-1)  # [1, V_draft]
        p_base_full  = F.softmax(ctx.logit_base  / t, dim=-1)  # [1, V_base]
        s_t = float(p_draft_full.max().item()) - float(p_base_full.max().item())
        c4_factor = s_t if s_t > self._tau else 0.0

        # ── C5 信号：H_t / H_max（Target 输出熵归一化）──────────────────────
        p_target_full = F.softmax(ctx.logit_target / t, dim=-1)  # [1, V_target]
        h_t = float(
            -torch.sum(p_target_full * torch.log(p_target_full + 1e-12)).item()
        )
        h_max = math.log(p_target_full.shape[-1])
        c5_factor = h_t / h_max  # ∈ [0, 1]

        # ── 联合动态 α（与 C6 相同）─────────────────────────────────────────
        # α_t = λ · I(S_t > τ) · S_t · H_t/H_max
        alpha_t = self._lam * c4_factor * c5_factor

        # ── 计算 ΔP 与各模型对 draft token 的概率 ───────────────────────────
        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c7_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # ── C3 框架：概率域直接加法，消除 P_d 隐式乘积 ──────────────────────
        # Step 1: P'_target(x) = P_target(x) + α_t · ΔP （纯概率域补贴）
        # Step 2: P'_accept = min(1, P'_target / P_draft)
        p_target_prime = p_target + alpha_t * delta_p
        p_accept_prime = min(1.0, p_target_prime / p_draft)

        # 贪婪模式（t_sample=0）：确定性接受（继承 C3 的判据）
        if ctx.t_sample == 0.0:
            if p_accept_prime >= 1.0:
                return AcceptResult(
                    accepted=True,
                    chosen_token_id=x,
                    reason="c7_greedy_dual_accepted",
                    delta_p=delta_p,
                    p_draft=p_draft,
                    p_target=p_target_prime,
                )
            else:
                chosen = _argmax_token(ctx.logit_target)
                return AcceptResult(
                    accepted=False,
                    chosen_token_id=chosen,
                    reason="c7_greedy_dual_rejected",
                    delta_p=delta_p,
                    p_draft=p_draft,
                    p_target=p_target_prime,
                )

        # 随机采样模式
        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="c7_dual_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c7_dual_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target_prime,
            )


# ---------------------------------------------------------------------------
# 策略 C8：C6 变体——门控信号改为 token 级 ΔP(x)（消融全局 vs token 级）
# ---------------------------------------------------------------------------

class SoftGuidanceC8(AcceptanceStrategy):
    """策略 C8：token 级双信号门控（Token-Level Confidence Gate）。

    动机与 C6 的唯一区别：
        C6 的门控信号 S_t = max(P_draft) - max(P_base) 是一个「步级（step-level）」
        全局信号——两个 max 可能来自不同的 token，衡量的是 Draft 整体上是否比 Base 更
        自信，与当前被验收的 token x 没有直接绑定。

        这会导致「误触发」：Draft 高置信在 token A 上，但我们在验收 token B，S_t 依然
        可能很高，从而对 B 施加了不该有的增强。

    C8 的修正：
        将门控信号改为「token 级（token-level）」：
            S_t(x) = P_draft(x) - P_base(x) = ΔP(x)

        即与 ΔP 使用完全相同的 token x，门控和 bonus 指向同一个问题：
            「Draft 对这个具体 token x，比 Base 更偏爱吗？」

    动态 α（token 级门控 + C5 熵权）：
        α_t = λ · I(ΔP(x) > τ) · ΔP(x) · (H_t / H_max)

    等价 bonus（在 C1 框架展开后）：
        bonus_C8 = α_t · ΔP(x) · P_d(x)
                 = λ · I(ΔP(x)>τ) · (ΔP(x))² · (H_t/H_max) · P_d(x)

    与 C6 的比较：
        C6: bonus = λ · I(S_t>τ) · S_t · ΔP(x) · (H_t/H_max) · P_d(x)   [S_t 和 ΔP 不同量]
        C8: bonus = λ · I(ΔP(x)>τ) · (ΔP(x))² · (H_t/H_max) · P_d(x)  [ΔP 出现两次，自放大]

    消融价值：
        - 若 C8 > C6：说明 token 级门控更精准，全局 S_t 引入了误触发噪音
        - 若 C8 < C6：说明全局 step-level S_t 携带了额外有用信息（Draft 全局自信≠ΔP 高）

    完整验收公式（C1 比值域加法框架）：
        P'_accept = min(1, P_target(x)/P_draft(x) + α_t · ΔP(x))

    超参数：
        lambda  (--alpha) : 最大注入强度 λ
        c4_tau  (--c4_tau): ΔP(x) 门控阈值 τ（默认 0.05，比 C6 的 0.1 略低，因为 ΔP 值域更小）
    """

    def __init__(
        self,
        lam:           float,
        c4_tau:        float,
        signal_params: DomainSignalParams,
    ) -> None:
        self._lam    = lam
        self._tau    = c4_tau
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── 计算 ΔP(x)：token 级领域信号（同时用于门控和 bonus）───────────
        # ΔP(x) = P_draft(x) - P_base(x)，在提案 token x 上计算
        # shape: logit_draft [1, V_draft], logit_base [1, V_base]
        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )

        # ── C8 token 级门控：I(ΔP(x) > τ) · ΔP(x) ───────────────────────
        # 仅当 Draft 对 token x 的优势超过阈值时才激活
        # （与 C6 的区别：C6 用 max(P_d)-max(P_b) 全局信号，C8 用 ΔP(x) token 信号）
        c8_factor = delta_p if delta_p > self._tau else 0.0   # ∈ [0, ~0.5]

        # ── C5 信号：H_t / H_max（Target 输出熵归一化） ───────────────────
        p_target_full = F.softmax(ctx.logit_target / t, dim=-1)  # [1, V_target]
        h_t = float(
            -torch.sum(p_target_full * torch.log(p_target_full + 1e-12)).item()
        )
        h_max     = math.log(p_target_full.shape[-1])   # log(V) ≈ 11.93
        c5_factor = h_t / h_max                          # ∈ [0, 1]

        # ── 联合动态 α（token 级双信号）────────────────────────────────────
        # α_t = λ · I(ΔP(x) > τ) · ΔP(x) · (H_t / H_max)
        # bonus 展开后 ∝ (ΔP(x))²，自放大效应比 C6 更锐利
        alpha_t = self._lam * c8_factor * c5_factor       # scalar

        # ── 计算 P_target(x) ──────────────────────────────────────────────
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c8_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # ── C1 框架：比值域加法 + token 级联合动态 α ─────────────────────
        # P'_accept = min(1, P_target(x)/P_draft(x) + α_t · ΔP(x))
        # shape: scalar
        p_accept_prime = min(1.0, (p_target / p_draft) + alpha_t * delta_p)

        token_gated = c8_factor > 0.0
        reason_tag  = "token_gated" if token_gated else "passthrough"

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason=f"c8_{reason_tag}_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason=f"c8_{reason_tag}_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 C9：二值 token 级门控 + 线性 ΔP（去掉 C8 的隐式平方）
# ---------------------------------------------------------------------------

class SoftGuidanceC9(AcceptanceStrategy):
    """策略 C9：二值 token 级门控 + 线性 ΔP（Binary Token-Level Gate + Linear ΔP）。

    动机：
        C8 将门控信号从步级 S_t 改为 token 级 ΔP(x)，带来精度提升，
        但由于 α_t 中已含 ΔP(x)，最终 bonus ∝ (ΔP(x))²（隐式平方）。
        平方放大是代数副产品，并非主动设计。

        C9 将门控改为"纯二值（binary）"：只用 I(ΔP(x) > τ) 决定是否激活，
        不再把 ΔP(x) 的大小带入 α_t，从而将 ΔP(x) 恢复为线性出现一次：

            α_t = λ · I(ΔP(x) > τ) · (H_t / H_max)

        bonus 展开后：
            bonus_C9 = α_t · ΔP(x)
                     = λ · I(ΔP(x) > τ) · ΔP(x) · (H_t / H_max) · P_d(x)

        对比 C8：
            bonus_C8 = λ · I(ΔP(x) > τ) · (ΔP(x))² · (H_t / H_max) · P_d(x)

        C9 vs C8 的唯一区别：bonus 中 ΔP(x) 的次数：C9 线性（一次），C8 平方（两次）。

    设计含义：
        - 门控：I(ΔP(x) > τ)         ——纯二值，判断"是否是领域词"
        - 幅度：ΔP(x) × H_t/H_max   ——领域优势 × Target 不确定性，两信号线性独立
        - 不存在因"用同一量计算门控和幅度"引起的自放大

    消融价值（C8 vs C9）：
        若 C9 > C8：说明线性 ΔP 已足够，平方放大过于激进（过注入）
        若 C9 < C8：说明平方放大的"超线性集中"本身有益，幂次比二值门控更精准

    完整验收公式（C1 比值域加法框架）：
        P'_accept = min(1, P_target(x)/P_draft(x) + α_t · ΔP(x))

    超参数：
        lambda  (--alpha) : 最大注入强度 λ
        c4_tau  (--c4_tau): ΔP(x) 激活阈值 τ（默认 0.05）
    """

    def __init__(
        self,
        lam:           float,
        c4_tau:        float,
        signal_params: DomainSignalParams,
    ) -> None:
        self._lam    = lam
        self._tau    = c4_tau
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── 计算 ΔP(x)：token 级领域信号 ─────────────────────────────────
        # ΔP(x) = P_draft(x) - P_base(x)  # shape: scalar
        delta_p, p_draft, _ = compute_delta_p(
            ctx.logit_draft, ctx.logit_base, x, self._params.t_fixed
        )

        # ── C9 二值 token 级门控：I(ΔP(x) > τ) ───────────────────────────
        # 仅判断是否激活（on/off），不把 ΔP(x) 的大小带入 α_t
        # 与 C8 的区别：C8 的 alpha_t 含 ΔP(x) → bonus ∝ (ΔP)²
        #               C9 的 alpha_t 不含 ΔP(x) → bonus ∝ ΔP（线性）
        gate = 1.0 if delta_p > self._tau else 0.0   # binary {0, 1}

        # ── C5 信号：H_t / H_max（Target 输出熵归一化）───────────────────
        p_target_full = F.softmax(ctx.logit_target / t, dim=-1)  # [1, V_target]
        h_t = float(
            -torch.sum(p_target_full * torch.log(p_target_full + 1e-12)).item()
        )
        h_max     = math.log(p_target_full.shape[-1])   # log(V) ≈ 11.93
        c5_factor = h_t / h_max                          # ∈ [0, 1]

        # ── 二值门控 × 熵权：α_t = λ · I(ΔP(x) > τ) · (H_t / H_max) ────
        # bonus = α_t · ΔP(x) = λ · I(ΔP(x)>τ) · ΔP(x) · (H_t/H_max)  [线性]
        alpha_t = self._lam * gate * c5_factor            # scalar

        # ── P_target(x) ──────────────────────────────────────────────────
        p_target = _prob_at(ctx.logit_target, x, self._params.t_fixed)

        if p_draft < 1e-12:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c9_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )

        # ── C1 框架：P'_accept = min(1, P_t/P_d + α_t · ΔP(x)) ──────────
        # bonus_C9 = α_t · ΔP(x) = λ · I(ΔP>τ) · ΔP(x) · H_t/H_max  [线性一次]
        p_accept_prime = min(1.0, (p_target / p_draft) + alpha_t * delta_p)

        reason_tag = "binary_gated" if gate > 0.0 else "passthrough"

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason=f"c9_{reason_tag}_accepted",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )
        else:
            chosen = _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason=f"c9_{reason_tag}_rejected",
                delta_p=delta_p,
                p_draft=p_draft,
                p_target=p_target,
            )


# ---------------------------------------------------------------------------
# 策略 C10：Logit 域 Product of Experts（最具理论基础的领域注入）
# ---------------------------------------------------------------------------

class SoftGuidanceC10(AcceptanceStrategy):
    """策略 C10：Logit 域 Product of Experts（PoE）注入。

    与 C1–C9 在概率域或比值域操作不同，C10 直接在 logit（对数概率）域做领域注入：

        logit_steered = logit_target + α · (logit_draft - logit_base)

    等价于（Softmax 后）：

        P_steered(x) ∝ P_target(x) · (P_draft(x) / P_base(x))^α

    这正是 Product of Experts（PoE / Bayesian 乘积更新）公式：
        - P_target       → 先验（通用大模型的通用知识）
        - P_draft/P_base → 领域似然比（专家相对常识基线的超额置信度）
        - α              → 似然更新强度

    物理可解释性：
        Δlogit(x) = logit_draft(x) - logit_base(x) = log(P_draft(x) / P_base(x))
        是领域似然比的对数——比概率差 ΔP(x) 更具理论基础（Bayes 更新的充分统计量）。
        注入后 P_steered 始终是合法的概率分布（Softmax 保证归一化，无负值问题）。

    与 C3 的区别：
        C3  在概率域做加法：P_steered = P_target + α · ΔP（相加后可能不归一化）
        C10 在 logit 域做加法：logit_steered = logit_target + α · Δlogit（相加后归一化天然满足）

    验收公式（C3 框架，直接用 P_steered 替换 P_target）：
        P'_accept = min(1, P_steered(x) / P_draft(x))

    拒绝时：
        - 贪婪模式（t_sample=0）：从 logit_steered 取 argmax
        - 随机模式：从 softmax(logit_steered / t_sample) 采样

    超参数：
        alpha (--alpha): 领域注入强度 α（建议搜索 0.1–2.0，量纲与 C1 类似）

    注意事项：
        - 需要三模型词表相同（Qwen2.5 系列均为 152064，满足）
        - 每步需做一次全词表向量加法（O(V)，V=152064），略贵于 C1 的标量操作
    """

    def __init__(self, alpha: float, signal_params: DomainSignalParams) -> None:
        self._alpha  = alpha
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── 取三模型 logit（在 t_fixed 温度缩放下） ───────────────────────
        # shape: [1, V]，所有 Qwen2.5 模型 V=152064
        logit_t = ctx.logit_target / t   # [1, V_target]
        logit_d = ctx.logit_draft  / t   # [1, V_draft]
        logit_b = ctx.logit_base   / t   # [1, V_base]

        # 词表大小一致性保证（截断到共同最小维度）
        v_min = min(logit_t.shape[-1], logit_d.shape[-1], logit_b.shape[-1])
        if x >= v_min:
            # 极端边界情况：token 超出最小词表，直接 passthrough
            chosen = _argmax_token(ctx.logit_target) if ctx.t_sample == 0.0 \
                     else _sample_token(ctx.logit_target, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c10_vocab_boundary",
                delta_p=0.0,
                p_draft=0.0,
                p_target=0.0,
            )
        logit_t = logit_t[..., :v_min]   # [1, v_min]
        logit_d = logit_d[..., :v_min]   # [1, v_min]
        logit_b = logit_b[..., :v_min]   # [1, v_min]

        # ── Logit 域 PoE 注入（全词表向量操作）──────────────────────────
        # Δlogit = logit_draft - logit_base = log(P_draft / P_base)  [1, v_min]
        # logit_steered = logit_target + α · Δlogit                  [1, v_min]
        # 等价于：P_steered(x) ∝ P_target(x) · (P_draft(x)/P_base(x))^α
        delta_logit   = logit_d - logit_b                         # [1, v_min]
        logit_steered = logit_t + self._alpha * delta_logit       # [1, v_min]

        # ── 从 steered logit 提取 P_steered(x)（acceptance 用）────────────
        p_steered_full = F.softmax(logit_steered, dim=-1)         # [1, v_min]
        p_steered_x    = float(p_steered_full[0, x].item())       # scalar

        # ── P_draft(x)（acceptance 分母） ────────────────────────────────
        p_draft_x = float(F.softmax(ctx.logit_draft / t, dim=-1)[0, x].item())

        # ── ΔP（仅遥测，不参与 C10 逻辑） ────────────────────────────────
        p_base_x  = float(F.softmax(ctx.logit_base  / t, dim=-1)[0, x].item())
        delta_p   = p_draft_x - p_base_x

        if p_draft_x < 1e-12:
            chosen = _argmax_token(ctx.logit_target[..., :v_min]) if ctx.t_sample == 0.0 \
                     else _sample_token(ctx.logit_target[..., :v_min], ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c10_draft_prob_zero",
                delta_p=delta_p,
                p_draft=p_draft_x,
                p_target=p_steered_x,
            )

        # ── 验收：P'_accept = min(1, P_steered(x) / P_draft(x)) ──────────
        # 使用 P_steered 替代原始 P_target，形式与 C3 相同，但分子来自 logit 域注入
        p_accept_prime = min(1.0, p_steered_x / p_draft_x)

        # ── 拒绝时从 logit_steered 采样（而非 logit_target）──────────────
        # 原始未缩放的 steered logit（供 _sample_token 使用）
        # logit_steered_raw = logit_target + α · (logit_draft - logit_base)（无温度缩放）
        logit_steered_raw = ctx.logit_target[..., :v_min] \
                            + self._alpha * (ctx.logit_draft[..., :v_min]
                                             - ctx.logit_base[..., :v_min])  # [1, v_min]

        if ctx.t_sample == 0.0:
            if p_accept_prime >= 1.0:
                return AcceptResult(
                    accepted=True,
                    chosen_token_id=x,
                    reason="c10_greedy_accepted",
                    delta_p=delta_p,
                    p_draft=p_draft_x,
                    p_target=p_steered_x,
                )
            else:
                chosen = _argmax_token(logit_steered_raw)
                return AcceptResult(
                    accepted=False,
                    chosen_token_id=chosen,
                    reason="c10_greedy_rejected",
                    delta_p=delta_p,
                    p_draft=p_draft_x,
                    p_target=p_steered_x,
                )

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True,
                chosen_token_id=x,
                reason="c10_accepted",
                delta_p=delta_p,
                p_draft=p_draft_x,
                p_target=p_steered_x,
            )
        else:
            chosen = _sample_token(logit_steered_raw, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c10_rejected",
                delta_p=delta_p,
                p_draft=p_draft_x,
                p_target=p_steered_x,
            )


# ---------------------------------------------------------------------------
# 策略 C11：Logit 域 PoE + C9 二值 token 级门控 + 熵权（动态 α）
# ---------------------------------------------------------------------------

class SoftGuidanceC11(AcceptanceStrategy):
    """策略 C11：Logit 域 PoE + 动态门控（C10 × C9 思想融合）。

    C10 在 logit 域做固定强度的全局 PoE 注入，等价于全程用同一 α 更新所有 token。
    C11 将 C9 的"二值 token 级门控 + Target 熵权"移植到 logit 域，使注入更精准：

        α_t = λ · I(Δlogit(x) > τ) · (H_t / H_max)

    其中门控信号从 C9 的 ΔP(x) 换为 **Δlogit(x) = logit_draft(x) - logit_base(x)**，
    即对数似然比。这使整个方法完全在 logit 域内自洽：

        信号 Δlogit ——用于门控判断"是否是领域词"（Δlogit > τ → log-ratio > τ → P_d/P_b > e^τ）
        信号 Δlogit ——用于注入幅度（logit_steered = logit_T + α_t · Δlogit）
        信号 H_t    ——Target 熵，决定"Target 有多困惑"

    等价 PoE 形式（Softmax 后）：

        P_steered(x) ∝ P_target(x) · (P_draft(x)/P_base(x))^{α_t}

    与 C10 的唯一区别：α 变为动态（token 级二值门控 + 熵权），未满足条件时 α_t=0，
    退化为纯 Target 贪婪——天然保护通用能力。

    与 C9 的唯一区别：注入域从概率域（ΔP 加法）变为 logit 域（Δlogit 加法）。

    消融价值：
        C11 vs C10：验证动态门控是否在 logit 域同样有益（减少误触发）
        C11 vs C9 ：验证 logit 域信号是否比概率域信号更精准（理论更优雅）

    完整流程：
        1. 计算 Δlogit(x) = logit_draft(x) - logit_base(x)
        2. 计算 H_t（Target 输出熵），得到 α_t = λ · I(Δlogit(x) > τ) · H_t/H_max
        3. logit_steered = logit_target + α_t · Δlogit  （全词表向量操作，仅当 α_t > 0）
        4. P'_accept = min(1, P_steered(x) / P_draft(x))

    超参数：
        lambda  (--alpha) : 最大注入强度 λ
        c4_tau  (--c4_tau): Δlogit(x) 门控阈值 τ（默认 0.1，对应 P_d/P_b > e^0.1 ≈ 1.1×）
    """

    def __init__(
        self,
        lam:           float,
        c4_tau:        float,
        signal_params: DomainSignalParams,
    ) -> None:
        self._lam    = lam
        self._tau    = c4_tau
        self._params = signal_params

    def evaluate(self, ctx: VerifyContext) -> AcceptResult:
        x = ctx.draft_token_id
        t = max(self._params.t_fixed, 1e-9)

        # ── 词表对齐 ──────────────────────────────────────────────────────
        v_min = min(
            ctx.logit_target.shape[-1],
            ctx.logit_draft.shape[-1],
            ctx.logit_base.shape[-1],
        )
        logit_t_raw = ctx.logit_target[..., :v_min]   # [1, v_min]（原始，无温度）
        logit_d_raw = ctx.logit_draft[..., :v_min]  # [1, v_min]
        logit_b_raw = ctx.logit_base  [..., :v_min]   # [1, v_min]

        if x >= v_min:
            chosen = _argmax_token(logit_t_raw) if ctx.t_sample == 0.0 \
                     else _sample_token(logit_t_raw, ctx.t_sample)
            return AcceptResult(
                accepted=False,
                chosen_token_id=chosen,
                reason="c11_vocab_boundary",
                delta_p=0.0, p_draft=0.0, p_target=0.0,
            )

        # ── Δlogit(x)：logit 域 token 级领域信号 ──────────────────────────
        # Δlogit(x) = logit_draft(x) - logit_base(x) = log(P_draft(x)/P_base(x))
        # 使用 t_fixed 温度缩放后的 logit（与其他策略保持一致）
        delta_logit_x = float((logit_d_raw[0, x] - logit_b_raw[0, x]).item()) / t   # scalar

        # ── 二值 token 级门控：I(Δlogit(x) > τ) ─────────────────────────
        # τ 作用于对数似然比域：Δlogit > τ ↔ P_d/P_b > e^τ（Draft 更偏爱 x）
        gate = 1.0 if delta_logit_x > self._tau else 0.0   # binary {0, 1}

        # ── C5 信号：H_t / H_max（Target 输出熵归一化） ───────────────────
        p_target_full = F.softmax(logit_t_raw / t, dim=-1)  # [1, v_min]
        h_t = float(
            -torch.sum(p_target_full * torch.log(p_target_full + 1e-12)).item()
        )
        h_max     = math.log(p_target_full.shape[-1])        # log(V)
        c5_factor = h_t / h_max                              # ∈ [0, 1]

        # ── 动态 α_t = λ · I(Δlogit > τ) · (H_t / H_max) ────────────────
        alpha_t = self._lam * gate * c5_factor               # scalar

        # ── Logit 域 PoE 注入（仅当 α_t > 0 时执行） ─────────────────────
        # logit_steered = logit_target + α_t · Δlogit  [1, v_min]
        if alpha_t > 0.0:
            delta_logit_full  = (logit_d_raw - logit_b_raw) / t   # [1, v_min]
            logit_steered_raw = logit_t_raw + alpha_t * (logit_d_raw - logit_b_raw)  # [1, v_min]（无温度缩放，供采样用）
            logit_steered_t   = logit_t_raw / t + alpha_t * delta_logit_full         # [1, v_min]（有温度缩放，供 P_steered 计算）
        else:
            logit_steered_raw = logit_t_raw
            logit_steered_t   = logit_t_raw / t

        # ── 提取 P_steered(x) 和 P_draft(x) ─────────────────────────────
        p_steered_full = F.softmax(logit_steered_t, dim=-1)         # [1, v_min]
        p_steered_x    = float(p_steered_full[0, x].item())         # scalar
        p_draft_x      = float(F.softmax(logit_d_raw / t, dim=-1)[0, x].item())  # scalar

        # ΔP（遥测用）
        p_base_x = float(F.softmax(logit_b_raw / t, dim=-1)[0, x].item())
        delta_p  = p_draft_x - p_base_x

        if p_draft_x < 1e-12:
            chosen = _argmax_token(logit_steered_raw) if ctx.t_sample == 0.0 \
                     else _sample_token(logit_steered_raw, ctx.t_sample)
            return AcceptResult(
                accepted=False, chosen_token_id=chosen,
                reason="c11_draft_prob_zero",
                delta_p=delta_p, p_draft=p_draft_x, p_target=p_steered_x,
            )

        # ── 验收：P'_accept = min(1, P_steered(x) / P_draft(x)) ──────────
        p_accept_prime = min(1.0, p_steered_x / p_draft_x)
        reason_tag = "gated" if gate > 0.0 else "passthrough"

        if ctx.t_sample == 0.0:
            if p_accept_prime >= 1.0:
                return AcceptResult(
                    accepted=True, chosen_token_id=x,
                    reason=f"c11_greedy_{reason_tag}_accepted",
                    delta_p=delta_p, p_draft=p_draft_x, p_target=p_steered_x,
                )
            else:
                chosen = _argmax_token(logit_steered_raw)
                return AcceptResult(
                    accepted=False, chosen_token_id=chosen,
                    reason=f"c11_greedy_{reason_tag}_rejected",
                    delta_p=delta_p, p_draft=p_draft_x, p_target=p_steered_x,
                )

        if torch.rand(1).item() < p_accept_prime:
            return AcceptResult(
                accepted=True, chosen_token_id=x,
                reason=f"c11_{reason_tag}_accepted",
                delta_p=delta_p, p_draft=p_draft_x, p_target=p_steered_x,
            )
        else:
            chosen = _sample_token(logit_steered_raw, ctx.t_sample)
            return AcceptResult(
                accepted=False, chosen_token_id=chosen,
                reason=f"c11_{reason_tag}_rejected",
                delta_p=delta_p, p_draft=p_draft_x, p_target=p_steered_x,
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
        alpha:         C1/C2/C3/C4/C5 软引导强度
        **kwargs:      策略专用扩展参数：
                         c2_variant (str):  C2 注入变体 full | onehot | topk（默认 full）
                         c2_topk    (int):  C2 topk 变体的 K 值（默认 5）
                         c4_tau     (float): C4 动态门控阈值 τ（默认 0.1）

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

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C3:
        return SoftGuidanceC3(alpha=alpha, signal_params=signal_params)

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C4:
        return SoftGuidanceC4(
            alpha_base=alpha,
            c4_tau=float(kwargs.get("c4_tau", 0.1)),
            signal_params=signal_params,
        )

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C5:
        return SoftGuidanceC5(lam=alpha, signal_params=signal_params)

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C6:
        return SoftGuidanceC6(
            lam=alpha,
            c4_tau=float(kwargs.get("c4_tau", 0.1)),
            signal_params=signal_params,
        )

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C7:
        return SoftGuidanceC7(
            lam=alpha,
            c4_tau=float(kwargs.get("c4_tau", 0.1)),
            signal_params=signal_params,
        )

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C8:
        return SoftGuidanceC8(
            lam=alpha,
            c4_tau=float(kwargs.get("c4_tau", 0.05)),
            signal_params=signal_params,
        )

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C9:
        return SoftGuidanceC9(
            lam=alpha,
            c4_tau=float(kwargs.get("c4_tau", 0.05)),
            signal_params=signal_params,
        )

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C10:
        return SoftGuidanceC10(
            alpha=alpha,
            signal_params=signal_params,
        )

    elif strategy_type == StrategyType.SOFT_GUIDANCE_C11:
        return SoftGuidanceC11(
            lam=alpha,
            # τ 作用于 Δlogit 域（默认 0.1 ≈ P_d/P_b > 1.1×）
            c4_tau=float(kwargs.get("c4_tau", 0.1)),
            signal_params=signal_params,
        )

    else:
        raise ValueError(f"未知策略类型: {strategy_type}")
