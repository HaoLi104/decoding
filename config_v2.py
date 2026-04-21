"""
配置管理模块 v2 — Qwen2.5 三模型对比置信度投机解码实验
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# 策略枚举
# ---------------------------------------------------------------------------

class StrategyType(Enum):
    STANDARD_SD       = "standard_sd"        # 策略 A：标准投机解码
    HARD_OVERRIDE_B0  = "hard_override_b0"   # 策略 B0：离散 argmax 对比强制放行
    HARD_OVERRIDE_B   = "hard_override_b"    # 策略 B：连续 ΔP 阈值硬覆盖
    SOFT_GUIDANCE_C1  = "soft_guidance_c1"   # 策略 C1：概率层加法（比值域补偿）
    SOFT_GUIDANCE_C2  = "soft_guidance_c2"   # 策略 C2：Logit 层 Z-score 残差注入
    SOFT_GUIDANCE_C3  = "soft_guidance_c3"   # 策略 C3：Target 概率局部校准（概率域直接补贴）
    SOFT_GUIDANCE_C4  = "soft_guidance_c4"   # 策略 C4：Draft 领域自信度动态门控（Confidence-Gated α）
    SOFT_GUIDANCE_C5  = "soft_guidance_c5"   # 策略 C5：Target 认知不确定性驱动路由（Entropy-Aware α）
    SOFT_GUIDANCE_C6  = "soft_guidance_c6"   # 策略 C6：双信号联合门控（C4 × C5，Draft 自信 AND Target 懵逼）
    SOFT_GUIDANCE_C7  = "soft_guidance_c7"   # 策略 C7：C3 框架 + C6 双信号动态 α（概率域直接补贴 + 联合门控）
    SOFT_GUIDANCE_C8  = "soft_guidance_c8"   # 策略 C8：C6 变体，门控信号 S_t 改为 token 级 ΔP(x)（消融全局 vs token 级门控）
    SOFT_GUIDANCE_C9  = "soft_guidance_c9"   # 策略 C9：二值 token 级门控 + 线性 ΔP（去掉 C8 的平方，α_t = λ·I(ΔP>τ)·H_t/H_max）
    SOFT_GUIDANCE_C10 = "soft_guidance_c10"  # 策略 C10：Logit 域 Product of Experts（logit_steered = logit_T + α·(logit_D - logit_B)，固定 α）
    SOFT_GUIDANCE_C11 = "soft_guidance_c11"  # 策略 C11：Logit 域 PoE + C9 二值 token 级门控 + 熵权（动态 α_t = λ·I(Δlogit>τ)·H_t/H_max）
    SOFT_GUIDANCE_C12 = "soft_guidance_c12"  # 策略 C12：C9 同构比值域验收 + 仅 x 处 logit 标量 bonus（无全词表注入）
    SOFT_GUIDANCE_C13 = "soft_guidance_c13"  # 策略 C13：局部 Logit PoE（只在 x 维做 PoE 注入，解析闭式验收）


class ExecutionArch(Enum):
    DUAL_STREAM   = "dual_stream"    # 架构 A：CUDA Stream 双流异步并发
    SHADOW_SYNC   = "shadow_sync"    # 架构 B：影子同步 + Lazy LM Head
    DEFERRED_BASE = "deferred_base"  # 架构 C：延迟 Base + CUDA Stream 并行（Base 在 Target 验证时并行执行）


# ---------------------------------------------------------------------------
# 模型路径
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPaths:
    """三模型路径配置（远端机器本地绝对路径）"""
    TARGET: str = "/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct"
    BASE:   str = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct"
    DRAFT:  str = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Surgery/checkpoint-1676"


# ---------------------------------------------------------------------------
# StaticCache 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheConfig:
    max_batch_size: int  = 1
    max_cache_len:  int  = 2048   # prompt + max_new_tokens 的最大序列长度
    device:         str  = "cuda:0"
    dtype:          str  = "bfloat16"


# ---------------------------------------------------------------------------
# 领域信号超参数
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainSignalParams:
    """ΔP 探针的超参数。

    t_fixed:    固定锐化温度，剥离全局采样温度干扰（论文 Section 4 Step 1）
    theta_high: Draft 最低置信度阈值 P_draft(x) > θ_high（Condition_Domain）
    tau:        ΔP 触发阈值 ΔP > τ（Condition_Domain）
    """
    t_fixed:    float = 1.0
    theta_high: float = 0.6
    tau:        float = 0.1


# ---------------------------------------------------------------------------
# 解码配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodeConfig:
    """单次实验的完整解码配置。"""
    strategy:        StrategyType    = StrategyType.STANDARD_SD
    arch:            ExecutionArch   = ExecutionArch.SHADOW_SYNC
    signal_params:   DomainSignalParams = DomainSignalParams()
    gamma:           int   = 5       # 投机窗口长度 K
    max_new_tokens:  int   = 256
    t_sample:        float = 0.0     # 全局采样温度（0=贪婪，0.6=随机采样）
    alpha:           float = 1.0     # C1/C2 软引导强度 α
    stop_on_eos:     bool  = True
    # C2 专用：注入变体与 Top-K 参数
    c2_variant:      str   = "full"  # "full" | "onehot" | "topk"
    c2_topk:         int   = 5       # topk 变体的 K 值
    # C4 专用：动态门控阈值 τ（信号强度 S_t = max(P_draft) - max(P_base) 的触发阈值）
    c4_tau:          float = 0.1     # 低于此阈值时 α_t = 0（稀疏激活）


# ---------------------------------------------------------------------------
# 硬件约束（单卡极致内聚）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareConfig:
    device:     str         = "cuda:0"
    dtype:      str         = "bfloat16"   # 全局严格使用 bfloat16
    device_map: str         = "cuda:0"     # 严格绑定，禁止 auto 跨卡
    compile_mode: Optional[str] = "reduce-overhead"  # torch.compile 模式；None 表示跳过编译


# ---------------------------------------------------------------------------
# 实验常量
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# 网格搜索候选值（run_benchmark.py 使用）
ALPHA_GRID:       list[float] = [0.1, 0.5, 1.0, 1.5, 2.0]
TEMPERATURE_GRID: list[float] = [0.0, 0.6]
