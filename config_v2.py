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
    SOFT_GUIDANCE_C1  = "soft_guidance_c1"   # 策略 C1：概率层线性补偿
    SOFT_GUIDANCE_C2  = "soft_guidance_c2"   # 策略 C2：Logit 层 Z-score 残差注入


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
