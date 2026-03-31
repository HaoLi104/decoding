"""
KV Cache 管理器 — StaticCache 预分配 + Base/Draft 前缀共享 + Copy-on-Write

生命周期（每个 sample）：
  1. allocate()       预分配 target_cache + shared_small_cache
  2. Prefill 阶段      Base/Draft 共享 shared_cache（一次 forward 填充，零拷贝）
  3. fork_caches()    Decode 开始前 CoW 分叉，生成独立的 draft_cache / base_cache
  4. Decode 阶段      三个 cache 独立演进
  5. rollback_cache() Token 被拒绝时将 draft/base cache 回退到 reject_pos
  6. reset()          Sample 结束，重置所有 cache 状态

核心约束：
  - 所有 StaticCache 在 __init__ 时一次性预分配显存，避免动态扩展
  - Base 与 Draft 架构相同（均为 Qwen2.5-3B），Prefill 阶段物理共享同一 cache
  - CoW 分叉通过 torch.clone() 实现，分叉后各自独立写入
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import StaticCache

logger = logging.getLogger(__name__)


def _make_static_cache(
    config: PretrainedConfig,
    max_batch_size: int,
    max_cache_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> StaticCache:
    """创建并预分配一个 StaticCache。"""
    cache = StaticCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )
    return cache


class PrefixSharedCacheManager:
    """三模型 StaticCache 独立分配管理器。

    每个模型（Target / Draft / Base）拥有独立的 StaticCache，
    在 allocate() 时一次性预分配，避免 deepcopy 带来的巨大开销。

    属性（只读 property）：
        target_cache:  Target(32B) 的独立 StaticCache
        draft_cache:   Draft(3B-Medical) 的独立 StaticCache
        base_cache:    Base(3B) 的独立 StaticCache
    """

    def __init__(
        self,
        target_config: PretrainedConfig,
        small_config:  PretrainedConfig,   # Draft 和 Base 共享同一架构配置（3B 相同）
        max_batch_size: int,
        max_cache_len:  int,
        device:  torch.device,
        dtype:   torch.dtype,
    ) -> None:
        self._target_config  = target_config
        self._small_config   = small_config
        self._max_batch_size = max_batch_size
        self._max_cache_len  = max_cache_len
        self._device = device
        self._dtype  = dtype

        # 内部 cache 引用（allocate() 后初始化）
        self._target_cache: Optional[StaticCache] = None
        self._draft_cache:  Optional[StaticCache] = None
        self._base_cache:   Optional[StaticCache] = None

    # ------------------------------------------------------------------
    # 生命周期 API
    # ------------------------------------------------------------------

    def allocate(self) -> None:
        """预分配三个独立的 StaticCache（每个 sample 开始前调用一次）。

        三个模型各自持有独立的 StaticCache，无 deepcopy 操作。
        Draft 和 Base 的 Prefill 分别在各自独立的 cache 上执行。
        """
        self._target_cache = _make_static_cache(
            self._target_config,
            self._max_batch_size,
            self._max_cache_len,
            self._device,
            self._dtype,
        )
        self._draft_cache = _make_static_cache(
            self._small_config,
            self._max_batch_size,
            self._max_cache_len,
            self._device,
            self._dtype,
        )
        self._base_cache = _make_static_cache(
            self._small_config,
            self._max_batch_size,
            self._max_cache_len,
            self._device,
            self._dtype,
        )
        logger.debug("三模型 StaticCache 独立预分配完毕  max_cache_len=%d", self._max_cache_len)

    def rollback_cache(self, cache: StaticCache, to_seq_len: int) -> None:
        """将 cache 的有效填充长度回退到 to_seq_len。

        StaticCache 使用固定大小的预分配 buffer，通过重置内部 seen_tokens
        计数器实现逻辑回退。新版 transformers（>= 4.40）移除了该计数器，
        改由外部 cache_position 参数全权管理写入位置；此时通过清零超出范围的
        KV Buffer 实现等效的物理回退，防止注意力旁路泄漏。

        注意：ModelContext.seq_len 在 rollback() 中已同步更新，
        下次 decode_step/batch_verify 会以正确的 position_id 和 attention_mask
        覆盖 buffer 中的 stale 数据，因此本函数的清零是防御性操作。

        Args:
            cache:      需要回退的 StaticCache（draft_cache 或 base_cache）
            to_seq_len: 回退后的目标序列长度（= prompt_len + accepted_prefix_len）
        """
        if cache is None:
            raise RuntimeError("传入的 cache 为 None，无法回退")

        # 尝试旧版 transformers（< 4.40）内部计数器重置
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = to_seq_len
            logger.debug("cache 回退（_seen_tokens）至 seq_len=%d", to_seq_len)
            return
        if hasattr(cache, "seen_tokens"):
            cache.seen_tokens = to_seq_len
            logger.debug("cache 回退（seen_tokens）至 seq_len=%d", to_seq_len)
            return

        # 新版 transformers（>= 4.40）：StaticCache 不维护内部序列计数器
        # 通过清零超出 to_seq_len 的 KV Buffer 实现等效物理回退
        # key_cache / value_cache: List[Tensor]，每层一个 Tensor
        # 标准布局: [batch, num_kv_heads, max_cache_len, head_dim]（dim 2 = 序列维度）
        if hasattr(cache, "key_cache"):
            # key_cache 可能是 list 或 tuple，统一按序列处理
            kc = cache.key_cache
            vc = cache.value_cache
            if isinstance(kc, (list, tuple)) and len(kc) > 0:
                for k_t in kc:
                    if isinstance(k_t, torch.Tensor) and k_t.ndim == 4:
                        # 标准布局: [batch, num_kv_heads, max_cache_len, head_dim]
                        k_t[:, :, to_seq_len:, :].zero_()
                for v_t in vc:
                    if isinstance(v_t, torch.Tensor) and v_t.ndim == 4:
                        v_t[:, :, to_seq_len:, :].zero_()
                logger.debug(
                    "cache 回退（KV Buffer 清零）至 seq_len=%d，共 %d 层",
                    to_seq_len, len(kc),
                )
            else:
                logger.warning(
                    "StaticCache.key_cache 类型未知（%s），跳过物理清零，"
                    "依赖 cache_position 隔离（回退目标 seq_len=%d）",
                    type(kc).__name__, to_seq_len,
                )
        else:
            # 未知布局：仅依赖外部 seq_len 控制（forward_ops 通过 cache_position
            # 和 attention_mask 确保正确注意力范围，stale 数据不会被 attend）
            logger.warning(
                "StaticCache 无 key_cache 属性，跳过物理清零，依赖 cache_position 隔离 "
                "（回退目标 seq_len=%d）", to_seq_len
            )

    def reset(self) -> None:
        """重置所有 cache（下一个 sample 开始前调用）。

        直接丢弃旧 cache 引用，下次 allocate() 时重新分配。
        """
        self._target_cache = None
        self._draft_cache  = None
        self._base_cache   = None
        logger.debug("PrefixSharedCacheManager 重置完毕")

    # ------------------------------------------------------------------
    # 只读 property
    # ------------------------------------------------------------------

    @property
    def target_cache(self) -> StaticCache:
        if self._target_cache is None:
            raise RuntimeError("target_cache 未分配，请先调用 allocate()")
        return self._target_cache

    @property
    def draft_cache(self) -> StaticCache:
        if self._draft_cache is None:
            raise RuntimeError("draft_cache 未分配，请先调用 allocate()")
        return self._draft_cache

    @property
    def base_cache(self) -> StaticCache:
        if self._base_cache is None:
            raise RuntimeError("base_cache 未分配，请先调用 allocate()")
        return self._base_cache
