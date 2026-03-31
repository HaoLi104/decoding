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

import copy
import logging
from typing import Optional, Tuple

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


def _clone_static_cache(src: StaticCache) -> StaticCache:
    """CoW 深拷贝：复制 StaticCache 中所有 KV 张量的数据。

    StaticCache 内部以 key_cache / value_cache 列表形式存储每层的 KV Tensor。
    此函数对每层 KV Tensor 执行 .clone()，实现物理独立的写时复制。

    Returns:
        与 src 等配置但 KV 数据完全独立的新 StaticCache 实例。
    """
    # 直接对整个对象做深拷贝（含 config 引用和 buffer Tensor）
    dst = copy.deepcopy(src)
    return dst


class PrefixSharedCacheManager:
    """Base 与 Draft 的 StaticCache 前缀共享与写时复制管理器。

    属性（只读 property）：
        target_cache:  Target(32B) 的独立 StaticCache
        shared_cache:  Prefill 阶段 Base/Draft 共用的 StaticCache
        draft_cache:   CoW 分叉后 Draft 独立的 StaticCache（fork 前为 None）
        base_cache:    CoW 分叉后 Base 独立的 StaticCache（fork 前为 None）
    """

    def __init__(
        self,
        target_config: PretrainedConfig,
        small_config:  PretrainedConfig,   # Base/Draft 共享同一架构配置
        max_batch_size: int,
        max_cache_len:  int,
        device:  torch.device,
        dtype:   torch.dtype,
    ) -> None:
        self._target_config = target_config
        self._small_config  = small_config
        self._max_batch_size = max_batch_size
        self._max_cache_len  = max_cache_len
        self._device = device
        self._dtype  = dtype

        # 内部 cache 引用（allocate() 后初始化）
        self._target_cache:  Optional[StaticCache] = None
        self._shared_cache:  Optional[StaticCache] = None
        self._draft_cache:   Optional[StaticCache] = None
        self._base_cache:    Optional[StaticCache] = None
        self._forked: bool = False

    # ------------------------------------------------------------------
    # 生命周期 API
    # ------------------------------------------------------------------

    def allocate(self) -> None:
        """预分配所有 StaticCache（每个 sample 开始前调用一次）。

        target_cache:  独立分配，配置来自 target 模型
        shared_cache:  Base/Draft 共用，配置来自 small 模型（3B 架构）
        draft/base_cache 在 fork_caches() 时才创建。
        """
        self._target_cache = _make_static_cache(
            self._target_config,
            self._max_batch_size,
            self._max_cache_len,
            self._device,
            self._dtype,
        )
        self._shared_cache = _make_static_cache(
            self._small_config,
            self._max_batch_size,
            self._max_cache_len,
            self._device,
            self._dtype,
        )
        self._draft_cache = None
        self._base_cache  = None
        self._forked = False
        logger.debug("StaticCache 预分配完毕  max_cache_len=%d", self._max_cache_len)

    def fork_caches(self) -> Tuple[StaticCache, StaticCache]:
        """CoW 分叉：将 shared_cache 深拷贝为独立的 draft_cache 和 base_cache。

        调用时机：Prefill 完成后、Decode 阶段第一步开始前。
        分叉后 shared_cache 不再被任何模型写入，仅供参考。

        Returns:
            (draft_cache, base_cache)
        """
        if self._shared_cache is None:
            raise RuntimeError("shared_cache 未分配，请先调用 allocate()")
        if self._forked:
            raise RuntimeError("fork_caches() 已调用过，禁止重复分叉")

        self._draft_cache = _clone_static_cache(self._shared_cache)
        self._base_cache  = _clone_static_cache(self._shared_cache)
        self._forked = True
        logger.debug("CoW 分叉完成  draft_cache 和 base_cache 已独立")
        return self._draft_cache, self._base_cache

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
        self._shared_cache = None
        self._draft_cache  = None
        self._base_cache   = None
        self._forked = False
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
    def shared_cache(self) -> StaticCache:
        """Prefill 阶段 Base/Draft 物理共享的 KV Buffer。"""
        if self._shared_cache is None:
            raise RuntimeError("shared_cache 未分配，请先调用 allocate()")
        return self._shared_cache

    @property
    def draft_cache(self) -> StaticCache:
        if self._draft_cache is None:
            raise RuntimeError("draft_cache 未初始化，请先调用 fork_caches()")
        return self._draft_cache

    @property
    def base_cache(self) -> StaticCache:
        if self._base_cache is None:
            raise RuntimeError("base_cache 未初始化，请先调用 fork_caches()")
        return self._base_cache

    @property
    def is_forked(self) -> bool:
        return self._forked
