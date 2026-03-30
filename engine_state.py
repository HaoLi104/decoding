"""
三模型协同状态管理层

ModelContext：单个模型的运行时上下文（模型引用 + StaticCache + 位置计数器 + 最新 logits）
TriModelOrchestrator：编排 Target / Draft / Base 三模型的 Prefill、同步、回退等生命周期操作

设计原则：
  - ModelContext 不负责 cache 的分配/分叉，只持有 cache 引用
  - cache 的生命周期由 PrefixSharedCacheManager 统一管理
  - TriModelOrchestrator 作为唯一的三模型协调入口，外层解码循环不直接操作 ModelContext
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

from cache_manager import PrefixSharedCacheManager
from forward_ops import (
    decode_batch_verify,
    decode_step,
    prefill,
    sample_token,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelContext：单个模型的运行时状态封装
# ---------------------------------------------------------------------------

class ModelContext:
    """单个模型的运行时上下文。

    封装：
      - model:       AutoModelForCausalLM 实例（已 eval + compile）
      - cache:       StaticCache 引用（生命周期由 PrefixSharedCacheManager 管理）
      - seq_len:     当前序列总长度（prompt_len + 已生成并接受的 token 数）
      - last_logits: 最近一次 forward 返回的 next-token logits，shape [1, vocab_size]
      - device:      模型所在设备
    """

    def __init__(
        self,
        model:       AutoModelForCausalLM,
        cache:       StaticCache,
        seq_len:     int,
        last_logits: torch.Tensor,
        device:      torch.device,
    ) -> None:
        self.model       = model
        self.cache       = cache
        self.seq_len     = seq_len
        self.last_logits = last_logits  # shape: [1, vocab_size]
        self.device      = device

    # ------------------------------------------------------------------
    # 单步推进
    # ------------------------------------------------------------------

    def step(self, token_id: int) -> None:
        """将单个已接受 token 推进到 cache，更新 seq_len 和 last_logits。

        Args:
            token_id: 已接受的 token（int）
        """
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        # position_id = 当前 seq_len（下一个写入位置）
        self.last_logits = decode_step(
            model=self.model,
            token_id=token_tensor,
            cache=self.cache,
            position_id=self.seq_len,
        )  # shape: [1, vocab_size]
        self.seq_len += 1

    # ------------------------------------------------------------------
    # 批量推进（用于追赶多个已接受 token）
    # ------------------------------------------------------------------

    def advance(self, token_ids: List[int]) -> None:
        """批量将多个已接受 token 推进到 cache。

        一次 forward 处理 k 个 token，效率高于逐步调用 step()。
        更新 seq_len 和 last_logits（取批次最后一个位置的 logits）。

        Args:
            token_ids: 已接受的 token 列表（不含当前轮拒绝/矫正 token）
        """
        if not token_ids:
            return
        k = len(token_ids)
        ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        # shape: [1, k, vocab_size]
        logits_batch = decode_batch_verify(
            model=self.model,
            token_ids=ids_tensor,
            cache=self.cache,
            start_position=self.seq_len,
        )
        self.last_logits = logits_batch[:, -1, :]  # shape: [1, vocab_size]
        self.seq_len += k

    # ------------------------------------------------------------------
    # 回退（Token 被拒绝时将 cache 逻辑回退）
    # ------------------------------------------------------------------

    def rollback(self, cache_mgr: PrefixSharedCacheManager, to_seq_len: int) -> None:
        """将该模型的 cache 回退到 to_seq_len 位置。

        Args:
            cache_mgr:  PrefixSharedCacheManager（执行实际回退操作）
            to_seq_len: 目标序列长度（= prompt_len + accepted_prefix_len）
        """
        cache_mgr.rollback_cache(self.cache, to_seq_len)
        self.seq_len = to_seq_len
        # last_logits 在回退后将在下次 forward 时被刷新，此处标记为无效
        self.last_logits = torch.zeros_like(self.last_logits)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def clone_logits(self) -> torch.Tensor:
        """返回 last_logits 的 detach 副本，防止后续 forward 覆盖引用。

        Returns:
            shape: [1, vocab_size]
        """
        return self.last_logits.detach().clone()


# ---------------------------------------------------------------------------
# TriModelOrchestrator：三模型协同编排器
# ---------------------------------------------------------------------------

class TriModelOrchestrator:
    """编排 Target / Draft / Base 三模型的协同生命周期。

    对外接口：
      - init_from_prompt():     三模型共同 Prefill（Base/Draft 共享 cache → CoW 分叉）
      - sync_accepted():        将已接受前缀同步到 Draft/Base（Target 已在 verify 时推进）
      - sync_on_correction():   拒绝矫正：三模型回退到 reject_pos，以矫正 token 重新推进
      - reset():                重置为下一个 sample
      - target_ctx / draft_ctx / base_ctx: 只读 property，供 Proposer/Verifier 访问 logits
    """

    def __init__(
        self,
        target_model: AutoModelForCausalLM,
        draft_model:  AutoModelForCausalLM,
        base_model:   AutoModelForCausalLM,
        cache_mgr:    PrefixSharedCacheManager,
        device:       torch.device,
    ) -> None:
        self._target_model = target_model
        self._draft_model  = draft_model
        self._base_model   = base_model
        self._cache_mgr    = cache_mgr
        self._device       = device

        self._target_ctx: Optional[ModelContext] = None
        self._draft_ctx:  Optional[ModelContext] = None
        self._base_ctx:   Optional[ModelContext] = None

        # 记录 prompt_len，用于回退边界检查
        self._prompt_len: int = 0

    # ------------------------------------------------------------------
    # Prefill 初始化
    # ------------------------------------------------------------------

    def init_from_prompt(self, prompt_ids: torch.Tensor) -> None:
        """三模型共同 Prefill，初始化所有 ModelContext。

        执行步骤：
          1. cache_mgr.allocate() 预分配 StaticCache
          2. Base/Draft 共享 shared_cache 做一次 Prefill（物理共享，零拷贝）
          3. Target 独立 Prefill target_cache
          4. cache_mgr.fork_caches() CoW 分叉，生成独立 draft_cache / base_cache
          5. 初始化三个 ModelContext

        Args:
            prompt_ids: shape [1, prompt_len]，已移至 self._device
        """
        self._cache_mgr.allocate()
        prompt_ids = prompt_ids.to(self._device)
        prompt_len = prompt_ids.shape[1]
        self._prompt_len = prompt_len

        # --- Step 1: Base/Draft 共享 Prefill ---
        # 由于两者架构完全相同（Qwen2.5-3B），共用一次 forward 填充 shared_cache
        # 注意：此处只用 draft_model forward，base_model 后续通过 fork 获得相同 KV
        logger.debug("Base/Draft 共享 Prefill  prompt_len=%d", prompt_len)
        shared_logits = prefill(
            model=self._draft_model,
            input_ids=prompt_ids,
            cache=self._cache_mgr.shared_cache,
        )  # shape: [1, vocab_size]

        # --- Step 2: Target 独立 Prefill ---
        logger.debug("Target 独立 Prefill  prompt_len=%d", prompt_len)
        target_logits = prefill(
            model=self._target_model,
            input_ids=prompt_ids,
            cache=self._cache_mgr.target_cache,
        )  # shape: [1, vocab_size]

        # --- Step 3: CoW 分叉 shared_cache → draft_cache + base_cache ---
        draft_cache, base_cache = self._cache_mgr.fork_caches()

        # --- Step 4: 构建三个 ModelContext ---
        self._target_ctx = ModelContext(
            model=self._target_model,
            cache=self._cache_mgr.target_cache,
            seq_len=prompt_len,
            last_logits=target_logits,
            device=self._device,
        )
        self._draft_ctx = ModelContext(
            model=self._draft_model,
            cache=draft_cache,
            seq_len=prompt_len,
            last_logits=shared_logits.clone(),
            device=self._device,
        )
        self._base_ctx = ModelContext(
            model=self._base_model,
            cache=base_cache,
            seq_len=prompt_len,
            last_logits=shared_logits.clone(),
            device=self._device,
        )
        logger.debug("三模型 Prefill 完成，各 ModelContext 初始化")

    # ------------------------------------------------------------------
    # 接受后同步
    # ------------------------------------------------------------------

    def sync_accepted(self, accepted_tokens: List[int]) -> None:
        """将本轮已接受的 token 序列同步到 Draft 和 Base。

        Target 在 verify 阶段（decode_batch_verify）已处理过这些 token，
        此方法只需推进 Draft 和 Base 的状态。

        Args:
            accepted_tokens: 本轮已接受的 token id 列表（按顺序）
        """
        if not accepted_tokens:
            return
        self._draft_ctx.advance(accepted_tokens)
        self._base_ctx.advance(accepted_tokens)

    # ------------------------------------------------------------------
    # 拒绝矫正同步
    # ------------------------------------------------------------------

    def sync_on_correction(self, correction_token: int, reject_pos: int) -> None:
        """拒绝矫正：将三模型状态统一回退到 reject_pos，并以矫正 token 重新推进。

        执行步骤：
          1. 计算回退目标：to_seq_len = prompt_len + reject_pos（已接受前缀长度）
          2. Draft / Base 回退 cache 到 to_seq_len
             （Target 已在 batch_verify 时停在了正确位置，无需回退）
          3. 三模型各推进一步 correction_token
             - Target 已通过 resample 生成了 correction_token，需 step() 更新 cache
             - Draft / Base 同样 step()

        Args:
            correction_token: Target resample 得到的矫正 token id
            reject_pos:       在本轮 K 个提案中的拒绝位置（0-based）
        """
        to_seq_len = self._prompt_len + self._draft_ctx.seq_len - self._prompt_len - (
            self._draft_ctx.seq_len - self._prompt_len - reject_pos
        )
        # 更简洁表达：draft_ctx.seq_len 已追赶到 prompt_len + 已接受 token 数
        # reject_pos 是相对本轮提案的偏移，Draft 此时已因 propose 写入了临时状态
        # 实际回退目标 = 当前 draft_ctx.seq_len - (gamma - reject_pos) 步
        # 但 propose 使用 temp_state 不改动 draft_ctx，draft_ctx.seq_len 为上轮同步后的值
        # 因此：to_seq_len = draft_ctx.seq_len + reject_pos（加上本轮已接受前缀）
        # 注：此处 draft_ctx.seq_len 是经 sync_accepted 更新后、进入本轮前的值
        to_seq_len_actual = self._draft_ctx.seq_len + reject_pos

        # Draft 和 Base 回退 cache
        self._draft_ctx.rollback(self._cache_mgr, to_seq_len_actual)
        self._base_ctx.rollback(self._cache_mgr, to_seq_len_actual)

        # 三模型推进矫正 token
        self._target_ctx.step(correction_token)
        self._draft_ctx.step(correction_token)
        self._base_ctx.step(correction_token)

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置编排器状态，为下一个 sample 做准备。"""
        self._cache_mgr.reset()
        self._target_ctx = None
        self._draft_ctx  = None
        self._base_ctx   = None
        self._prompt_len = 0
        logger.debug("TriModelOrchestrator 重置完毕")

    # ------------------------------------------------------------------
    # 只读 property（供 Proposer / DecodeLoop 访问 logits）
    # ------------------------------------------------------------------

    @property
    def target_ctx(self) -> ModelContext:
        if self._target_ctx is None:
            raise RuntimeError("请先调用 init_from_prompt()")
        return self._target_ctx

    @property
    def draft_ctx(self) -> ModelContext:
        if self._draft_ctx is None:
            raise RuntimeError("请先调用 init_from_prompt()")
        return self._draft_ctx

    @property
    def base_ctx(self) -> ModelContext:
        if self._base_ctx is None:
            raise RuntimeError("请先调用 init_from_prompt()")
        return self._base_ctx

    @property
    def prompt_len(self) -> int:
        return self._prompt_len
