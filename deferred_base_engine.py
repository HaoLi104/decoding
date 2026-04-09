"""
架构 C：延迟 Base + CUDA Stream 并行（Deferred Base）

核心思路：
  将 Shadow Sync 中 Base 的 batch hidden forward 从"提案阶段"后移到
  "Target 验证阶段"，两者在不同 CUDA Stream 上真正并行执行：

    [主 Stream]  Draft K步 → Target batch verify → (同步 Base stream) → Accept/Reject
    [Base Stream]            ↑ Base batch hidden 在此并行启动 ↑

  这样 Base 的计算时间被 Target 的验证时间"掩盖"，消除了 Shadow Sync 中
  Base 串在 Draft 之后的额外延迟。

  理论节省：ΔT ≈ min(K × t_base_hidden, t_target_verify)

实现接口：
  propose_k_tokens() — 提案阶段：只跑 Draft，顺带在 Base Stream 上异步凯动 Base
  finalize_base_logits() — Target 验证后调用，等待 Base Stream 并提取 logits

与 ShadowSyncProposer 的差异（改动集中在 propose_k_tokens）：
  - Phase 2（Base batch hidden）改为在独立 CUDA Stream 上启动（非阻塞）
  - 返回的 ProposalResult 中 base_logits_per_pos 为占位列表（None entry）
  - LM Head 提取推迟到 finalize_base_logits() 中（在 Target 验证后）

注意事项：
  - 不依赖 torch.compile，避免 CUDAGraph 捕获问题
  - Base Stream 在 __init__ 时创建，整个推理周期复用，避免频繁分配开销
  - finalize_base_logits() 通过 current_stream.wait_stream(base_stream) 实现
    主流等待副流，保证 Base hidden 已写入后才做 LM Head 矩阵乘
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from dual_stream_engine import ProposalResult
from engine_state import ModelContext
from forward_ops import decode_batch_hidden_only, decode_step, extract_logits_at_positions
from shadow_sync_engine import _TempDraftSnapshot   # 复用 Draft 临时句柄

logger = logging.getLogger(__name__)


class DeferredBaseProposer:
    """架构 C：延迟 Base + CUDA Stream 并行提案引擎。

    两阶段调用：
      1. propose_k_tokens(k)        → 提案阶段（Draft + 异步 Base hidden）
      2. finalize_base_logits(result) → Target 验证后同步 Base 并提取 logits

    decode_loop.py 在 Target batch verify 后调用 finalize_base_logits()，
    此时 Base hidden 计算已在副流上与 Target 并行完成（或即将完成）。
    """

    def __init__(
        self,
        draft_ctx: ModelContext,
        base_ctx:  ModelContext,
        device:    torch.device,
    ) -> None:
        self._draft_ctx = draft_ctx
        self._base_ctx  = base_ctx
        self._device    = device

        # Base 专用 CUDA Stream（整个推理周期固定复用，避免频繁分配）
        self._base_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)

        # 跨 propose/finalize 两次调用的中间状态
        self._pending_hidden:       Optional[torch.Tensor] = None  # [1, k, hidden]
        self._pending_prev_logits:  Optional[torch.Tensor] = None  # [1, V]
        self._pending_k:            int = 0

    # ------------------------------------------------------------------
    # Phase 1：提案（仅 Draft；Base 异步启动）
    # ------------------------------------------------------------------

    def propose_k_tokens(self, k: int) -> ProposalResult:
        """生成 K 个候选 token，同时在 Base Stream 上异步启动 Base hidden forward。

        返回的 ProposalResult.base_logits_per_pos 为占位零张量，
        必须在 Target verify 后调用 finalize_base_logits() 填入真实值。

        Args:
            k: 提案步数（= DecodeConfig.gamma）

        Returns:
            ProposalResult（base_logits_per_pos 为占位，等待 finalize）
        """
        if k <= 0:
            raise ValueError(f"k 必须 >= 1，当前: {k}")

        # ------------------------------------------------------------------
        # Draft Phase：与 Shadow Sync 相同，串行 K 步提案
        # ------------------------------------------------------------------
        draft_temp = _TempDraftSnapshot(self._draft_ctx)

        proposed_tokens:      List[int]          = []
        draft_logits_per_pos: List[torch.Tensor] = []

        for step_i in range(k):
            cur_draft_logits = draft_temp.last_logits.clone()   # shape: [1, V]
            next_token = int(cur_draft_logits.argmax(dim=-1).item())
            proposed_tokens.append(next_token)
            draft_logits_per_pos.append(cur_draft_logits)

            if step_i < k - 1:
                draft_temp.step(next_token)

        draft_temp.rollback_seq_len(self._draft_ctx)

        # ------------------------------------------------------------------
        # Base Phase（异步）：在 Base Stream 上启动 batch hidden forward
        # 主 Stream 立刻返回，不等待，让 Target verify 与 Base 并行执行
        # ------------------------------------------------------------------
        token_ids_tensor = torch.tensor(
            [proposed_tokens], dtype=torch.long, device=self._device
        )  # shape: [1, K]

        base_start_seq_len        = self._base_ctx.seq_len
        self._pending_prev_logits = self._base_ctx.last_logits.clone()   # shape: [1, V]
        self._pending_k           = k

        # ★ 关键同步：让 Base Stream 等待主 Stream 的当前进度。
        #   上一轮 sync_accepted()/base_ctx.step() 在主 Stream 上写入了 base_ctx.cache，
        #   若不同步，Base Stream 可能读到上轮写入前的旧 KV，导致 logit_base 偏差，
        #   进而影响 C1 接受概率（即便 T=0 也会系统性偏移 acc）。
        #   此 wait_stream 不会破坏优化目标：等待结束后主 Stream 立即进入 Target verify，
        #   Base Stream 与 Target verify 仍真正并行。
        self._base_stream.wait_stream(torch.cuda.current_stream())

        # 用 context manager 将后续 CUDA kernel 路由到 Base Stream
        with torch.cuda.stream(self._base_stream):
            # decode_batch_hidden_only 的 CUDA kernel 在 Base Stream 上异步执行
            # 主 Stream 不阻塞，Target verify 可立即在主 Stream 上启动
            self._pending_hidden = decode_batch_hidden_only(
                model=self._base_ctx.model,
                token_ids=token_ids_tensor,
                cache=self._base_ctx.cache,          # 直接写入正式 cache（不拷贝）
                start_position=self._base_ctx.seq_len,
            )  # shape: [1, K, hidden_dim]

        # 回退 base_ctx.seq_len（与 Shadow Sync 一致）
        self._base_ctx.seq_len = base_start_seq_len

        # 占位 base_logits：shape 与 draft_logits 相同的全零张量
        # finalize_base_logits() 调用后会被真实值覆盖
        placeholder = torch.zeros_like(draft_logits_per_pos[0])   # [1, V]
        base_logits_placeholder: List[torch.Tensor] = [placeholder] * k

        logger.debug(
            "DeferredBase propose_k=%d  tokens=%s  (Base hidden launched on Base Stream)",
            k, proposed_tokens,
        )

        return ProposalResult(
            proposed_tokens=proposed_tokens,
            draft_logits_per_pos=draft_logits_per_pos,
            base_logits_per_pos=base_logits_placeholder,   # 占位，待 finalize 填充
        )

    # ------------------------------------------------------------------
    # Phase 2：同步 Base Stream，提取 LM Head logits（在 Target verify 后调用）
    # ------------------------------------------------------------------

    def finalize_base_logits(self, result: ProposalResult) -> None:
        """等待 Base Stream 完成，提取 LM Head logits 并就地写入 result。

        应在 decode_loop.py 的 Target batch verify 之后、Accept/Reject 之前调用。
        此时 Base hidden forward 已在副流上与 Target 并行执行，
        wait_stream 的阻塞时间趋近于零（Base 通常比 Target 快）。

        Args:
            result: propose_k_tokens() 返回的 ProposalResult（将被就地修改）
        """
        if self._pending_hidden is None:
            raise RuntimeError(
                "finalize_base_logits() 在 propose_k_tokens() 之前调用，或重复调用"
            )

        k = self._pending_k

        # 主 Stream 等待 Base Stream：确保 Base hidden 已写入，再做 LM Head 矩阵乘
        # 若 Base 已完成，此处几乎不阻塞；若 Target 很快，此处短暂等待
        torch.cuda.current_stream().wait_stream(self._base_stream)

        # LM Head 提取：与 Shadow Sync Phase 3 逻辑完全相同
        if k > 1:
            positions = list(range(k - 1))           # 提取前 K-1 个隐层位置
            base_logits_batch = extract_logits_at_positions(
                model=self._base_ctx.model,
                hidden_states=self._pending_hidden,
                positions=positions,
            )  # shape: [1, K-1, vocab_size]

            # 位移修正（同 ShadowSync）：
            #   pos_i=0 → prev_base_logits（提案前的 context logit）
            #   pos_i=j → base_logits_batch[:, j-1, :]
            base_logits_per_pos: List[torch.Tensor] = (
                [self._pending_prev_logits] +
                [base_logits_batch[:, i, :] for i in range(k - 1)]
            )
        else:
            base_logits_per_pos = [self._pending_prev_logits]

        # 就地修改 result，使 decode_loop 的 _verify_and_accept 能直接读取真实 logits
        result.base_logits_per_pos = base_logits_per_pos

        logger.debug(
            "DeferredBase finalize_base_logits k=%d  (Base Stream synced)", k
        )

        # 清理跨调用中间状态
        self._pending_hidden      = None
        self._pending_prev_logits = None
        self._pending_k           = 0
