"""
架构 B：影子同步 + Lazy LM Head 提案引擎（Shadow Sync）

对应实验计划 Section 5 架构 B：
  Draft 正常逐步生成 K 个候选 token + logits（每步一次完整 forward）。
  Base 以 Draft 相同的 token 序列做一次 batch forward，仅计算到 hidden_state，
  跳过全序列 LM Head 矩阵乘；之后仅在 Draft 提案的候选位置执行 LM Head（Lazy Eval）。

对比架构 A 的优势：
  - 避免 Base 完整 LM Head（V=152K vocab 的超大矩阵乘）在 K 步中被调用 K 次
  - Base hidden forward 一次性处理 K 个 token，利用 batch 并行性
  - 拒绝后无需 Mini-Prefill Latency Spike（因 Base 已同步到 K 步位置）

重要设计说明（CoW 策略）：
  Proposal 阶段直接写入正式 draft_ctx.cache 和 base_ctx.cache，
  完成后将 seq_len 重置到提案前的位置（不 deepcopy cache）。
  后续 sync_accepted/sync_on_correction 会以正确的 position 覆盖 proposal 写入的 KV。
  这完全规避了 copy.deepcopy(cache) 对 CUDA 张量的极高开销
  （deepcopy 会通过 config 引用触发整个模型参数的 CPU 拷贝）。
"""

from __future__ import annotations

import logging
from typing import List

import torch

from dual_stream_engine import ProposalResult  # 共享 ProposalResult 数据类
from engine_state import ModelContext
from forward_ops import decode_batch_hidden_only, decode_step, extract_logits_at_positions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助：轻量 Draft 提案句柄（直接使用正式 cache，不拷贝）
# ---------------------------------------------------------------------------

class _TempDraftSnapshot:
    """Draft 提案阶段的轻量句柄。

    直接引用 draft_ctx.cache，不做 deepcopy，在正式 cache 上写入提案 KV。
    提案完成后调用 rollback_seq_len() 将 draft_ctx.seq_len 重置，
    使后续 sync_accepted/sync_on_correction 以正确位置覆盖提案数据。
    """

    def __init__(self, ctx: ModelContext) -> None:
        self.model            = ctx.model
        self.cache            = ctx.cache        # 直接引用，不拷贝
        self.seq_len          = ctx.seq_len
        self.last_logits      = ctx.last_logits.clone()
        self.device           = ctx.device
        self._start_seq_len   = ctx.seq_len      # 记录起始位置，用于 seq_len 回退

    def step(self, token_id: int) -> None:
        """执行单步 decode_step，更新 last_logits 和 seq_len。"""
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        self.last_logits = decode_step(
            model=self.model,
            token_id=token_tensor,
            cache=self.cache,
            position_id=self.seq_len,
        )  # shape: [1, vocab_size]
        self.seq_len += 1

    def rollback_seq_len(self, ctx: ModelContext) -> None:
        """提案结束后将 ctx.seq_len 回退到提案前位置。

        proposal 期间写入 cache 的 KV 是"临时数据"，
        sync 阶段会以正确的 position_id 覆盖这些位置，无需清零。
        """
        ctx.seq_len = self._start_seq_len


# ---------------------------------------------------------------------------
# ShadowSyncProposer
# ---------------------------------------------------------------------------

class ShadowSyncProposer:
    """架构 B：影子同步 + Lazy LM Head 提案引擎。

    两阶段执行：
      Phase 1 — Draft 串行 K 步提案（每步一次 decode_step）
      Phase 2 — Base batch hidden forward + Lazy LM Head（一次 batch forward + K 次小矩阵乘）

    这比架构 A 每步都调 Base decode_step 更轻量，
    因为 decode_batch_hidden_only 跳过了 K 次 LM Head，改用一次 batch 计算 hidden。
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

    def propose_k_tokens(self, k: int) -> ProposalResult:
        """生成 K 个候选 token（proposal 后 seq_len 回退，不修改正式 ctx 的 seq_len）。

        Args:
            k: 提案步数（= DecodeConfig.gamma）

        Returns:
            ProposalResult：K 个候选 token + 各位置 Draft/Base logits
        """
        if k <= 0:
            raise ValueError(f"k 必须 >= 1，当前: {k}")

        # ---------------------------------------------------------------
        # Phase 1：Draft 串行步进，收集 K 个候选 token 和 draft_logits
        # 直接写入 draft_ctx.cache（不 deepcopy）；完成后回退 seq_len
        # ---------------------------------------------------------------
        draft_temp = _TempDraftSnapshot(self._draft_ctx)

        proposed_tokens:      List[int]          = []
        draft_logits_per_pos: List[torch.Tensor] = []

        for step_i in range(k):
            # 本位置的 Draft logits（当前步的 last_logits 预测本位置 token）
            cur_draft_logits = draft_temp.last_logits.clone()  # shape: [1, V]

            # 贪婪采样下一个候选 token
            next_token = int(cur_draft_logits.argmax(dim=-1).item())

            proposed_tokens.append(next_token)
            draft_logits_per_pos.append(cur_draft_logits)

            if step_i < k - 1:
                # 推进 Draft 临时状态（最后一步不需要，K 个 token 已确定）
                draft_temp.step(next_token)

        # 回退 draft_ctx.seq_len；sync_accepted 会以正确位置覆盖 proposal KV
        draft_temp.rollback_seq_len(self._draft_ctx)

        # ---------------------------------------------------------------
        # Phase 2：Base batch hidden forward（直接写入 base_ctx.cache，不拷贝）
        # 完成后回退 base_ctx.seq_len
        # ---------------------------------------------------------------
        token_ids_tensor = torch.tensor(
            [proposed_tokens], dtype=torch.long, device=self._device
        )  # shape: [1, K]

        base_start_seq_len = self._base_ctx.seq_len

        # batch_verify 之前保存 base 的 last_logits：验证 t₀ 需要 P(t₀ | context)
        prev_base_logits = self._base_ctx.last_logits.clone()  # shape: [1, V]

        # shape: [1, K, hidden_dim]
        base_hidden = decode_batch_hidden_only(
            model=self._base_ctx.model,
            token_ids=token_ids_tensor,
            cache=self._base_ctx.cache,   # 直接使用，不拷贝
            start_position=self._base_ctx.seq_len,
        )

        # 回退 base_ctx.seq_len；sync 会覆盖 proposal 在此范围写入的 KV
        self._base_ctx.seq_len = base_start_seq_len

        # ---------------------------------------------------------------
        # Phase 3：Lazy LM Head —— 对 K-1 个中间位置提取 Base logits
        # base_logits_batch[:, i, :] = P(t_{i+1} | context, t₀..tᵢ)，比验证需求晚一步
        # 位移修正：pos_i=0 用 prev_base_logits，pos_i=j 用 logits_batch[:, j-1, :]
        # ---------------------------------------------------------------
        # 只需提取 k-1 个中间位置（位置 0..k-2 对应验证 pos 1..k-1）
        positions = list(range(k - 1)) if k > 1 else []

        if positions:
            # shape: [1, K-1, vocab_size]
            base_logits_batch = extract_logits_at_positions(
                model=self._base_ctx.model,
                hidden_states=base_hidden,
                positions=positions,
            )
            # 正确的逐位置 Base logits（位移修正）：
            #   pos_i=0: P(t₀ | context)               = prev_base_logits
            #   pos_i=j: P(tⱼ | context, t₀..t_{j-1}) = base_logits_batch[:, j-1, :]
            base_logits_per_pos: List[torch.Tensor] = (
                [prev_base_logits] +
                [base_logits_batch[:, i, :] for i in range(k - 1)]
            )
        else:
            # k=1：只有一个位置，直接用 prev_base_logits
            base_logits_per_pos = [prev_base_logits]

        logger.debug(
            "ShadowSync propose_k=%d  tokens=%s",
            k, proposed_tokens
        )

        return ProposalResult(
            proposed_tokens=proposed_tokens,
            draft_logits_per_pos=draft_logits_per_pos,
            base_logits_per_pos=base_logits_per_pos,
        )
