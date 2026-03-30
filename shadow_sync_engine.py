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

工作流程（每步 i）：
  Phase 1 (Draft propose, 串行 K 步):
    draft_temp.step(token[i]) → draft_logits[i], token[i+1] = argmax

  Phase 2 (Base batch forward, 一次 forward):
    decode_batch_hidden_only(base_model, [token[0]..token[K-1]]) → hidden [1, K, H]

  Phase 3 (Lazy LM Head, 仅对 K 个候选位置):
    extract_logits_at_positions(base_model, hidden, [0..K-1]) → base_logits [1, K, V]
"""

from __future__ import annotations

import copy
import logging
from typing import List

import torch

from dual_stream_engine import ProposalResult  # 共享 ProposalResult 数据类
from engine_state import ModelContext
from forward_ops import decode_batch_hidden_only, decode_step, extract_logits_at_positions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助：临时 Draft 快照（propose 阶段不污染正式 ctx）
# ---------------------------------------------------------------------------

class _TempDraftSnapshot:
    """Draft 的临时快照，用于 propose 阶段串行步进。"""

    def __init__(self, ctx: ModelContext) -> None:
        self.model       = ctx.model
        self.cache       = copy.deepcopy(ctx.cache)  # CoW 隔离
        self.seq_len     = ctx.seq_len
        self.last_logits = ctx.last_logits.clone()
        self.device      = ctx.device

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
        """生成 K 个候选 token（不修改正式 ctx）。

        Args:
            k: 提案步数（= DecodeConfig.gamma）

        Returns:
            ProposalResult：K 个候选 token + 各位置 Draft/Base logits
        """
        if k <= 0:
            raise ValueError(f"k 必须 >= 1，当前: {k}")

        # ---------------------------------------------------------------
        # Phase 1：Draft 串行步进，收集 K 个候选 token 和 draft_logits
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

        # ---------------------------------------------------------------
        # Phase 2：Base batch hidden forward（一次 forward 处理 K 个 token）
        # ---------------------------------------------------------------
        # proposed_tokens: [token_0, token_1, ..., token_{K-1}]
        # Base 以相同 token 序列 batch forward，得到 K 个位置的 hidden states
        token_ids_tensor = torch.tensor(
            [proposed_tokens], dtype=torch.long, device=self._device
        )  # shape: [1, K]

        # 使用 base_ctx 的正式 cache（不拷贝，因为 hidden_only 不写入 cache 中间结果）
        # 注意：decode_batch_hidden_only 会写入 cache——因此也需要 CoW 隔离
        base_cache_copy = copy.deepcopy(self._base_ctx.cache)

        # shape: [1, K, hidden_dim]
        base_hidden = decode_batch_hidden_only(
            model=self._base_ctx.model,
            token_ids=token_ids_tensor,
            cache=base_cache_copy,
            start_position=self._base_ctx.seq_len,
        )

        # ---------------------------------------------------------------
        # Phase 3：Lazy LM Head —— 对 K 个候选位置提取 Base logits
        # ---------------------------------------------------------------
        # 所有 K 个位置都需要 base_logits（每个候选都要计算 ΔP）
        # positions: [0, 1, ..., K-1]（相对于本次 batch 中的位置）
        positions = list(range(k))

        # shape: [1, K, vocab_size]
        base_logits_batch = extract_logits_at_positions(
            model=self._base_ctx.model,
            hidden_states=base_hidden,
            positions=positions,
        )

        # 拆分为逐位置列表，每个 shape: [1, vocab_size]
        base_logits_per_pos: List[torch.Tensor] = [
            base_logits_batch[:, i, :]  # shape: [1, vocab_size]
            for i in range(k)
        ]

        logger.debug(
            "ShadowSync propose_k=%d  tokens=%s",
            k, proposed_tokens
        )

        return ProposalResult(
            proposed_tokens=proposed_tokens,
            draft_logits_per_pos=draft_logits_per_pos,
            base_logits_per_pos=base_logits_per_pos,
        )
