"""
架构 A：双流异步并发提案引擎（Dual-Stream Concurrent Execution）

对应实验计划 Section 5 架构 A：
  在 Draft 提案阶段，使用 torch.cuda.Stream() 为 Draft 和 Base 开辟独立流。
  在单个时间步内，两个 3B 模型在不同 SM 簇上同时执行前向传播，实时计算每步 ΔP。

工作流程（每步 i）：
  1. Draft stream:  decode_step(draft_model, proposed_token[i-1]) -> draft_logits[i]
                    -> argmax -> proposed_token[i]
  2. Base  stream:  decode_step(base_model,  proposed_token[i-1]) -> base_logits[i]
     （与 Draft 在不同 CUDA Stream 上并发执行）
  3. 同步屏障:      wait_stream，确保两者均完成后再进入下一步

重要约束：
  - propose_k_tokens() 使用「临时状态」（temp 变量），不修改 draft_ctx / base_ctx
    的真实 cache，避免 propose 阶段污染正式状态（对应 k_spec_kernels.py 的设计）
  - draft_ctx / base_ctx 的正式推进由 TriModelOrchestrator.sync_accepted() 完成
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List

import torch

from engine_state import ModelContext
from forward_ops import decode_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 共享数据类：K 步提案结果
# ---------------------------------------------------------------------------

@dataclass
class ProposalResult:
    """K 步提案的完整结果，供 DecodeLoop 验证和 Orchestrator 同步使用。

    Attributes:
        proposed_tokens:      Draft 逐步 argmax 生成的 K 个候选 token id
        draft_logits_per_pos: 每步 Draft 的 next-token logits，shape per elem: [1, vocab_size]
        base_logits_per_pos:  每步 Base  的 next-token logits，shape per elem: [1, vocab_size]
    """
    proposed_tokens:      List[int]
    draft_logits_per_pos: List[torch.Tensor]  # each: [1, vocab_size]
    base_logits_per_pos:  List[torch.Tensor]  # each: [1, vocab_size]


# ---------------------------------------------------------------------------
# 辅助：临时上下文快照（不修改原始 ctx）
# ---------------------------------------------------------------------------

@dataclass
class _TempCtxSnapshot:
    """Draft/Base 的临时状态快照，用于 propose 阶段避免污染正式 cache。

    由于 StaticCache 是可变对象，propose 阶段必须在独立 cache 副本上操作，
    完成后丢弃，不影响 draft_ctx / base_ctx 的正式状态。
    """
    model:      object   # AutoModelForCausalLM
    cache:      object   # StaticCache 深拷贝
    seq_len:    int
    last_logits: torch.Tensor  # shape: [1, vocab_size]
    device:     torch.device

    def step(self, token_id: int, stream: torch.cuda.Stream) -> None:
        """在指定 CUDA stream 上执行单步 decode_step，更新快照状态。"""
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with torch.cuda.stream(stream):
            self.last_logits = decode_step(
                model=self.model,
                token_id=token_tensor,
                cache=self.cache,
                position_id=self.seq_len,
            )  # shape: [1, vocab_size]
        self.seq_len += 1


def _snapshot_from_ctx(ctx: ModelContext) -> _TempCtxSnapshot:
    """从 ModelContext 深拷贝出临时快照（仅拷贝 cache，模型权重共享引用）。"""
    return _TempCtxSnapshot(
        model=ctx.model,
        cache=copy.deepcopy(ctx.cache),   # CoW：深拷贝 cache 以隔离 propose 阶段写入
        seq_len=ctx.seq_len,
        last_logits=ctx.last_logits.clone(),
        device=ctx.device,
    )


# ---------------------------------------------------------------------------
# DualStreamProposer
# ---------------------------------------------------------------------------

class DualStreamProposer:
    """架构 A：Draft 和 Base 在独立 CUDA Stream 上并发执行 K 步提案。

    每一步 i：
      - Draft stream 执行 decode_step → 得到 draft_logits[i] → argmax → token[i]
      - Base  stream 同步执行 decode_step(token[i-1]) → 得到 base_logits[i]
      - 同步屏障保证两个 stream 均完成后进入下一步

    注意：两个 stream 在同一 cuda:0 上运行，SM 簇级别的并发性由驱动调度，
    实际并发程度取决于显存带宽是否成为瓶颈（本实验旨在测量此瓶颈）。
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

        # 创建两个独立 CUDA Stream
        self.draft_stream = torch.cuda.Stream(device=device)
        self.base_stream  = torch.cuda.Stream(device=device)

    def propose_k_tokens(self, k: int) -> ProposalResult:
        """在临时快照状态上滚动生成 K 个候选 token（不修改正式 ctx）。

        执行逻辑（每步 i = 0..k-1）：
          step-0 输入：draft_ctx.last_logits（Prefill 或上轮 sync 后的 logits）
          step-i：
            1. argmax(draft_temp.last_logits) → proposed_token[i]
            2. Draft stream: decode_step(proposed_token[i]) → draft_logits[i+1]
            3. Base  stream: decode_step(proposed_token[i]) → base_logits[i]
               （Base 消费 Draft 同步提案的 token，维持相同的上下文）
            4. 同步屏障

        Returns:
            ProposalResult 包含 K 个候选 token 和各位置的 Draft/Base logits
        """
        if k <= 0:
            raise ValueError(f"k 必须 >= 1，当前: {k}")

        # 深拷贝 cache，隔离 propose 阶段
        draft_temp = _snapshot_from_ctx(self._draft_ctx)
        base_temp  = _snapshot_from_ctx(self._base_ctx)

        proposed_tokens:      List[int]           = []
        draft_logits_per_pos: List[torch.Tensor]  = []
        base_logits_per_pos:  List[torch.Tensor]  = []

        # 当前有效的 Draft logits（用于 argmax 决定下一个 token）
        current_draft_logits = draft_temp.last_logits  # shape: [1, vocab_size]

        for step_i in range(k):
            # --- 1. 从当前 Draft logits 贪婪选出候选 token ---
            next_token = int(current_draft_logits.argmax(dim=-1).item())
            proposed_tokens.append(next_token)

            # 记录本位置的 Draft logits（当前步的 logits 预测 next_token）
            draft_logits_per_pos.append(current_draft_logits.clone())

            # --- 2. Draft stream：推进到 next_token，得到下一步 logits ---
            draft_temp.step(next_token, self.draft_stream)

            # --- 3. Base stream：以相同 next_token 推进，得到 base_logits ---
            base_temp.step(next_token, self.base_stream)

            # --- 4. 同步屏障：等待两个 stream 均完成 ---
            # 让默认 stream 等待 draft_stream 和 base_stream
            torch.cuda.current_stream(self._device).wait_stream(self.draft_stream)
            torch.cuda.current_stream(self._device).wait_stream(self.base_stream)

            # 记录 Base 本位置的 logits（step 执行后 last_logits 已更新为下一步）
            # 注意：base_temp.step() 后，last_logits 是 next_token 之后的预测
            # 我们需要的是"位于 next_token 之前"的 base 分布（即推进前的 logits）
            # → 因此 Base logits 需在 step() 之前记录（与 Draft 相同位置）
            # 实现：先记录 base_temp.last_logits（step 前），再执行 step
            # 但因 step() 在 stream 上异步，需调整为：先记录，再 step

        # 上述逻辑需要重新组织：先记录再步进
        # 重写为更清晰的实现：

        return self._propose_k_tokens_impl(k)

    def _propose_k_tokens_impl(self, k: int) -> ProposalResult:
        """正确的双流并发实现（先记录 logits，再在 stream 上步进）。"""
        draft_temp = _snapshot_from_ctx(self._draft_ctx)
        base_temp  = _snapshot_from_ctx(self._base_ctx)

        proposed_tokens:      List[int]          = []
        draft_logits_per_pos: List[torch.Tensor] = []
        base_logits_per_pos:  List[torch.Tensor] = []

        for step_i in range(k):
            # 本位置的 Draft / Base logits（predict token at step_i）
            # draft_temp.last_logits 是前一步（或 Prefill）输出的 logits
            cur_draft_logits = draft_temp.last_logits.clone()  # shape: [1, V]
            cur_base_logits  = base_temp.last_logits.clone()   # shape: [1, V]

            # 从 Draft logits 贪婪选出候选 token
            next_token = int(cur_draft_logits.argmax(dim=-1).item())
            proposed_tokens.append(next_token)

            draft_logits_per_pos.append(cur_draft_logits)
            base_logits_per_pos.append(cur_base_logits)

            if step_i < k - 1:
                # 只有非最后一步才需要推进（最后一步的 next logits 由 Target verify 提供）
                token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self._device)

                # Draft stream 推进
                with torch.cuda.stream(self.draft_stream):
                    new_draft_logits = decode_step(
                        model=draft_temp.model,
                        token_id=token_tensor,
                        cache=draft_temp.cache,
                        position_id=draft_temp.seq_len,
                    )  # shape: [1, V]

                # Base stream 推进
                with torch.cuda.stream(self.base_stream):
                    new_base_logits = decode_step(
                        model=base_temp.model,
                        token_id=token_tensor,
                        cache=base_temp.cache,
                        position_id=base_temp.seq_len,
                    )  # shape: [1, V]

                # 同步屏障
                torch.cuda.current_stream(self._device).wait_stream(self.draft_stream)
                torch.cuda.current_stream(self._device).wait_stream(self.base_stream)

                draft_temp.last_logits = new_draft_logits
                draft_temp.seq_len    += 1
                base_temp.last_logits  = new_base_logits
                base_temp.seq_len     += 1

        logger.debug(
            "DualStream propose_k=%d  tokens=%s",
            k, proposed_tokens
        )

        return ProposalResult(
            proposed_tokens=proposed_tokens,
            draft_logits_per_pos=draft_logits_per_pos,
            base_logits_per_pos=base_logits_per_pos,
        )
