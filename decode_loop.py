"""
主解码循环编排器 — SpeculativeDecodeLoop

串联 Phase 1-3 的所有子系统，实现完整的投机解码主循环：
  Propose → Target Verify → Accept/Reject（Strategy Routing） → Sync → Telemetry

核心循环（每轮）：
  1. proposer.propose_k_tokens(γ)
       → ProposalResult（K 个候选 + Draft/Base logits）
  2. target_ctx.advance(proposed_tokens) 批量 verify
       → target_logits_per_pos [K, 1, V]
  3. 逐位验收（strategy.evaluate）
       → 找到首拒绝位置 j（-1 = 全部接受）
  4. 同步状态
       全部接受：orchestrator.sync_accepted(K tokens)
       拒绝于 j：orchestrator.sync_on_correction(chosen, j)
  5. 遥测记录 + Hard Override 后续熵监控
  6. 检查 EOS

设计约束：
  - 整个方法用 @torch.inference_mode() 保护，无梯度计算
  - Target 的 advance() 同时完成 verify 和 cache 写入，一次 forward
  - Proposer 使用临时 cache 快照，不影响正式 ctx，通过 Orchestrator 统一 sync
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from acceptance import AcceptResult, AcceptanceStrategy, VerifyContext
from config_v2 import DecodeConfig
from dual_stream_engine import DualStreamProposer, ProposalResult
from engine_state import TriModelOrchestrator
from forward_ops import decode_batch_verify, sample_token
from shadow_sync_engine import ShadowSyncProposer
from telemetry import StepTelemetry, TelemetryLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 解码结果数据类
# ---------------------------------------------------------------------------

@dataclass
class DecodeResult:
    """单个 sample 的完整解码结果。

    Attributes:
        generated_token_ids: 生成的 token id 列表（不含 prompt）
        total_rounds:        总解码轮数（每轮 propose K tokens 为一轮）
        accepted_tokens:     总接受 token 数（含 target-match 和 override）
        rejected_tokens:     总拒绝 token 数（被矫正替换的位置数）
        override_count:      触发硬覆盖（B0/B 策略）的次数
        duration_sec:        生成阶段耗时（不含模型加载和 prompt 处理）
        tokens_per_sec:      真实吞吐量 = len(generated) / duration_sec
        mean_acceptance_rate: 平均验收率 = accepted_tokens / total_proposed_tokens
    """
    generated_token_ids:  List[int]
    total_rounds:         int
    accepted_tokens:      int
    rejected_tokens:      int
    override_count:       int
    duration_sec:         float
    tokens_per_sec:       float
    mean_acceptance_rate: float = 0.0


# ---------------------------------------------------------------------------
# 主解码循环
# ---------------------------------------------------------------------------

class SpeculativeDecodeLoop:
    """投机解码主循环编排器。

    Args:
        orchestrator: 三模型状态编排器（管理 Prefill/Sync/Correction）
        proposer:     提案引擎（DualStreamProposer 或 ShadowSyncProposer）
        strategy:     验收策略（5 种之一）
        telemetry:    遥测收集器
        tokenizer:    用于 EOS 检测
        config:       完整解码配置
    """

    def __init__(
        self,
        orchestrator: TriModelOrchestrator,
        proposer:     Union[DualStreamProposer, ShadowSyncProposer],
        strategy:     AcceptanceStrategy,
        telemetry:    TelemetryLogger,
        tokenizer,
        config:       DecodeConfig,
    ) -> None:
        self._orch      = orchestrator
        self._proposer  = proposer
        self._strategy  = strategy
        self._telemetry = telemetry
        self._tokenizer = tokenizer
        self._config    = config

        self._eos_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run(self, prompt_ids: torch.Tensor) -> DecodeResult:
        """执行完整的投机解码循环。

        Args:
            prompt_ids: 已 tokenize 的 prompt，shape [1, prompt_len]

        Returns:
            DecodeResult
        """
        # --- 三模型 Prefill（Base/Draft 共享 cache → CoW 分叉）---
        self._orch.init_from_prompt(prompt_ids)

        generated:          List[int] = []
        total_rounds:       int = 0
        accepted_count:     int = 0
        rejected_count:     int = 0
        override_count:     int = 0
        total_proposed:     int = 0
        global_step:        int = 0  # 全局 token 步序号（用于遥测）

        t_start = time.perf_counter()

        while len(generated) < self._config.max_new_tokens:
            total_rounds += 1
            gamma_now = min(
                self._config.gamma,
                self._config.max_new_tokens - len(generated),
            )

            # -------------------------------------------------------
            # Step 1：Draft + Base 提案（在临时 cache 上，不影响正式 ctx）
            # -------------------------------------------------------
            proposal: ProposalResult = self._proposer.propose_k_tokens(gamma_now)
            total_proposed += len(proposal.proposed_tokens)

            # -------------------------------------------------------
            # Step 2：Target 批量 verify（一次 forward 处理 gamma_now 个 token）
            # 调用 target_ctx.advance() 同时推进 target_cache
            # -------------------------------------------------------
            target_ctx = self._orch.target_ctx
            proposed_ids_tensor = torch.tensor(
                [proposal.proposed_tokens], dtype=torch.long, device=target_ctx.device
            )  # shape: [1, gamma_now]

            # batch_verify 之前保存 last_logits：这是验证 t₀ 所需的 P(t₀ | context)
            # target_logits_full[:, i, :] = P(t_{i+1} | context, t₀..tᵢ)，比验证需求晚一步
            prev_target_logits = target_ctx.last_logits.clone()  # shape: [1, V]

            # shape: [1, gamma_now, vocab_size]
            target_logits_full = decode_batch_verify(
                model=target_ctx.model,
                token_ids=proposed_ids_tensor,
                cache=target_ctx.cache,
                start_position=target_ctx.seq_len,
            )
            # 更新 target_ctx seq_len（cache 已写入）
            target_ctx.seq_len += gamma_now
            # 更新 last_logits 为批次最后一位（全部接受时用于下轮位置 0 的验证基础）
            target_ctx.last_logits = target_logits_full[:, -1, :]  # shape: [1, V]

            # 正确的逐位验证 logits（位移修正）：
            #   pos_i=0: P(t₀ | context)               = prev_target_logits
            #   pos_i=j: P(tⱼ | context, t₀..t_{j-1}) = target_logits_full[:, j-1, :]
            target_logits_per_pos: List[torch.Tensor] = (
                [prev_target_logits] +
                [target_logits_full[:, i, :] for i in range(gamma_now - 1)]
            )  # len = gamma_now，每个 shape: [1, V]

            # -------------------------------------------------------
            # Step 2.5：架构 C（DeferredBase）同步 Base Stream
            # 此时 Target verify 已结束，Base 在副 Stream 上已并行执行完毕（或即将完成）
            # wait_stream 阻塞时间趋近于零；同步后 result 的 base_logits_per_pos 被填充
            # -------------------------------------------------------
            if hasattr(self._proposer, "finalize_base_logits"):
                self._proposer.finalize_base_logits(proposal)

            # -------------------------------------------------------
            # Step 3：逐位验收（Strategy Routing）
            # -------------------------------------------------------
            accepted_tokens_this_round, first_reject_pos, round_results = (
                self._verify_and_accept(proposal, target_logits_per_pos, global_step)
            )

            # 统计
            is_all_accepted = (first_reject_pos < 0)
            n_accepted_prefix = len(accepted_tokens_this_round) - (0 if is_all_accepted else 1)
            accepted_count += len(accepted_tokens_this_round)
            rejected_count += (0 if is_all_accepted else 1)

            # Override 计数
            for res in round_results:
                if res.override_triggered:
                    override_count += 1

            # -------------------------------------------------------
            # Step 4：同步状态
            # -------------------------------------------------------
            if is_all_accepted:
                # 全部接受：Draft/Base 追赶 K 步
                self._orch.sync_accepted(accepted_tokens_this_round)
            else:
                # 拒绝于 first_reject_pos：
                # accepted_tokens_this_round[-1] = correction_token（由策略决定）
                correction_token = accepted_tokens_this_round[-1]
                prefix_tokens    = accepted_tokens_this_round[:-1]  # 拒绝前的接受前缀

                # Target cache 已在 batch_verify 时写入了全部 gamma_now 步
                # 需要回退 target cache 到 prefix + correction 位置
                # 先将多余的位置回退
                rollback_target_to = (
                    self._orch.prompt_len
                    + len(generated)
                    + first_reject_pos
                )
                self._orch._cache_mgr.rollback_cache(
                    target_ctx.cache, rollback_target_to
                )
                target_ctx.seq_len = rollback_target_to

                # 推进矫正 token 到 target
                target_ctx.step(correction_token)

                # Draft/Base 同步：从上轮 sync 后位置 → prefix → correction
                self._orch.sync_accepted(prefix_tokens)
                self._orch.draft_ctx.step(correction_token)
                self._orch.base_ctx.step(correction_token)

            generated.extend(accepted_tokens_this_round)
            global_step += len(accepted_tokens_this_round)

            # -------------------------------------------------------
            # Step 5：遥测 + Override 后续熵监控
            # -------------------------------------------------------
            self._record_telemetry(
                round_results=round_results,
                proposal=proposal,
                target_logits_per_pos=target_logits_per_pos,
                accepted_tokens=accepted_tokens_this_round,
                global_step_start=global_step - len(accepted_tokens_this_round),
            )

            # 推进 Override 后续熵监控（对每个已接受 token 步）
            for token_step_logits in target_logits_per_pos[: len(accepted_tokens_this_round)]:
                self._telemetry.log_target_entropy_for_probes(token_step_logits)

            # -------------------------------------------------------
            # Step 6：EOS 检测
            # -------------------------------------------------------
            if self._config.stop_on_eos and self._eos_id is not None:
                if generated and generated[-1] == self._eos_id:
                    logger.debug("EOS detected at step %d", global_step)
                    break

        t_end = time.perf_counter()
        duration = t_end - t_start

        self._telemetry.finalize_probes()

        n_gen = len(generated)
        return DecodeResult(
            generated_token_ids=generated,
            total_rounds=total_rounds,
            accepted_tokens=accepted_count,
            rejected_tokens=rejected_count,
            override_count=override_count,
            duration_sec=duration,
            tokens_per_sec=(n_gen / duration) if duration > 0 else 0.0,
            mean_acceptance_rate=(
                accepted_count / total_proposed if total_proposed > 0 else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # 逐位验收
    # ------------------------------------------------------------------

    def _verify_and_accept(
        self,
        proposal:               ProposalResult,
        target_logits_per_pos:  List[torch.Tensor],
        global_step_start:      int,
    ) -> tuple[List[int], int, List[StepTelemetry]]:
        """逐位验证提案，返回（已接受 token 列表，首拒绝位置，遥测列表）。

        逻辑：
          对 i = 0..gamma-1：
            构造 VerifyContext（draft/target/base logits at pos i）
            调用 strategy.evaluate()
            若接受：append to accepted
            若拒绝：append correction_token, break

          返回 first_reject_pos = -1 表示全部接受。

        Args:
            proposal:              当前轮的提案结果
            target_logits_per_pos: Target verify 返回的逐位 logits
            global_step_start:     本轮首个 token 的全局步序号（用于遥测）

        Returns:
            (accepted_tokens, first_reject_pos, telemetry_list)
            accepted_tokens: 全部接受时 = proposed_tokens；
                             拒绝时   = accepted_prefix + [correction_token]
        """
        accepted_tokens: List[int]          = []
        telemetry_list:  List[StepTelemetry] = []
        first_reject_pos: int = -1

        draft_ctx  = self._orch.draft_ctx
        base_ctx   = self._orch.base_ctx

        for pos_i, draft_token in enumerate(proposal.proposed_tokens):
            logit_target = target_logits_per_pos[pos_i]        # shape: [1, V]
            logit_draft  = proposal.draft_logits_per_pos[pos_i] # shape: [1, V]
            logit_base   = proposal.base_logits_per_pos[pos_i]  # shape: [1, V]

            ctx = VerifyContext(
                draft_token_id=draft_token,
                logit_target=logit_target,
                logit_draft=logit_draft,
                logit_base=logit_base,
                t_sample=self._config.t_sample,
            )

            result: AcceptResult = self._strategy.evaluate(ctx)

            # 判断是否触发了 Hard Override（B0/B 策略的强制放行）
            override_triggered = result.accepted and (
                "override" in result.reason.lower()
            )

            # 构造本步遥测
            step_tel = StepTelemetry(
                step=global_step_start + pos_i,
                draft_token_id=draft_token,
                target_top1_id=int(logit_target.argmax(dim=-1).item()),
                base_top1_id=int(logit_base.argmax(dim=-1).item()),
                delta_p=result.delta_p,
                p_draft=result.p_draft,
                p_target=result.p_target,
                accepted=result.accepted,
                override_triggered=override_triggered,
                strategy_reason=result.reason,
            )
            telemetry_list.append(step_tel)
            self._telemetry.log_step(step_tel)

            # Override 触发时注册后续探针
            if override_triggered:
                self._telemetry.register_override(override_step=global_step_start + pos_i)

            if result.accepted:
                accepted_tokens.append(result.chosen_token_id)
            else:
                # 拒绝：追加矫正 token，停止本轮
                accepted_tokens.append(result.chosen_token_id)
                first_reject_pos = pos_i
                break

        return accepted_tokens, first_reject_pos, telemetry_list

    # ------------------------------------------------------------------
    # 遥测记录（内部）
    # ------------------------------------------------------------------

    def _record_telemetry(
        self,
        round_results:         List[StepTelemetry],
        proposal:              ProposalResult,
        target_logits_per_pos: List[torch.Tensor],
        accepted_tokens:       List[int],
        global_step_start:     int,
    ) -> None:
        """（当前为空壳）遥测已在 _verify_and_accept 中逐步记录，此处保留扩展点。"""
        pass
