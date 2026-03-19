from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM


# 统一管理每个模型的“当前位置”状态：
# - past_key_values: KV cache
# - next_logits: 在当前前缀下预测“下一个 token”的 logits
# - seq_len: 当前前缀长度（用于构造 attention_mask）
@dataclass
class ModelState:
    past_key_values: Optional[tuple]
    next_logits: torch.Tensor
    seq_len: int


@torch.inference_mode()
def init_state_from_prompt(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    device: torch.device,
) -> ModelState:
    # 用完整 prompt 做一次前向，得到初始 KV cache 和 next_logits。
    out = model(
        input_ids=prompt_ids.to(device),
        attention_mask=prompt_mask.to(device),
        use_cache=True,
        past_key_values=None,
        return_dict=True,
    )
    next_logits = out.logits[:, -1, :]
    return ModelState(past_key_values=out.past_key_values, next_logits=next_logits, seq_len=int(prompt_ids.shape[1]))


@torch.inference_mode()
def step_state_with_token(
    model: AutoModelForCausalLM,
    state: ModelState,
    token_id: int,
    device: torch.device,
) -> ModelState:
    # 将一个“已接受 token”推进到状态中。
    # 注意：这里是增量前向（带 past_key_values），避免重复计算整段前缀。
    token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    attn = torch.ones((1, state.seq_len + 1), dtype=torch.long, device=device)
    out = model(
        input_ids=token,
        attention_mask=attn,
        use_cache=True,
        past_key_values=state.past_key_values,
        return_dict=True,
    )
    next_logits = out.logits[:, -1, :]
    return ModelState(
        past_key_values=out.past_key_values,
        next_logits=next_logits,
        seq_len=state.seq_len + 1,
    )


@torch.inference_mode()
def advance_state_with_tokens(
    model: AutoModelForCausalLM,
    state: ModelState,
    token_ids: Sequence[int],
    device: torch.device,
) -> ModelState:
    # 批量推进多个已接受 token（一次前向），
    # 比逐 token 调 step_state_with_token 更省调度开销。
    if not token_ids:
        return state

    input_ids = torch.tensor([list(int(t) for t in token_ids)], dtype=torch.long, device=device)
    attn = torch.ones((1, state.seq_len + input_ids.shape[1]), dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        past_key_values=state.past_key_values,
        return_dict=True,
    )
    next_logits = out.logits[:, -1, :]
    return ModelState(
        past_key_values=out.past_key_values,
        next_logits=next_logits,
        seq_len=state.seq_len + input_ids.shape[1],
    )


def argmax_id(logits: torch.Tensor) -> int:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return int(logits.argmax(dim=-1).item())


def logp_of(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.log_softmax(logits, dim=-1)[0, int(token_id)].item())


@torch.inference_mode()
def propose_k_tokens_with_temp_state(
    model: AutoModelForCausalLM,
    state: ModelState,
    k: int,
    device: torch.device,
) -> Tuple[List[int], List[torch.Tensor]]:
    # draft 在“临时状态”上滚动生成 K 个候选 token。
    # 这里不会改动真实 draft_state（相当于只做 proposal，不提交）。
    proposed: List[int] = []
    draft_logits_per_pos: List[torch.Tensor] = []

    temp_state = ModelState(
        past_key_values=state.past_key_values,
        next_logits=state.next_logits,
        seq_len=state.seq_len,
    )

    for _ in range(k):
        cur_logits = temp_state.next_logits
        token = argmax_id(cur_logits)
        proposed.append(int(token))
        draft_logits_per_pos.append(cur_logits)
        temp_state = step_state_with_token(model=model, state=temp_state, token_id=token, device=device)

    return proposed, draft_logits_per_pos


@torch.inference_mode()
def get_target_verify_logits(
    model: AutoModelForCausalLM,
    state: ModelState,
    proposed_tokens: Sequence[int],
    device: torch.device,
) -> List[torch.Tensor]:
    # 目标：一次拿到 target 对 K 个候选位置的验证 logits。
    # 约定：返回的第 i 个 logits 对应 “proposal 第 i 个 token”的验证分布。
    if not proposed_tokens:
        return []

    logits_per_pos: List[torch.Tensor] = [state.next_logits]

    if len(proposed_tokens) == 1:
        return logits_per_pos

    input_ids = torch.tensor([list(int(t) for t in proposed_tokens[:-1])], dtype=torch.long, device=device)
    attn = torch.ones((1, state.seq_len + input_ids.shape[1]), dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        past_key_values=state.past_key_values,
        return_dict=True,
    )
    seq_logits = out.logits  # [1, K-1, V]
    for i in range(seq_logits.shape[1]):
        logits_per_pos.append(seq_logits[:, i, :])

    return logits_per_pos


def find_first_reject_pos(
    proposed_tokens: Sequence[int],
    target_logits_per_pos: Sequence[torch.Tensor],
) -> int:
    # 找到第一个不通过 target top-1 验收的位置。
    # 返回 -1 表示 K 个候选全部通过。
    for idx, token in enumerate(proposed_tokens):
        tgt_pred = argmax_id(target_logits_per_pos[idx])
        if int(token) != int(tgt_pred):
            return idx
    return -1


def should_override(
    mode: str,
    draft_token_id: int,
    base_token_id: int,
    delta_logp: float,
    target_opp: float,
    tau_delta: float,
    tau_target_opp: float,
) -> Tuple[bool, str]:
    # 拒绝点复判规则（v0/v1/v2）
    # - 先要求 draft 与 base 分歧
    # - 再按模式追加阈值约束
    if draft_token_id == base_token_id:
        return False, "no_draft_base_divergence"

    if mode == "divergence_v0":
        return True, "v0_divergence"

    if mode == "divergence_v1":
        if delta_logp > tau_delta:
            return True, "v1_delta_pass"
        return False, "v1_delta_fail"

    if mode == "divergence_v2":
        if delta_logp <= tau_delta:
            return False, "v2_delta_fail"
        if target_opp >= tau_target_opp:
            return False, "v2_target_opp_fail"
        return True, "v2_pass"

    return False, "mode_no_override"
