"""
前向传播原子操作层 — 严格基于 model.forward()，禁用 model.generate()

所有上层模块（engine_state / dual_stream / shadow_sync）均通过此模块
的函数与模型交互，不得直接调用 model.forward()，以便统一管理：
  - attention_mask / position_ids 的正确构造
  - StaticCache 的 cache_position 参数传递
  - 输出张量的维度注释与切片

提供的 5 个原子操作：
  1. prefill()                  Prefill 阶段：全 prompt 一次前向
  2. decode_step()              单 Token 增量解码步
  3. decode_batch_verify()      批量验证/追赶（Target verify / Base sync）
  4. decode_batch_hidden_only() 仅到 hidden_state，跳过 LM Head（Shadow Sync 用）
  5. extract_logits_at_positions() 对指定位置的 hidden_state 执行 LM Head（Lazy Eval）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache


# ---------------------------------------------------------------------------
# 1. Prefill
# ---------------------------------------------------------------------------

@torch.inference_mode()
def prefill(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,   # shape: [1, prompt_len]
    cache: StaticCache,
) -> torch.Tensor:
    """Prefill 阶段：将完整 prompt 一次性前向，填充 StaticCache。

    Args:
        model:     目标模型（已 eval + compile）
        input_ids: prompt token id 序列，shape [1, prompt_len]
        cache:     预分配好的 StaticCache，将被本次 forward 填充

    Returns:
        logits: prompt 最后一个 token 位置的 next-token logits
                shape: [1, vocab_size]
    """
    prompt_len = input_ids.shape[1]
    device = input_ids.device

    # attention_mask: 全 1（无 padding），shape [1, prompt_len]
    attention_mask = torch.ones((1, prompt_len), dtype=torch.long, device=device)

    # cache_position: [0, 1, ..., prompt_len-1]，StaticCache 用于定位写入位置
    cache_position = torch.arange(prompt_len, dtype=torch.long, device=device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    # out.logits: shape [1, prompt_len, vocab_size]
    # 取最后一个位置作为"下一个 token"的预测 logits
    logits = out.logits[:, -1, :]  # shape: [1, vocab_size]
    return logits


# ---------------------------------------------------------------------------
# 2. decode_step（单步增量解码）
# ---------------------------------------------------------------------------

@torch.inference_mode()
def decode_step(
    model: AutoModelForCausalLM,
    token_id: torch.Tensor,    # shape: [1, 1]
    cache: StaticCache,
    position_id: int,
) -> torch.Tensor:
    """单 Token 增量解码步。

    以 token_id 为输入，利用 cache 中已有的 KV 做增量 forward，
    返回该 token 之后的 next-token logits。

    Args:
        model:       目标模型
        token_id:    当前 token，shape [1, 1]
        cache:       当前 StaticCache（将被本步 forward 追加写入）
        position_id: 当前 token 在序列中的绝对位置（= prompt_len + 已生成步数）

    Returns:
        logits: shape [1, vocab_size]
    """
    device = token_id.device

    # attention_mask 需覆盖历史 + 当前位置，长度 = position_id + 1
    attention_mask = torch.ones((1, position_id + 1), dtype=torch.long, device=device)

    # cache_position 指明本次写入 cache 的位置
    cache_position = torch.tensor([position_id], dtype=torch.long, device=device)

    out = model(
        input_ids=token_id,            # shape: [1, 1]
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    logits = out.logits[:, -1, :]  # shape: [1, vocab_size]
    return logits


# ---------------------------------------------------------------------------
# 3. decode_batch_verify（批量验证/追赶）
# ---------------------------------------------------------------------------

@torch.inference_mode()
def decode_batch_verify(
    model: AutoModelForCausalLM,
    token_ids: torch.Tensor,   # shape: [1, k]
    cache: StaticCache,
    start_position: int,
) -> torch.Tensor:
    """批量验证/追赶：一次前向处理 k 个 token，返回每个位置的 logits。

    用途：
      - Target 批量验证 Draft 提案的 K 个候选 token
      - Base/Draft 批量追赶已接受的 K 个 token

    Args:
        model:          目标模型
        token_ids:      k 个 token，shape [1, k]
        cache:          当前 StaticCache
        start_position: 第一个 token 在序列中的绝对位置

    Returns:
        logits: 每个输入 token 位置的 next-token logits
                shape: [1, k, vocab_size]
                logits[:, i, :] 对应"基于前 i+1 个 token 预测第 i+1 位置之后"
    """
    k = token_ids.shape[1]
    device = token_ids.device

    # attention_mask: 覆盖 [0, start_position + k)，历史 + 当前批次
    seq_len_total = start_position + k
    attention_mask = torch.ones((1, seq_len_total), dtype=torch.long, device=device)

    # cache_position: [start_position, ..., start_position + k - 1]
    cache_position = torch.arange(
        start_position, start_position + k, dtype=torch.long, device=device
    )

    out = model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    # out.logits: shape [1, k, vocab_size]
    return out.logits


# ---------------------------------------------------------------------------
# 4. decode_batch_hidden_only（仅到 hidden_state，跳过 LM Head）
# ---------------------------------------------------------------------------

@torch.inference_mode()
def decode_batch_hidden_only(
    model: AutoModelForCausalLM,
    token_ids: torch.Tensor,   # shape: [1, k]
    cache: StaticCache,
    start_position: int,
) -> torch.Tensor:
    """仅前向到最后一层 Transformer hidden state，跳过 LM Head 矩阵乘。

    用于 Shadow Sync 架构中 Base 模型的 Lazy Evaluation：
      - Base 先以 Draft 相同 token 序列做一次高效批量 forward（不执行 LM Head）
      - 后续仅在需要 logits 的候选位置按需执行 LM Head（见 extract_logits_at_positions）

    实现方式：设置 output_hidden_states=True 并直接截断 lm_head 调用
    （通过 output_logits=False 或直接返回 hidden_states）

    Args:
        model:          目标模型（AutoModelForCausalLM）
        token_ids:      k 个 token，shape [1, k]
        cache:          当前 StaticCache
        start_position: 第一个 token 的绝对位置

    Returns:
        hidden_states: 最后一层 Transformer 输出，shape [1, k, hidden_dim]
    """
    k = token_ids.shape[1]
    device = token_ids.device

    seq_len_total = start_position + k
    attention_mask = torch.ones((1, seq_len_total), dtype=torch.long, device=device)
    cache_position  = torch.arange(
        start_position, start_position + k, dtype=torch.long, device=device
    )

    out = model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    # out.hidden_states: tuple，最后一个元素为最终 Transformer 层输出
    # shape: [1, k, hidden_dim]
    hidden_states = out.hidden_states[-1]  # shape: [1, k, hidden_dim]
    return hidden_states


# ---------------------------------------------------------------------------
# 5. extract_logits_at_positions（Lazy LM Head）
# ---------------------------------------------------------------------------

def extract_logits_at_positions(
    model: AutoModelForCausalLM,
    hidden_states: torch.Tensor,   # shape: [1, k, hidden_dim]
    positions: list[int],          # 相对位置索引（在 k 个 token 中的下标）
) -> torch.Tensor:
    """对指定位置的 hidden state 执行 LM Head 矩阵乘，提取局部 logits。

    Shadow Sync 架构中，Base 仅对 Draft 提案的候选位置（而非全 K 位）
    计算 LM Head，避免全序列 Softmax + 采样的额外开销。

    Args:
        model:         AutoModelForCausalLM（需含 lm_head 属性）
        hidden_states: Transformer 最后层输出，shape [1, k, hidden_dim]
        positions:     需要提取 logits 的位置索引列表（0-based，相对于 k）

    Returns:
        logits: shape [1, len(positions), vocab_size]
    """
    if not positions:
        raise ValueError("positions 列表不能为空")

    # 提取指定位置的 hidden states，shape: [1, len(positions), hidden_dim]
    pos_tensor = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
    selected_hidden = hidden_states[:, pos_tensor, :]  # shape: [1, P, hidden_dim]

    # 执行 LM Head（线性投影：hidden_dim -> vocab_size）
    # lm_head 通常是 nn.Linear，不含 bias 或含 bias，直接调用即可
    lm_head = model.lm_head
    logits = lm_head(selected_hidden)  # shape: [1, P, vocab_size]
    return logits


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def sample_token(
    logits: torch.Tensor,   # shape: [1, vocab_size]
    temperature: float = 0.0,
) -> int:
    """从 logits 采样下一个 token。

    temperature=0 时退化为贪婪解码（argmax）。

    Args:
        logits:      next-token logits，shape [1, vocab_size]
        temperature: 采样温度（0=贪婪）

    Returns:
        token_id (int)
    """
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())

    # 随机采样：对 logits / T 做 softmax 后 multinomial
    probs = F.softmax(logits / temperature, dim=-1)  # shape: [1, vocab_size]
    token_id = torch.multinomial(probs, num_samples=1).item()
    return int(token_id)


def prob_of_token(
    logits: torch.Tensor,   # shape: [1, vocab_size]
    token_id: int,
    temperature: float = 1.0,
) -> float:
    """计算 token_id 在给定温度下的归一化概率。

    Args:
        logits:      next-token logits，shape [1, vocab_size]
        token_id:    目标 token
        temperature: Softmax 温度（t_fixed 或 t_sample）

    Returns:
        P(token_id | logits, temperature) ∈ (0, 1)
    """
    probs = F.softmax(logits / temperature, dim=-1)  # shape: [1, vocab_size]
    return float(probs[0, token_id].item())
