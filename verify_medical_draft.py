"""
Draft 模型快速验证脚本

微调完成后运行本脚本，验证：
  1. Chat Template 未被破坏（<|im_start|> 格式正常）
  2. 模型对医学 MCQ 能给出合理输出
  3. ΔP 信号基础验证：Draft 对医学 token 的置信度高于 Base

严格遵守架构规范：基于 model.forward() 手写解码循环，禁止 model.generate()

用法（远端机器）：
  cd /data/ocean/decoding
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=0
  python verify_medical_draft.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH  = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct"
DRAFT_PATH = "/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Medical"
DEVICE     = torch.device("cuda:0")
MAX_NEW    = 64

# 测试问题：社区获得性肺炎首选抗生素
TEST_MESSAGES = [
    {"role": "system", "content": "You are a medical expert."},
    {
        "role": "user",
        "content": (
            "Which antibiotic is first-line for community-acquired pneumonia?\n"
            "A) Amoxicillin\nB) Vancomycin\nC) Meropenem\nD) Azithromycin"
        ),
    },
]

# 期望医学领域高置信 token（用于验证 ΔP 信号）
MEDICAL_PROBE_TOKENS = ["Azithromycin", "azithromycin", "amoxicillin", "Amoxicillin"]


def _greedy_decode(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,     # shape: [1, prompt_len]
    max_new: int,
    eos_id: int,
) -> list[int]:
    """手写贪婪解码循环（model.forward() only，禁用 generate()）。"""
    generated = []
    past_key_values = None
    seq_len = input_ids.shape[1]

    with torch.inference_mode():
        # Prefill
        out = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]          # shape: [1, vocab_size]
        past_key_values = out.past_key_values

        for _ in range(max_new):
            next_id = int(logits.argmax(dim=-1).item())
            generated.append(next_id)
            if next_id == eos_id:
                break

            next_token = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
            seq_len += 1
            attn = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)

            out = model(
                input_ids=next_token,
                attention_mask=attn,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]      # shape: [1, vocab_size]
            past_key_values = out.past_key_values

    return generated


def _compute_delta_p(
    logit_draft: torch.Tensor,   # shape: [1, vocab_size]
    logit_base:  torch.Tensor,   # shape: [1, vocab_size]
    token_id:    int,
    t_fixed:     float = 1.0,
) -> tuple[float, float, float]:
    """计算 ΔP = p_draft(x) - p_base(x)，固定温度 T_fixed。"""
    p_draft = F.softmax(logit_draft / t_fixed, dim=-1)
    p_base  = F.softmax(logit_base  / t_fixed, dim=-1)
    pd = float(p_draft[0, token_id].item())
    pb = float(p_base[0,  token_id].item())
    return pd - pb, pd, pb


def main() -> None:
    print("=" * 60)
    print("Draft 模型验证：Chat Template + 医学置信度 + ΔP 信号")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 加载 Base 和 Draft
    # -----------------------------------------------------------------------
    print(f"\n[1/3] 加载 Base 模型: {BASE_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map=str(DEVICE)
    )
    base_model.eval()

    print(f"[1/3] 加载 Draft 模型: {DRAFT_PATH}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, torch_dtype=torch.bfloat16, device_map=str(DEVICE)
    )
    draft_model.eval()

    # -----------------------------------------------------------------------
    # 验证 1：Chat Template 格式
    # -----------------------------------------------------------------------
    print("\n[2/3] 验证 Chat Template（<|im_start|> 格式）")
    prompt_text = tokenizer.apply_chat_template(
        TEST_MESSAGES, tokenize=False, add_generation_prompt=True
    )
    print(f"  prompt 前 120 字符：{repr(prompt_text[:120])}")
    assert "<|im_start|>" in prompt_text, "❌ Chat Template 格式异常！缺少 <|im_start|>"
    print("  ✓ Chat Template 正常")

    # -----------------------------------------------------------------------
    # 验证 2：Draft 生成输出
    # -----------------------------------------------------------------------
    print("\n[2/3] Draft 生成输出（贪婪解码 model.forward()）")
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(DEVICE)
    eos_id    = tokenizer.eos_token_id or 0

    gen_ids  = _greedy_decode(draft_model, input_ids, MAX_NEW, eos_id)
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"  Draft 回复：{response[:200]}")

    # -----------------------------------------------------------------------
    # 验证 3：ΔP 信号（Draft vs Base 对医学 token 的置信度差）
    # -----------------------------------------------------------------------
    print("\n[3/3] ΔP 信号验证（医学 Token 置信度差）")
    with torch.inference_mode():
        draft_out = draft_model(input_ids=input_ids, use_cache=False, return_dict=True)
        base_out  = base_model( input_ids=input_ids, use_cache=False, return_dict=True)

    draft_logits = draft_out.logits[:, -1, :]  # shape: [1, vocab_size]
    base_logits  = base_out.logits[:, -1, :]   # shape: [1, vocab_size]

    print(f"  {'Token':<20}  {'ΔP':>8}  {'p_draft':>9}  {'p_base':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*9}  {'-'*8}")
    for token_text in MEDICAL_PROBE_TOKENS:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if not token_ids:
            continue
        tid = token_ids[0]
        delta_p, p_d, p_b = _compute_delta_p(draft_logits, base_logits, tid)
        marker = " ← 正向信号" if delta_p > 0.01 else ""
        print(f"  {token_text:<20}  {delta_p:+.4f}  {p_d:.6f}  {p_b:.6f}{marker}")

    print("\n✓ 验证完成。若 ΔP > 0.01 的医学 token 存在，说明微调注入了领域偏置。")
    print("=" * 60)


if __name__ == "__main__":
    main()
