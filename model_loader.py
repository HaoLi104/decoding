"""
模型加载模块：负责同时加载三套模型与共享 Tokenizer
"""

from typing import Dict, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import Hardware, ModelIDs


def _build_bnb_config() -> BitsAndBytesConfig:
    """构建 bitsandbytes 配置，确保 4bit 量化"""

    if not Hardware.LOAD_IN_4BIT:
        return BitsAndBytesConfig(load_in_4bit=False)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer() -> AutoTokenizer:
    """加载通用 Tokenizer（所有模型共享）

    返回:
        AutoTokenizer: 适配 Llama-3 chat 模型的 Tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        ModelIDs.TARGET,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_models() -> Dict[str, AutoModelForCausalLM]:
    """一次性加载 Target / Draft Base / Draft Expert 三个模型

    返回:
        dict: {"target": model, "base": model, "expert": model}
    """

    quant_config = _build_bnb_config()
    common_kwargs = {
        "torch_dtype": (
            None if Hardware.TORCH_DTYPE == "auto" else getattr(torch, Hardware.TORCH_DTYPE)
        ),
        "device_map": Hardware.DEVICE_MAP,
        "trust_remote_code": True,
        "quantization_config": quant_config,
    }

    target = AutoModelForCausalLM.from_pretrained(ModelIDs.TARGET, **common_kwargs)
    draft_base = AutoModelForCausalLM.from_pretrained(ModelIDs.DRAFT_BASE, **common_kwargs)
    expert_kwargs = dict(common_kwargs)

    # 某些小模型（如 alpha-ai/Medical-Guide-COT-llama3.2-1B）在 4bit 下会报错，
    # 对专家模型关闭量化，直接用 FP16 加载。
    if "Medical-Guide-COT-llama3.2-1B" in ModelIDs.DRAFT_EXPERT:
        expert_kwargs["quantization_config"] = None
        expert_kwargs["torch_dtype"] = torch.float16
    draft_expert = AutoModelForCausalLM.from_pretrained(ModelIDs.DRAFT_EXPERT, **expert_kwargs)

    return {"target": target, "base": draft_base, "expert": draft_expert}


def get_model_and_tokenizer() -> Tuple[Dict[str, AutoModelForCausalLM], AutoTokenizer]:
    """便捷函数：同时加载模型与 tokenizer"""

    tokenizer = load_tokenizer()
    models = load_models()
    return models, tokenizer


