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

    # TARGET 模型（70B FP8）：已经是 FP8 格式，完全移除量化配置参数
    # 使用 max_memory 限制显存使用，为其他模型留出空间
    target_kwargs = dict(common_kwargs)
    if "FP8" in ModelIDs.TARGET or "70B" in ModelIDs.TARGET:
        # 完全移除 quantization_config，而不是设为 None
        target_kwargs.pop("quantization_config", None)
        # FP8 模型让 transformers 自动处理，不强制指定 dtype
        if Hardware.TORCH_DTYPE == "auto":
            target_kwargs["torch_dtype"] = None
        # 限制 70B 模型显存使用：只使用 GPU 0 和 GPU 2（避免占用正在使用的 GPU）
        # 每张 GPU 最多 100GB，留出空间给其他模型
        max_memory = {0: "100GiB", 2: "100GiB"}
        max_memory["cpu"] = "200GiB"  # CPU offloading 备用
        target_kwargs["max_memory"] = max_memory
        # 让 transformers 自动分配到 GPU 0 和 2（通过 max_memory 限制）
        target_kwargs["device_map"] = "auto"
    target = AutoModelForCausalLM.from_pretrained(ModelIDs.TARGET, **target_kwargs)
    
    # 8B 模型分配到 GPU 6（避免占用其他 GPU）
    base_kwargs = dict(common_kwargs)
    base_kwargs["max_memory"] = {6: "20GiB"}  # 8B 模型 4bit 量化后约 4-5GB，留出余量
    draft_base = AutoModelForCausalLM.from_pretrained(ModelIDs.DRAFT_BASE, **base_kwargs)

    # 专家模型：使用用户配置的路径/ID；仅对特定模型关闭量化
    expert_id = ModelIDs.DRAFT_EXPERT
    expert_kwargs = dict(common_kwargs)
    expert_kwargs["max_memory"] = {6: "20GiB"}  # 8B 模型 4bit 量化后约 4-5GB，留出余量
    if "Medical-Guide-COT-llama3.2-1B" in expert_id:
        expert_kwargs["quantization_config"] = None
        expert_kwargs["torch_dtype"] = torch.float16
    draft_expert = AutoModelForCausalLM.from_pretrained(expert_id, **expert_kwargs)

    return {"target": target, "base": draft_base, "expert": draft_expert}


def get_model_and_tokenizer() -> Tuple[Dict[str, AutoModelForCausalLM], AutoTokenizer]:
    """便捷函数：同时加载模型与 tokenizer"""

    tokenizer = load_tokenizer()
    models = load_models()
    return models, tokenizer


