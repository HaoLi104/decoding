"""
三模型加载管线 v2 — 严格绑定 cuda:0 / bfloat16 / torch.compile
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_v2 import HardwareConfig, ModelPaths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类：三模型 Bundle
# ---------------------------------------------------------------------------

@dataclass
class TriModelBundle:
    """持有三个已加载模型和共享 tokenizer 的容器。

    target:    Qwen2.5-32B-Instruct（验收官）
    draft:     Qwen2.5-3B-Instruct-Medical（提案者）
    base:      Qwen2.5-3B-Instruct（常识对照组）
    tokenizer: 与 target 模型共享，Qwen 系列词表一致
    """
    target:    AutoModelForCausalLM
    draft:     AutoModelForCausalLM
    base:      AutoModelForCausalLM
    tokenizer: AutoTokenizer


# ---------------------------------------------------------------------------
# 辅助：解析 dtype 字符串
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
    }
    key = dtype_str.strip().lower()
    if key not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_str}，可选: {list(mapping.keys())}")
    return mapping[key]


# ---------------------------------------------------------------------------
# 核心：加载单个模型
# ---------------------------------------------------------------------------

def load_single_model(
    model_path: str,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    compile_mode: Optional[str] = "reduce-overhead",
) -> AutoModelForCausalLM:
    """加载单个因果语言模型并应用 torch.compile。

    Args:
        model_path:   HuggingFace Hub ID 或本地路径。
        device:       目标设备，严格绑定 cuda:0。
        dtype:        模型精度，全局使用 bfloat16。
        compile_mode: torch.compile 模式；传 None 跳过编译（调试用）。

    Returns:
        处于 eval 模式的 AutoModelForCausalLM 实例。
    """
    logger.info("加载模型: %s  device=%s  dtype=%s", model_path, device, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=str(device),   # 严格绑定单卡，禁止 auto 跨卡分配
        trust_remote_code=True,
    )
    model.eval()

    if compile_mode is not None:
        logger.info("torch.compile 编译模型: mode=%s", compile_mode)
        # 对 forward 方法应用编译优化，规避 CUDA Graph 手写复杂度
        model = torch.compile(model, mode=compile_mode)

    return model


# ---------------------------------------------------------------------------
# 核心：加载 Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_path: str) -> AutoTokenizer:
    """加载 tokenizer，确保 pad_token 有效。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ---------------------------------------------------------------------------
# 核心：三模型一站式加载
# ---------------------------------------------------------------------------

def load_tri_models(
    paths: ModelPaths = ModelPaths(),
    hw: HardwareConfig = HardwareConfig(),
) -> TriModelBundle:
    """按实验规范加载 Target / Draft / Base 三个模型。

    加载顺序：Target(32B) → Base(3B) → Draft(3B)，依次释放加载峰值显存压力。
    三个模型均绑定 hw.device，精度统一为 hw.dtype，并应用 torch.compile。

    Args:
        paths: 三模型路径配置。
        hw:    硬件约束配置（device / dtype / compile_mode）。

    Returns:
        TriModelBundle
    """
    device       = torch.device(hw.device)
    dtype        = _resolve_dtype(hw.dtype)
    compile_mode = hw.compile_mode

    logger.info("=== 开始加载三模型（单卡极致内聚架构）===")
    logger.info("device=%s  dtype=%s  compile=%s", device, dtype, compile_mode)

    # #region agent log - debug ecc61b
    import json, time, os as _os
    logger.info("[DBG-ecc61b] ModelPaths.TARGET=%s", paths.TARGET)
    logger.info("[DBG-ecc61b] ModelPaths.BASE=%s", paths.BASE)
    logger.info("[DBG-ecc61b] ModelPaths.DRAFT=%s", paths.DRAFT)
    logger.info("[DBG-ecc61b] config_v2 file=%s", _os.path.abspath(__import__('config_v2').__file__))
    logger.info("[DBG-ecc61b] DRAFT local exists=%s", _os.path.isdir(paths.DRAFT))
    # #endregion

    # 1. 加载 tokenizer（以 target 为基准，Qwen 系列词表一致）
    tokenizer = load_tokenizer(paths.TARGET)

    # 2. 加载 Target（32B，约 64GB）
    target = load_single_model(paths.TARGET, device=device, dtype=dtype, compile_mode=compile_mode)
    logger.info("Target 加载完成: %s", paths.TARGET)

    # 3. 加载 Base（3B，约 6GB）
    base = load_single_model(paths.BASE, device=device, dtype=dtype, compile_mode=compile_mode)
    logger.info("Base 加载完成: %s", paths.BASE)

    # 4. 加载 Draft（3B-Medical，约 6GB）
    draft = load_single_model(paths.DRAFT, device=device, dtype=dtype, compile_mode=compile_mode)
    logger.info("Draft 加载完成: %s", paths.DRAFT)

    logger.info("=== 三模型加载完毕 ===")

    return TriModelBundle(
        target=target,
        draft=draft,
        base=base,
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# 工具：打印当前显存占用（便于调试）
# ---------------------------------------------------------------------------

def log_gpu_memory(tag: str = "") -> None:
    """打印当前 cuda:0 显存占用（MiB）。"""
    if not torch.cuda.is_available():
        return
    alloc  = torch.cuda.memory_allocated(0)  / 1024 ** 2
    reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
    logger.info("[显存] %s  allocated=%.1f MiB  reserved=%.1f MiB", tag, alloc, reserved)
