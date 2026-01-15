"""
配置管理模块
"""

from dataclasses import dataclass

# 随机种子确保复现性
RANDOM_SEED: int = 42


@dataclass(frozen=True)
class ModelIDs:
    """模型 ID 常量"""

    # 使用本地 ModelScope 下载后的路径，避免外网访问受限
    TARGET: str = "/data/ocean/decoding/model/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"
    DRAFT_BASE: str = "/data/ocean/decoding/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # 领域专家模型 - MedReason-8B
    DRAFT_EXPERT: str = "/data/ocean/decoding/model/UCSC-VLAA/MedReason-8B"
    # 之前的专家模型路径（已注释）
    # DRAFT_EXPERT: str = "/data/ocean/decoding/model/dir"  # HuatuoGPT-o1-8B
    


@dataclass(frozen=True)
class HyperParams:
    """超参数配置"""

    BASE_ALPHA: float = 1.0  # 初始引导强度
    ADAPTIVE_SCALE: float = 4.0  # 自适应放大因子
    MAX_BOOST: float = 0.5  # 最大额外放大倍率（适当降低避免偏移过大）
    DISTANCE_METRIC: str = "l2"  # 可选 l2 / jsd，默认 l2
    REPETITION_PENALTY: float = 1.1  # 简单重复惩罚，防止单 token 循环


@dataclass(frozen=True)
class Hardware:
    """硬件加载相关参数"""

    LOAD_IN_4BIT: bool = True  # 指定 4bit 量化以节省显存
    DEVICE_MAP: str = "auto"  # 让 transformers 自动分配设备
    TORCH_DTYPE: str = "auto"  # 数据类型自动推断


