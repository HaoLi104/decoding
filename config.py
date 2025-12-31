"""
配置管理模块
"""

from dataclasses import dataclass

# 随机种子确保复现性
RANDOM_SEED: int = 42


@dataclass(frozen=True)
class ModelIDs:
    """模型 ID 常量"""

    TARGET: str = "meta-llama/Llama-3.1-8B-Instruct"
    DRAFT_BASE: str = "meta-llama/Llama-3.2-1B-Instruct"
    DRAFT_EXPERT: str = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"


@dataclass(frozen=True)
class HyperParams:
    """超参数配置"""

    BASE_ALPHA: float = 1.0  # 初始引导强度
    ADAPTIVE_SCALE: float = 4.0  # 自适应放大因子
    MAX_BOOST: float = 1.5  # 最大额外放大倍率
    DISTANCE_METRIC: str = "l2"  # 可选 l2 / jsd，默认 l2


@dataclass(frozen=True)
class Hardware:
    """硬件加载相关参数"""

    LOAD_IN_4BIT: bool = True  # 指定 4bit 量化以节省显存
    DEVICE_MAP: str = "auto"  # 让 transformers 自动分配设备
    TORCH_DTYPE: str = "auto"  # 数据类型自动推断


