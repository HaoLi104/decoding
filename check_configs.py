"""
排查脚本：读取三套模型配置并打印关键信息
运行：python check_configs.py
"""

import os

from transformers import AutoConfig, __version__ as transformers_version
import torch

from config import ModelIDs


def print_config(model_id: str, label: str) -> None:
    """加载并打印单个模型的关键信息"""

    print(f"\n=== {label} ===")
    if os.path.isabs(model_id):
        cfg_path = os.path.join(model_id, "config.json")
        print(f"config.json 路径: {os.path.abspath(cfg_path)} (存在: {os.path.isfile(cfg_path)})")
    else:
        print(f"使用 HF repo: {model_id}")

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"_name_or_path: {getattr(cfg, '_name_or_path', None)}")
    print(f"vocab_size   : {getattr(cfg, 'vocab_size', None)}")
    print(f"hidden_size  : {getattr(cfg, 'hidden_size', None)}")


def main() -> None:
    print_config(ModelIDs.TARGET, "TARGET")
    print_config(ModelIDs.DRAFT_BASE, "DRAFT_BASE")
    print_config(ModelIDs.DRAFT_EXPERT, "DRAFT_EXPERT")

    print("\n=== 版本信息 ===")
    print(f"transformers: {transformers_version}")
    print(f"torch       : {torch.__version__}")


if __name__ == "__main__":
    main()


