"""
简单脚本：读取 ModelIDs.TARGET 路径下 config.json 的 vocab_size
运行：python print_vocab_size.py
"""

import json
import os

from config import ModelIDs


def main() -> None:
    config_path = os.path.join(ModelIDs.TARGET, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到 config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    vocab_size = cfg.get("vocab_size")
    print(f"config.json vocab_size = {vocab_size}")


if __name__ == "__main__":
    main()


