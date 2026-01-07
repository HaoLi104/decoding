"""
离线下载脚本：将需要的评测数据集缓存到指定目录
默认下载到 /data/ocean/decoding/datasets
运行：python download_datasets.py
"""

import os

from datasets import load_dataset

TARGET_CACHE = "/data/ocean/decoding/datasets"


def main() -> None:
    os.environ["HF_DATASETS_CACHE"] = TARGET_CACHE
    os.makedirs(TARGET_CACHE, exist_ok=True)

    print(f"缓存目录: {TARGET_CACHE}")

    print("下载 MMLU - Professional Medicine ...")
    load_dataset("cais/mmlu", "professional_medicine", split="test")

    print("下载 MMLU - Medical Genetics ...")
    load_dataset("cais/mmlu", "medical_genetics", split="test")

    print("下载 MedMCQA ...")
    load_dataset("medmcqa", split="validation")

    print("完成。")


if __name__ == "__main__":
    main()


