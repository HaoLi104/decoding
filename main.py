"""
项目入口：加载模型与数据，运行 Baseline 与 Steered 评测
"""

import random
from typing import Dict

import numpy as np
import torch

from config import RANDOM_SEED
from data_loader import load_medqa, prepare_batch_prompts
from evaluator import run_baseline, run_single, run_steered
from model_loader import get_model_and_tokenizer


def set_seed(seed: int = RANDOM_SEED) -> None:
    """设置随机种子，保证复现"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_results(title: str, accuracy: float) -> None:
    """简洁打印评测结果"""

    print(f"[{title}] 准确率: {accuracy:.4f}")


def main() -> None:
    set_seed()

    print("正在加载模型与 tokenizer ...")
    models, tokenizer = get_model_and_tokenizer()

    print("正在加载 MedQA 测试数据 ...")
    dataset = load_medqa(split="validation", limit=100)
    prompts = prepare_batch_prompts(tokenizer, dataset, limit=50)

    print("开始 Baseline 评测 ...")
    baseline_acc, _ = run_baseline(models["target"], tokenizer, prompts)
    report_results("Baseline", baseline_acc)

    print("开始 Expert-only 评测 ...")
    expert_acc, _ = run_single(models["expert"], tokenizer, prompts)
    report_results("Expert-only", expert_acc)

    print("开始 Steered 评测 ...")
    steered_acc, _ = run_steered(models, tokenizer, prompts)
    report_results("Steered", steered_acc)

    print("评测完成，对比总结：")
    print(f"- Baseline 准确率: {baseline_acc:.4f}")
    print(f"- Expert-only 准确率: {expert_acc:.4f}")
    print(f"- Steered  准确率: {steered_acc:.4f}")


if __name__ == "__main__":
    main()


