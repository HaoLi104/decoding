"""
项目入口：加载模型与数据，运行 Baseline 与 Steered 评测
"""

import os
import random
from typing import Dict, List

import numpy as np
import torch

from config import RANDOM_SEED
from data_loader import load_medqa, load_medmcqa, load_mmlu, load_medreason, prepare_batch_prompts
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

    # 可通过环境变量 TASKS 控制任务列表，逗号分隔，支持：
    # medreason, pro_med, med_gen, medmcqa
    task_env = os.environ.get("TASKS", "").split(",")
    task_env = [t.strip() for t in task_env if t.strip()]

    default_tasks = [
        ("medreason", "MedReason (医疗推理，与微调数据一致)", lambda: load_medreason(split="validation", limit=200)),
        ("pro_med", "MMLU - Professional Medicine", lambda: load_mmlu("professional_medicine", split="test", limit=200)),
        ("med_gen", "MMLU - Medical Genetics", lambda: load_mmlu("medical_genetics", split="test", limit=200)),
        ("medmcqa", "MedMCQA", lambda: load_medmcqa(split="validation", limit=200)),
    ]

    if task_env:
        task_keys = set(task_env)
        tasks = [(k, name, fn) for k, name, fn in default_tasks if k in task_keys]
    else:
        tasks = default_tasks

    # 每个任务取前 20 个样本做快速对比，可按需调大
    prompt_limit = 20
    debug_n = 5  # 每个任务打印少量示例尾部
    gen_len = 1024  # 生成上限

    for task_key, task_name, loader in tasks:
        print(f"\n==== 开始任务：{task_name} ====")
        try:
            dataset = loader()
        except Exception as e:
            print(f"[WARN] 加载任务 {task_name} 失败，跳过。错误：{e}")
            continue
        prompts = prepare_batch_prompts(tokenizer, dataset, limit=prompt_limit)
        if not prompts:
            print(f"[WARN] 任务 {task_name} 无可用样本（可能缺少答案/选项），已跳过。")
            continue

        print("开始 Baseline 评测 ...")
        baseline_acc, baseline_preds, baseline_gts = run_baseline(
            models["target"], tokenizer, prompts, max_new_tokens=gen_len, log_first_n=debug_n
        )
        report_results(f"{task_name} - Baseline", baseline_acc)
        print("Baseline 明细：")
        for i, (p, g) in enumerate(zip(baseline_preds, baseline_gts)):
            print(f"- #{i:02d} GT={g} | Pred={p}")

        print("开始 Expert-only 评测 ...")
        expert_acc, expert_preds, expert_gts = run_single(
            models["expert"], tokenizer, prompts, max_new_tokens=gen_len, log_first_n=debug_n
        )
        report_results(f"{task_name} - Expert-only", expert_acc)
        print("Expert-only 明细：")
        for i, (p, g) in enumerate(zip(expert_preds, expert_gts)):
            print(f"- #{i:02d} GT={g} | Pred={p}")

        print("开始 Steered 评测 ...")
        steered_acc, steered_preds, steered_gts = run_steered(
            models, tokenizer, prompts, max_new_tokens=gen_len, log_first_n=debug_n
        )
        report_results(f"{task_name} - Steered", steered_acc)
        print("Steered 明细：")
        for i, (p, g) in enumerate(zip(steered_preds, steered_gts)):
            print(f"- #{i:02d} GT={g} | Pred={p}")

    print("评测完成，对比总结：")
    print(f"- Baseline 准确率: {baseline_acc:.4f}")
    print(f"- Expert-only 准确率: {expert_acc:.4f}")
    print(f"- Steered  准确率: {steered_acc:.4f}")


if __name__ == "__main__":
    main()


