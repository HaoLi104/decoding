"""DAF — 根据 layer_scores_round{k}.json 生成 LLaMA-Factory LoRA YAML

输入：
  layer_scores_round{k}.json  （由 daf.fdlp_score 产出，含 4 个 subset 打分）
  template_yaml               （以 train_medmcqa_surgery_3b.yaml 为模板）

逻辑（plan 2.4.1 + 8 节决议：首版用统一 rank）：
  1. 取 subset='fdlp' 的 Top-K 模块名（含 'model.layers.{i}.self_attn.q_proj' 风格）；
  2. 把 lora_target 设为 Top-K 模块的逗号分隔列表；
  3. lora_rank 取 r_total / K（向上取整），与 plan 一致；
  4. dataset 字段指向 build_flip_sft_data 生成的 daf_round{k}_train；
  5. output_dir 自动指向 Qwen2.5-32B-Instruct-DAF-v{round+1}-lora。

输出：
  train_daf_round{k}.yaml

用法（远端）：
  python -m daf.gen_lora_yaml \
      --layer_scores logs/daf_round0/layer_scores_round0.json \
      --template     train_medmcqa_surgery_3b.yaml \
      --base_model   /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --dataset_key  daf_round0_train \
      --output_yaml  train_daf_round0.yaml \
      --output_dir   /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v1-lora \
      --top_k 8 --r_total 128 --num_train_epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.gen_lora_yaml")


# ---------------------------------------------------------------------------
# 极简 YAML 写出器（避免引入 pyyaml 依赖；只做 dict→key:value 行写）
# ---------------------------------------------------------------------------

def _dump_yaml(d: Dict[str, Any], comments_header: List[str]) -> str:
    lines: List[str] = list(comments_header)
    lines.append("")
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, str):
            # 字符串：如果含特殊字符就加引号
            if any(c in v for c in [":", "#", "\n"]) or v == "":
                escaped = v.replace('"', '\\"')
                lines.append(f'{k}: "{escaped}"')
            else:
                lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _select_top_modules(
    layer_scores: Dict[str, Any],
    subset:       str,
    top_k:        int,
) -> List[str]:
    """从 layer_scores_round{k}.json 抽 Top-K 模块名。"""
    summary = layer_scores.get("summary", {}).get(subset)
    if not summary:
        raise ValueError(f"layer_scores 中找不到 subset='{subset}'，可选: "
                         f"{list(layer_scores.get('summary', {}).keys())}")
    modules = summary.get("top_k_modules", [])
    if not modules:
        raise ValueError(f"subset='{subset}' 的 top_k_modules 为空")
    names = [m["name"] for m in modules[:top_k]]
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF — layer_scores → LLaMA-Factory LoRA YAML")
    p.add_argument("--layer_scores", required=True, help="layer_scores_round{k}.json")
    p.add_argument("--template",     required=True, help="模板 yaml（train_medmcqa_surgery_3b.yaml）")
    p.add_argument("--base_model",   required=True, help="LoRA 基座模型路径（一般为 32B Target）")
    p.add_argument("--dataset_key",  required=True, help="LLaMA-Factory dataset_info.json 中注册的训练集 key")
    p.add_argument("--output_yaml",  required=True, help="生成的 yaml 路径")
    p.add_argument("--output_dir",   required=True, help="LoRA adapter 输出目录")

    p.add_argument("--subset",       default="fdlp",
                   help="使用哪个 subset 的 Top-K 模块作为 lora_target（默认 fdlp）")
    p.add_argument("--top_k",        type=int, default=8,   help="Top-K 模块数（plan: 8）")
    p.add_argument("--r_total",      type=int, default=128, help="LoRA 总 rank 预算（plan: 128）")
    p.add_argument("--num_train_epochs", type=float, default=1.0,
                   help="LoRA 训练 epoch（plan: 控制 ~1000 步以内）")
    p.add_argument("--per_device_train_batch_size", type=int, default=2,
                   help="32B LoRA 单卡 micro-batch（默认 2，可压到 1）")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1.0e-4,
                   help="LoRA lr（PEFT 通常 5e-5 ~ 2e-4，默认 1e-4）")
    p.add_argument("--cutoff_len",    type=int, default=1024)
    p.add_argument("--lora_alpha",    type=int, default=None,
                   help="默认 = 2 × lora_rank（PEFT 常用配置）")
    p.add_argument("--lora_dropout",  type=float, default=0.05)

    p.add_argument("--dataset_dir", default="/data/ocean/decoding/LLaMA-Factory/data")
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    layer_scores = json.loads(Path(args.layer_scores).read_text(encoding="utf-8"))
    top_modules = _select_top_modules(
        layer_scores=layer_scores, subset=args.subset, top_k=args.top_k,
    )

    # LLaMA-Factory 的 lora_target 接收"模块的最后一个名字"或"完整路径"。
    # 实测 LLaMA-Factory >= 0.9 支持完整路径（包含 `model.layers.18.self_attn.q_proj`）；
    # 为兼容性更高，这里保留完整路径，依赖 LLaMA-Factory 的层级匹配。
    lora_target_str = ",".join(top_modules)

    lora_rank = max(1, math.ceil(args.r_total / max(args.top_k, 1)))
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * lora_rank

    yaml_dict: Dict[str, Any] = {
        # --- 1. 模型设置 ---
        "model_name_or_path":  args.base_model,
        "trust_remote_code":   True,
        "stage":               "sft",
        "do_train":            True,
        "finetuning_type":     "lora",
        "lora_target":         lora_target_str,
        "lora_rank":           lora_rank,
        "lora_alpha":          lora_alpha,
        "lora_dropout":        args.lora_dropout,
        # --- 2. 数据集设置 ---
        "dataset_dir":         args.dataset_dir,
        "dataset":             args.dataset_key,
        "template":            "qwen",
        "cutoff_len":          args.cutoff_len,
        "val_size":            0.0,
        "overwrite_cache":     True,
        "preprocessing_num_workers": 8,
        # --- 3. 训练核心参数 ---
        "learning_rate":       args.learning_rate,
        "num_train_epochs":    args.num_train_epochs,
        "lr_scheduler_type":   "cosine",
        "warmup_ratio":        0.05,
        "weight_decay":        0.01,
        "max_grad_norm":       1.0,
        # --- 4. 单卡 H200 批次设置 ---
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size":  args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16":                True,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "flash_attn":          "auto",
        # --- 5. 保存与日志 ---
        "output_dir":          args.output_dir,
        "logging_steps":       args.logging_steps,
        "save_strategy":       args.save_strategy,
        "save_total_limit":    args.save_total_limit,
        "plot_loss":           True,
        "report_to":           "none",
    }

    comments_header = [
        "### ============================================================================",
        "### LLaMA-Factory LoRA YAML — DAF Round (auto-generated by daf.gen_lora_yaml)",
        f"### subset             : {args.subset}",
        f"### top_k              : {args.top_k}",
        f"### r_total            : {args.r_total}",
        f"### lora_rank/module   : {lora_rank}",
        f"### lora_alpha         : {lora_alpha}",
        f"### lora_target ({len(top_modules)} 个模块):",
        *[f"###   - {m}" for m in top_modules],
        f"### dataset_key        : {args.dataset_key}",
        f"### base_model         : {args.base_model}",
        f"### output_dir         : {args.output_dir}",
        "### ============================================================================",
    ]

    yaml_text = _dump_yaml(yaml_dict, comments_header=comments_header)
    out_path = Path(args.output_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")

    logger.info("✓ 已生成 LoRA YAML: %s", out_path)
    logger.info("  Top-%d 模块: %s", len(top_modules), top_modules)
    logger.info("  lora_rank=%d  lora_alpha=%d", lora_rank, lora_alpha)


if __name__ == "__main__":
    main()
