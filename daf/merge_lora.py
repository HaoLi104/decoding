"""DAF — peft.merge_and_unload 把 LoRA adapter 合并回基座，得到 Target_v{k+1}

策略（DAF 文档 2.4.2 节决议方式 B）：
  base + LoRA → 重 base 化为新 Target，下一轮飞轮直接使用合并后的权重。
  这样 Round k+1 的 FDLP forward / decode 都走原生权重，不再有 PEFT runtime 开销。

用法（远端 H200）：
  cd /data/ocean/decoding
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=0
  python -m daf.merge_lora \
      --base_model    /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --lora_adapter  /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v1-lora \
      --output_model  /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.merge_lora")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF — 合并 LoRA → 重 base 化的 Target_v{k+1}")
    p.add_argument("--base_model",   required=True, help="LoRA 基座（一般是上一轮 Target）")
    p.add_argument("--lora_adapter", required=True, help="LLaMA-Factory output_dir（含 adapter_config.json）")
    p.add_argument("--output_model", required=True, help="合并后的新 Target 路径")
    p.add_argument("--device_map",   default="cuda:0",
                   help="加载 base 时的 device_map（默认 cuda:0；32B 加载与合并均在单卡完成）")
    p.add_argument("--dtype",        default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--safe_serialization", action="store_true", default=True)
    return p


def _resolve_dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def main() -> None:
    args = build_arg_parser().parse_args()

    base_path  = args.base_model
    lora_path  = args.lora_adapter
    out_path   = Path(args.output_model)
    out_path.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(args.dtype)
    logger.info("加载 base: %s  dtype=%s  device_map=%s", base_path, dtype, args.device_map)

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    logger.info("加载 LoRA adapter: %s", lora_path)
    from peft import PeftModel  # 延迟导入避免不必要依赖
    peft_model = PeftModel.from_pretrained(base, lora_path, torch_dtype=dtype)

    logger.info("merge_and_unload ...")
    merged = peft_model.merge_and_unload()
    merged.eval()

    logger.info("保存到: %s  safe_serialization=%s", out_path, args.safe_serialization)
    merged.save_pretrained(str(out_path), safe_serialization=args.safe_serialization)

    # 同步 tokenizer（直接复制 base 的）
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tok.save_pretrained(str(out_path))

    logger.info("✓ 合并完成 → %s", out_path)


if __name__ == "__main__":
    main()
