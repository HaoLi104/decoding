"""
Checkpoint 扫描评测脚本 — 找 Delta(Draft-Base) 拐点，选最优 epoch。

注意：必须在所有 import 之前设置 HF_DATASETS_OFFLINE=1，
      否则 datasets 库在 builder 初始化阶段会尝试联网验证文件列表，
      在无法访问 huggingface.co 的机器上会超时报错。
      medmcqa 已在训练阶段下载到本地缓存，离线模式直接复用缓存。
"""

# from __future__ 必须紧跟 docstring，不能在任何其他 import 语句之后
from __future__ import annotations

import os
# 必须在 datasets 相关 import 之前设置，防止联网验证 medmcqa 文件列表
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# data_loader 的 setdefault 在此之前已由顶部代码处理
from data_loader import format_prompt, load_medmcqa

# ---------------------------------------------------------------------------
# 答案抽取（与 run_baseline.py 完全对齐）
# ---------------------------------------------------------------------------
import re as _re

_RE_THINK        = _re.compile(r"<think>.*?</think>", _re.DOTALL | _re.IGNORECASE)
_RE_ANSWER_BLOCK = _re.compile(r"<answer>(.*?)</answer>", _re.DOTALL | _re.IGNORECASE)
_STRONG_PATTERNS = [
    _re.compile(r"Final\s+answer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", _re.IGNORECASE),
    _re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", _re.IGNORECASE),
    _re.compile(r"\bAnswer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", _re.IGNORECASE),
]
_RE_LASTLINE = _re.compile(r"^\s*(?:option\s*)?([A-D])\s*[.)]*\s*$", _re.IGNORECASE)


def extract_answer(response: str) -> str:
    if not response:
        return ""
    text  = _RE_THINK.sub("", response)
    m     = _RE_ANSWER_BLOCK.search(text)
    scope = m.group(1) if m else text[-1500:]
    for pat in _STRONG_PATTERNS:
        hits = list(pat.finditer(scope))
        if hits:
            return hits[-1].group(1).upper()
    lines = [ln.strip() for ln in scope.splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        mm = _RE_LASTLINE.match(ln)
        if mm:
            return mm.group(1).upper()
    return ""


# ---------------------------------------------------------------------------
# 单个 checkpoint 评测
# ---------------------------------------------------------------------------

def eval_one_checkpoint(ckpt_path: str, dataset, max_new_tokens: int = 256) -> dict:
    print(f"\n[加载] {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total_tokens = 0
    total_time   = 0.0

    for idx in tqdm(range(len(dataset)), desc=f"评测 {Path(ckpt_path).name}"):
        item = dataset[idx]
        prompt_text = format_prompt(tokenizer, item["question"], item["options"],
                                    dataset_name="medmcqa")
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids  = enc["input_ids"].to(device)    # shape: [1, L_in]
        prompt_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        gen_ids  = out_ids[0, prompt_len:]          # shape: [L_gen]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_gen    = (gen_ids != tokenizer.eos_token_id).sum().item()
        total_tokens += n_gen
        total_time   += elapsed

        pred = extract_answer(gen_text)
        gold = str(item.get("answer_idx", item.get("answer", ""))).strip().upper()
        correct += int(pred == gold and pred != "")

    n = len(dataset)
    acc = correct / n if n else 0.0
    tps = total_tokens / total_time if total_time else 0.0

    # 显式释放显存，避免下一个 checkpoint 加载时 OOM
    del model
    torch.cuda.empty_cache()

    return {"ckpt": ckpt_path, "acc": acc, "n": n, "correct": correct, "tps": tps}


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="扫描所有 checkpoint，找最优 epoch")
    # --ckpt_dir 与 --checkpoint_dir 均可接受（兼容旧调用方式）
    parser.add_argument("--ckpt_dir", "--checkpoint_dir", dest="ckpt_dir", required=True,
                        help="模型输出目录（含 checkpoint-* 子目录）")
    parser.add_argument("--base_acc",  type=float, default=None,
                        help="Base-3B 在同数据集上的 acc（手动指定；与 --base_model 互斥）")
    parser.add_argument("--base_model", type=str, default=None,
                        help="若指定，则自动运行 Base 模型并计算 base_acc（会消耗额外时间）")
    parser.add_argument("--dataset",   type=str,   default="medmcqa",
                        choices=["medmcqa"], help="评测数据集（当前仅支持 medmcqa）")
    parser.add_argument("--subject",   type=str,   default="",
                        help="若指定，仅评测该 subject_name 的题目（如 Surgery）")
    parser.add_argument("--split",     type=str,   default="validation",
                        help="数据集分片（default: validation）")
    parser.add_argument("--limit",     type=int,   default=300,
                        help="评测样本数上限（0=全量）")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out",       required=True, help="结果 JSON 输出路径")
    args = parser.parse_args()

    if args.base_acc is None and args.base_model is None:
        parser.error("必须指定 --base_acc 或 --base_model 之一")

    # 收集所有 checkpoint 目录（按 step 数排序）
    ckpt_root = Path(args.ckpt_dir)
    ckpt_dirs = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    # 也包含最终模型（output_dir 本身，如果有 config.json）
    if (ckpt_root / "config.json").exists():
        ckpt_dirs.append(ckpt_root)

    if not ckpt_dirs:
        print(f"[ERROR] 在 {ckpt_root} 中未找到任何 checkpoint-* 目录！")
        return

    print(f"发现 {len(ckpt_dirs)} 个 checkpoint：")
    for d in ckpt_dirs:
        print(f"  {d}")

    print(f"\n加载 MedMCQA {args.split}（limit={args.limit}，subject='{args.subject or '全科'}'）")
    dataset = load_medmcqa(split=args.split, limit=args.limit, subject=args.subject)
    print(f"  共 {len(dataset)} 条样本")

    # 计算 base_acc（若未手动指定则自动运行 Base 模型）
    base_acc: float
    if args.base_acc is not None:
        base_acc = args.base_acc
        print(f"  [base_acc] 使用手动指定值: {base_acc:.4f}")
    else:
        print(f"\n  [base_acc] 正在用 {args.base_model} 计算 Base baseline ...")
        base_result = eval_one_checkpoint(
            args.base_model, dataset, max_new_tokens=args.max_new_tokens
        )
        base_acc = base_result["acc"]
        print(f"  [base_acc] 计算完成: {base_acc:.4f}  (n={base_result['n']})")

    results = []
    for ckpt_path in ckpt_dirs:
        r = eval_one_checkpoint(
            str(ckpt_path), dataset,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(r)

    # 打印对比表
    print(f"\n{'='*65}")
    print(f"{'Checkpoint':<40} {'acc':>6} {'Delta(D-B)':>12} {'tps':>7}")
    print(f"{'='*65}")
    print(f"  {'[Base-3B baseline]':<38} {base_acc:.4f} {'---':>12}")
    best = max(results, key=lambda r: r["acc"])
    for r in results:
        name  = Path(r["ckpt"]).name
        delta = r["acc"] - base_acc
        flag  = " ← best" if r["ckpt"] == best["ckpt"] else ""
        print(f"  {name:<38} {r['acc']:.4f} {delta:+12.4f} {r['tps']:>7.1f}{flag}")
    print(f"{'='*65}")
    print(f"\n最优 checkpoint：{best['ckpt']}")
    print(f"  acc={best['acc']:.4f}  Delta={best['acc']-base_acc:+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果已保存：{args.out}")


if __name__ == "__main__":
    main()
