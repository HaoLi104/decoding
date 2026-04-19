"""DAF — flip 事件 → LLaMA-Factory Alpaca SFT 数据

输入：
  flip_events_round{k}.jsonl  （由 daf.run_flip_logger 产出）

输出（plan 2.4.1）：
  daf_round{k}_train.json     LLaMA-Factory alpaca 格式
  daf_round{k}_val.json       小规模验证集（默认 5%）
  daf_round{k}_train_with_meta.json   含 _meta 调试字段的副本

  并打印一段可粘贴到 LLaMA-Factory `data/dataset_info.json` 的注册片段，
  自动写入 dataset_info.json（若 --dataset_info_json 指定）。

样本构造规则：
  对每个 flip 事件 (prefix_ids, A=target_top1, B=draft_token, F=True)：
    - 正样本 (flip)     : instruction=<prefix 文本>, output=<token B 文本> → 教 Target 学习 Draft 的领域知识
    - 平衡样本 (anti-flip): instruction=<prefix 文本>, output=<token A 文本> → 教 Target 保持自身判断
  由此每个 flip 事件产生 1:1 正/平衡对，避免单边过拟合。
  额外可加 25% tatsu-lab/alpaca 通用数据作为格式锚。

内容过滤（v2 增强，针对结构 token 主导问题）：
  --exclude_special_tokens : 跳过 token_B 为 tokenizer.all_special_ids 的 flip
                             （如 <|im_end|>, <|im_start|>, eos）
  --exclude_template_words : 跳过 token_B 解码后属于答题模板词的 flip
                             （内置：Final/answer/Answer/answer:/Final answer/单字母 A-D 等）
  --min_token_id           : 跳过 token_B id < N 的 flip（默认 0=不过滤）；
                             Qwen 词表里 < ~200 的多为 ASCII 标点 / 数字 / 空白
  --keep_meta              : train.json 中保留 meta 字段（默认 False，向后兼容 LLaMA-Factory）

用法（远端）：
  # v1: 不过滤
  python -m daf.build_flip_sft_data \
      --flip_jsonl logs/daf_round0/flip_events_round0.jsonl \
      --tokenizer  /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --out_dir    /data/ocean/decoding/data --round 0 \
      --max_samples 2500 \
      --dataset_info_json /data/ocean/decoding/LLaMA-Factory/data/dataset_info.json

  # v2: 过滤结构/模板 token
  python -m daf.build_flip_sft_data \
      --flip_jsonl logs/daf_round0/flip_events_round0.jsonl \
      --tokenizer  /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --out_dir    /data/ocean/decoding/data --round 0_v2 \
      --max_samples 2500 \
      --exclude_special_tokens --exclude_template_words --min_token_id 200 \
      --dataset_info_json /data/ocean/decoding/LLaMA-Factory/data/dataset_info.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 国内服务器从镜像加载 alpaca
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from daf.flip_definition import iter_flip_records  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.build_flip_sft_data")


# ---------------------------------------------------------------------------
# tokenizer 工具
# ---------------------------------------------------------------------------

def _load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _decode_token(tok, token_id: int) -> str:
    """单 token 解码 → 字符串（保留前导空格 / 子词形态）。"""
    text = tok.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text


# 答题模板词黑名单：用于过滤"结构 flip"。
# 设计原则：以「strip().lower()」后的纯文本匹配，覆盖 MedMCQA / 通用考试模板。
_TEMPLATE_WORDS = {
    "final", "answer", "answers", "final answer", "the answer is", "therefore",
    "the", "is", "of", "and", "or", "to", "in", "a", "an",
    ":", ".", ",", "?", "!", ";", "-", "—", "(", ")", "[", "]", "{", "}",
    "</s>", "<s>", "<|endoftext|>",
    # 选项字母（单字母不能作为 flip 监督，因为它正是答案本身）
    "a", "b", "c", "d", "e",
}


def _is_template_token_text(text: str) -> bool:
    """判断 token 解码文本是否属于答题模板词。"""
    if not text:
        return True
    norm = text.strip().lower()
    if not norm:
        return True
    return norm in _TEMPLATE_WORDS


# ---------------------------------------------------------------------------
# 通用 alpaca 锚点
# ---------------------------------------------------------------------------

def _load_general_alpaca(limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out: List[Dict[str, Any]] = []
    for item in ds:
        instruction = str(item.get("instruction", "")).strip()
        inp         = str(item.get("input", "")).strip()
        output      = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        out.append({
            "instruction": instruction,
            "input":       inp,
            "output":      output,
        })
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------

def build_sft_records(
    flip_jsonl:              Path,
    tokenizer,
    max_flip_events:         int,
    max_prefix_len:          int,
    add_balance:             bool,
    rng:                     random.Random,
    exclude_special_tokens:  bool = False,
    exclude_template_words:  bool = False,
    min_token_id:            int = 0,
) -> Dict[str, Any]:
    """读 flip_jsonl，每个 flip 事件展开为 (1 + add_balance) 条 alpaca 样本。

    Args:
        flip_jsonl:             flip 事件 jsonl 路径
        tokenizer:              与 Target 共享的 tokenizer
        max_flip_events:        主循环最多消费多少 flip 事件
        max_prefix_len:         prefix token 截断上限（左截断）
        add_balance:            是否对每个 flip 事件再加一条 anti-flip(=A_t) 样本
        exclude_special_tokens: 跳过 token_B 在 tokenizer.all_special_ids 中的 flip
        exclude_template_words: 跳过 token_B 解码后属于答题模板词的 flip
        min_token_id:           跳过 token_B id < min_token_id 的 flip（默认 0=不过滤）

    Returns:
        {
          "records": list of alpaca dict (含 _meta 字段),
          "stats":   {n_consumed, n_dropped_*: ..., n_pos: ..., n_balance: ...}
        }
    """
    records: List[Dict[str, Any]] = []
    n_consumed       = 0
    n_pos            = 0
    n_balance        = 0
    n_drop_special   = 0
    n_drop_template  = 0
    n_drop_min_id    = 0
    n_drop_decode    = 0
    n_drop_eq_AB     = 0

    special_ids = set()
    if exclude_special_tokens:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    for rec in iter_flip_records(flip_jsonl):
        if n_consumed >= max_flip_events:
            break
        n_consumed += 1

        prefix = rec.prefix_ids
        if len(prefix) > max_prefix_len:
            prefix = prefix[-max_prefix_len:]
        if not prefix:
            continue

        # ---- 内容过滤（针对正样本 token_B） ----
        if exclude_special_tokens and rec.B in special_ids:
            n_drop_special += 1
            continue
        if min_token_id > 0 and rec.B < min_token_id:
            n_drop_min_id += 1
            continue

        try:
            instruction_text = tokenizer.decode(
                prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            logger.warning("prefix 解码失败 qid=%s step=%d: %s", rec.qid, rec.step, exc)
            n_drop_decode += 1
            continue

        token_B = _decode_token(tokenizer, rec.B)
        if not token_B:
            n_drop_decode += 1
            continue

        if exclude_template_words and _is_template_token_text(token_B):
            n_drop_template += 1
            continue

        records.append({
            "instruction": instruction_text,
            "input":       "",
            "output":      token_B,
            "_meta": {
                "qid": rec.qid, "step": rec.step, "kind": "flip_positive",
                "B_id": rec.B, "A_id": rec.A,
                "delta_p": rec.delta_p, "h_t": rec.h_t,
            },
        })
        n_pos += 1

        if add_balance:
            token_A = _decode_token(tokenizer, rec.A)
            if not token_A or token_A == token_B:
                n_drop_eq_AB += 1
                continue
            records.append({
                "instruction": instruction_text,
                "input":       "",
                "output":      token_A,
                "_meta": {
                    "qid": rec.qid, "step": rec.step, "kind": "flip_balance",
                    "B_id": rec.B, "A_id": rec.A,
                    "delta_p": rec.delta_p, "h_t": rec.h_t,
                },
            })
            n_balance += 1

    rng.shuffle(records)

    stats = {
        "n_consumed":       n_consumed,
        "n_pos":            n_pos,
        "n_balance":        n_balance,
        "n_drop_special":   n_drop_special,
        "n_drop_template":  n_drop_template,
        "n_drop_min_id":    n_drop_min_id,
        "n_drop_decode":    n_drop_decode,
        "n_drop_eq_AB":     n_drop_eq_AB,
        "filter_kept_ratio": (n_pos / n_consumed) if n_consumed else 0.0,
    }
    logger.info("flip 过滤统计: %s", stats)
    return {"records": records, "stats": stats}


def _strip_meta(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去除内部 _meta 字段，输出供 LLaMA-Factory 直接消费的纯 alpaca 格式。"""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]


def _expose_meta(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 _meta 暴露为 meta 字段（不带下划线），用于 analyze 脚本读取。
    若 LLaMA-Factory 严格 schema 检查，仍建议用 _strip_meta 输出 train/val。"""
    out: List[Dict[str, Any]] = []
    for r in records:
        new = {k: v for k, v in r.items() if not k.startswith("_")}
        if "_meta" in r:
            new["meta"] = r["_meta"]
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF flip 事件 → LLaMA-Factory Alpaca SFT 数据")
    p.add_argument("--flip_jsonl", required=True)
    p.add_argument("--tokenizer",  required=True, help="与 Target 一致的 tokenizer 路径")
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--round",      type=int, required=True)

    p.add_argument("--max_samples",      type=int, default=30000,
                   help="经 add_balance + 通用锚点 后的最大总样本数（默认 30000）")
    p.add_argument("--max_flip_events",  type=int, default=20000,
                   help="最多消费多少 flip 事件（默认 20000）")
    p.add_argument("--max_prefix_len",   type=int, default=1024)
    p.add_argument("--add_balance",      action="store_true", default=True,
                   help="对每个 flip 事件加一条 anti-flip 平衡样本（默认开启）")
    p.add_argument("--no_balance",       dest="add_balance", action="store_false")
    p.add_argument("--general_ratio",    type=float, default=0.25,
                   help="通用 Alpaca 锚点占总样本比例（默认 0.25）")
    p.add_argument("--val_ratio",        type=float, default=0.05)
    p.add_argument("--seed",             type=int, default=42)

    # ---- v2 内容过滤开关（针对结构 token 主导问题） ----
    p.add_argument("--exclude_special_tokens", action="store_true",
                   help="跳过 token_B 为 tokenizer.all_special_ids 的 flip（如 <|im_end|>）")
    p.add_argument("--exclude_template_words", action="store_true",
                   help="跳过 token_B 解码后属于答题模板词的 flip（如 'Final', 'answer', ':', 单字母）")
    p.add_argument("--min_token_id",  type=int, default=0,
                   help="跳过 token_B id 小于该值的 flip（默认 0=不过滤；建议 200 过滤 ASCII 标点/数字）")
    p.add_argument("--keep_meta", action="store_true",
                   help="train/val.json 中保留 meta 字段（默认 False，保持 LLaMA-Factory 兼容）")

    p.add_argument("--dataset_info_json", default="",
                   help="若指定，自动写入 LLaMA-Factory dataset_info.json")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flip_jsonl = Path(args.flip_jsonl)
    assert flip_jsonl.exists(), f"flip_jsonl 不存在: {flip_jsonl}"

    logger.info("加载 tokenizer: %s", args.tokenizer)
    tok = _load_tokenizer(args.tokenizer)

    logger.info("从 flip 事件构造 SFT 样本 ...")
    build_out = build_sft_records(
        flip_jsonl=flip_jsonl,
        tokenizer=tok,
        max_flip_events=args.max_flip_events,
        max_prefix_len=args.max_prefix_len,
        add_balance=args.add_balance,
        rng=rng,
        exclude_special_tokens=args.exclude_special_tokens,
        exclude_template_words=args.exclude_template_words,
        min_token_id=args.min_token_id,
    )
    flip_records = build_out["records"]
    flip_stats   = build_out["stats"]
    logger.info("flip 派生样本数 (含 anti-flip): %d", len(flip_records))

    # 通用锚点
    if args.general_ratio > 0 and len(flip_records) > 0:
        general_limit = int(
            len(flip_records) * args.general_ratio / max(1 - args.general_ratio, 1e-9)
        )
        logger.info("加载通用 Alpaca 锚点: limit=%d", general_limit)
        general_records = _load_general_alpaca(general_limit)
        logger.info("通用样本数: %d", len(general_records))
    else:
        general_records = []

    # 合并、限量、再 shuffle
    all_records = flip_records + general_records
    if args.max_samples and len(all_records) > args.max_samples:
        all_records = all_records[:args.max_samples]
    rng.shuffle(all_records)
    logger.info("合并后总样本数: %d  (flip-derived=%d  general=%d)",
                len(all_records), len(flip_records), len(general_records))

    # train/val 切分
    n_val = max(1, int(len(all_records) * args.val_ratio))
    val_records   = all_records[:n_val]
    train_records = all_records[n_val:]
    logger.info("train: %d  val: %d", len(train_records), len(val_records))

    train_path = out_dir / f"daf_round{args.round}_train.json"
    val_path   = out_dir / f"daf_round{args.round}_val.json"

    # train/val 是否带 meta，由 --keep_meta 决定（默认不带，保持 LLaMA-Factory 兼容）
    _serialize = _expose_meta if args.keep_meta else _strip_meta
    train_path.write_text(
        json.dumps(_serialize(train_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    val_path.write_text(
        json.dumps(_serialize(val_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("✓ 写入 train: %s  (keep_meta=%s)", train_path, args.keep_meta)
    logger.info("✓ 写入 val:   %s", val_path)

    # 同时保留含 meta 的调试副本（始终生成，便于 analyze 脚本统计）
    train_dbg = out_dir / f"daf_round{args.round}_train_with_meta.json"
    train_dbg.write_text(
        json.dumps(_expose_meta(train_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_key = f"daf_round{args.round}_train"
    val_key   = f"daf_round{args.round}_val"

    snippet = {
        train_key: {"file_name": str(train_path), "formatting": "alpaca"},
        val_key:   {"file_name": str(val_path),   "formatting": "alpaca"},
    }
    logger.info("LLaMA-Factory dataset_info.json 注册片段:\n%s",
                json.dumps(snippet, ensure_ascii=False, indent=2))

    if args.dataset_info_json:
        info_path = Path(args.dataset_info_json)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
        else:
            info = {}
        info.update(snippet)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("✓ 已注册 %s / %s 到 %s", train_key, val_key, info_path)

    # 写一份汇总到 out_dir
    summary = {
        "round_id":            args.round,
        "n_flip_records":      len(flip_records),
        "n_general_records":   len(general_records),
        "n_train":             len(train_records),
        "n_val":               len(val_records),
        "train_path":          str(train_path),
        "val_path":            str(val_path),
        "dataset_train_key":   train_key,
        "dataset_val_key":     val_key,
        "tokenizer_path":      args.tokenizer,
        "flip_jsonl":          str(flip_jsonl),
        "max_flip_events":     args.max_flip_events,
        "max_prefix_len":      args.max_prefix_len,
        "add_balance":         args.add_balance,
        "general_ratio":       args.general_ratio,
        "exclude_special_tokens": args.exclude_special_tokens,
        "exclude_template_words": args.exclude_template_words,
        "min_token_id":           args.min_token_id,
        "keep_meta":              args.keep_meta,
        "flip_filter_stats":      flip_stats,
    }
    (out_dir / f"daf_round{args.round}_sft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
