"""DAF — Flip-Driven Layer Placement (FDLP) 反向传播打分

核心思想：
  对每个 flip 事件 (qid, prefix_ids, A=target_top1, B=draft_token_id) 做一次
  Target 模型的 forward + backward：
    loss = -log_softmax(target.forward(prefix_ids).logits[-1])[B]
    loss.backward()
  通过 register_post_accumulate_grad_hook 即时抓取每个候选模块（q/k/v/o/up/down/gate）
  权重的 ‖∇W‖_F / sqrt(numel)（参数规模归一化的梯度范数），并立即把 .grad 释放，
  避免 32B 模型的全量梯度长时间驻留显存。

  最终：S_ℓ = (1/N_flip) * Σ_{flip events} ‖∇W_ℓ‖_F / sqrt(numel)
  作为「flip-token 监督下的层位重要性」打分。

显存控制（H200 141 GB）：
  * bf16 32B 权重 ~64 GB；
  * gradient_checkpointing 启用，activation 峰值 ~5–8 GB；
  * post_accumulate_grad_hook 抓 norm 后立刻 set .grad = None，避免梯度堆积；
  * batch=1，prefix 截断到 max_prefix_len（默认 1024）以兜住峰值；
  * 必要时可用 --per_layer_backward 退化为「每次只对 1 层做 backward」（牺牲速度换显存）。

输出：
  layer_scores_round{k}.json:
    {
      "subsets": {
        "fdlp":               {<module_name>: {"score": float, "n_event": int}, ...},
        "fdlp_top_entropy":   {...},
        "fdlp_top_disagreement": {...},
        "fdlp_random_subset": {...}
      },
      "summary": {
        "fdlp": {
          "top_k_modules":    [{"name": ..., "score": ..., "rank": int}, ...],
          "top_k_layer_idxs": [...]
        },
        ...
      },
      "meta": {
        "round_id": int, "target_model": str, "n_flip_total": int,
        "max_prefix_len": int, "subset_sizes": {...},
        "candidate_modules": [...],
        "flip_jsonl": str
      }
    }

用法（远端 H200）：
  cd /data/ocean/decoding
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=0
  python -m daf.fdlp_score \
      --target_model /data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct \
      --flip_jsonl logs/daf_round0/flip_events_round0.jsonl \
      --out logs/daf_round0/layer_scores_round0.json \
      --top_k 8 --r_total 128 \
      --max_events 2000 --max_prefix_len 1024
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from model_loader_v2 import load_single_model  # noqa: E402

from daf.flip_definition import FlipRecord, iter_flip_records  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("daf.fdlp_score")


# ---------------------------------------------------------------------------
# 候选模块识别
# ---------------------------------------------------------------------------

# Qwen2.5 / LLaMA 风格的目标模块名（出现在 model.layers.{i}.* 路径中）
_DEFAULT_CANDIDATE_PATTERNS = [
    r"\.self_attn\.q_proj$",
    r"\.self_attn\.k_proj$",
    r"\.self_attn\.v_proj$",
    r"\.self_attn\.o_proj$",
    r"\.mlp\.gate_proj$",
    r"\.mlp\.up_proj$",
    r"\.mlp\.down_proj$",
]
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _is_candidate_module(name: str) -> bool:
    """name 形如 'model.layers.18.self_attn.q_proj'"""
    return any(re.search(p, name) for p in _DEFAULT_CANDIDATE_PATTERNS)


def _layer_idx_of(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _module_path_of(name: str) -> str:
    """从 'model.layers.18.self_attn.q_proj.weight' 抽出 'model.layers.18.self_attn.q_proj'。"""
    if name.endswith(".weight"):
        return name[: -len(".weight")]
    return name


# ---------------------------------------------------------------------------
# 梯度范数收集器
# ---------------------------------------------------------------------------

class FlipGradAccumulator:
    """对每个候选模块累计 ‖∇W‖_F / sqrt(numel)。

    支持多个 subset bucket（fdlp / entropy / disagreement / random_subset），
    一次 backward 后按当前事件所属的 bucket 集合分别累加，避免重复 forward。

    使用 register_post_accumulate_grad_hook 在每个候选 weight 完成梯度累积后
    立即抓 norm 并把 .grad 置 None，将峰值显存压在「单层梯度」量级。
    """

    def __init__(self, model: torch.nn.Module, subsets: List[str]) -> None:
        self._model = model
        self._subsets = subsets
        # subsets[s] -> {module_path: float}
        self.acc:    Dict[str, Dict[str, float]] = {s: defaultdict(float) for s in subsets}
        self.n_event: Dict[str, int] = {s: 0 for s in subsets}

        # 暂存本次 backward 的 per-module grad-norm
        self._tmp_norms: Dict[str, float] = {}
        # 候选 weight 参数列表
        self._candidate_params: List[Tuple[str, torch.nn.Parameter]] = []
        # 已注册的 hook handle
        self._hook_handles: List[Any] = []

        self._register_hooks()
        logger.info("候选模块数: %d  示例: %s",
                    len(self._candidate_params),
                    [n for n, _ in self._candidate_params[:5]])

    # ----------------------- hook 注册 -----------------------

    def _register_hooks(self) -> None:
        for name, p in self._model.named_parameters():
            module_path = _module_path_of(name)
            if not _is_candidate_module(module_path):
                continue
            if not p.requires_grad:
                continue
            self._candidate_params.append((module_path, p))

            handle = p.register_post_accumulate_grad_hook(
                self._make_hook(module_path)
            )
            self._hook_handles.append(handle)

    def _make_hook(self, module_path: str) -> Callable[[torch.nn.Parameter], None]:
        tmp = self._tmp_norms

        def _hook(p: torch.nn.Parameter) -> None:
            g = p.grad
            if g is None:
                return
            # 用 fp32 累加器算 L2 范数避免 bf16 溢出
            n = float(g.detach().to(torch.float32).norm().item())
            numel = max(p.numel(), 1)
            tmp[module_path] = tmp.get(module_path, 0.0) + n / math.sqrt(numel)
            # 立即释放梯度，避免堆积
            p.grad = None

        return _hook

    # ----------------------- 单事件接口 -----------------------

    def reset_event(self) -> None:
        """每个事件 forward+backward 前调用：清空本次的临时缓存与所有 .grad。"""
        self._tmp_norms.clear()
        # 双保险：把所有候选参数的 grad 置 None
        for _, p in self._candidate_params:
            p.grad = None
        # 也把整模型其他参数 grad 释放（如 embedding / lm_head）
        self._model.zero_grad(set_to_none=True)

    def commit_event(self, buckets: Iterable[str]) -> None:
        """backward 完成后，将本次的 per-module norm 写入指定 buckets。"""
        if not self._tmp_norms:
            return
        for b in buckets:
            if b not in self.acc:
                continue
            self.n_event[b] += 1
            for mp, v in self._tmp_norms.items():
                self.acc[b][mp] += v
        self._tmp_norms.clear()

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    # ----------------------- 输出 -----------------------

    def to_scores(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for s in self._subsets:
            n = max(self.n_event[s], 1)
            scored: Dict[str, Dict[str, Any]] = {}
            for mp, total in self.acc[s].items():
                scored[mp] = {"score": total / n, "n_event": self.n_event[s]}
            out[s] = scored
        return out


# ---------------------------------------------------------------------------
# 子集划分（基于预扫描的 h_t / delta_p 分位数）
# ---------------------------------------------------------------------------

def _percentile(values: List[float], q: float) -> Optional[float]:
    """简单实现，q 取 [0,1]，避免引入 numpy 仅为单一调用。"""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _prepass_thresholds(
    flip_jsonl: Path,
    max_events: int,
) -> Dict[str, Optional[float]]:
    """单遍扫一次，估出 h_t / delta_p 的 90% 分位数（用于 top-10% 子集）。

    若总事件数超过 max_events，按时间顺序截断（与主循环采样口径一致）。
    """
    h_ts: List[float] = []
    dps: List[float] = []
    n = 0
    for r in iter_flip_records(flip_jsonl):
        if r.h_t is not None:
            h_ts.append(r.h_t)
        dps.append(r.delta_p)
        n += 1
        if n >= max_events:
            break
    return {
        "h_t_p90":     _percentile(h_ts, 0.90),
        "delta_p_p90": _percentile(dps, 0.90),
        "n_seen":      n,
    }


# ---------------------------------------------------------------------------
# 单事件 forward + backward
# ---------------------------------------------------------------------------

def _score_one_event(
    model:        torch.nn.Module,
    rec:          FlipRecord,
    device:       torch.device,
    max_prefix_len: int,
) -> bool:
    """对单个 flip 事件做一次 forward + backward。

    Returns:
        True 成功（hook 已抓到 norm），False 跳过（如 prefix 过短）。
    """
    prefix = rec.prefix_ids
    if len(prefix) < 1:
        return False

    # 截断左侧（保留靠右的上下文，对 next-token 预测最关键）
    if len(prefix) > max_prefix_len:
        prefix = prefix[-max_prefix_len:]

    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)  # shape: [1, L]

    # forward
    # 关闭 cache 与 dropout（model 已 eval()）
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, -1, :]                  # shape: [1, V]
    log_probs = F.log_softmax(logits.float(), dim=-1)  # fp32 稳定 softmax
    target_id = int(rec.B)
    if target_id < 0 or target_id >= log_probs.shape[-1]:
        return False
    loss = -log_probs[0, target_id]                # scalar

    # backward
    loss.backward()
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _classify_event(
    rec:           FlipRecord,
    h_t_p90:       Optional[float],
    delta_p_p90:   Optional[float],
    random_keep:   bool,
) -> List[str]:
    """决定本事件落入哪些 bucket。fdlp 永远收。"""
    buckets = ["fdlp"]
    if h_t_p90 is not None and rec.h_t is not None and rec.h_t >= h_t_p90:
        buckets.append("fdlp_top_entropy")
    if delta_p_p90 is not None and rec.delta_p >= delta_p_p90:
        buckets.append("fdlp_top_disagreement")
    if random_keep:
        buckets.append("fdlp_random_subset")
    return buckets


def _build_top_k_summary(
    scores: Dict[str, Dict[str, Any]],
    top_k:  int,
    r_total: int,
) -> Dict[str, Any]:
    """从单个 subset 的 {module: {score, n_event}} 抽取 Top-K 模块。

    同时给出按层粒度聚合的 top_k_layer_idxs（按 sum(score per layer) 降序）。
    """
    items = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top_modules = [
        {"name": name, "score": meta["score"], "rank": i + 1}
        for i, (name, meta) in enumerate(items[:top_k])
    ]
    # 平均 rank 分配（plan 用统一 rank；这里也给逐模块建议 rank 以便 future work）
    if top_modules:
        per_module_rank = max(1, r_total // len(top_modules))
    else:
        per_module_rank = 0
    for m in top_modules:
        m["suggested_rank"] = per_module_rank

    # 按层聚合
    per_layer: Dict[int, float] = defaultdict(float)
    for name, meta in items:
        idx = _layer_idx_of(name)
        if idx is not None:
            per_layer[idx] += meta["score"]
    top_layers = sorted(per_layer.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    top_layer_idxs = [{"layer": i, "agg_score": v} for i, v in top_layers]

    return {
        "top_k_modules":    top_modules,
        "top_k_layer_idxs": top_layer_idxs,
        "per_module_rank":  per_module_rank,
        "n_modules_total":  len(items),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAF FDLP 反向传播打分（32B Target，含 4 套对照）")
    p.add_argument("--target_model", required=True, help="32B Target 模型路径")
    p.add_argument("--flip_jsonl",   required=True, help="run_flip_logger 产出的 flip_events_round{k}.jsonl")
    p.add_argument("--out",          required=True, help="layer_scores_round{k}.json 输出路径")

    p.add_argument("--top_k",   type=int, default=8,   help="Top-K 模块（plan: 8）")
    p.add_argument("--r_total", type=int, default=128, help="LoRA 总 rank 预算（plan: 128）")

    p.add_argument("--max_events",     type=int, default=2000,
                   help="主循环最多处理多少 flip 事件（控制总耗时；默认 2000）")
    p.add_argument("--max_prefix_len", type=int, default=1024,
                   help="单事件 prefix 截断上限（控制 activation 峰值；默认 1024）")
    p.add_argument("--random_subset_ratio", type=float, default=0.10,
                   help="随机对照子集采样比例（默认 10%）")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="开启 gradient_checkpointing（默认开启，bf16 32B 必备）")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    p.add_argument("--log_every", type=int, default=20)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flip_jsonl = Path(args.flip_jsonl)
    assert flip_jsonl.exists(), f"flip_jsonl 不存在: {flip_jsonl}"

    device = torch.device("cuda:0")

    # ---------- 预扫：决定 h_t / delta_p 90% 分位数 ----------
    logger.info("预扫 flip_jsonl 估算 h_t / delta_p 90% 分位数 ...")
    thr = _prepass_thresholds(flip_jsonl, max_events=args.max_events)
    logger.info("阈值估算完成: %s", thr)

    # ---------- 加载 32B Target（单模型）----------
    logger.info("加载 32B Target ... %s", args.target_model)
    model = load_single_model(
        model_path=args.target_model,
        device=device,
        dtype=torch.bfloat16,
        compile_mode=None,  # FDLP backward 不与 torch.compile 兼容
    )
    # FDLP 需要梯度，必须 train() 否则 dropout 关，但梯度记录开关在 requires_grad
    # 这里保持 eval() 关 dropout，但通过 requires_grad=True 让权重参与 backward
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # 与 use_cache=False 配合
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            logger.info("gradient_checkpointing 已开启")
        except Exception as exc:
            logger.warning("gradient_checkpointing 开启失败: %s", exc)

    # ---------- 收集器 ----------
    subsets = ["fdlp", "fdlp_top_entropy", "fdlp_top_disagreement", "fdlp_random_subset"]
    accumulator = FlipGradAccumulator(model=model, subsets=subsets)

    # ---------- 主循环 ----------
    n_processed = 0
    n_skipped   = 0
    t_start     = time.perf_counter()

    logger.info("开始 FDLP 主循环  max_events=%d  max_prefix_len=%d",
                args.max_events, args.max_prefix_len)

    for rec in iter_flip_records(flip_jsonl):
        if n_processed + n_skipped >= args.max_events:
            break

        random_keep = (rng.random() < args.random_subset_ratio)
        buckets = _classify_event(
            rec,
            h_t_p90=thr.get("h_t_p90"),
            delta_p_p90=thr.get("delta_p_p90"),
            random_keep=random_keep,
        )

        accumulator.reset_event()
        try:
            ok = _score_one_event(
                model=model,
                rec=rec,
                device=device,
                max_prefix_len=args.max_prefix_len,
            )
        except torch.cuda.OutOfMemoryError as exc:
            logger.error("CUDA OOM @ event qid=%s step=%d prefix_len=%d: %s",
                         rec.qid, rec.step, len(rec.prefix_ids), exc)
            torch.cuda.empty_cache()
            gc.collect()
            n_skipped += 1
            continue
        except Exception as exc:
            logger.warning("event 处理失败 qid=%s step=%d: %s", rec.qid, rec.step, exc)
            n_skipped += 1
            continue

        if not ok:
            n_skipped += 1
            continue

        accumulator.commit_event(buckets)
        # 主动释放计算图缓存
        del rec  # rec 自身只是普通对象，但帮助提醒
        n_processed += 1

        if (n_processed % args.log_every == 0):
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            mem_resv  = torch.cuda.memory_reserved(device)  / 1024**3
            elapsed   = time.perf_counter() - t_start
            tps = n_processed / max(elapsed, 1e-6)
            logger.info(
                "[%d/%d] processed (skipped=%d)  events/sec=%.2f  "
                "GPU alloc=%.1f GiB  reserved=%.1f GiB",
                n_processed, args.max_events, n_skipped, tps, mem_alloc, mem_resv,
            )

    # ---------- 卸 hook + 出分 ----------
    accumulator.remove_hooks()
    scores_by_subset = accumulator.to_scores()
    summary = {
        s: _build_top_k_summary(scores_by_subset[s], top_k=args.top_k, r_total=args.r_total)
        for s in subsets
    }
    subset_sizes = {s: accumulator.n_event[s] for s in subsets}

    out_obj: Dict[str, Any] = {
        "subsets": scores_by_subset,
        "summary": summary,
        "meta": {
            "round_id":         _infer_round_id_from_filename(flip_jsonl),
            "target_model":     args.target_model,
            "flip_jsonl":       str(flip_jsonl),
            "n_flip_total":     n_processed,
            "n_flip_skipped":   n_skipped,
            "max_prefix_len":   args.max_prefix_len,
            "max_events":       args.max_events,
            "random_subset_ratio": args.random_subset_ratio,
            "subset_sizes":     subset_sizes,
            "top_k":            args.top_k,
            "r_total":          args.r_total,
            "candidate_module_patterns": _DEFAULT_CANDIDATE_PATTERNS,
            "thresholds":       thr,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
    }

    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("✓ FDLP 打分完成  n_processed=%d  →  %s", n_processed, out_path)
    for s in subsets:
        top = summary[s]["top_k_modules"][:5]
        logger.info("  [%s] n_event=%d  top5=%s",
                    s, subset_sizes[s], [(m["name"], round(m["score"], 4)) for m in top])


def _infer_round_id_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"round(\d+)", path.name)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    main()
