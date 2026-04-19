"""DAF — Token Flip 事件的唯一口径

约束：
  所有下游脚本（fdlp_score / build_flip_sft_data / convergence_check）必须从本模块
  import is_flip / iter_flip_records，禁止在别处重新定义 flip。
  这样保证「飞轮回归 / 飞轮收敛 / 飞轮训练」三处口径一致，避免漂移。

定义（与 telemetry.StepTelemetry / decode_loop._verify_and_accept 完全对应）：
  flip_t = accepted_t ∧ (draft_token_id_t ≠ target_top1_id_t)

也即：Draft 提议被 Target 接受，但 Target 自己 argmax 并不会选这个 token；
此时 Draft 把 Target 的解码轨迹"挟持"到了一个 Target 不会主动走的方向，
正是 DAF 关心的"领域知识注入事件"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Flip 事件结构（持久化到 flip_events_round{k}.jsonl 的一行）
# ---------------------------------------------------------------------------

@dataclass
class FlipRecord:
    """单条 flip 事件（已展开，便于 fdlp_score 直接使用）。

    Attributes:
        round_id:       飞轮轮次编号（0=Round 0，1=Round 1，...）
        qid:            sample id（与 telemetry 文件名一致）
        step:           本 sample 内的全局 token 步序号
        prefix_ids:     至本步为止的全部 token id 序列（不含本步 final_token）
                        DAF FDLP forward 的 input_ids 即为此
        A:              target_top1_id (Target 自己 argmax 的 token)
        B:              draft_token_id  (Draft 提议且被接受的 token)
        F:              is_flip         (恒为 True，留作字段对称性 / 抽样校验)
        delta_p:        ΔP = p_draft(B) - p_base(B)
        h_t:            Target 分布熵（nats）
        accepted:       是否接受（恒为 True，flip 事件必接受）
    """
    round_id:   int
    qid:        str
    step:       int
    prefix_ids: List[int]
    A:          int
    B:          int
    F:          bool
    delta_p:    float
    h_t:        Optional[float]
    accepted:   bool


# ---------------------------------------------------------------------------
# 口径函数
# ---------------------------------------------------------------------------

def is_flip(step: Dict[str, Any]) -> bool:
    """根据一条 telemetry step (dict) 判断是否为 flip 事件。

    优先读取 telemetry 已写入的 is_flip 字段（M0 微改后）；
    若旧日志缺该字段，则按等价规则回退判定，保证向后兼容。

    Args:
        step: telemetry.jsonl 中 type=='step' 的一行 dict
    Returns:
        bool
    """
    if step.get("type", "step") != "step":
        return False
    if step.get("is_flip") is not None:
        return bool(step["is_flip"])
    accepted   = bool(step.get("accepted", False))
    draft_id   = step.get("draft_token_id")
    target_id  = step.get("target_top1_id")
    return bool(accepted and draft_id is not None and target_id is not None
                and draft_id != target_id)


# ---------------------------------------------------------------------------
# 流式读取
# ---------------------------------------------------------------------------

def iter_flip_records(jsonl_path: Path | str) -> Iterator[FlipRecord]:
    """流式读取 flip_events_round{k}.jsonl，yield FlipRecord。

    Args:
        jsonl_path: run_flip_logger 产出的 jsonl 文件路径
    Yields:
        FlipRecord 实例（仅 F=True 的事件，写入侧已过滤）
    """
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield FlipRecord(
                round_id=int(rec["round_id"]),
                qid=str(rec["qid"]),
                step=int(rec["step"]),
                prefix_ids=list(rec["prefix_ids"]),
                A=int(rec["A"]),
                B=int(rec["B"]),
                F=bool(rec.get("F", True)),
                delta_p=float(rec["delta_p"]),
                h_t=(None if rec.get("h_t") is None else float(rec["h_t"])),
                accepted=bool(rec.get("accepted", True)),
            )


def write_flip_records(records: List[FlipRecord], out_path: Path | str) -> Path:
    """将 FlipRecord 列表写入 jsonl，主要给单元测试 / 离线整理使用。

    Args:
        records: FlipRecord 列表
        out_path: 输出路径
    Returns:
        out_path（Path）
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "round_id":   r.round_id,
                "qid":        r.qid,
                "step":       r.step,
                "prefix_ids": r.prefix_ids,
                "A":          r.A,
                "B":          r.B,
                "F":          r.F,
                "delta_p":    r.delta_p,
                "h_t":        r.h_t,
                "accepted":   r.accepted,
            }, ensure_ascii=False))
            f.write("\n")
    return out


# ---------------------------------------------------------------------------
# 简易统计
# ---------------------------------------------------------------------------

def summarize_flip_jsonl(jsonl_path: Path | str) -> Dict[str, Any]:
    """对 flip_events_round{k}.jsonl 做一次扫描汇总。

    Returns:
        {
          "n_flip": int,            # 总 flip 数
          "n_qid":  int,            # 涉及 sample 数
          "mean_flip_per_qid": float,
          "mean_delta_p": float,
          "mean_h_t":     float,
        }
    """
    n_flip = 0
    qids: set[str] = set()
    sum_dp = 0.0
    sum_ht = 0.0
    n_ht   = 0
    for rec in iter_flip_records(jsonl_path):
        n_flip += 1
        qids.add(rec.qid)
        sum_dp += rec.delta_p
        if rec.h_t is not None:
            sum_ht += rec.h_t
            n_ht   += 1
    return {
        "n_flip":             n_flip,
        "n_qid":              len(qids),
        "mean_flip_per_qid":  (n_flip / max(len(qids), 1)),
        "mean_delta_p":       (sum_dp / max(n_flip, 1)),
        "mean_h_t":           (sum_ht / max(n_ht, 1)) if n_ht else None,
    }
