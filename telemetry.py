"""
遥测监控系统 — TelemetryLogger

功能（对应实验计划 Section 6 监控系统要求）：
  1. 逐步记录每个 token 位置的核心信号（ΔP, p_draft, p_target, 验收结果）
  2. Hard Override 触发后，监控接下来 5 步 Target 分布熵与 Top-1 置信度
  3. 将 Sample 级别遥测数据以 JSON Lines 格式写入磁盘
  4. 提供 summary() 返回 Sample 级别的汇总统计（覆盖率、平均 ΔP 等）

设计原则：
  - TelemetryLogger 无副作用，仅收集数据，不影响解码逻辑
  - 遥测采集是实时的（逐步追加），不依赖解码结束后的批量处理
  - PostOverrideProbe 由解码循环在 Override 后连续 5 步填入
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 数据类：单步遥测
# ---------------------------------------------------------------------------

@dataclass
class StepTelemetry:
    """单个 token 位置的遥测数据。

    Attributes:
        step:              解码步序号（0-based）
        draft_token_id:    Draft 提案的 token id
        target_top1_id:    Target argmax token id
        base_top1_id:      Base   argmax token id
        delta_p:           ΔP = p_draft(x) - p_base(x)，T_fixed 温度下
        p_draft:           Draft 对 draft_token_id 的概率（T_fixed）
        p_target:          Target 对 draft_token_id 的概率（T_fixed）
        accepted:          是否接受 draft_token_id
        override_triggered: 是否触发了 Hard Override（B0/B 策略）
        strategy_reason:   来自 AcceptResult.reason 的决策字符串
        final_token_id:    实际进入序列的 token id（accept→draft；reject→correction；DAF 第二点扩展）
        is_flip:           本步是否为 token flip 事件，定义为：accepted ∧ draft_token_id != target_top1_id
                           （DAF 飞轮第二点的核心监督信号；可空保持向后兼容）
        target_entropy:    本步 Target 分布熵 H_t（nats，DAF 用于熵权与 entropy 对照）
    """
    step:               int
    draft_token_id:     int
    target_top1_id:     int
    base_top1_id:       int
    delta_p:            float
    p_draft:            float
    p_target:           float
    accepted:           bool
    override_triggered: bool
    strategy_reason:    str
    final_token_id:     Optional[int]  = None
    is_flip:            Optional[bool] = None
    target_entropy:     Optional[float] = None


# ---------------------------------------------------------------------------
# 数据类：Override 后续探针
# ---------------------------------------------------------------------------

@dataclass
class PostOverrideProbe:
    """Hard Override 触发后，连续监控 Target 分布变化的探针数据。

    实验计划要求：
      - 监控触发 Override 后接下来 5 步的 Target 分布熵
      - 监控对应的 Target Top-1 置信度
    以此探究 OOD token 对 Target KV Cache 的污染程度（Context Collapse 现象）。

    Attributes:
        override_step:         发生 Override 的步序号
        subsequent_entropies:  Override 后 5 步的 Target 分布熵（nats）
        subsequent_top1_confs: Override 后 5 步的 Target Top-1 置信度
    """
    override_step:         int
    subsequent_entropies:  List[float] = field(default_factory=list)
    subsequent_top1_confs: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TelemetryLogger 主类
# ---------------------------------------------------------------------------

class TelemetryLogger:
    """单个 sample 的遥测收集器。

    每个 sample 创建一个独立实例，解码结束后调用 flush() 持久化到磁盘。

    Args:
        log_dir:   遥测日志输出目录
        sample_id: 当前 sample 的唯一标识（用于文件命名）
    """

    _POST_OVERRIDE_WINDOW: int = 5  # 监控 Override 后续的步数

    def __init__(self, log_dir: Path, sample_id: str) -> None:
        self._log_dir   = Path(log_dir)
        self._sample_id = sample_id

        self._steps:          List[StepTelemetry]   = []
        self._override_probes: List[PostOverrideProbe] = []

        # Override 后续监控状态机
        self._pending_probes: List[PostOverrideProbe] = []  # 还未收集满 5 步的 probe

    # ------------------------------------------------------------------
    # 核心记录接口
    # ------------------------------------------------------------------

    def log_step(self, telemetry: StepTelemetry) -> None:
        """记录单步遥测数据，并推进所有 pending PostOverrideProbe 的收集。

        每次调用时，检查是否有正在等待收集的 PostOverrideProbe，
        如果是，则将本步的 Target logits 熵和 Top-1 置信度追加进去。

        Args:
            telemetry: 本步完整遥测数据
        """
        self._steps.append(telemetry)

        # 推进所有正在等待的 Override 后续探针
        # 注意：logits 在此时已不可直接访问，熵数据需由解码循环单独提交
        # （log_target_entropy_for_probes 方法）

    def log_target_entropy_for_probes(
        self,
        target_logits: torch.Tensor,   # shape: [1, vocab_size]
    ) -> None:
        """将本步 Target logits 的熵和 Top-1 置信度写入所有 pending probe。

        由解码循环在 Override 触发后的接下来每步调用（与 log_step 分离，
        避免对非 Override 步造成计算开销）。

        Args:
            target_logits: 本步 Target 的 next-token logits，shape [1, vocab_size]
        """
        entropy    = self.compute_entropy(target_logits)
        top1_conf  = float(F.softmax(target_logits, dim=-1).max().item())

        completed_probes = []
        still_pending    = []
        for probe in self._pending_probes:
            probe.subsequent_entropies.append(entropy)
            probe.subsequent_top1_confs.append(top1_conf)
            if len(probe.subsequent_entropies) >= self._POST_OVERRIDE_WINDOW:
                completed_probes.append(probe)
            else:
                still_pending.append(probe)

        self._override_probes.extend(completed_probes)
        self._pending_probes = still_pending

    def register_override(self, override_step: int) -> None:
        """在 Override 触发时注册一个新的 PostOverrideProbe。

        应在 log_step() 之后调用，表明从下一步开始进入监控窗口。

        Args:
            override_step: Override 发生的步序号
        """
        probe = PostOverrideProbe(override_step=override_step)
        self._pending_probes.append(probe)

    def finalize_probes(self) -> None:
        """Sample 结束时，将未收集满 5 步的 probe 也记录入结果（可能不足 5 步）。"""
        self._override_probes.extend(self._pending_probes)
        self._pending_probes = []

    # ------------------------------------------------------------------
    # 工具：计算信息熵
    # ------------------------------------------------------------------

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """计算 logits 对应分布的信息熵（单位：nats）。

        H(P) = -Σ P(x) · log P(x)

        Args:
            logits: shape [1, vocab_size]

        Returns:
            entropy (float, >= 0)
        """
        probs = F.softmax(logits, dim=-1).squeeze(0)    # shape: [vocab_size]
        # 数值稳定：log(0) 取 0
        log_probs = torch.log(probs.clamp(min=1e-12))   # shape: [vocab_size]
        entropy = float(-(probs * log_probs).sum().item())
        return entropy

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def flush(self) -> Path:
        """将本 sample 的完整遥测数据写入 JSON Lines 文件。

        文件路径：{log_dir}/{sample_id}_telemetry.jsonl

        Returns:
            写入文件的 Path 对象
        """
        self.finalize_probes()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        out_path = self._log_dir / f"{self._sample_id}_telemetry.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            # 写入 summary 行
            f.write(json.dumps({"type": "summary", **self.summary()}, ensure_ascii=False))
            f.write("\n")
            # 写入逐步遥测
            for step_data in self._steps:
                f.write(json.dumps({"type": "step", **asdict(step_data)}, ensure_ascii=False))
                f.write("\n")
            # 写入 Override 后续探针
            for probe in self._override_probes:
                f.write(json.dumps({"type": "override_probe", **asdict(probe)}, ensure_ascii=False))
                f.write("\n")

        return out_path

    # ------------------------------------------------------------------
    # 统计汇总
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """返回本 sample 的汇总统计字典。

        统计指标：
          - total_steps:       总解码步数
          - override_count:    Hard Override 触发次数
          - override_rate:     触发率 = override_count / total_steps
          - mean_delta_p:      平均 ΔP（仅计算 override_triggered=True 的步）
          - mean_delta_p_all:  所有步的平均 ΔP
          - acceptance_rate:   接受率 = accepted / total_steps
          - mean_post_override_entropy: Override 后 5 步平均熵
          - mean_post_override_top1:    Override 后 5 步平均 Top-1 置信度
        """
        n = len(self._steps)
        if n == 0:
            return {"total_steps": 0}

        overrides       = [s for s in self._steps if s.override_triggered]
        accepted_count  = sum(1 for s in self._steps if s.accepted)
        all_delta_p     = [s.delta_p for s in self._steps]
        override_delta_p = [s.delta_p for s in overrides]

        # Override 后续监控汇总
        all_subsequent_entropies  = [e for p in self._override_probes
                                       for e in p.subsequent_entropies]
        all_subsequent_top1_confs = [c for p in self._override_probes
                                       for c in p.subsequent_top1_confs]

        def safe_mean(lst: list) -> Optional[float]:
            return sum(lst) / len(lst) if lst else None

        return {
            "sample_id":                    self._sample_id,
            "total_steps":                  n,
            "override_count":               len(overrides),
            "override_rate":                len(overrides) / n,
            "acceptance_rate":              accepted_count / n,
            "mean_delta_p_all":             safe_mean(all_delta_p),
            "mean_delta_p_on_override":     safe_mean(override_delta_p),
            "mean_post_override_entropy":   safe_mean(all_subsequent_entropies),
            "mean_post_override_top1_conf": safe_mean(all_subsequent_top1_confs),
            "n_override_probes":            len(self._override_probes),
        }
