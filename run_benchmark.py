"""
评测驱动器 — run_benchmark.py

执行 MedQA-USMLE 和 GSM8K 基准评测，支持：
  - 所有基线组（Pure Target / Pure Draft / Standard SD）
  - 所有实验组（B0 / B / C1 / C2）
  - 执行架构对比（dual_stream vs shadow_sync）
  - 超参数网格搜索（α × T_sample）

用法示例（远端机器）：
  cd /data/ocean/decoding
  conda activate kvner
  export CUDA_VISIBLE_DEVICES=0

  # 跑所有组合（MedQA，300 条，贪婪）
  python run_benchmark.py \\
    --dataset medqa --limit 300 --t_sample 0.0 \\
    --arch shadow_sync --gamma 5 \\
    --out_dir logs/benchmark_$(date +%Y%m%d)

  # 只跑 α 网格搜索（C1/C2）
  python run_benchmark.py \\
    --dataset medqa --limit 300 \\
    --strategies soft_guidance_c1 soft_guidance_c2 \\
    --alpha_grid 0.1 0.5 1.0 1.5 2.0 \\
    --t_sample 0.0 0.6 \\
    --out_dir logs/alpha_grid_$(date +%Y%m%d)
"""

from __future__ import annotations

# region agent log (debug ecc61b)
import json as _json, time as _time, os as _os, glob as _glob, sys as _sys
_REMOTE_LOG = "/tmp/debug-ecc61b-remote.log"
def _dblog(hyp, msg, data):
    e = {"sessionId":"ecc61b","timestamp":int(_time.time()*1000),
         "location":"run_benchmark.py:top","hypothesisId":hyp,
         "message":msg,"data":data,"runId":"bench_diag"}
    print(f"  [DIAG/{hyp}] {msg}: {_json.dumps(data, ensure_ascii=False)}", flush=True)
    with open(_REMOTE_LOG,"a") as _f: _f.write(_json.dumps(e)+"\n")

_cwd = _os.getcwd()
_py_files = sorted(_os.path.basename(p) for p in _glob.glob("/data/ocean/decoding/*.py"))
_key_modules = {m: _os.path.exists(f"/data/ocean/decoding/{m}.py")
                for m in ["model_loader_v2","data_loader","evaluator","acceptance","cache_manager"]}
_dblog("H1", "file_existence_check", {"cwd": _cwd, "key_modules": _key_modules})
_dblog("H2", "sys_path_sample", {"sys_path_0_3": _sys.path[:3]})
_dblog("H3", "argv", {"argv": _sys.argv})
del _json, _time, _os, _glob, _sys, _cwd, _py_files, _key_modules, _REMOTE_LOG, _dblog
# endregion agent log

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from acceptance import create_strategy
from cache_manager import PrefixSharedCacheManager
from config_v2 import (
    ALPHA_GRID,
    TEMPERATURE_GRID,
    CacheConfig,
    DecodeConfig,
    DomainSignalParams,
    ExecutionArch,
    HardwareConfig,
    ModelPaths,
    StrategyType,
)
from data_loader import format_prompt, load_medqa
from decode_loop import DecodeResult, SpeculativeDecodeLoop
from dual_stream_engine import DualStreamProposer
from engine_state import TriModelOrchestrator
from evaluator import extract_answer
from model_loader_v2 import TriModelBundle, load_tri_models
from shadow_sync_engine import ShadowSyncProposer
from telemetry import TelemetryLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类：单次实验结果
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """单次配置（Strategy × α × T_sample × Arch）的评测汇总。

    Attributes:
        strategy:             策略名称字符串
        arch:                 执行架构字符串
        alpha:                软引导强度（A/B0/B 时为 N/A，记为 -1）
        t_sample:             全局采样温度
        dataset:              评测数据集名称
        n_cases:              评测样本数
        accuracy:             准确率（correct / n_cases）
        correct:              正确数
        tokens_per_sec:       平均真实吞吐量（tokens/sec）
        mean_acceptance_rate: 平均验收率
        override_rate:        平均 Override 率（B0/B 专用）
        total_generated:      总生成 token 数
        total_duration_sec:   总耗时（秒）
    """
    strategy:             str
    arch:                 str
    alpha:                float
    t_sample:             float
    dataset:              str
    n_cases:              int
    accuracy:             float
    correct:              int
    tokens_per_sec:       float
    mean_acceptance_rate: float
    override_rate:        float
    total_generated:      int
    total_duration_sec:   float


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

def _load_dataset(dataset_name: str, limit: int, split: str = "test") -> List[Dict[str, Any]]:
    """加载指定数据集，返回格式化后的样本列表。

    Args:
        dataset_name: "medqa" 或 "gsm8k"
        limit:        最多加载的样本数
        split:        数据集分片

    Returns:
        list of {"id", "prompt_text", "gt_answer", ...}
    """
    if dataset_name == "medqa":
        raw_ds = load_medqa(split=split, limit=limit)
        cases = []
        for idx, item in enumerate(raw_ds):
            q    = item.get("question", "")
            opts = item.get("options", {})
            gt   = str(item.get("answer_idx", "")).strip().upper()
            if not q or not opts or gt not in {"A", "B", "C", "D"}:
                continue
            cases.append({
                "id":      str(item.get("id", idx)),
                "question": q,
                "options":  opts,
                "gt":       gt,
            })
        return cases[:limit]

    elif dataset_name == "gsm8k":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("gsm8k", "main", split=split)
        cases = []
        for idx, item in enumerate(ds):
            if idx >= limit:
                break
            cases.append({
                "id":      str(idx),
                "question": item["question"],
                "answer":   item["answer"],
                "gt":       item["answer"].split("####")[-1].strip(),
            })
        return cases

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}，可选: medqa / gsm8k")


def _format_prompt(item: Dict[str, Any], tokenizer, dataset_name: str) -> str:
    """将样本格式化为 prompt 字符串。"""
    if dataset_name == "medqa":
        return format_prompt(tokenizer, item["question"], item["options"])
    elif dataset_name == "gsm8k":
        return (
            f"<|im_start|>system\nYou are a math problem solver. "
            f"Show your reasoning step by step, then give the final answer after ####.<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    raise ValueError(f"未知数据集: {dataset_name}")


def _extract_pred(response: str, dataset_name: str) -> str:
    """从生成文本中提取预测答案。"""
    if dataset_name == "medqa":
        full = response
        pred = extract_answer(full)
        return pred if pred in {"A", "B", "C", "D"} else ""
    elif dataset_name == "gsm8k":
        # 提取 #### 后的数字
        import re
        match = re.search(r"####\s*([\-\d,\.]+)", response)
        return match.group(1).replace(",", "").strip() if match else ""
    return ""


def _check_correct(pred: str, gt: str, dataset_name: str) -> bool:
    """判断预测是否正确。"""
    if not pred:
        return False
    if dataset_name == "medqa":
        return pred.upper() == gt.upper()
    elif dataset_name == "gsm8k":
        try:
            return abs(float(pred) - float(gt.replace(",", ""))) < 1e-3
        except ValueError:
            return pred.strip() == gt.strip()
    return False


# ---------------------------------------------------------------------------
# 构建 Proposer
# ---------------------------------------------------------------------------

def _build_proposer(
    arch: ExecutionArch,
    orch: TriModelOrchestrator,
    device: torch.device,
) -> DualStreamProposer | ShadowSyncProposer:
    """根据架构枚举构造对应的提案引擎。"""
    if arch == ExecutionArch.DUAL_STREAM:
        return DualStreamProposer(
            draft_ctx=orch.draft_ctx,
            base_ctx=orch.base_ctx,
            device=device,
        )
    elif arch == ExecutionArch.SHADOW_SYNC:
        return ShadowSyncProposer(
            draft_ctx=orch.draft_ctx,
            base_ctx=orch.base_ctx,
            device=device,
        )
    raise ValueError(f"未知架构: {arch}")


# ---------------------------------------------------------------------------
# 单次配置完整评测
# ---------------------------------------------------------------------------

def run_single_config(
    config:     DecodeConfig,
    dataset:    List[Dict[str, Any]],
    bundle:     TriModelBundle,
    cache_mgr:  PrefixSharedCacheManager,
    orch:       TriModelOrchestrator,
    dataset_name: str,
    out_dir:    Path,
) -> BenchmarkResult:
    """在给定 DecodeConfig 下跑完整个数据集，返回汇总 BenchmarkResult。

    Args:
        config:       完整解码配置（策略、架构、α、T_sample 等）
        dataset:      已加载的样本列表
        bundle:       三模型 Bundle
        cache_mgr:    PrefixSharedCacheManager（每个 sample 调用 reset()）
        orch:         TriModelOrchestrator
        dataset_name: "medqa" 或 "gsm8k"
        out_dir:      输出目录（存放逐 sample 结果和遥测日志）

    Returns:
        BenchmarkResult
    """
    device = torch.device("cuda:0")
    tokenizer = bundle.tokenizer

    strategy = create_strategy(
        strategy_type=config.strategy,
        signal_params=config.signal_params,
        alpha=config.alpha,
    )

    config_tag = (
        f"{config.strategy.value}_arch-{config.arch.value}"
        f"_alpha{config.alpha:.2f}_t{config.t_sample:.1f}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = out_dir / "telemetry" / config_tag

    results_path = out_dir / f"{config_tag}.jsonl"
    results_fh   = results_path.open("w", encoding="utf-8")

    correct       = 0
    total_tokens  = 0
    total_time    = 0.0
    total_acc     = 0.0
    total_ovr     = 0.0
    n_processed   = 0

    logger.info("开始评测  config=%s  n=%d", config_tag, len(dataset))

    for idx, item in enumerate(dataset):
        sample_id = str(item.get("id", idx))

        # 构造 prompt
        prompt_text = _format_prompt(item, tokenizer, dataset_name)
        prompt_ids  = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        prompt_ids  = prompt_ids.to(device)

        # 遥测 logger（每 sample 独立）
        telemetry = TelemetryLogger(log_dir=telemetry_dir, sample_id=sample_id)

        # Orchestrator Prefill + 提案引擎
        orch.reset()
        orch.init_from_prompt(prompt_ids)

        proposer = _build_proposer(config.arch, orch, device)

        decode_loop = SpeculativeDecodeLoop(
            orchestrator=orch,
            proposer=proposer,
            strategy=strategy,
            telemetry=telemetry,
            tokenizer=tokenizer,
            config=config,
        )

        # 执行解码
        try:
            result: DecodeResult = decode_loop.run(prompt_ids)
        except Exception as exc:
            logger.error("sample %s 解码失败: %s", sample_id, exc, exc_info=True)
            orch.reset()
            continue

        # 解码结果转文本
        gen_ids = result.generated_token_ids
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred     = _extract_pred(response, dataset_name)
        gt       = str(item.get("gt", "")).strip()
        is_correct = _check_correct(pred, gt, dataset_name)

        if is_correct:
            correct += 1

        total_tokens += len(gen_ids)
        total_time   += result.duration_sec
        total_acc    += result.mean_acceptance_rate
        total_ovr    += (result.override_count / max(len(gen_ids), 1))
        n_processed  += 1

        # 写入逐 sample 结果
        sample_record = {
            "id":               sample_id,
            "gt":               gt,
            "pred":             pred,
            "correct":          is_correct,
            "generated_tokens": len(gen_ids),
            "duration_sec":     result.duration_sec,
            "tokens_per_sec":   result.tokens_per_sec,
            "acceptance_rate":  result.mean_acceptance_rate,
            "override_count":   result.override_count,
            "response":         response[:500],  # 截断避免日志过大
        }
        results_fh.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
        results_fh.flush()

        # 遥测持久化
        telemetry.flush()

        # 重置 cache（为下一 sample 准备）
        cache_mgr.reset()

        if (idx + 1) % 20 == 0:
            acc_so_far = correct / n_processed if n_processed else 0.0
            tps = total_tokens / total_time if total_time > 0 else 0.0
            logger.info(
                "[%d/%d] acc=%.3f  tps=%.1f  config=%s",
                idx + 1, len(dataset), acc_so_far, tps, config_tag,
            )

    results_fh.close()

    n = n_processed
    accuracy        = correct / n if n else 0.0
    avg_tps         = total_tokens / total_time if total_time > 0 else 0.0
    avg_accept_rate = total_acc / n if n else 0.0
    avg_ovr_rate    = total_ovr / n if n else 0.0

    bench = BenchmarkResult(
        strategy=config.strategy.value,
        arch=config.arch.value,
        alpha=config.alpha,
        t_sample=config.t_sample,
        dataset=dataset_name,
        n_cases=n,
        accuracy=accuracy,
        correct=correct,
        tokens_per_sec=avg_tps,
        mean_acceptance_rate=avg_accept_rate,
        override_rate=avg_ovr_rate,
        total_generated=total_tokens,
        total_duration_sec=total_time,
    )

    # 写入 summary
    summary_path = out_dir / f"{config_tag}_summary.json"
    summary_path.write_text(
        json.dumps(asdict(bench), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("✓ 完成  config=%s  acc=%.3f  tps=%.1f", config_tag, accuracy, avg_tps)
    return bench


# ---------------------------------------------------------------------------
# 网格搜索
# ---------------------------------------------------------------------------

def run_grid_search(
    bundle:       TriModelBundle,
    cache_mgr:    PrefixSharedCacheManager,
    orch:         TriModelOrchestrator,
    dataset:      List[Dict[str, Any]],
    dataset_name: str,
    strategies:   List[StrategyType],
    alphas:       List[float],
    temperatures: List[float],
    arch:         ExecutionArch,
    signal_params: DomainSignalParams,
    gamma:        int,
    max_new_tokens: int,
    out_dir:      Path,
) -> List[BenchmarkResult]:
    """Strategy × α × T_sample 的正交网格搜索。

    对于不需要 α 的策略（A / B0 / B），α 固定为 -1，只跑一次。

    Args:
        ...（见参数注释）

    Returns:
        所有组合的 BenchmarkResult 列表
    """
    no_alpha_strategies = {
        StrategyType.STANDARD_SD,
        StrategyType.HARD_OVERRIDE_B0,
        StrategyType.HARD_OVERRIDE_B,
    }

    all_results: List[BenchmarkResult] = []

    for strategy in strategies:
        alpha_list = [-1.0] if strategy in no_alpha_strategies else alphas
        for alpha in alpha_list:
            for t_sample in temperatures:
                actual_alpha = max(alpha, 0.0)  # -1 时使用 0（无意义，策略不使用）
                cfg = DecodeConfig(
                    strategy=strategy,
                    arch=arch,
                    signal_params=signal_params,
                    gamma=gamma,
                    max_new_tokens=max_new_tokens,
                    t_sample=t_sample,
                    alpha=actual_alpha,
                )
                result = run_single_config(
                    config=cfg,
                    dataset=dataset,
                    bundle=bundle,
                    cache_mgr=cache_mgr,
                    orch=orch,
                    dataset_name=dataset_name,
                    out_dir=out_dir,
                )
                result.alpha = alpha  # 保留 -1 标记（无 α 策略）
                all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# 汇总表生成
# ---------------------------------------------------------------------------

def generate_summary_table(results: List[BenchmarkResult], out_path: Path) -> None:
    """将所有实验结果写入 JSON + Markdown 汇总表。

    Markdown 表格按 accuracy 降序排列，便于快速对比帕累托最优。

    Args:
        results:  所有 BenchmarkResult 的列表
        out_path: 输出 Markdown 文件路径（同时生成 .json 版本）
    """
    # JSON 版本
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 按 accuracy 降序排列
    sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)

    # Markdown 表格
    header = (
        "| Strategy | Arch | α | T | Dataset | N | Accuracy | "
        "Tokens/sec | Accept Rate | Override Rate |"
    )
    divider = "|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for r in sorted_results:
        alpha_str = f"{r.alpha:.2f}" if r.alpha >= 0 else "N/A"
        rows.append(
            f"| {r.strategy} | {r.arch} | {alpha_str} | {r.t_sample:.1f} "
            f"| {r.dataset} | {r.n_cases} | {r.accuracy:.4f} "
            f"| {r.tokens_per_sec:.1f} | {r.mean_acceptance_rate:.3f} "
            f"| {r.override_rate:.3f} |"
        )

    md_content = "\n".join(["# 实验结果汇总", "", header, divider] + rows)
    out_path.write_text(md_content, encoding="utf-8")

    logger.info("汇总表已写入: %s  %s", out_path, json_path)


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Domain-Steered Speculative Decoding — 评测驱动器"
    )

    # 数据集
    p.add_argument("--dataset", choices=["medqa", "gsm8k"], default="medqa")
    p.add_argument("--split",   default="test")
    p.add_argument("--limit",   type=int, default=300)

    # 模型路径（可覆盖默认值）
    p.add_argument("--target_model", default=None, help="Target 模型路径（默认 Qwen2.5-32B）")
    p.add_argument("--draft_model",  default=None, help="Draft 模型路径")
    p.add_argument("--base_model",   default=None, help="Base 模型路径")

    # 解码参数
    p.add_argument("--arch",     choices=["dual_stream", "shadow_sync"], default="shadow_sync")
    p.add_argument("--gamma",    type=int,   default=5,   help="投机窗口长度 K")
    p.add_argument("--max_new_tokens", type=int, default=256)

    # 策略选择（可多选）
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=[s.value for s in StrategyType],
        default=[s.value for s in StrategyType],
        help="要评测的策略列表（默认全部）",
    )

    # 网格搜索参数
    p.add_argument(
        "--alpha_grid", nargs="+", type=float,
        default=ALPHA_GRID,
        help="α 网格（C1/C2 专用）",
    )
    p.add_argument(
        "--t_sample", nargs="+", type=float,
        default=[0.0],
        help="全局采样温度列表（0=贪婪，0.6=随机）",
    )

    # 领域信号参数
    p.add_argument("--t_fixed",    type=float, default=1.0)
    p.add_argument("--theta_high", type=float, default=0.6)
    p.add_argument("--tau",        type=float, default=0.1)

    # StaticCache 配置
    p.add_argument("--max_cache_len", type=int, default=2048)

    # 输出
    p.add_argument("--out_dir", required=True, help="结果输出目录")

    # 编译模式
    p.add_argument(
        "--no_compile", action="store_true",
        help="跳过 torch.compile（调试用）",
    )

    return p


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    # 构建模型路径
    paths = ModelPaths(
        TARGET=args.target_model or ModelPaths.TARGET,
        BASE=args.base_model    or ModelPaths.BASE,
        DRAFT=args.draft_model  or ModelPaths.DRAFT,
    )

    hw = HardwareConfig(
        compile_mode=None if args.no_compile else "reduce-overhead",
    )

    # 加载三模型
    logger.info("=== 加载三模型 ===")
    bundle = load_tri_models(paths=paths, hw=hw)

    # 构建 PrefixSharedCacheManager
    target_config = bundle.target.config
    small_config  = bundle.base.config   # Base/Draft 同架构

    cache_mgr = PrefixSharedCacheManager(
        target_config=target_config,
        small_config=small_config,
        max_batch_size=1,
        max_cache_len=args.max_cache_len,
        device=device,
        dtype=torch.bfloat16,
    )

    # 构建 Orchestrator
    orch = TriModelOrchestrator(
        target_model=bundle.target,
        draft_model=bundle.draft,
        base_model=bundle.base,
        cache_mgr=cache_mgr,
        device=device,
    )

    # 加载数据集
    logger.info("=== 加载数据集: %s  limit=%d ===", args.dataset, args.limit)
    dataset = _load_dataset(args.dataset, limit=args.limit, split=args.split)
    logger.info("加载完成，共 %d 条", len(dataset))

    # 解析策略枚举列表
    strategies = [StrategyType(s) for s in args.strategies]

    # 执行架构枚举
    arch = ExecutionArch(args.arch)

    # 领域信号超参数
    signal_params = DomainSignalParams(
        t_fixed=args.t_fixed,
        theta_high=args.theta_high,
        tau=args.tau,
    )

    # 启动网格搜索
    logger.info("=== 开始网格搜索 ===")
    logger.info(
        "strategies=%s  alphas=%s  temperatures=%s  arch=%s",
        [s.value for s in strategies], args.alpha_grid, args.t_sample, arch.value,
    )

    all_results = run_grid_search(
        bundle=bundle,
        cache_mgr=cache_mgr,
        orch=orch,
        dataset=dataset,
        dataset_name=args.dataset,
        strategies=strategies,
        alphas=args.alpha_grid,
        temperatures=args.t_sample,
        arch=arch,
        signal_params=signal_params,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        out_dir=out_dir,
    )

    # 生成汇总表
    summary_path = out_dir / "summary_table.md"
    generate_summary_table(all_results, summary_path)

    # 打印核心指标
    logger.info("=== 评测完成，核心结果 ===")
    for r in sorted(all_results, key=lambda x: x.accuracy, reverse=True):
        alpha_str = f"{r.alpha:.2f}" if r.alpha >= 0 else "N/A"
        logger.info(
            "  %-25s  α=%-5s  T=%.1f  acc=%.4f  tps=%.1f  acc_rate=%.3f",
            r.strategy, alpha_str, r.t_sample, r.accuracy, r.tokens_per_sec, r.mean_acceptance_rate,
        )


if __name__ == "__main__":
    main()
