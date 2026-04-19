#!/usr/bin/env bash
# =============================================================================
# DAF Flywheel PoC — 主驱动脚本（M1 → M7 全流程，Round 0 + Round 1）
#
# 流程：
#   M1: Round 0 解码采集 flip 事件
#   M2: Round 0 FDLP 反向传播打分（4 套对照）
#   M3: Round 0 SFT 数据 + LoRA yaml 自动生成
#   M4: Round 0 LLaMA-Factory LoRA 训练（llamafactory311 conda）
#   M5: Round 0 merge_lora → Target_v1，纯 Target 评测（Surgery + MMLU）→ Go/No-Go
#   M6: Round 1 重复 M1-M5（用 Target_v1 作为新 Target）→ Target_v2
#   M7: 跨轮收敛分析 + 热点稳定性 + 汇总报告
#
# 使用方式：
#   cd /data/ocean/decoding
#   bash scripts/run_daf_flywheel.sh                  # 跑完整 Round 0 + Round 1
#   bash scripts/run_daf_flywheel.sh round0_only      # 只跑 Round 0
#   bash scripts/run_daf_flywheel.sh m7_only          # 仅做 Round 0+1 已完成后的 M7 汇总
# =============================================================================

set -euo pipefail
cd /data/ocean/decoding

MODE="${1:-full}"      # full | round0_only | m7_only

# ---------- conda 激活（与 run_finetune_medmcqa_surgery_3b.sh 同口径）----------
_CONDA_SH=""
for _d in "${CONDA_ROOT:-}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
  [[ -z "$_d" ]] && continue
  if [[ -f "${_d}/etc/profile.d/conda.sh" ]]; then
    _CONDA_SH="${_d}/etc/profile.d/conda.sh"
    break
  fi
done
if [[ -n "$_CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$_CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: 未找到 conda" >&2
  exit 1
fi
conda activate kvner

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=0
export HF_ENDPOINT=https://hf-mirror.com

# ---------- 路径常量 ----------
TARGET_V0="/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct"
DRAFT="/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Surgery/checkpoint-1676"
BASE="/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct"

DATA_DIR="/data/ocean/decoding/data"
LF_DIR="/data/ocean/decoding/LLaMA-Factory"
LF_DATASET_INFO="${LF_DIR}/data/dataset_info.json"

OUT_BASE="logs/daf_flywheel_$(date +%Y%m%d_%H%M)"
OUT_BASE_OVERRIDE="${OUT_BASE_OVERRIDE:-}"
if [[ -n "$OUT_BASE_OVERRIDE" ]]; then
  OUT_BASE="$OUT_BASE_OVERRIDE"
fi
mkdir -p "${OUT_BASE}"
echo "OUT_BASE = ${OUT_BASE}"

# ---------- 关键超参 ----------
FLIP_LIMIT=5000             # Round k 解码题量（plan: 5000）
FDLP_MAX_EVENTS=2000        # FDLP 单轮处理最多 flip 事件
FDLP_MAX_PREFIX_LEN=1024
TOP_K=8                     # plan: 8
R_TOTAL=128                 # plan: 128

LORA_EPOCHS=1.0             # 控制 ~1000 步以内
LORA_BSZ=2
LORA_GRAD_ACCUM=8
LORA_LR=1.0e-4

EVAL_SURGERY_LIMIT=200
MMLU_LIMIT_PER_SUBJECT=100

# =============================================================================
# 通用：跑完整一轮 Round k
#   $1 = round_id (0 / 1)
#   $2 = 当轮 Target 模型路径（输入）
#   $3 = 输出目录
#   $4 = 训练完成后合并出来的 Target_{k+1} 路径（输出）
# =============================================================================
run_one_round() {
  local R="$1"
  local TGT_IN="$2"
  local OUT_DIR="$3"
  local TGT_OUT="$4"

  mkdir -p "${OUT_DIR}"
  echo
  echo "###################  ROUND ${R}  ###################"
  echo "  TARGET_IN  = ${TGT_IN}"
  echo "  OUT_DIR    = ${OUT_DIR}"
  echo "  TARGET_OUT = ${TGT_OUT}"

  # ---------- M1: 解码采集 flip ----------
  echo "----- [Round ${R}  M1] flip 采集 -----"
  python -m daf.run_flip_logger \
      --round "${R}" \
      --target_model "${TGT_IN}" \
      --draft_model  "${DRAFT}" \
      --base_model   "${BASE}" \
      --dataset      medmcqa \
      --subject      Surgery \
      --split        train \
      --limit        "${FLIP_LIMIT}" \
      --arch         shadow_sync \
      --gamma        5 \
      --max_new_tokens 256 \
      --alpha        50 \
      --c4_tau       0.05 \
      --t_sample     0.0 \
      --seed         $((42 + R)) \
      --out_dir      "${OUT_DIR}" \
      2>&1 | tee "${OUT_DIR}/m1_flip_logger.log"

  local FLIP_JSONL="${OUT_DIR}/flip_events_round${R}.jsonl"

  # ---------- M2: FDLP 打分 ----------
  echo "----- [Round ${R}  M2] FDLP 打分 -----"
  local LAYER_SCORES="${OUT_DIR}/layer_scores_round${R}.json"
  python -m daf.fdlp_score \
      --target_model    "${TGT_IN}" \
      --flip_jsonl      "${FLIP_JSONL}" \
      --out             "${LAYER_SCORES}" \
      --top_k           "${TOP_K}" \
      --r_total         "${R_TOTAL}" \
      --max_events      "${FDLP_MAX_EVENTS}" \
      --max_prefix_len  "${FDLP_MAX_PREFIX_LEN}" \
      --random_subset_ratio 0.10 \
      --seed            $((42 + R)) \
      2>&1 | tee "${OUT_DIR}/m2_fdlp_score.log"

  # ---------- M3: SFT 数据 + yaml ----------
  echo "----- [Round ${R}  M3] SFT 数据 + LoRA yaml -----"
  python -m daf.build_flip_sft_data \
      --flip_jsonl       "${FLIP_JSONL}" \
      --tokenizer        "${TGT_IN}" \
      --out_dir          "${DATA_DIR}" \
      --round            "${R}" \
      --max_samples      30000 \
      --max_flip_events  20000 \
      --max_prefix_len   1024 \
      --general_ratio    0.25 \
      --val_ratio        0.05 \
      --seed             $((42 + R)) \
      --dataset_info_json "${LF_DATASET_INFO}" \
      2>&1 | tee "${OUT_DIR}/m3_build_sft.log"

  local LORA_YAML="${OUT_DIR}/train_daf_round${R}.yaml"
  local LORA_OUT="/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v$((R+1))-lora"
  python -m daf.gen_lora_yaml \
      --layer_scores  "${LAYER_SCORES}" \
      --template      train_medmcqa_surgery_3b.yaml \
      --base_model    "${TGT_IN}" \
      --dataset_key   "daf_round${R}_train" \
      --output_yaml   "${LORA_YAML}" \
      --output_dir    "${LORA_OUT}" \
      --subset        fdlp \
      --top_k         "${TOP_K}" \
      --r_total       "${R_TOTAL}" \
      --num_train_epochs "${LORA_EPOCHS}" \
      --per_device_train_batch_size "${LORA_BSZ}" \
      --gradient_accumulation_steps "${LORA_GRAD_ACCUM}" \
      --learning_rate "${LORA_LR}" \
      --cutoff_len    1024 \
      --dataset_dir   "${LF_DIR}/data" \
      2>&1 | tee "${OUT_DIR}/m3_gen_yaml.log"

  # ---------- M4: LLaMA-Factory LoRA 训练 ----------
  echo "----- [Round ${R}  M4] LLaMA-Factory LoRA 训练 -----"
  conda run -n llamafactory311 --no-capture-output \
      env CUDA_VISIBLE_DEVICES=0 \
      llamafactory-cli train "${LORA_YAML}" \
      2>&1 | tee "${OUT_DIR}/m4_llamafactory.log"

  # ---------- M5a: merge LoRA ----------
  echo "----- [Round ${R}  M5a] merge_lora → ${TGT_OUT} -----"
  python -m daf.merge_lora \
      --base_model   "${TGT_IN}" \
      --lora_adapter "${LORA_OUT}" \
      --output_model "${TGT_OUT}" \
      --device_map   cuda:0 \
      --dtype        bfloat16 \
      2>&1 | tee "${OUT_DIR}/m5a_merge_lora.log"

  # ---------- M5b: 评测 Target_{R+1} ----------
  echo "----- [Round ${R}  M5b] 纯 Target 评测 (Surgery + MMLU) -----"
  python -m daf.run_eval_round \
      --target_model            "${TGT_OUT}" \
      --round                   $((R+1)) \
      --out                     "${OUT_DIR}/eval_round$((R+1)).json" \
      --surgery_limit           "${EVAL_SURGERY_LIMIT}" \
      --mmlu_limit_per_subject  "${MMLU_LIMIT_PER_SUBJECT}" \
      --max_new_tokens          256 \
      2>&1 | tee "${OUT_DIR}/m5b_eval.log"

  # ---------- 同时评测一下当轮的 baseline (Target_R) 以便比较 ----------
  if [[ ! -f "${OUT_DIR}/eval_round${R}.json" ]]; then
      echo "----- [Round ${R}  M5c] 当轮 Target_${R} baseline 评测（对照基线）-----"
      python -m daf.run_eval_round \
          --target_model            "${TGT_IN}" \
          --round                   "${R}" \
          --out                     "${OUT_DIR}/eval_round${R}.json" \
          --surgery_limit           "${EVAL_SURGERY_LIMIT}" \
          --mmlu_limit_per_subject  "${MMLU_LIMIT_PER_SUBJECT}" \
          --max_new_tokens          256 \
          2>&1 | tee "${OUT_DIR}/m5c_eval_baseline.log"
  fi

  echo "###################  ROUND ${R} END  ###################"
}


# =============================================================================
# 主流程
# =============================================================================

ROUND0_OUT="${OUT_BASE}/round0"
ROUND1_OUT="${OUT_BASE}/round1"
TARGET_V1="/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v1"
TARGET_V2="/data/ocean/decoding/model/Qwen/Qwen2.5-32B-Instruct-DAF-v2"

if [[ "$MODE" != "m7_only" ]]; then
  # -------- Round 0 --------
  run_one_round 0 "${TARGET_V0}" "${ROUND0_OUT}" "${TARGET_V1}"

  if [[ "$MODE" != "round0_only" ]]; then
    # -------- Round 1 --------
    run_one_round 1 "${TARGET_V1}" "${ROUND1_OUT}" "${TARGET_V2}"
  fi
fi

# =============================================================================
# M7: 收敛 + 热点 + 汇总
# =============================================================================
echo
echo "###################  M7  汇总分析  ###################"
M7_OUT="${OUT_BASE}/m7"
mkdir -p "${M7_OUT}"

if [[ -f "${ROUND0_OUT}/flip_events_round0.jsonl" && -f "${ROUND1_OUT}/flip_events_round1.jsonl" ]]; then
    python -m daf.convergence_check \
        --flip_jsonls "${ROUND0_OUT}/flip_events_round0.jsonl" \
                       "${ROUND1_OUT}/flip_events_round1.jsonl" \
        --eval_jsons  "${ROUND0_OUT}/eval_round0.json" \
                       "${ROUND0_OUT}/eval_round1.json" \
        --out         "${M7_OUT}/convergence_round1.json" \
        2>&1 | tee "${M7_OUT}/m7_convergence.log"

    python -m daf.hotspot_stability \
        --layer_scores "${ROUND0_OUT}/layer_scores_round0.json" \
                       "${ROUND1_OUT}/layer_scores_round1.json" \
        --subset       fdlp \
        --top_k        "${TOP_K}" \
        --out          "${M7_OUT}/hotspot_stability.json" \
        2>&1 | tee "${M7_OUT}/m7_hotspot.log"

    # ---------- 一表汇总 ----------
    python - <<PYEOF
import json, pathlib
m7  = pathlib.Path("${M7_OUT}")
r0  = pathlib.Path("${ROUND0_OUT}")
r1  = pathlib.Path("${ROUND1_OUT}")

def maybe(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

summary = {
    "out_base":    "${OUT_BASE}",
    "round0": {
        "flip_logger_summary": maybe(r0 / "flip_logger_round0_summary.json"),
        "layer_scores_meta":   (maybe(r0 / "layer_scores_round0.json") or {}).get("meta"),
        "eval_round0":         maybe(r0 / "eval_round0.json"),
        "eval_round1_after_v1":maybe(r0 / "eval_round1.json"),
    },
    "round1": {
        "flip_logger_summary": maybe(r1 / "flip_logger_round1_summary.json"),
        "layer_scores_meta":   (maybe(r1 / "layer_scores_round1.json") or {}).get("meta"),
        "eval_round1":         maybe(r1 / "eval_round1.json"),
        "eval_round2_after_v2":maybe(r1 / "eval_round2.json"),
    },
    "convergence": maybe(m7 / "convergence_round1.json"),
    "hotspot":     maybe(m7 / "hotspot_stability.json"),
}
out = m7 / "daf_flywheel_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✓ DAF 飞轮汇总写入 {out}")
PYEOF
else
    echo "缺少 flip_events_round{0,1}.jsonl，跳过 M7。"
fi

echo
echo "########## DAF FLYWHEEL POC 全流程完成 ##########"
echo "查看汇总:    cat ${M7_OUT}/daf_flywheel_summary.json | python -m json.tool"
echo "查看收敛:    cat ${M7_OUT}/convergence_round1.json | python -m json.tool"
echo "查看热点:    cat ${M7_OUT}/hotspot_stability.json  | python -m json.tool"
