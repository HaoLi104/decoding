#!/usr/bin/env bash
set -euo pipefail

# One-click serial runner for 300-case divergence-override experiments.
# Default sequence:
#   1) baseline
#   2) divergence_v0
#   3) divergence_v2 (tau_delta=0.5, tau_target_opp=1.0)
#   4) summary markdown

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TARGET_MODEL="${TARGET_MODEL:-/data/ocean/decoding/model/Qwen/Qwen3-14B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/ocean/decoding/model/II-Medical-8B}"
SMALL_BASE_MODEL="${SMALL_BASE_MODEL:-/data/ocean/decoding/model/Qwen/Qwen3-8B-Base}"
TOKENIZER="${TOKENIZER:-$TARGET_MODEL}"

LIMIT="${LIMIT:-300}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bf16}"

TARGET_DEVICE_MAP="${TARGET_DEVICE_MAP:-auto}"
DRAFT_DEVICE_MAP="${DRAFT_DEVICE_MAP:-auto}"
SMALL_BASE_DEVICE_MAP="${SMALL_BASE_DEVICE_MAP:-auto}"

TAU_DELTA="${TAU_DELTA:-0.5}"
TAU_TARGET_OPP="${TAU_TARGET_OPP:-1.0}"

RUN_TAG="${RUN_TAG:-medqa300}"
OUT_DIR="${OUT_DIR:-logs/divergence_override/${RUN_TAG}}"
mkdir -p "$OUT_DIR"

BASELINE_OUT="$OUT_DIR/baseline_${LIMIT}.json"
V0_OUT="$OUT_DIR/divergence_v0_${LIMIT}.json"
V2_OUT="$OUT_DIR/divergence_v2_d${TAU_DELTA//./p}_t${TAU_TARGET_OPP//./p}_${LIMIT}.json"
SUMMARY_OUT="$OUT_DIR/summary.md"

run_step() {
  local title="$1"
  shift
  echo ""
  echo "============================================================"
  echo "[RUN] $title"
  echo "============================================================"
  "$@"
}

run_step "baseline (${LIMIT} cases)" \
  python divergence_override_eval.py \
    --mode baseline \
    --target_model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --target_device_map "$TARGET_DEVICE_MAP" \
    --dtype "$DTYPE" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --out "$BASELINE_OUT"

run_step "divergence_v0 (${LIMIT} cases)" \
  python divergence_override_eval.py \
    --mode divergence_v0 \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --small_base_model "$SMALL_BASE_MODEL" \
    --tokenizer "$TOKENIZER" \
    --target_device_map "$TARGET_DEVICE_MAP" \
    --draft_device_map "$DRAFT_DEVICE_MAP" \
    --small_base_device_map "$SMALL_BASE_DEVICE_MAP" \
    --dtype "$DTYPE" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --out "$V0_OUT"

run_step "divergence_v2 (${LIMIT} cases, tau_delta=${TAU_DELTA}, tau_target_opp=${TAU_TARGET_OPP})" \
  python divergence_override_eval.py \
    --mode divergence_v2 \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --small_base_model "$SMALL_BASE_MODEL" \
    --tokenizer "$TOKENIZER" \
    --target_device_map "$TARGET_DEVICE_MAP" \
    --draft_device_map "$DRAFT_DEVICE_MAP" \
    --small_base_device_map "$SMALL_BASE_DEVICE_MAP" \
    --dtype "$DTYPE" \
    --tau_delta "$TAU_DELTA" \
    --tau_target_opp "$TAU_TARGET_OPP" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --out "$V2_OUT"

run_step "summary" \
  python summarize_divergence_results.py \
    --in_dir "$OUT_DIR" \
    --out_md "$SUMMARY_OUT"

echo ""
echo "Done."
echo "Outputs: $OUT_DIR"
echo "Summary: $SUMMARY_OUT"
