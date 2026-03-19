#!/usr/bin/env bash
set -euo pipefail

# 300题串行脚本（快速主线实验）
# 默认顺序：
# 1) baseline
# 2) standard_speculative (K-token verify, 无 override)
# 3) divergence_v2 (K-token + 拒绝点复判)
# 4) 汇总表 summary.md

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
SPEC_TOKENS="${SPEC_TOKENS:-4}"
DTYPE="${DTYPE:-bf16}"

TARGET_DEVICE_MAP="${TARGET_DEVICE_MAP:-auto}"
DRAFT_DEVICE_MAP="${DRAFT_DEVICE_MAP:-auto}"
SMALL_BASE_DEVICE_MAP="${SMALL_BASE_DEVICE_MAP:-auto}"

TAU_DELTA="${TAU_DELTA:-0.5}"
TAU_TARGET_OPP="${TAU_TARGET_OPP:-1.0}"

RUN_TAG="${RUN_TAG:-k_spec300}"
OUT_DIR="${OUT_DIR:-logs/k_spec_divergence/${RUN_TAG}}"
mkdir -p "$OUT_DIR"

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
  python k_spec_decode_divergence_eval.py \
    --mode baseline \
    --target_model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --target_device_map "$TARGET_DEVICE_MAP" \
    --dtype "$DTYPE" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --out "$OUT_DIR/baseline_${LIMIT}.json"

run_step "standard speculative (${LIMIT} cases, K=${SPEC_TOKENS})" \
  python k_spec_decode_divergence_eval.py \
    --mode standard_speculative \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --tokenizer "$TOKENIZER" \
    --target_device_map "$TARGET_DEVICE_MAP" \
    --draft_device_map "$DRAFT_DEVICE_MAP" \
    --dtype "$DTYPE" \
    --speculative_tokens "$SPEC_TOKENS" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --out "$OUT_DIR/standard_speculative_k${SPEC_TOKENS}_${LIMIT}.json"

run_step "divergence_v2 K-spec (${LIMIT} cases, K=${SPEC_TOKENS})" \
  python k_spec_decode_divergence_eval.py \
    --mode divergence_v2 \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --small_base_model "$SMALL_BASE_MODEL" \
    --tokenizer "$TOKENIZER" \
    --target_device_map "$TARGET_DEVICE_MAP" \
    --draft_device_map "$DRAFT_DEVICE_MAP" \
    --small_base_device_map "$SMALL_BASE_DEVICE_MAP" \
    --dtype "$DTYPE" \
    --speculative_tokens "$SPEC_TOKENS" \
    --tau_delta "$TAU_DELTA" \
    --tau_target_opp "$TAU_TARGET_OPP" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --out "$OUT_DIR/divergence_v2_k${SPEC_TOKENS}_d${TAU_DELTA//./p}_t${TAU_TARGET_OPP//./p}_${LIMIT}.json"

run_step "summary" \
  python summarize_divergence_results.py \
    --in_dir "$OUT_DIR" \
    --out_md "$OUT_DIR/summary.md"

echo ""
echo "Done."
echo "Outputs: $OUT_DIR"
echo "Summary: $OUT_DIR/summary.md"
