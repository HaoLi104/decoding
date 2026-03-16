#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
TARGET_MODEL="${TARGET_MODEL:-/data/ocean/decoding/model/Qwen/Qwen3-14B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/ocean/decoding/model/II-Medical-8B}"
SMALL_BASE_MODEL="${SMALL_BASE_MODEL:-/data/ocean/decoding/model/Qwen/Qwen3-8B-Base}"
TOKENIZER="${TOKENIZER:-$TARGET_MODEL}"

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

OUT_DIR="${OUT_DIR:-logs/divergence_override}"
mkdir -p "$OUT_DIR"

# Threshold grids
TAU_DELTA_LIST=(${TAU_DELTA_LIST:-0.0 0.2 0.5 1.0 1.5})
TAU_TARGET_OPP_LIST=(${TAU_TARGET_OPP_LIST:-0.5 1.0 1.5 2.0})

SCRIPT="divergence_override_eval.py"

common_args=(
  --target_model "$TARGET_MODEL"
  --draft_model "$DRAFT_MODEL"
  --small_base_model "$SMALL_BASE_MODEL"
  --tokenizer "$TOKENIZER"
  --split "$SPLIT"
  --limit "$LIMIT"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --device_map "$DEVICE_MAP"
)

echo "[1/5] baseline"
python "$SCRIPT" \
  --mode baseline \
  --target_model "$TARGET_MODEL" \
  --tokenizer "$TOKENIZER" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device_map "$DEVICE_MAP" \
  --out "$OUT_DIR/baseline.json"

echo "[2/5] strict"
python "$SCRIPT" \
  --mode strict \
  --target_model "$TARGET_MODEL" \
  --draft_model "$DRAFT_MODEL" \
  --tokenizer "$TOKENIZER" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device_map "$DEVICE_MAP" \
  --out "$OUT_DIR/strict.json"

echo "[3/5] divergence_v0"
python "$SCRIPT" \
  --mode divergence_v0 \
  "${common_args[@]}" \
  --out "$OUT_DIR/divergence_v0.json"

echo "[4/5] divergence_v1 grid"
for tau_delta in "${TAU_DELTA_LIST[@]}"; do
  tag="d${tau_delta//./p}"
  python "$SCRIPT" \
    --mode divergence_v1 \
    "${common_args[@]}" \
    --tau_delta "$tau_delta" \
    --out "$OUT_DIR/divergence_v1_${tag}.json"
done

echo "[5/5] divergence_v2 grid"
for tau_delta in "${TAU_DELTA_LIST[@]}"; do
  for tau_opp in "${TAU_TARGET_OPP_LIST[@]}"; do
    d_tag="d${tau_delta//./p}"
    t_tag="t${tau_opp//./p}"
    python "$SCRIPT" \
      --mode divergence_v2 \
      "${common_args[@]}" \
      --tau_delta "$tau_delta" \
      --tau_target_opp "$tau_opp" \
      --out "$OUT_DIR/divergence_v2_${d_tag}_${t_tag}.json"
  done
done

echo "All runs done. Outputs in: $OUT_DIR"
