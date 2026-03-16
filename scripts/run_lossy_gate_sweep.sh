#!/usr/bin/env bash
set -euo pipefail

# Phase-A threshold sweep for classifier-gated lossy speculative decoding

TARGET_MODEL="${TARGET_MODEL:-/data/ocean/decoding/model/Qwen/Qwen3-14B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/ocean/decoding/model/II-Medical-8B}"
TOKENIZER="${TOKENIZER:-/data/ocean/decoding/model/Qwen/Qwen3-14B}"
GATE_CKPT="${GATE_CKPT:-checkpoints/token_classifier/student_mlp_mined_full_test_qwen_dp.pt}"
LIMIT="${LIMIT:-300}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
OUT_DIR="${OUT_DIR:-logs/lossy_gate_sweep}"

mkdir -p "$OUT_DIR"

echo "[1/3] Baseline target-only"
python lossy_spec_decode_eval.py \
  --mode baseline \
  --target_model "$TARGET_MODEL" \
  --tokenizer "$TOKENIZER" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device_map "$DEVICE_MAP" \
  --out "$OUT_DIR/baseline_target.json"

echo "[2/3] Strict speculative (no gate override)"
python lossy_spec_decode_eval.py \
  --mode strict \
  --target_model "$TARGET_MODEL" \
  --draft_model "$DRAFT_MODEL" \
  --tokenizer "$TOKENIZER" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device_map "$DEVICE_MAP" \
  --out "$OUT_DIR/strict_spec.json"

echo "[3/3] Gate sweep"
for TAU in 0.30 0.40 0.50 0.60 0.70; do
  echo "  - tau=${TAU}"
  python lossy_spec_decode_eval.py \
    --mode gate \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --tokenizer "$TOKENIZER" \
    --gate_ckpt "$GATE_CKPT" \
    --tau "$TAU" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device_map "$DEVICE_MAP" \
    --out "$OUT_DIR/gate_tau_${TAU}.json"
done

echo "Sweep done. Outputs in: $OUT_DIR"
