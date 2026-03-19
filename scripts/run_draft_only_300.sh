#!/usr/bin/env bash
set -euo pipefail

# Draft-only 评测脚本（用于对比：当前架构是否超过专家小模型）
# 说明：这里用 baseline 模式，但把 target_model 指向 draft 模型，
# 等价于“只跑 draft 单模型 greedy 生成”。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DRAFT_MODEL="${DRAFT_MODEL:-/data/ocean/decoding/model/II-Medical-8B}"
TOKENIZER="${TOKENIZER:-$DRAFT_MODEL}"

LIMIT="${LIMIT:-300}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bf16}"
DRAFT_DEVICE_MAP="${DRAFT_DEVICE_MAP:-auto}"

RUN_TAG="${RUN_TAG:-draft_only_300}"
OUT_DIR="${OUT_DIR:-logs/draft_only/${RUN_TAG}}"
mkdir -p "$OUT_DIR"

OUT_JSON="$OUT_DIR/draft_baseline_${LIMIT}.json"

echo "============================================================"
echo "[RUN] draft-only baseline (${LIMIT} cases)"
echo "============================================================"
python k_spec_decode_divergence_eval.py \
  --mode baseline \
  --target_model "$DRAFT_MODEL" \
  --tokenizer "$TOKENIZER" \
  --target_device_map "$DRAFT_DEVICE_MAP" \
  --dtype "$DTYPE" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out "$OUT_JSON"

echo ""
echo "Done."
echo "Output: $OUT_JSON"
