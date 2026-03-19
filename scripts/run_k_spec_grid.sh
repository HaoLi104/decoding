#!/usr/bin/env bash
set -euo pipefail

# 一键网格扫描脚本（串行）
# 扫描维度：
#   - K: SPEC_TOKENS_LIST（默认 2 4 6）
#   - tau_delta: TAU_DELTA_LIST（默认 0.2 0.5 1.0）
#   - tau_target_opp: TAU_TARGET_OPP_LIST（默认 1.0 1.5 2.0）
# 每个组合运行：
#   1) standard_speculative
#   2) divergence_v2
# 另外会先跑一次 baseline 作为统一参照。

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

SPEC_TOKENS_LIST=(${SPEC_TOKENS_LIST:-2 4 6})
TAU_DELTA_LIST=(${TAU_DELTA_LIST:-0.2 0.5 1.0})
TAU_TARGET_OPP_LIST=(${TAU_TARGET_OPP_LIST:-1.0 1.5 2.0})

RUN_TAG="${RUN_TAG:-k_spec_grid300}"
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

# 0) baseline（只跑一次）
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

# 1) K 网格：standard_speculative + v2
for k in "${SPEC_TOKENS_LIST[@]}"; do
  run_step "standard speculative K=${k} (${LIMIT} cases)" \
    python k_spec_decode_divergence_eval.py \
      --mode standard_speculative \
      --target_model "$TARGET_MODEL" \
      --draft_model "$DRAFT_MODEL" \
      --tokenizer "$TOKENIZER" \
      --target_device_map "$TARGET_DEVICE_MAP" \
      --draft_device_map "$DRAFT_DEVICE_MAP" \
      --dtype "$DTYPE" \
      --speculative_tokens "$k" \
      --split "$SPLIT" \
      --limit "$LIMIT" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --out "$OUT_DIR/standard_speculative_k${k}_${LIMIT}.json"

  for tau_delta in "${TAU_DELTA_LIST[@]}"; do
    for tau_opp in "${TAU_TARGET_OPP_LIST[@]}"; do
      d_tag="${tau_delta//./p}"
      t_tag="${tau_opp//./p}"
      run_step "v2 K=${k}, d=${tau_delta}, t=${tau_opp} (${LIMIT} cases)" \
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
          --speculative_tokens "$k" \
          --tau_delta "$tau_delta" \
          --tau_target_opp "$tau_opp" \
          --split "$SPLIT" \
          --limit "$LIMIT" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          --out "$OUT_DIR/divergence_v2_k${k}_d${d_tag}_t${t_tag}_${LIMIT}.json"
    done
  done
done

# 2) 汇总
run_step "summary" \
  python summarize_divergence_results.py \
    --in_dir "$OUT_DIR" \
    --out_md "$OUT_DIR/summary.md"

echo ""
echo "Done."
echo "Outputs: $OUT_DIR"
echo "Summary: $OUT_DIR/summary.md"
