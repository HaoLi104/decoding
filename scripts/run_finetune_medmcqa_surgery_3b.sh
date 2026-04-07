#!/usr/bin/env bash
# =============================================================================
# 外科专项微调脚本 — MedMCQA Surgery → Qwen2.5-3B-Instruct-Surgery
#
# 执行顺序：
#   1. 准备专项微调数据（subject=Surgery）
#   2. 注册数据集到 LLaMA-Factory
#   3. 启动 FFT 微调（5 epochs，checkpoint sweep）
#   4. 扫描全部 checkpoint，找 Delta(D-B) 最大拐点
#
# 使用方式：
#   cd /data/ocean/decoding
#   bash scripts/run_finetune_medmcqa_surgery_3b.sh
# =============================================================================

set -e
cd /data/ocean/decoding

# 非交互式 bash 不会加载 ~/.bashrc，必须先 source conda.sh，否则 conda activate 报错：
#   CondaError: Run 'conda init' before 'conda activate'
_CONDA_SH=""
# 若安装路径不在下列列表，可先：export CONDA_ROOT=/path/to/conda 再运行本脚本
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
  echo "ERROR: 未找到 conda（已尝试 miniforge3/miniconda3/anaconda3/opt/conda）" >&2
  exit 1
fi
conda activate kvner

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

LLAMAFACTORY_DIR=/data/ocean/decoding/LLaMA-Factory
DATA_DIR=/data/ocean/decoding/data
BASE_MODEL=/data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct

# =============================================================================
# Step 1：准备外科专项微调数据
# =============================================================================
echo "========== Step 1: 准备 Surgery 专项微调数据 =========="
python prepare_finetune_data_medmcqa.py \
    --out_dir       "${DATA_DIR}" \
    --subject       "Surgery" \
    --domain_limit  0 \
    --general_ratio 0.25 \
    --val_size      0.05 \
    --seed          42

# =============================================================================
# Step 2：注册数据集到 LLaMA-Factory
# =============================================================================
echo "========== Step 2: 注册数据集 =========="
DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"

python - <<'PYEOF'
import json, pathlib

info_path = pathlib.Path("/data/ocean/decoding/LLaMA-Factory/data/dataset_info.json")
data = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

data["medmcqa_surgery_mix_train"] = {
    "file_name": "/data/ocean/decoding/data/medmcqa_surgery_mix_train.json",
    "formatting": "alpaca"
}
data["medmcqa_surgery_mix_val"] = {
    "file_name": "/data/ocean/decoding/data/medmcqa_surgery_mix_val.json",
    "formatting": "alpaca"
}

info_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已注册 medmcqa_surgery_mix_train / medmcqa_surgery_mix_val 到 {info_path}")
PYEOF

# =============================================================================
# Step 3：启动微调
# =============================================================================
echo "========== Step 3: 启动 Surgery 专项微调 =========="
# llamafactory-cli 默认按 CWD 下相对路径 "data/dataset_info.json" 查找数据集注册表；
# 用 --dataset_dir 显式指定 LLaMA-Factory 内置的 data/ 目录，避免路径不一致报错。
conda run -n llamafactory311 --no-capture-output \
    llamafactory-cli train /data/ocean/decoding/train_medmcqa_surgery_3b.yaml \
    --dataset_dir "${LLAMAFACTORY_DIR}/data"

# =============================================================================
# Step 4：扫描 checkpoint，找最优模型
# =============================================================================
echo "========== Step 4: 扫描 checkpoint =========="
python eval_checkpoints.py \
    --ckpt_dir  /data/ocean/decoding/model/Qwen/Qwen2.5-3B-Instruct-Surgery \
    --base_acc  0.4940 \
    --dataset   medmcqa \
    --subject   Surgery \
    --split     validation \
    --limit     0 \
    --out       results/baseline/surgery_checkpoint_sweep.json

echo "========== 全流程完成 =========="
echo "查看结果："
echo "  cat results/baseline/surgery_checkpoint_sweep.json | python -m json.tool"
