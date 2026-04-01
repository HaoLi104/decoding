#!/usr/bin/env bash
# =============================================================================
# 一键启动脚本：微调 Qwen2.5-3B-Instruct → Qwen2.5-3B-Instruct-Law
#
# 完整流程：
#   Step 0  检查 GPU 负载
#   Step 1  准备混合训练数据（75% 法律 + 25% 通用）
#   Step 2  将数据集路径注册到 LLaMA-Factory
#   Step 3  启动 LLaMA-Factory 单卡全参微调
#   Step 4  打印结果查看命令
#
# 用法（远端 H200 机器）：
#   cd /data/ocean/decoding
#   conda activate kvner
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/run_finetune_law_3b.sh
#
# 如只重跑微调（数据已准备好）：
#   bash scripts/run_finetune_law_3b.sh --skip_data_prep
# =============================================================================

set -euo pipefail

# ---- 路径配置 ---------------------------------------------------------------
WORK_DIR="/data/ocean/decoding"
DATA_DIR="${WORK_DIR}/data"
MODEL_BASE="${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct"
MODEL_OUTPUT="${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct-Law"
LLAMAFACTORY_DIR="${WORK_DIR}/LLaMA-Factory"
TRAIN_YAML="${WORK_DIR}/train_law_3b.yaml"
DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
LOG_DIR="${WORK_DIR}/logs/finetune_law_3b_$(date +%Y%m%d_%H%M%S)"
LLAMAFACTORY_ENV="${LLAMAFACTORY_ENV:-llamafactory311}"

# ---- 参数解析 ----------------------------------------------------------------
SKIP_DATA_PREP=false
for arg in "$@"; do
    if [[ "${arg}" == "--skip_data_prep" ]]; then
        SKIP_DATA_PREP=true
    fi
done

# ---- 单卡绑定 ---------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"
echo "[$(date '+%H:%M:%S')] 日志目录: ${LOG_DIR}"

# =============================================================================
# Step 0：检查 GPU 负载
# =============================================================================
echo ""
echo "========== Step 0: GPU 负载检查 =========="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
           --format=csv,noheader,nounits
echo ""

# =============================================================================
# Step 1：准备混合训练数据
# =============================================================================
if [[ "${SKIP_DATA_PREP}" == "false" ]]; then
    echo "========== Step 1: 准备法律混合训练数据 =========="
    cd "${WORK_DIR}"
    python prepare_finetune_data_law.py \
        --out_dir         "${DATA_DIR}" \
        --law_limit       15000 \
        --general_ratio   0.25 \
        --val_size        0.05 \
        --seed            42 \
        2>&1 | tee "${LOG_DIR}/data_prep.log"
    echo "数据准备完成，日志: ${LOG_DIR}/data_prep.log"
else
    echo "========== Step 1: 跳过数据准备（--skip_data_prep）=========="
    if [[ ! -f "${DATA_DIR}/law_mix_train.json" ]]; then
        echo "[ERROR] 找不到 ${DATA_DIR}/law_mix_train.json，请先运行数据准备！"
        exit 1
    fi
fi

# =============================================================================
# Step 2：注册数据集到 LLaMA-Factory dataset_info.json
# =============================================================================
echo ""
echo "========== Step 2: 注册数据集到 LLaMA-Factory =========="

if [[ ! -d "${LLAMAFACTORY_DIR}" ]]; then
    echo "[ERROR] 未找到 LLaMA-Factory 目录: ${LLAMAFACTORY_DIR}"
    echo "        请先执行: git clone https://github.com/hiyouga/LLaMA-Factory.git ${LLAMAFACTORY_DIR}"
    exit 1
fi

python - <<EOF
import json
from pathlib import Path

info_path = Path("${DATASET_INFO}")
if info_path.exists():
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
else:
    info = {}

info["law_mix_train"] = {
    "file_name": "${DATA_DIR}/law_mix_train.json",
    "formatting": "alpaca"
}
info["law_mix_val"] = {
    "file_name": "${DATA_DIR}/law_mix_val.json",
    "formatting": "alpaca"
}

with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"  已写入 dataset_info.json: {info_path}")
print(f"    注册条目: law_mix_train, law_mix_val")
EOF

# =============================================================================
# Step 2.5：定位 LLaMA-Factory 专用环境的 Python 绝对路径
# =============================================================================
echo ""
echo "========== Step 2.5: 定位专用环境 Python =========="
echo "  目标环境名: ${LLAMAFACTORY_ENV}"

CONDA_ENVS_DIR="$(dirname "${CONDA_PREFIX:-}")"

LF_PYTHON=""
for CANDIDATE in \
    "${CONDA_ENVS_DIR}/${LLAMAFACTORY_ENV}/bin/python3" \
    "${CONDA_ENVS_DIR}/${LLAMAFACTORY_ENV}/bin/python" \
    "/opt/conda/envs/${LLAMAFACTORY_ENV}/bin/python3" \
    "/opt/conda/envs/${LLAMAFACTORY_ENV}/bin/python" \
    "${HOME}/.conda/envs/${LLAMAFACTORY_ENV}/bin/python3" \
    "${HOME}/.conda/envs/${LLAMAFACTORY_ENV}/bin/python"
do
    if [[ -x "${CANDIDATE}" ]]; then
        LF_PYTHON="${CANDIDATE}"
        break
    fi
done

if [[ -z "${LF_PYTHON}" ]]; then
    echo "[ERROR] 无法找到 ${LLAMAFACTORY_ENV} 的 Python 可执行文件。"
    echo ""
    echo "请先执行："
    echo "  bash scripts/setup_llamafactory_env.sh"
    echo ""
    echo "或手动检查路径："
    echo "  conda env list"
    exit 1
fi

echo "  Python 绝对路径: ${LF_PYTHON}"

LF_PY_VERSION="$("${LF_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || true)"
echo "  Python 版本: ${LF_PY_VERSION}"

LF_PY_MAJOR="$(echo "${LF_PY_VERSION}" | cut -d. -f1)"
LF_PY_MINOR="$(echo "${LF_PY_VERSION}" | cut -d. -f2)"
if [[ "${LF_PY_MAJOR:-0}" -lt 3 ]] || [[ "${LF_PY_MAJOR:-0}" -eq 3 && "${LF_PY_MINOR:-0}" -lt 11 ]]; then
    echo "[ERROR] Python 版本过低（${LF_PY_VERSION}），LLaMA-Factory 需要 >= 3.11"
    echo "请重新执行：bash scripts/setup_llamafactory_env.sh"
    exit 1
fi

if ! "${LF_PYTHON}" -c "import peft,trl,tyro,llamafactory" >/dev/null 2>&1; then
    echo "[ERROR] 缺少训练依赖（peft / trl / tyro / llamafactory）"
    echo "请执行：bash scripts/setup_llamafactory_env.sh"
    exit 1
fi
echo "  专用环境检查通过（Python ${LF_PY_VERSION}）"

# =============================================================================
# Step 3：启动 LLaMA-Factory 单卡全参微调
# =============================================================================
echo ""
echo "========== Step 3: 启动全参微调（单卡 H200，bfloat16）=========="
echo "  Base 模型:  ${MODEL_BASE}"
echo "  输出路径:   ${MODEL_OUTPUT}"
echo "  配置文件:   ${TRAIN_YAML}"
echo "  训练 Python: ${LF_PYTHON}"
echo "  CUDA 设备:  ${CUDA_VISIBLE_DEVICES}"
echo ""

cd "${LLAMAFACTORY_DIR}"

"${LF_PYTHON}" src/train.py "${TRAIN_YAML}" \
    2>&1 | tee "${LOG_DIR}/train.log"

TRAIN_EXIT=${PIPESTATUS[0]}
if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo "[ERROR] 训练失败，退出码: ${TRAIN_EXIT}，请查看日志: ${LOG_DIR}/train.log"
    exit ${TRAIN_EXIT}
fi

echo ""
echo "训练完成！模型已保存至: ${MODEL_OUTPUT}"

# =============================================================================
# Step 4：结果查看命令
# =============================================================================
echo ""
echo "========== Step 4: 后续评测命令 =========="
echo ""
echo "# 查看训练 loss 曲线："
echo "  ls -lh ${MODEL_OUTPUT}/training_loss.png"
echo ""
echo "# Baseline 评测（JEC-QA，三个模型）："
echo "  cd ${WORK_DIR} && conda activate kvner"
echo ""
echo "  # Target-32B"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  python run_baseline.py \\"
echo "      --model ${WORK_DIR}/model/Qwen/Qwen2.5-32B-Instruct \\"
echo "      --dataset jecqa --limit 200 \\"
echo "      --out results/baseline/target_only_jecqa_200.json"
echo ""
echo "  # Base-3B"
echo "  python run_baseline.py \\"
echo "      --model ${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct \\"
echo "      --dataset jecqa --limit 200 \\"
echo "      --out results/baseline/base_only_jecqa_200.json"
echo ""
echo "  # Draft-Law-3B"
echo "  python run_baseline.py \\"
echo "      --model ${MODEL_OUTPUT} \\"
echo "      --dataset jecqa --limit 200 \\"
echo "      --out results/baseline/draft_law_jecqa_200.json"

echo ""
echo "========== 全流程完成 =========="
