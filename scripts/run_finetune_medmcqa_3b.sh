#!/usr/bin/env bash
# =============================================================================
# 一键启动脚本：微调 Qwen2.5-3B-Instruct → Qwen2.5-3B-Instruct-MedMCQA
#
# 完整流程：
#   Step 0  检查 GPU 负载
#   Step 1  准备混合训练数据（75% MedMCQA + 25% 通用）
#   Step 2  注册数据集到 LLaMA-Factory dataset_info.json
#   Step 2.5 定位 llamafactory311 环境 Python
#   Step 3  启动 LLaMA-Factory 单卡全参微调（1 epoch）
#   Step 4  打印后续评测命令
#
# 用法（远端 H200 机器）：
#   cd /data/ocean/decoding
#   conda activate kvner
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/run_finetune_medmcqa_3b.sh
#
# 如只重跑微调（数据已准备好）：
#   bash scripts/run_finetune_medmcqa_3b.sh --skip_data_prep
# =============================================================================

set -euo pipefail

# ---- 路径配置 ---------------------------------------------------------------
WORK_DIR="/data/ocean/decoding"
DATA_DIR="${WORK_DIR}/data"
MODEL_BASE="${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct"
MODEL_OUTPUT="${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct-MedMCQA"
LLAMAFACTORY_DIR="${WORK_DIR}/LLaMA-Factory"
TRAIN_YAML="${WORK_DIR}/train_medmcqa_3b.yaml"
DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
LOG_DIR="${WORK_DIR}/logs/finetune_medmcqa_3b_$(date +%Y%m%d_%H%M%S)"
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
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[网络] HF_ENDPOINT=${HF_ENDPOINT}"

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
    echo "========== Step 1: 准备 MedMCQA 混合训练数据 =========="
    cd "${WORK_DIR}"
    python prepare_finetune_data_medmcqa.py \
        --out_dir       "${DATA_DIR}" \
        --domain_limit  15000 \
        --general_ratio 0.25 \
        --val_size      0.05 \
        --seed          42 \
        2>&1 | tee "${LOG_DIR}/data_prep.log"
    echo "数据准备完成，日志: ${LOG_DIR}/data_prep.log"
else
    echo "========== Step 1: 跳过数据准备（--skip_data_prep）=========="
    if [[ ! -f "${DATA_DIR}/medmcqa_mix_train.json" ]]; then
        echo "[ERROR] 找不到 ${DATA_DIR}/medmcqa_mix_train.json，请先运行数据准备！"
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

info["medmcqa_mix_train"] = {
    "file_name": "${DATA_DIR}/medmcqa_mix_train.json",
    "formatting": "alpaca"
}
info["medmcqa_mix_val"] = {
    "file_name": "${DATA_DIR}/medmcqa_mix_val.json",
    "formatting": "alpaca"
}

with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"  已写入 dataset_info.json: {info_path}")
print(f"    注册条目: medmcqa_mix_train, medmcqa_mix_val")
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
    echo "请检查: conda env list"
    exit 1
fi

echo "  Python 绝对路径: ${LF_PYTHON}"
LF_PY_VERSION="$("${LF_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || true)"
echo "  Python 版本: ${LF_PY_VERSION}"

LF_PY_MAJOR="$(echo "${LF_PY_VERSION}" | cut -d. -f1)"
LF_PY_MINOR="$(echo "${LF_PY_VERSION}" | cut -d. -f2)"
if [[ "${LF_PY_MAJOR:-0}" -lt 3 ]] || [[ "${LF_PY_MAJOR:-0}" -eq 3 && "${LF_PY_MINOR:-0}" -lt 11 ]]; then
    echo "[ERROR] Python 版本过低（${LF_PY_VERSION}），LLaMA-Factory 需要 >= 3.11"
    exit 1
fi

if ! "${LF_PYTHON}" -c "import peft,trl,tyro,llamafactory" >/dev/null 2>&1; then
    echo "[ERROR] 缺少训练依赖（peft / trl / tyro / llamafactory）"
    exit 1
fi
echo "  专用环境检查通过（Python ${LF_PY_VERSION}）"

# =============================================================================
# Step 3：启动 LLaMA-Factory 单卡全参微调
# =============================================================================
echo ""
echo "========== Step 3: 启动全参微调（单卡 H200，bfloat16，1 epoch）=========="
echo "  Base 模型:   ${MODEL_BASE}"
echo "  输出路径:    ${MODEL_OUTPUT}"
echo "  配置文件:    ${TRAIN_YAML}"
echo "  训练 Python: ${LF_PYTHON}"
echo "  CUDA 设备:   ${CUDA_VISIBLE_DEVICES}"
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
# Step 4：后续评测命令
# =============================================================================
echo ""
echo "========== Step 4: 后续评测命令 =========="
echo ""
echo "# 三模型 MedMCQA Baseline 对比（先跑，确认 Draft 超过 Base）："
echo "  cd ${WORK_DIR} && conda activate kvner"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo ""
echo "  # Draft-MedMCQA-3B（微调后）"
echo "  python run_baseline.py \\"
echo "      --model ${MODEL_OUTPUT} \\"
echo "      --dataset medmcqa --split validation --limit 300 \\"
echo "      --out results/baseline/draft_medmcqa_val300.json"
echo ""
echo "  # Base-3B（对照组）"
echo "  python run_baseline.py \\"
echo "      --model ${MODEL_BASE} \\"
echo "      --dataset medmcqa --split validation --limit 300 \\"
echo "      --out results/baseline/base3b_medmcqa_val300.json"
echo ""
echo "  # 对比结果（Pass 条件：Draft acc > Base acc + 0.10）"
echo "  python -c \""
echo "  import json"
echo "  d=json.load(open('results/baseline/draft_medmcqa_val300.json'))"
echo "  b=json.load(open('results/baseline/base3b_medmcqa_val300.json'))"
echo "  print(f'Draft={d[\\\"accuracy\\\"]:.4f}  Base={b[\\\"accuracy\\\"]:.4f}  Delta={d[\\\"accuracy\\\"]-b[\\\"accuracy\\\"]:.4f}')"
echo "  \""
echo ""
echo "========== 全流程完成 =========="
