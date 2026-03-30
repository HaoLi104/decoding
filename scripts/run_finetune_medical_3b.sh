#!/usr/bin/env bash
# =============================================================================
# 一键启动脚本：微调 Qwen2.5-3B-Instruct → Qwen2.5-3B-Instruct-Medical
#
# 完整流程：
#   Step 0  检查 GPU 负载
#   Step 1  准备混合训练数据（75% 医学 + 25% 通用）
#   Step 2  将数据集路径注册到 LLaMA-Factory
#   Step 3  启动 LLaMA-Factory 单卡全参微调
#   Step 4  打印结果查看命令
#
# 用法（远端 H200 机器）：
#   cd /data/ocean/decoding
#   conda activate kvner
#   bash scripts/run_finetune_medical_3b.sh
#
# 如只重跑微调（数据已准备好）：
#   bash scripts/run_finetune_medical_3b.sh --skip_data_prep
# =============================================================================

set -euo pipefail

# ---- 路径配置 ---------------------------------------------------------------
WORK_DIR="/data/ocean/decoding"
DATA_DIR="${WORK_DIR}/data"
MODEL_BASE="${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct"
MODEL_OUTPUT="${WORK_DIR}/model/Qwen/Qwen2.5-3B-Instruct-Medical"
LLAMAFACTORY_DIR="${WORK_DIR}/LLaMA-Factory"
TRAIN_YAML="${WORK_DIR}/train_medical_3b.yaml"
DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
LOG_DIR="${WORK_DIR}/logs/finetune_medical_3b_$(date +%Y%m%d_%H%M%S)"

# ---- 参数解析 ----------------------------------------------------------------
SKIP_DATA_PREP=false
for arg in "$@"; do
    if [[ "${arg}" == "--skip_data_prep" ]]; then
        SKIP_DATA_PREP=true
    fi
done

# ---- 单卡绑定 ---------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

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
    echo "========== Step 1: 准备混合训练数据 =========="
    cd "${WORK_DIR}"
    python prepare_finetune_data.py \
        --out_dir         "${DATA_DIR}" \
        --medical_limit   20000 \
        --general_ratio   0.25 \
        --val_size        0.05 \
        --seed            42 \
        2>&1 | tee "${LOG_DIR}/data_prep.log"
    echo "数据准备完成，日志: ${LOG_DIR}/data_prep.log"
else
    echo "========== Step 1: 跳过数据准备（--skip_data_prep）=========="
    if [[ ! -f "${DATA_DIR}/medical_mix_train.json" ]]; then
        echo "[ERROR] 找不到 ${DATA_DIR}/medical_mix_train.json，请先运行数据准备！"
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

# 使用 Python 安全地合并 dataset_info.json，不破坏已有条目
python - <<EOF
import json
from pathlib import Path

info_path = Path("${DATASET_INFO}")
if info_path.exists():
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
else:
    info = {}

# 注册训练集
info["medical_mix_train"] = {
    "file_name": "${DATA_DIR}/medical_mix_train.json",
    "formatting": "alpaca"
}
# 注册验证集（供手动评估使用）
info["medical_mix_val"] = {
    "file_name": "${DATA_DIR}/medical_mix_val.json",
    "formatting": "alpaca"
}

with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"  ✓ 已写入 dataset_info.json: {info_path}")
print(f"    注册条目: medical_mix_train, medical_mix_val")
EOF

# =============================================================================
# Step 3：启动 LLaMA-Factory 单卡全参微调
# =============================================================================
echo ""
echo "========== Step 3: 启动全参微调（单卡 H200，bfloat16）=========="
echo "  Base 模型:  ${MODEL_BASE}"
echo "  输出路径:   ${MODEL_OUTPUT}"
echo "  配置文件:   ${TRAIN_YAML}"
echo ""

cd "${LLAMAFACTORY_DIR}"

# llamafactory-cli train 读取 YAML 配置启动训练
llamafactory-cli train "${TRAIN_YAML}" \
    2>&1 | tee "${LOG_DIR}/train.log"

TRAIN_EXIT=$?
if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo "[ERROR] 训练失败，退出码: ${TRAIN_EXIT}，请查看日志: ${LOG_DIR}/train.log"
    exit ${TRAIN_EXIT}
fi

echo ""
echo "✓ 训练完成！模型已保存至: ${MODEL_OUTPUT}"

# =============================================================================
# Step 4：结果查看命令
# =============================================================================
echo ""
echo "========== Step 4: 结果查看命令 =========="
echo ""
echo "# 查看训练 loss 曲线（PNG）："
echo "  ls -lh ${MODEL_OUTPUT}/training_loss.png"
echo ""
echo "# 查看最终 checkpoint："
echo "  ls -lh ${MODEL_OUTPUT}/"
echo ""
echo "# 快速推理验证（确认 Chat Template 未破坏）："
echo "  cd ${WORK_DIR}"
echo "  conda activate kvner"
cat << 'PYEOF'
  python verify_medical_draft.py
PYEOF

echo ""
echo "========== 全流程完成 =========="
