#!/usr/bin/env bash
# =============================================================================
# 创建 / 修复 LLaMA-Factory 专用 Python 3.11 环境
#
# 目的：
#   - 避免和当前 kvner (Python 3.10) 冲突
#   - 满足 LLaMA-Factory 对 Python >= 3.11 的要求
#   - 安装 peft / trl / tyro / llamafactory 等训练依赖
#
# 用法：
#   cd /data/ocean/decoding
#   conda activate kvner
#   bash scripts/setup_llamafactory_env.sh
# =============================================================================

set -euo pipefail

WORK_DIR="/data/ocean/decoding"
LLAMAFACTORY_DIR="${WORK_DIR}/LLaMA-Factory"
LLAMAFACTORY_ENV="${LLAMAFACTORY_ENV:-llamafactory311}"

echo "========== 创建 LLaMA-Factory 专用环境 =========="
echo "  环境名: ${LLAMAFACTORY_ENV}"
echo "  LLaMA-Factory 目录: ${LLAMAFACTORY_DIR}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 当前 shell 找不到 conda。"
    exit 1
fi

if [[ ! -d "${LLAMAFACTORY_DIR}" ]]; then
    echo "[ERROR] 未找到 ${LLAMAFACTORY_DIR}"
    echo "请先执行："
    echo "  cd ${WORK_DIR}"
    echo "  git clone https://github.com/hiyouga/LLaMA-Factory.git ${LLAMAFACTORY_DIR}"
    exit 1
fi

echo ""
echo "========== Step 0: GPU 负载检查 =========="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
           --format=csv,noheader,nounits || true

echo ""
echo "========== Step 1: 创建 Python 3.11 环境 =========="
if conda env list | awk '{print $1}' | grep -qx "${LLAMAFACTORY_ENV}"; then
    echo "  环境已存在，跳过创建"
else
    conda create -n "${LLAMAFACTORY_ENV}" python=3.11 -y
fi

echo ""
echo "========== Step 2: 升级 pip / setuptools / wheel =========="
conda run -n "${LLAMAFACTORY_ENV}" python -m pip install --upgrade pip setuptools wheel

echo ""
echo "========== Step 3: 安装 PyTorch CUDA 依赖 =========="
# 如远端已安装 torch，可按需跳过；这里显式安装以保证环境自洽。
conda run -n "${LLAMAFACTORY_ENV}" python -m pip install torch torchvision torchaudio

echo ""
echo "========== Step 4: 安装 LLaMA-Factory 及训练依赖 =========="
cd "${LLAMAFACTORY_DIR}"
conda run -n "${LLAMAFACTORY_ENV}" python -m pip install -e ".[torch,metrics]"

echo ""
echo "========== Step 5: 验证依赖 =========="
conda run -n "${LLAMAFACTORY_ENV}" python - <<'EOF'
import sys
import peft
import trl
import tyro
import llamafactory

print("python:", sys.version)
print("peft:", peft.__version__)
print("trl:", trl.__version__)
print("tyro:", tyro.__version__)
print("llamafactory:", getattr(llamafactory, "__version__", "unknown"))
EOF

echo ""
echo "✓ 专用环境准备完成"
echo ""
echo "下一步运行命令："
echo "  cd /data/ocean/decoding"
echo "  conda activate kvner"
echo "  export CUDA_VISIBLE_DEVICES=1"
echo "  bash scripts/run_finetune_medical_3b.sh --skip_data_prep"
