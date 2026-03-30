#!/usr/bin/env bash
# =============================================================================
# 创建 / 修复 LLaMA-Factory 专用 Python 3.11 环境
#
# 目的：
#   - 避免和当前 kvner (Python 3.10) 冲突
#   - 满足 LLaMA-Factory 对 Python >= 3.11 的要求
#   - 安装 peft / trl / tyro / llamafactory 等训练依赖
#
# 核心设计：
#   环境内所有 Python 调用均使用 conda 环境的**绝对路径**，
#   完全绕开 pyenv shim 对 "python" 命令名的拦截。
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

# ---- 解析 conda 环境的绝对 Python 路径，绕开 pyenv shim ----
# CONDA_PREFIX 指向当前激活的环境（kvner），由此推导 envs 根目录。
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
    echo "[ERROR] 创建环境后仍无法找到 Python 可执行文件："
    for P in "${CONDA_ENVS_DIR}/${LLAMAFACTORY_ENV}/bin/python3" \
             "${HOME}/.conda/envs/${LLAMAFACTORY_ENV}/bin/python3"; do
        echo "  尝试路径: ${P} → 不存在"
    done
    echo "请检查 conda 安装目录后重试。"
    exit 1
fi

echo "  Python 绝对路径: ${LF_PYTHON}"
echo "  Python 版本: $("${LF_PYTHON}" --version 2>&1)"

echo ""
echo "========== Step 2: 升级 pip / setuptools / wheel =========="
"${LF_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo ""
echo "========== Step 3: 安装 PyTorch CUDA 依赖 =========="
# 使用 CUDA 12.4 编译的轮子（cu124），向上兼容 CUDA 12.x 驱动（如服务器的 12.8）。
# 不使用 pip install torch（无版本锁定），防止装到要求 CUDA > 12.8 的过新版本。
"${LF_PYTHON}" -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "========== Step 4: 安装 LLaMA-Factory 及训练依赖 =========="
cd "${LLAMAFACTORY_DIR}"

# 先安装 LLaMA-Factory 本体（editable）
"${LF_PYTHON}" -m pip install -e .

# 再显式补齐训练依赖，避免 pyproject extras 名称变化导致依赖未装全
"${LF_PYTHON}" -m pip install \
    transformers datasets accelerate peft trl tyro sentencepiece scipy

echo ""
echo "========== Step 5: 验证依赖 =========="
"${LF_PYTHON}" - <<'EOF'
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
