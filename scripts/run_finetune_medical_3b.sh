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
#   export CUDA_VISIBLE_DEVICES=1
#   bash scripts/run_finetune_medical_3b.sh
#
# 如只重跑微调（数据已准备好）：
#   bash scripts/run_finetune_medical_3b.sh --skip_data_prep
#
# 说明：
#   - 数据准备阶段沿用当前环境（kvner）
#   - LLaMA-Factory 训练阶段使用专用 Python 3.11 环境的绝对路径，
#     完全绕开 pyenv shim 和 conda run 的干扰
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
LLAMAFACTORY_ENV="${LLAMAFACTORY_ENV:-llamafactory311}"

# ---- 参数解析 ----------------------------------------------------------------
SKIP_DATA_PREP=false
for arg in "$@"; do
    if [[ "${arg}" == "--skip_data_prep" ]]; then
        SKIP_DATA_PREP=true
    fi
done

# ---- 单卡绑定 ---------------------------------------------------------------
# 如果用户已在外部显式设置 CUDA_VISIBLE_DEVICES，则尊重用户设置；
# 否则默认绑定到 0 号卡。
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

info["medical_mix_train"] = {
    "file_name": "${DATA_DIR}/medical_mix_train.json",
    "formatting": "alpaca"
}
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
# Step 2.5：定位 LLaMA-Factory 专用环境的 Python 绝对路径
#
# 核心设计：pyenv shim 会拦截所有通过命令名 "python" 发起的调用（包括
# conda run 内部），导致 llamafactory311 的 Python 永远无法被正确解析。
# 解决方案：直接使用 conda 环境目录下 Python 二进制的绝对路径，完全绕开
# pyenv 的 PATH shim 机制。
# =============================================================================
echo ""
echo "========== Step 2.5: 定位专用环境 Python =========="
echo "  目标环境名: ${LLAMAFACTORY_ENV}"

# 从当前激活 conda 环境的路径推导 envs 根目录
# CONDA_PREFIX 通常为 /home/<user>/.conda/envs/<current_env>
# 或 /opt/conda/envs/<current_env>
CONDA_ENVS_DIR="$(dirname "${CONDA_PREFIX:-}")"

# 尝试常见的 Python 二进制位置
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

# 验证版本
LF_PY_VERSION="$("${LF_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || true)"
echo "  Python 版本: ${LF_PY_VERSION}"

LF_PY_MAJOR="$(echo "${LF_PY_VERSION}" | cut -d. -f1)"
LF_PY_MINOR="$(echo "${LF_PY_VERSION}" | cut -d. -f2)"
if [[ "${LF_PY_MAJOR:-0}" -lt 3 ]] || [[ "${LF_PY_MAJOR:-0}" -eq 3 && "${LF_PY_MINOR:-0}" -lt 11 ]]; then
    echo "[ERROR] Python 版本过低（${LF_PY_VERSION}），LLaMA-Factory 需要 >= 3.11"
    echo "请重新执行：bash scripts/setup_llamafactory_env.sh"
    exit 1
fi

# 验证关键依赖
if ! "${LF_PYTHON}" -c "import peft,trl,tyro,llamafactory" >/dev/null 2>&1; then
    echo "[ERROR] 缺少训练依赖（peft / trl / tyro / llamafactory）"
    echo "请执行：bash scripts/setup_llamafactory_env.sh"
    exit 1
fi
echo "  ✓ 专用环境检查通过（Python ${LF_PY_VERSION}）"

# region agent log (debug ecc61b)
echo ""
echo "---- [DEBUG ecc61b] CUDA/Torch 诊断 ----"
"${LF_PYTHON}" - <<'PYEOF'
import json, time, os, glob
from pathlib import Path

REMOTE_LOG = Path("/tmp/debug-ecc61b-remote.log")

def log(hyp, msg, data):
    e = {"sessionId":"ecc61b","timestamp":int(time.time()*1000),
         "location":"run_finetune:step2.5","hypothesisId":hyp,
         "message":msg,"data":data,"runId":"run_diag"}
    print(f"  [DIAG/{hyp}] {msg}: {json.dumps(data, ensure_ascii=False)}")
    with open(REMOTE_LOG, "a") as f:
        f.write(json.dumps(e)+"\n")

# H1+H2: torch 版本、CUDA 编译目标、安装时间
try:
    import torch, importlib.util
    spec = importlib.util.find_spec("torch")
    mtime = os.path.getmtime(spec.origin) if spec else None
    log("H1-H2", "torch_info", {
        "version": torch.__version__,
        "cuda_built_for": torch.version.cuda,
        "install_mtime": time.ctime(mtime) if mtime else "unknown"
    })
except Exception as e:
    log("H1-H2", "torch_import_error", {"err": str(e)})

# H1+H3: torchaudio 版本及导入结果
try:
    import torchaudio
    log("H1-H3", "torchaudio_ok", {"version": torchaudio.__version__})
except ImportError as e:
    log("H1-H3", "torchaudio_import_error", {"err": str(e)})

# H4+H5: 服务器上实际存在的 libcudart 文件
patterns = [
    "/usr/local/cuda*/lib64/libcudart.so*",
    "/usr/lib/x86_64-linux-gnu/libcudart.so*",
    "/usr/local/lib/libcudart.so*",
]
found = []
for p in patterns:
    found.extend(glob.glob(p))
log("H4-H5", "libcudart_scan", {"found": found})

# H4: LD_LIBRARY_PATH
log("H4", "ld_library_path", {"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "NOT_SET")})
PYEOF
echo "---- [DEBUG ecc61b END] ----"
echo ""
# endregion agent log

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

# 使用绝对路径直接调用，完全绕开 pyenv shim 和 conda run
"${LF_PYTHON}" src/train.py "${TRAIN_YAML}" \
    2>&1 | tee "${LOG_DIR}/train.log"

TRAIN_EXIT=${PIPESTATUS[0]}
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
echo "  cd ${WORK_DIR} && conda activate kvner && python verify_medical_draft.py"

echo ""
echo "========== 全流程完成 =========="
