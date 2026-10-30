#!/bin/bash

# ===================== 仅需修改这几行（WSL内配置） =====================
PORT=8001                          # 端口号（和BAT脚本保持一致）
VENV=/mnt/g/UGit/qwen3-vl/.venv-ground   # 独立虚拟环境（共享 torch/CUDA，勿删原 .venv）
USE_MOGE2=1                        # 1：MoGe-2 替换 DepthPro；0：使用 DepthPro
MOGE2_FP16=1                       # 1：MoGe-2 使用 FP16；0：FP32（DepthPro 不受影响）
# GroundingDINO 权重在 HuggingFace 缓存中；离线启动避免联网超时
export HF_HOME="${HF_HOME:-/home/art/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# 启用 P0 姿势推理时，设置为 ImagePose_Resources 根目录（其中包含 models/ 与 vendor/）。
export SAM3D_RESOURCES="${SAM3D_RESOURCES:-/mnt/g/UGit/QingGeTools/ImagePose_Resources}"
# =====================================================================

pause_on_exit() {
    local code=$?
    echo ""
    if [ "$code" -ne 0 ]; then
        echo "[FAIL] exit code: $code"
    else
        echo "[DONE] service stopped"
    fi
    read -r -p "Press Enter to close..."
    exit "$code"
}
trap pause_on_exit EXIT

set -e

cd /mnt/g/UGit/Phy-SIC

echo "===== activate venv ====="
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[FAIL] $VENV not found. 请先按 WSL_SETUP.md 创建 .venv-ground。"
    exit 1
fi
source "$VENV/bin/activate"

DEPTH_ARGS=(--depth-backend depthpro)
if [ "$USE_MOGE2" = "1" ]; then
    DEPTH_ARGS=(--depth-backend moge2)
    if [ "$MOGE2_FP16" = "1" ]; then
        DEPTH_ARGS+=(--depth-fp16)
    fi
fi

echo "===== HF_HOME=$HF_HOME, HF_HUB_OFFLINE=$HF_HUB_OFFLINE ====="
echo "===== depth: ${DEPTH_ARGS[*]} ====="
echo "===== start ground_service.py (port=$PORT) ====="
python ./ground_pipeline/ground_service.py --port "$PORT" --host 0.0.0.0 "${DEPTH_ARGS[@]}"
