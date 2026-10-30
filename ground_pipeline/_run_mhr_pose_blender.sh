#!/usr/bin/env bash
# 用无头 Blender 把每张图的目标姿势套到 MHR-lod5.tpose.fbx 上并导出（FBX 7.4）。
#   bash ground_pipeline/_run_mhr_pose_blender.sh [目标json目录] [输出目录]
set -e
cd /mnt/g/UGit/Phy-SIC
BLENDER=${BLENDER:-/home/art/blender-4.2.9-linux-x64/blender}
SRC=${1:-images/_wr_mhrpose/fbx}
DST=${2:-images/_wr_mhrpose/fbx_blender}
mkdir -p "$DST"
for j in "$SRC"/*_pose_targets.json; do
  stem=$(basename "$j" _pose_targets.json)
  echo "== $stem"
  "$BLENDER" -b --factory-startup --python ground_pipeline/mhr_pose_blender.py -- \
    --targets "$j" --out "$DST/${stem}_mhr_pose_blender.fbx" 2>&1 \
    | grep -E '^\[(pose|verify|out)\]' || true
done
