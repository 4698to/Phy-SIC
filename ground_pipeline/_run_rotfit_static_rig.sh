#!/usr/bin/env bash
set -e
cd /mnt/g/UGit/Phy-SIC
export POSEEXTRACTOR_SAM3D_BONE_SNAP=1
export POSEEXTRACTOR_SAM3D_BONE_SNAP_MODE=legacy
export POSEEXTRACTOR_SAM3D_SCALE_FIT=torso
export POSEEXTRACTOR_SAM3D_BONE_ROTATION_FIT=1
OUT=images/_wr_all_rotfit
/home/art/envs/blade310/bin/python -m py_compile ground_pipeline/mhr_mesh_fbx.py ground_pipeline/render_world_align.py
echo "compile ok"
/home/art/envs/blade310/bin/python ground_pipeline/render_world_align.py \
  images/test_image/5306c1d3gy1hsktgsqco6j20sg11wq4s.jpg \
  images/test_image/5306c1d3gy1hsktgstrvxj20sg11wgoo.jpg \
  images/test_image/5306c1d3gy1hsktgsw6d1j20sg11wgon.jpg \
  images/test_image/5306c1d3gy1hsktgsz459j20sg0ugmyv.jpg \
  images/test_image2/006dplNhly1h2kv65f0ixj30u019079d.jpg \
  images/test_image2/90dc5050afe423ba9696bf3e365e0333.jpg \
  --outdir "$OUT" \
  --mesh-rig-fbx "$OUT/fbx" 2>&1 | grep -Ev '^\[[0-9a-f]{16}\] \{' || true
