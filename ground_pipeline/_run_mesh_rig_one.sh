#!/usr/bin/env bash
set -e
cd /mnt/g/UGit/Phy-SIC
export POSEEXTRACTOR_SAM3D_BONE_SNAP=1
/home/art/envs/blade310/bin/python ground_pipeline/render_world_align.py \
  images/test_image2/90dc5050afe423ba9696bf3e365e0333.jpg \
  --outdir images/_wr_mhr_rig_fbx \
  --mesh-rig-fbx images/_wr_mhr_rig_fbx
