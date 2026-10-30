#!/usr/bin/env bash
set -e
cd /mnt/g/UGit/Phy-SIC/ground_pipeline
PY=/home/art/envs/blade310/bin/python
$PY -m py_compile mhr_mesh_fbx.py bone_depth_snap.py render_world_align.py
echo "compile ok"
$PY - <<'EOF'
import numpy as np, tempfile, os
import mhr_mesh_fbx as m

rig = m.get_rig()
print("bones:", len(rig["bone_order"]), "root:", rig["root"])
print("rest:", rig["rest_path"])
first = rig["bone_order"][:5]
print("first bones:", first)

# 造假人形：127 关节 + 小网格，检查结构
rng = np.random.default_rng(0)
joints = rng.uniform(0, 1.8, size=(127, 3))
verts = rng.uniform(0, 1.8, size=(200, 3))
faces = np.array([[i, i + 1, i + 2] for i in range(0, 198)], dtype=np.int64)
out = os.path.join(tempfile.gettempdir(), "rig_probe.fbx")
info = m.write_mesh_rig(out, verts, faces, joints, object_name="probe", rig=rig)
print(info)
txt = open(out, encoding="utf-8").read()
print("Model:", txt.count("\n\tModel: "), "NodeAttribute:", txt.count("\n\tNodeAttribute: "))
print("C:", txt.count('\n\tC: "OO"'))
EOF
