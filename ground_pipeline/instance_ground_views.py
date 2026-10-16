"""地面数据可视化：相机视角（RGB+地面掩码标记）与世界 3D 视角（对齐地面法向量后的点云+地面平面标记）。"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/mnt/g/UGit/Phy-SIC")
NPZ = sys.argv[1] if len(sys.argv) > 1 else "/tmp/man_sofa_ground.npz"
IMG = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "images/man_sofa.jpg")
stem = Path(NPZ).stem
OUT_CAM = ROOT / "data" / f"{stem}_camera_view.png"
OUT_WORLD = ROOT / "data" / f"{stem}_world_3d.png"

data = np.load(NPZ)
rgb = np.asarray(Image.open(IMG).convert("RGB"))
depth = data["depth"].astype(np.float64)
valid = data["valid_mask"].astype(bool)
floor = data["floor_mask"].astype(bool)
inlier = data["plane_inlier_mask"].astype(bool)
points = data["points"]
normal = data["plane_normal"]
plane_d = float(data["plane_d"])
plane_points = data["plane_points"]
inlier_pts = data["plane_inlier_points"]


def rotate_to_up(n: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rodrigues 旋转：把平面法向量 n 对齐到 up，返回 3x3 旋转矩阵。"""
    n = n / np.linalg.norm(n)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    v = np.cross(n, up)
    s = np.linalg.norm(v)
    c = float(np.dot(n, up))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


# 相机系 -> 世界系：把法向量对齐到 -z，并用离地高度 z = -(n·p + d) 作为世界 z。
# 否则 n·p 会令物理上更高的物体反而 z 更小（上下颠倒）。
R = rotate_to_up(normal, up=(0.0, 0.0, -1.0))
wpts = points.reshape(-1, 3) @ R.T
win = inlier_pts @ R.T
center = win.mean(axis=0)
wground = plane_points @ R.T
z_world = lambda q: q[..., 2] - plane_d          # 地面 z=0，物体离地高度为正
wpts[..., 2] = z_world(wpts)
win[..., 2] = z_world(win)
wground[..., 2] = z_world(wground)
center[2] = z_world(center)
cam_world = np.array([0.0, 0.0, -plane_d], dtype=float)  # 相机在世界系的位置（地面 z=0）

# ---------- 图 1：相机视角 ----------
scale = 0.5
w, h = int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)
rgb_s = np.asarray(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))
inlier_s = np.asarray(Image.fromarray((inlier * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0

fig1, ax = plt.subplots(figsize=(11, 7.4))
ax.imshow(rgb_s)
overlay = np.zeros((*inlier_s.shape, 4), dtype=np.uint8)
overlay[inlier_s] = (60, 255, 90, 150)
ax.imshow(overlay, alpha=0.55)
ax.contour(inlier_s, levels=[0.5], colors="#0aa03c", linewidths=1.6)
ax.text(0.985, 0.025, "GROUND (camera view)", transform=ax.transAxes, ha="right", va="bottom",
        color="#0aa03c", fontsize=15, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0aa03c", alpha=0.9))
ax.axis("off")
fig1.tight_layout()
fig1.savefig(OUT_CAM, dpi=120)
plt.close(fig1)

# ---------- 图 2：世界 3D 视角 ----------
rng = np.random.default_rng(1)
valid_idx = np.flatnonzero(valid.ravel())
n_scene = min(150_000, len(valid_idx))
scene = wpts[rng.choice(valid_idx, size=n_scene, replace=False)]
n_in = min(80_000, len(win))
wsub = win[rng.choice(len(win), size=n_in, replace=False)]

zmin, zmax = np.percentile(wpts[valid_idx, 2], [2, 98])

fig2 = plt.figure(figsize=(10.5, 9))
ax3d = fig2.add_subplot(111, projection="3d")
sc = ax3d.scatter(scene[:, 0], scene[:, 1], scene[:, 2], c=scene[:, 2], cmap="viridis",
                  s=0.4, alpha=0.35, vmin=zmin, vmax=zmax, label="scene points")
gw = ax3d.scatter(wsub[:, 0], wsub[:, 1], wsub[:, 2], c="#2ef05c", s=0.6, alpha=0.55, label="ground inliers")
gg = wground.reshape(100, 100, 3)
ax3d.plot_surface(gg[:, :, 0], gg[:, :, 1], gg[:, :, 2], color="red", alpha=0.45, label="fitted ground plane")
ax3d.quiver(center[0], center[1], center[2], 0, 0, 1, length=1.5, color="k", arrow_length_ratio=0.15)
ax3d.scatter([cam_world[0]], [cam_world[1]], [cam_world[2]], c="red", s=60, marker="*",
             label="camera (h=%.2f m)" % cam_world[2])
ax3d.set_xlabel("world x (m)")
ax3d.set_ylabel("world y (m)")
ax3d.set_zlabel("height above ground (m)")
ax3d.set_title("GROUND in world 3D | ground at z=0, normal->up", fontsize=13)
ax3d.set_box_aspect((1, 1, 1))
ax3d.view_init(elev=25, azim=-60)
fig2.tight_layout()
fig2.savefig(OUT_WORLD, dpi=120)
plt.close(fig2)

print("SAVED", OUT_CAM, OUT_CAM.stat().st_size)
print("SAVED", OUT_WORLD, OUT_WORLD.stat().st_size)
print("ground world z median", float(np.median(win[:, 2])))
print("camera world height", float(cam_world[2]))
