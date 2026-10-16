"""可视化 man_sofa 实例测试结果：分割掩码叠加、深度图、3D 地面点云+拟合平面、内点残差。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/mnt/g/UGit/Phy-SIC")
NPZ = "/tmp/man_sofa_ground.npz"
OUT = ROOT / "data" / "man_sofa_ground_viz.png"

data = np.load(NPZ)
rgb = np.asarray(Image.open(ROOT / "images/man_sofa.jpg").convert("RGB"))
depth = data["depth"].astype(np.float64)
floor = data["floor_mask"].astype(bool)
inlier = data["plane_inlier_mask"].astype(bool)
points = data["points"]
normal = data["plane_normal"]
plane_d = float(data["plane_d"])
plane_points = data["plane_points"]
inlier_pts = data["plane_inlier_points"]

# ---- 降采样到可展示尺寸 ----
scale = 0.2
w, h = int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)
rgb_s = np.asarray(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))
dmin, dmax = np.nanpercentile(depth, 1), np.nanpercentile(depth, 99)
depth_n = np.clip((depth - dmin) / (dmax - dmin), 0, 1)
depth_s = np.asarray(
    Image.fromarray((depth_n * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
).astype(np.float32) / 255 * (dmax - dmin) + dmin
floor_s = np.asarray(Image.fromarray((floor * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
inlier_s = np.asarray(Image.fromarray((inlier * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0

# ---- 图 1：RGB + 平面内点叠加 ----
overlay = rgb_s.copy()
green = np.array([60, 220, 60])
overlay[inlier_s] = (overlay[inlier_s] * 0.45 + green * 0.55).astype(np.uint8)

# ---- 图 3：3D 内点点云 + 拟合平面 ----
rng = np.random.default_rng(0)
sub = inlier_pts[rng.choice(len(inlier_pts), size=min(40000, len(inlier_pts)), replace=False)]
center = sub.mean(axis=0)
grid = plane_points.reshape(100, 100, 3)
axis_max = max(np.ptp(sub[:, 0]), np.ptp(sub[:, 1]), np.ptp(sub[:, 2]))

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(overlay)
ax1.set_title("RGB + ground-plane inliers (green)")
ax1.axis("off")

ax2 = fig.add_subplot(2, 2, 2)
im = ax2.imshow(depth_s, cmap="turbo", vmin=dmin, vmax=dmax)
ax2.set_title("DepthPro metric depth")
ax2.axis("off")
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cb.set_label("depth (m)")

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
sc = ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], s=1, cmap="viridis", alpha=0.5)
ax3.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], color="tab:red", alpha=0.35, label="fitted plane")
ax3.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=axis_max * 0.5, color="k")
ax3.set_title("Floor inlier points + fitted plane")
ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("z (m)")
ax3.set_box_aspect((1, 1, 1))

ax4 = fig.add_subplot(2, 2, 4)
resid = np.abs(inlier_pts @ normal + plane_d)
ax4.hist(resid, bins=80, color="tab:blue")
ax4.axvline(np.median(resid), color="k", ls="--", label=f"median {np.median(resid):.4f} m")
ax4.set_title("Plane-fit residuals (|n·p + d|)")
ax4.set_xlabel("distance to plane (m)")
ax4.legend()

fig.suptitle(
    f"ground_pipeline instance test - man_sofa.jpg | normal={normal.tolist()} d={plane_d:.4f} | "
    f"inliers={inlier.sum():,}/{floor.sum():,} floor px",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=140)
print("SAVED", OUT, "size", OUT.stat().st_size)
