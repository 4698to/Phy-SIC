"""MoGe-2 + SAM-3D 场景验收面板（顶排 5 图 + 底排 3 图）。

版面：
  顶排 5 个：
    1. 相机视角：原图 + COCO17 骨架 + 地面网格（MoGe-2 深度遮挡）
    2. 原图 + 半透明 SAM-3D 网格叠加（camera_v1 相机系，未经地面变换）
    3. 正交 x-z（正视）
    4. 正交 y-z（侧视）
    5. 正交 x-y（俯视）
  底排 3 个：
    6. 世界视角：不透明 SAM-3D 实体网格 + 红色网格地面（无骨骼）
    7. 世界视角点云：MoGe-2 场景点云（按前后深度着色）+ 相机视锥 + 红色网格地面
    8. 深度图：MoGe-2 depth（viridis，米制色标）

正交/3D 中的 SAM-3D 网格统一用带光照着色的实体三角面片（PolyCollection /
Poly3DCollection），不再用点云点线渲染。地面平面使用红色 plot_surface +
plot_wireframe（与 ground_service 的可视化风格一致）。

依赖运行中的服务（默认 http://127.0.0.1:8001）：
  POST /v1/ground      {image, return_data: true, visualize: false}
  POST /v1/pose/single {image, want_overlay: true, want_world: true,
                        want_mesh: true, align_gravity: true}

用法（WSL，需 matplotlib/cv2/PIL 环境）：
  python ground_pipeline/render_scene_panel.py [IMG ...] \
      [--base http://127.0.0.1:8001] [--outdir <dir>] [--dpi 150] [--no-cache]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import urllib.request

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

DEFAULT_BASE = "http://127.0.0.1:8001"
DEFAULT_ROOT = "/mnt/g/UGit/Phy-SIC/images/test_image"
CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "phy_sic_moge2_scene_cache"

C_MESH = "#4fc3e8"
C_MESH_RGB = np.array([0.31, 0.765, 0.91])
C_JOINT = "#ffd23f"
C_GROUND_RED = "#e5453c"
C_ANKLE = "#c9a227"

M_CAM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
ANKLES = (15, 16)

# COCO17 骨组（覆盖全部 16 条边）
BONE_GROUPS = (
    (((0, 1), (0, 2), (1, 3), (2, 4)), "#f2a63b"),          # 头部
    (((5, 7), (7, 9)), "#f2a63b"),                          # 左臂
    (((5, 6), (5, 11), (6, 12), (11, 12)), "#7b5cd6"),      # 躯干
    (((6, 8), (8, 10)), "#e98a2e"),                         # 右臂
    (((11, 13), (13, 15)), "#2f7de1"),                      # 左腿
    (((12, 14), (14, 16)), "#3aa76d"),                      # 右腿
)

LIGHT = np.array([0.35, -0.55, -0.75])

# 3D 面板透视强度：focal_length 越小视角越广、透视越明显（<1 为广角）
PERSP_FOCAL = 0.5


# ------------------------------------------------------------------ 基础设施
def _load_font() -> None:
    for path in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"):
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            return


def post(base: str, path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_health(base: str) -> None:
    with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=20) as resp:
        print(f"[service] health {resp.status} @ {base}")


def _b64_payload(s: str) -> str:
    return s.split(",", 1)[1] if s.startswith("data:") and "," in s else s


def mask_from_b64(s: str) -> np.ndarray:
    arr = Image.open(io.BytesIO(base64.b64decode(_b64_payload(s)))).convert("L")
    return np.asarray(arr) > 127


def img_from_b64(s: str) -> np.ndarray:
    arr = Image.open(io.BytesIO(base64.b64decode(_b64_payload(s)))).convert("RGB")
    return np.asarray(arr)


def to_local_path(p: str) -> str:
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", p):
        return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}"
    return p


def pose_cache_suffix() -> str:
    """缓存名跟随 SAM-3D 拟合开关，旋转优先且不复用旧尺度缓存。"""
    rotation = os.environ.get("POSEEXTRACTOR_SAM3D_ROTATION_FIT", "").strip().lower()
    snap = os.environ.get("POSEEXTRACTOR_SAM3D_BONE_SNAP", "").strip().lower()
    legacy = os.environ.get("POSEEXTRACTOR_SAM3D_BONE_SNAP_MODE", "native").strip().lower() == "legacy"
    true_values = ("1", "on", "true", "yes", "mhr")
    if rotation in true_values or (snap in true_values and not legacy):
        return "_rotationfit_native_v2"
    legacy_suffix = "_bonesnap_legacy" if snap in true_values and legacy else ""
    value = os.environ.get("POSEEXTRACTOR_SAM3D_SCALE_FIT", "").strip().lower()
    if value in ("limb", "per_limb", "limbs", "local"):
        return "_scalefit_limb" + legacy_suffix
    if value in ("torso", "body", "center"):
        return "_scalefit_torso" + legacy_suffix
    return ("_scalefit" if value in ("moge", "depth", "on", "1", "true") else "_native_v2") + legacy_suffix


def fetch_case(base: str, image_path: str, use_cache: bool) -> tuple[dict, dict, str]:
    raw = pathlib.Path(image_path).read_bytes()
    data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    tag = hashlib.sha1(raw).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    gpath = CACHE_DIR / f"{tag}_ground.json"
    ppath = CACHE_DIR / f"{tag}_pose_vitpose{pose_cache_suffix()}.json"
    if use_cache and gpath.exists() and ppath.exists():
        print(f"  cache hit {tag}")
        return (json.loads(gpath.read_text(encoding="utf-8")),
                json.loads(ppath.read_text(encoding="utf-8")), tag)

    g = post(base, "/v1/ground",
             {"image": data_uri, "return_data": True, "visualize": False}, timeout=1800)
    pose_req = {
        "image": data_uri, "want_overlay": True, "want_raw": False,
        "want_world": True, "want_mesh": True, "align_gravity": True,
    }
    # 把 MoGe 实测内参交给 SAM-3D；否则它会退回默认 FOV 先验，网格整体偏近/偏远。
    cam_int = (g.get("camera") or {}).get("intrinsics")
    if cam_int is not None:
        pose_req["cam_int"] = cam_int
    p = post(base, "/v1/pose/single", pose_req, timeout=1800)
    if pose_cache_suffix().startswith("_rotationfit") and p.get("rotation_fit") is None:
        raise RuntimeError("Native bone snap requested, but service did not return rotation_fit; restart/configure the service before caching")
    if use_cache:
        gpath.write_text(json.dumps(g), encoding="utf-8")
        ppath.write_text(json.dumps(p), encoding="utf-8")
    return g, p, tag


# ------------------------------------------------------------------ 几何/图元
def to_display(points: np.ndarray, world: bool) -> np.ndarray:
    """统一绘图坐标 (right, depth, up)。世界系 (x,y,z) 恒等；相机系 (x,z,-y)。"""
    if world:
        return np.ascontiguousarray(points)
    return points @ M_CAM.T


def shade_faces(verts: np.ndarray, faces: np.ndarray, base_rgb: np.ndarray,
                alpha: float, light: np.ndarray = LIGHT) -> np.ndarray:
    """按面法向做半兰伯特着色，返回逐面 RGBA（base_rgb 支持单色或逐面配色）。"""
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    l = np.asarray(light, dtype=float)
    l = l / np.linalg.norm(l)
    s = 0.55 + 0.45 * np.abs(nrm @ l)
    base = np.asarray(base_rgb, dtype=float)
    if base.ndim == 1:
        base = base.reshape(1, 3)
    rgba = np.empty((len(faces), 4), dtype=float)
    rgba[:, :3] = np.clip(base * s[:, None], 0.0, 1.0)
    rgba[:, 3] = alpha
    return rgba


def mesh_overlay_image(rgb: np.ndarray, verts_cam: np.ndarray, faces: np.ndarray,
                       focal: float, out_size: tuple[int, int],
                       shift_m: np.ndarray | None = None,
                       shift_lo: float = 0.01, shift_hi: float = 0.03
                       ) -> tuple[np.ndarray, int]:
    """深度排序三角面片，半透明青色覆盖到原图上（远->近 fillPoly）。

    ``shift_m`` 给定时按逐顶点位移量把面片从青色插值到红色（>=shift_hi 全红），
    用于标出 limb 档实际被改动的区域；默认 None 时输出与原实现完全一致。
    """
    h, w = rgb.shape[0], rgb.shape[1]
    z = np.maximum(verts_cam[:, 2], 1e-6)
    uv = np.stack([focal * verts_cam[:, 0] / z + w / 2.0,
                   focal * verts_cam[:, 1] / z + h / 2.0], axis=1)

    keep = (verts_cam[faces, 2] > 1e-3).all(axis=1)
    face_idx = np.nonzero(keep)[0]
    tris = faces[face_idx]
    if not len(tris):
        return rgb, 0
    order = np.argsort(-z[tris].mean(axis=1))

    tri3 = verts_cam[tris]
    nrm = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    light = LIGHT / np.linalg.norm(LIGHT)
    shade = 0.40 + 0.60 * np.abs(nrm @ light)

    import cv2

    # 逐面位移（三个顶点均值）→ 青色到红色的混合权重
    if shift_m is None:
        face_t = None
    else:
        sh = np.asarray(shift_m, dtype=np.float64).reshape(-1)
        face_t = np.clip((sh[faces].mean(axis=1) - shift_lo)
                         / max(shift_hi - shift_lo, 1e-9), 0.0, 1.0)

    layer = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA
    for idx in order:
        pts = np.rint(uv[tris[idx]]).astype(np.int32)
        s = float(shade[idx])
        if face_t is None:
            bgr = (int(232 * s), int(195 * s), int(79 * s))
        else:
            t = float(face_t[face_idx[idx]])
            bgr = (int((232 + (48 - 232) * t) * s),
                   int((195 + (61 - 195) * t) * s),
                   int((79 + (219 - 79) * t) * s))
        cv2.fillPoly(layer, [pts], bgr + (96,), lineType=cv2.LINE_8)
    for idx in order[::6]:
        pts = np.rint(uv[tris[idx]]).astype(np.int32)
        cv2.polylines(layer, [pts], True, (150, 110, 30, 60), 1, cv2.LINE_8)

    base = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGBA")
    over = Image.fromarray(layer[:, :, [2, 1, 0, 3]], mode="RGBA")
    return np.asarray(Image.alpha_composite(base, over).convert("RGB")), int(len(tris))


def draw_camera(ax, cam_disp: np.ndarray, dir_matrix: np.ndarray, w: int, h: int,
                focal: float, length: float = 0.55) -> None:
    def disp_dir(u, v):
        d = np.array([(u - w / 2.0) / focal, (v - h / 2.0) / focal, 1.0])
        d /= np.linalg.norm(d)
        return dir_matrix @ d

    corners = [cam_disp + length * disp_dir(u, v)
               for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    for c in corners:
        ax.plot([cam_disp[0], c[0]], [cam_disp[1], c[1]], [cam_disp[2], c[2]],
                color="#222222", lw=0.8, zorder=7)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="#222222", lw=0.9, zorder=7)
    ax.scatter([cam_disp[0]], [cam_disp[1]], [cam_disp[2]], s=44, c="#111111",
               marker="^", depthshade=False, zorder=8)


def project_ground_grid(ax_img, rot, tvec, focal, w, h, depth, half=2.5, stride=0.5,
                        silhouette=None, center_xy=(0.0, 0.0)):
    """世界系 z=0 网格投到图像上（相机系深度 + 人物轮廓做遮挡剔除）。"""
    cx, cy = center_xy
    xs = np.arange(cx - half, cx + half + 1e-6, stride)
    ys = np.arange(cy - half, cy + half + 1e-6, stride)
    lines = [np.stack([np.full_like(ys, x), ys, np.zeros_like(ys)], axis=1) for x in xs]
    lines += [np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1) for y in ys]
    for pts in lines:
        cam = (pts - tvec) @ rot
        zc = cam[:, 2]
        ok = zc > 0.2
        uu = focal * cam[:, 0] / np.maximum(zc, 1e-6) + w / 2.0
        vv = focal * cam[:, 1] / np.maximum(zc, 1e-6) + h / 2.0
        ui = np.rint(np.where(ok, uu, -1)).astype(int)
        vi = np.rint(np.where(ok, vv, -1)).astype(int)
        ins = ok & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        dd = np.full(uu.shape, np.nan)
        dd[ins] = depth[vi[ins], ui[ins]]
        keep = ins & np.isfinite(dd) & (np.abs(dd - zc) < 0.20)
        if silhouette is not None:
            sv = np.rint(vv).astype(int)
            su = np.rint(uu).astype(int)
            inside = ok & (su >= 0) & (su < w) & (sv >= 0) & (sv < h)
            hit = np.zeros(uu.shape, dtype=bool)
            hit[inside] = silhouette[sv[inside], su[inside]]
            keep &= ~hit
        ax_img.plot(np.where(keep, uu, np.nan), np.where(keep, vv, np.nan),
                    color="#e8873a", lw=0.9, alpha=0.9)


def draw_bones(ax, pts: np.ndarray, lw: float = 1.5, three_d: bool = False,
               color: str | None = None) -> None:
    zero = np.all(np.abs(pts) < 1e-9, axis=1)
    for edges, group_color in BONE_GROUPS:
        for a, b in edges:
            if zero[a] or zero[b]:
                continue
            if three_d:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                        [pts[a, 2], pts[b, 2]], color=color or group_color, lw=lw,
                        solid_capstyle="round", zorder=7)
            else:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                        color=color or group_color, lw=lw, solid_capstyle="round",
                        zorder=6)


def draw_joints(ax, pts: np.ndarray, size: float = 12.0, three_d: bool = False) -> None:
    zero = np.all(np.abs(pts) < 1e-9, axis=1)
    pts = pts[~zero]
    if not len(pts):
        return
    if three_d:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=C_JOINT,
                   edgecolors="#8a6d00", linewidths=0.4, depthshade=False, zorder=8)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=size * 0.8, c=C_JOINT,
                   edgecolors="#8a6d00", linewidths=0.4, zorder=7)


def red_ground_plane(ax, xlim, ylim, z: float, stride: float = 0.5) -> None:
    """红色低透明面 + 红色网格线框（与 ground_service 的拟合平面风格一致）。"""
    nx = max(2, min(60, int(round((xlim[1] - xlim[0]) / stride)) + 1))
    ny = max(2, min(60, int(round((ylim[1] - ylim[0]) / stride)) + 1))
    gx = np.linspace(xlim[0], xlim[1], nx)
    gy = np.linspace(ylim[0], ylim[1], ny)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = np.full_like(XX, z)
    ax.plot_surface(XX, YY, ZZ, color="red", alpha=0.16, linewidth=0,
                    shade=False, antialiased=True, zorder=1)
    ax.plot_wireframe(XX, YY, ZZ, color="red", alpha=0.85, lw=0.5, zorder=2)


def style_3d_axes(ax) -> None:
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#d5dbe3")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update({"color": "#cfd6df", "linewidth": 0.5,
                                     "linestyle": "-"})


# ------------------------------------------------------------------ 面板
def ortho_panel(ax, disp, faces, joints, title, xlabel, ylabel, ix, iy, window,
                z_plane=None) -> None:
    ax.set_facecolor("white")
    tri = disp[faces]
    if len(tri):
        rgba = shade_faces(disp, faces, C_MESH_RGB, alpha=0.62)
        depth_axis = 3 - ix - iy
        order = np.argsort(-tri[:, :, depth_axis].mean(axis=1))
        coll = PolyCollection(tri[order][:, :, [ix, iy]], facecolors=rgba[order],
                              edgecolors=(0.06, 0.35, 0.45, 0.22), linewidths=0.10,
                              rasterized=True, zorder=3)
        ax.add_collection(coll)
    if z_plane is not None and iy == 2:
        ax.axhline(z_plane, color=C_GROUND_RED, lw=1.1, linestyle=(0, (4, 3)), zorder=2)
    if len(joints) >= 17:
        draw_bones(ax, joints[:, [ix, iy]])
        draw_joints(ax, joints[:, [ix, iy]])
        if z_plane is not None and iy == 2:
            for a in ANKLES:
                if np.all(np.abs(joints[a]) < 1e-9):
                    continue
                ax.plot([joints[a, ix], joints[a, ix]], [joints[a, iy], z_plane],
                        color=C_ANKLE, lw=0.9, linestyle=(0, (2, 2)), zorder=4)
                ax.scatter([joints[a, ix]], [z_plane], s=8, c=C_ANKLE, zorder=4)
    ax.set_xlim(*window[0])
    ax.set_ylim(*window[1])
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(which="major", color="#c9d0da", lw=0.7, alpha=0.9, zorder=0)
    ax.grid(which="minor", color="#e6eaf0", lw=0.4, alpha=0.9, zorder=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _style_3d_view(ax, win3) -> None:
    ax.set_xlim(*win3[0])
    ax.set_ylim(*win3[1])
    ax.set_zlim(*win3[2])
    ax.set_xlabel("right (m)", fontsize=7.5)
    ax.set_ylabel("depth (m)", fontsize=7.5)
    ax.set_zlabel("up (m)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.set_box_aspect((win3[0][1] - win3[0][0],
                       win3[1][1] - win3[1][0],
                       win3[2][1] - win3[2][0]))


def scene3d_panel(ax, disp, faces, cam_disp, dir_matrix, w, h, focal,
                  z_plane, win3, title) -> None:
    """世界视角：不透明 SAM-3D 实体网格 + 红色网格地面 + 相机视锥（无骨骼）。"""
    ax.computed_zorder = False
    ax.set_proj_type("persp", focal_length=PERSP_FOCAL)
    style_3d_axes(ax)

    red_ground_plane(ax, (win3[0][0], win3[0][1]), (win3[1][0], win3[1][1]), z_plane)
    if len(faces):
        rgba = shade_faces(disp, faces, C_MESH_RGB, alpha=1.0)
        coll = Poly3DCollection(disp[faces], facecolors=rgba,
                                edgecolors=(0.05, 0.26, 0.34, 0.35), linewidths=0.10,
                                zsort="average")
        coll.set_zorder(5)
        ax.add_collection3d(coll)
    draw_camera(ax, cam_disp, dir_matrix, w, h, focal)

    _style_3d_view(ax, win3)
    ax.view_init(elev=18, azim=-64)
    ax.set_title(title, fontsize=9, pad=2)


def pointcloud3d_panel(ax, pts, cam_disp, dir_matrix, w, h, focal,
                       z_plane, win3, title) -> None:
    """世界视角点云：MoGe-2 反投影点云（按前后深度着色）+ 红色网格地面 + 相机视锥。"""
    ax.computed_zorder = False
    ax.set_proj_type("persp", focal_length=PERSP_FOCAL)
    style_3d_axes(ax)

    red_ground_plane(ax, (win3[0][0], win3[0][1]), (win3[1][0], win3[1][1]), z_plane)
    if len(pts):
        dd = pts[:, 1]  # 绘图坐标第 1 轴 = 前后深度
        vmin, vmax = np.percentile(dd, 2.0), np.percentile(dd, 98.0)
        if not np.isfinite(vmax) or vmax - vmin < 1e-3:
            vmin, vmax = float(np.min(dd)), float(np.min(dd)) + 1.0
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dd, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=1.1, alpha=0.95,
                        depthshade=False, zorder=4)
        cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
        cb.set_label("depth (m)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    draw_camera(ax, cam_disp, dir_matrix, w, h, focal)

    _style_3d_view(ax, win3)
    ax.view_init(elev=18, azim=-64)
    ax.set_title(title, fontsize=9, pad=2)


def depth_panel(fig, ax, depth, title) -> None:
    """MoGe-2 深度图（viridis + 米制色标）。"""
    d = np.asarray(depth, dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)
    if ok.any():
        vmin, vmax = (float(v) for v in np.percentile(d[ok], (2.0, 98.0)))
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(d, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02, shrink=0.78)
    cb.set_label("depth (m)", fontsize=8)
    cb.ax.tick_params(labelsize=6.5)


# ------------------------------------------------------------------ 主渲染
def render(g: dict, p: dict, image_path: str, stem: str, out_path: str,
           dpi: int = 150) -> dict:
    world = p.get("world") or {}
    grounded = (world.get("status") == "ok"
                and bool((world.get("mesh") or {}).get("vertices")))
    mesh_block = (world.get("mesh") or {}) if grounded else (p.get("camera_space") or {})

    verts = np.asarray(mesh_block.get("vertices") or [], dtype=np.float64)
    faces = np.asarray(mesh_block.get("faces") or [], dtype=np.int64)
    # keypoints 与 mesh 不同层：grounded 在 world 顶层，回退时在 camera_space
    kp_src = world if grounded else (p.get("camera_space") or {})
    kp = np.asarray(kp_src.get("keypoints_3d_coco17") or [], dtype=np.float64)

    depth = np.load(io.BytesIO(base64.b64decode(_b64_payload(g["depth_b64"])))).astype(np.float32)
    dh, dw = depth.shape
    focal = float(g["focal_px"])
    cam_meta = p.get("camera") or {}
    f_sam = float(cam_meta.get("focal_px") or focal)

    if grounded:
        rot = np.asarray(world["rotation"], dtype=np.float64)
        tvec = np.asarray(world["camera_position"], dtype=np.float64)
        cam_h = float(world.get("camera_height_m") or 0.0)
        disp = to_display(verts, True)
        kdisp = to_display(kp, True)
        verts_cam = (verts - tvec) @ rot
        cam_disp = tvec
        dir_matrix = rot
        z_plane = 0.0
        wins = _ortho_windows(disp, kdisp, grounded=True, z_plane=z_plane)
        scene_disp = _scene_points(depth, focal, rot, tvec,
                                   grounded=True, cam_world=tvec)
        frame_name = "frame=ground_world_v1"
        ground_tag = f"ground=ok ({world.get('source') or 'inferred'})"
    else:
        rot = np.eye(3)
        tvec = np.zeros(3)
        cam_h = 0.0
        disp = to_display(verts, False)
        kdisp = to_display(kp, False)
        verts_cam = verts
        cam_disp = np.zeros(3)
        dir_matrix = M_CAM
        z_plane = float(np.percentile(disp[:, 2], 0.5)) if len(disp) else 0.0
        wins = _ortho_windows(disp, kdisp, grounded=False, z_plane=z_plane)
        scene_disp = _scene_points(depth, focal, None, None,
                                   grounded=False, cam_world=None)
        frame_name = "frame=camera_v1"
        ground_tag = "ground=rejected"

    base_rgb = np.asarray(Image.open(to_local_path(image_path)).convert("RGB"))
    ih, iw = base_rgb.shape[:2]
    f_img = f_sam * iw / float(cam_meta.get("image_width") or iw)
    # 叠加图只用未经地面变换的 camera_v1 几何：world 系几何含落地/重力对齐平移，
    # 反解回相机系会整体平移 40~55 px，与原图错位。
    cam_block = p.get("camera_space") or {}
    ov_verts = np.asarray(cam_block.get("vertices") or [], dtype=np.float64)
    ov_faces = np.asarray(cam_block.get("faces") or [], dtype=np.int64)
    if not len(ov_verts):  # 旧响应无 camera_space 时退回世界系反解
        ov_verts, ov_faces = verts_cam, faces
    mesh_img, ntri = mesh_overlay_image(base_rgb, ov_verts, ov_faces, f_img, (iw, ih))

    # ---------------- 版面 ----------------
    fig = plt.figure(figsize=(21.0, 11.4), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          left=0.028, right=0.985, top=0.905, bottom=0.068,
                          wspace=0.05, hspace=0.13)
    top = gs[0, :].subgridspec(1, 5, width_ratios=[1.14, 1.14, 1.0, 1.0, 1.0],
                               wspace=0.20)
    ax_img, ax_msh, ax_f, ax_s, ax_t = [fig.add_subplot(top[i]) for i in range(5)]
    bottom = gs[1, :].subgridspec(1, 3, width_ratios=[1.06, 1.06, 0.80], wspace=0.06)
    ax_ov = fig.add_subplot(bottom[0], projection="3d")
    ax_pc = fig.add_subplot(bottom[1], projection="3d")
    ax_dp = fig.add_subplot(bottom[2])

    # 1) 相机视角
    ax_img.imshow(img_from_b64(p["overlay_jpeg_b64"]))
    if grounded:
        cx, cy = float(np.median(disp[:, 0])), float(np.median(disp[:, 1]))
        project_ground_grid(ax_img, rot, tvec, focal, iw, ih, depth,
                            center_xy=(cx, cy), silhouette=None)
    ax_img.set_xlim(0, iw)
    ax_img.set_ylim(ih, 0)
    ax_img.axis("off")
    ax_img.set_title("camera view: skeleton + ground grid" if grounded
                     else "camera view: skeleton (ground rejected)", fontsize=9)

    # 2) 原图 + 半透明 SAM-3D 网格
    ax_msh.imshow(mesh_img)
    ax_msh.set_xlim(0, iw)
    ax_msh.set_ylim(ih, 0)
    ax_msh.axis("off")
    ax_msh.set_title(f"original image + semi-transparent SAM-3D mesh\n"
                     f"{ntri} faces, focal={f_img:.0f} px", fontsize=9)

    # 3-5) 正交三视图（实体网格）
    if grounded:
        labels = (("world x-z (front)", "x right (m)", "z up (m)"),
                  ("world y-z (side)", "y depth (m)", "z up (m)"),
                  ("world x-y (top)", "x right (m)", "y depth (m)"))
    else:
        labels = (("camera x-up (front)", "x right (m)", "-y up (m)"),
                  ("camera depth-up (side)", "z depth (m)", "-y up (m)"),
                  ("camera x-depth (top)", "x right (m)", "z depth (m)"))
    for ax, (ix, iy), (xr, yr), lab in zip((ax_f, ax_s, ax_t), ((0, 2), (1, 2), (0, 1)),
                                          wins, labels):
        ortho_panel(ax, disp, faces, kdisp, lab[0], lab[1], lab[2], ix, iy,
                    (xr, yr), z_plane=z_plane)

    # 6) 世界视角：不透明实体网格（无骨骼）
    ref = np.vstack([disp, cam_disp[None, :]]) if len(disp) else cam_disp[None, :]
    ov = _fit3(ref, pad=0.30)
    ov[2] = (min(ov[2][0], z_plane - 0.05), ov[2][1])
    frame_txt = "world" if grounded else "camera"
    scene3d_panel(ax_ov, disp, faces, cam_disp, dir_matrix, iw, ih, f_img,
                  z_plane, ov,
                  f"{frame_txt} view: opaque SAM-3D mesh + ground plane")

    # 7) 世界视角点云
    pc_ref = np.vstack([scene_disp, cam_disp[None, :]])
    pc = _fit3(pc_ref, pad=0.10)
    pc[2] = (min(pc[2][0], z_plane - 0.04), pc[2][1])
    pointcloud3d_panel(ax_pc, scene_disp, cam_disp, dir_matrix, iw, ih, f_img,
                       z_plane, pc,
                       f"{frame_txt} view point cloud: depth-colored ({len(scene_disp)} pts)")

    # 8) 深度图
    depth_panel(fig, ax_dp, depth,
                f"MoGe-2 depth map ({dh}x{dw})")

    # ---------------- 图例 / 标题 ----------------
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_JOINT,
               markeredgecolor="#8a6d00", markersize=5, label="COCO17 keypoints"),
        Line2D([0], [0], color="#7b5cd6", lw=2.0, label="COCO17 skeleton"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#21918c",
               markersize=5, label="MoGe-2 point cloud (depth-colored)"),
        Patch(facecolor=(0.31, 0.765, 0.91, 1.0), edgecolor=(0.05, 0.26, 0.34),
              label="SAM-3D opaque mesh"),
        Patch(facecolor=(0.31, 0.765, 0.91, 0.35), edgecolor="none",
              label="SAM-3D mesh on image"),
        Line2D([0], [0], color="red", lw=2.0,
               label="ground plane z=0 (red grid)" if grounded
               else "foot-level reference plane (red grid)"),
        Line2D([0], [0], color=C_ANKLE, lw=1.2, linestyle=(0, (2, 2)),
               label="ankle→ground"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#111111",
               markersize=6, label="camera + frustum"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8.6)

    cam_h_txt = f"camera_height={cam_h:.3f} m" if grounded else "camera_height=n/a"
    fig.suptitle(
        f"{stem}  |  {frame_name}  |  {ground_tag}  |  {cam_h_txt}  |  "
        f"SAM-3D {len(verts)} verts / {len(faces)} faces  |  focal={f_sam:.0f} px  |  "
        f"image {iw}x{ih}",
        fontsize=9.5, y=0.972,
    )
    if not grounded and world.get("error"):
        fig.text(0.985, 0.930, f"ground rejected: {world['error']}",
                 fontsize=8.5, color="#b3402f", ha="right")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "stem": stem,
        "frame": "ground_world_v1" if grounded else "camera_v1",
        "ground_status": world.get("status"),
        "ground_error": world.get("error"),
        "camera_height_m": cam_h if grounded else None,
        "mesh_verts": int(len(verts)),
        "mesh_faces": int(len(faces)),
        "mesh_faces_drawn": int(ntri),
        "keypoints": int(len(kp)),
        "focal_sam3d_px": f_sam,
        "focal_moge2_px": focal,
        "image_hw": [ih, iw],
        "out": out_path,
    }


def _fit3(pts: np.ndarray, pad: float) -> list[tuple[float, float]]:
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    span = max(float(np.max(hi - lo)), 0.4)
    pad_abs = max(pad, 0.25 * span)
    lo = lo - pad_abs
    hi = hi + pad_abs
    return [(float(lo[i]), float(hi[i])) for i in range(3)]


def _ortho_windows(disp, kdisp, grounded: bool, z_plane: float) -> list[tuple]:
    """每个正交视图各自的正方形窗口，避免某个轴跨度小导致图形缩成一小团。"""
    ref = np.vstack([a for a in (disp, kdisp) if len(a)])
    if not len(ref):
        ref = np.zeros((4, 3))
    out = []
    for ix, iy in ((0, 2), (1, 2), (0, 1)):
        lo = np.percentile(ref[:, [ix, iy]], 0.5, axis=0)
        hi = np.percentile(ref[:, [ix, iy]], 99.5, axis=0)
        span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 0.6) * 1.12
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        yr = (float(cy - span / 2), float(cy + span / 2))
        if grounded and iy == 2:  # 竖直轴锚定地面
            yr = (float(z_plane - 0.06 * span), float(z_plane + 0.94 * span))
        out.append(((float(cx - span / 2), float(cx + span / 2)), yr))
    return out


def _scene_points(depth, focal, rot, tvec, grounded: bool,
                  cam_world) -> np.ndarray:
    """MoGe-2 深度反投影点云（含人物表面，去掉过近/过远噪点）。

    地面变换的平移满足 ``plane_d_camera = -translation[2]``，即世界系 z=0 就是
    拟合地面，因此 ``p @ rot.T + tvec`` 之后点云已天然落地，不能再叠加
    ``ground_snap.offset_m``（该平移只属于 SAM-3D 网格的落地修正）。
    """
    dh, dw = depth.shape
    step = max(1, int(np.sqrt(dh * dw / 12000.0)))
    vs, us = np.mgrid[0:dh:step, 0:dw:step]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0.3) & (z < 8.0)
    zz = z[ok].astype(float)
    uu = us[ok].astype(float)
    vv = vs[ok].astype(float)
    pc = np.stack([(uu - dw / 2.0) * zz / focal,
                   (vv - dh / 2.0) * zz / focal, zz], axis=-1)
    if not grounded:
        return to_display(pc, False)
    pw = pc @ rot.T + tvec
    keep = np.linalg.norm(pw - cam_world, axis=1) < 4.0
    return to_display(pw[keep], True)


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MoGe-2 + SAM-3D 场景验收面板")
    ap.add_argument("images", nargs="*", help="图片路径，默认 test_image 下 4 张 jpg")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--outdir",
                    default="/mnt/g/UGit/Phy-SIC/images/test_image/moge2_visualization/scene_panel")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    _load_font()
    check_health(args.base)

    images = args.images or [
        str(p) for p in sorted(pathlib.Path(DEFAULT_ROOT).glob("*.jpg"))
        if not p.name.startswith("pose_overlay")]
    os.makedirs(args.outdir, exist_ok=True)
    stats = []
    for path in images:
        local = to_local_path(path)
        if not os.path.exists(local):
            print(f"[skip] {path}")
            continue
        stem = pathlib.Path(local).stem
        print(f"[{stem}] ...")
        g, p, tag = fetch_case(args.base, local, not args.no_cache)
        out = os.path.join(args.outdir, f"{stem}_scene_panel.png")
        st = render(g, p, local, stem, out, dpi=args.dpi)
        stats.append(st)
        print(f"[{stem}] {json.dumps(st, ensure_ascii=False)}")

    summary = os.path.join(args.outdir, "scene_panel_summary.json")
    pathlib.Path(summary).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
