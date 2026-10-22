"""姿势推理 + DepthPro 四面板可视化（可复用）。

两条渲染路径（按地面质量自动选择）：

A. 地面可信 -> 世界系四面板（render_panel）
  1. 相机视角：原图 + COCO17 骨架 + 地面网格（按 DepthPro 深度做遮挡剔除）
  2. SAM-3D Body 网格叠加原图（按 SAM-3D 自身 focal / pred_cam_t 投影）
  3. 世界系（ground_world_v1）3D 视图：DepthPro 场景点云 + 人物点云 + SAM-3D 网格 + z=0 地面 + 相机视锥
  4. DepthPro 米制深度图（turbo + 人物轮廓 + colorbar）

B. 地面不可信 -> 相机锚定四面板（render_camera_panel）
  不做地面平面换基/落地平移，只保留 camera_v1 下的 SAM-3D 数据：
  1. 原图 + COCO17 骨架（无地面网格）
  2. SAM-3D 网格叠加原图
  3. camera_v1 3D 视图：SAM-3D 网格 + COCO17 3D 骨架 + 原点相机视锥
  4. DepthPro 深度图（仅参考）

依赖运行中的服务（默认 http://127.0.0.1:8001）：
  POST /v1/ground      {image, return_data: true, visualize: false}
  POST /v1/pose/single {image, want_overlay: true, want_world: <bool>, want_mesh: true}
                       want_world=false 时服务返回 camera_space（相机锚定网格/骨架）

用法（WSL，需 matplotlib/cv2/PIL 环境）：
  python ground_pipeline/pose_pointcloud_panel.py IMG1 IMG2 ... \
      [--base http://127.0.0.1:8001] [--outdir images] [--suffix _api_pc_mesh_depth_panel] \
      [--no-cache] [--dpi 130]

输出：<outdir>/<stem><suffix>.png
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
import urllib.error
import urllib.request

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

DEFAULT_BASE = "http://127.0.0.1:8001"
DEFAULT_SUFFIX = "_api_pc_mesh_depth_panel"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "phy_sic_pose_panel_cache")

C_SCENE = "#9aa7bb"
C_PERSON = "#ff8a1f"
C_MESH = "#2fd4e8"
C_GROUND = "#8f99a8"

# COCO17 骨架 16 条边
COCO17_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
                (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12))


# ------------------------------------------------------------------ 基础设施
def to_local_path(p: str) -> str:
    """把 Windows 路径转成当前系统可用路径（WSL 下 G:\\a\\b -> /mnt/g/a/b）。"""
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", p):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p


def post(base: str, path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_health(base: str) -> None:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=20) as resp:
            print(f"[service] health {resp.status} @ {base}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"服务不可用：{base}/health ({exc})。请先启动 ground_service。"
        )


def fetch_ground(base: str, image_path: str, use_cache: bool) -> tuple[dict, str, str]:
    """取 /v1/ground 响应，按图片内容哈希做磁盘缓存。返回 (响应, data-uri, tag)。"""
    raw = pathlib.Path(image_path).read_bytes()
    img = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    tag = hashlib.sha1(raw).hexdigest()[:16]
    gpath = os.path.join(CACHE_DIR, f"{tag}_ground.json")
    if use_cache and os.path.exists(gpath):
        print(f"  cache hit {tag}/ground")
        return json.load(open(gpath, encoding="utf-8")), img, tag

    g = post(base, "/v1/ground", {"image": img, "return_data": True, "visualize": False})
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(g, open(gpath, "w", encoding="utf-8"))
    return g, img, tag


def fetch_pose(base: str, img: str, tag: str, use_cache: bool,
               want_world: bool = True) -> dict:
    """取 /v1/pose/single 响应；want_world=false 时服务只给相机锚定数据。"""
    ppath = os.path.join(CACHE_DIR, f"{tag}_pose{'' if want_world else '_cam'}.json")
    if use_cache and os.path.exists(ppath):
        print(f"  cache hit {tag}/pose{'' if want_world else '_cam'}")
        return json.load(open(ppath, encoding="utf-8"))

    p = post(base, "/v1/pose/single", {
        "image": img, "want_overlay": True, "want_world": want_world, "want_mesh": True,
    })
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(p, open(ppath, "w", encoding="utf-8"))
    return p


def _b64_payload(s: str) -> str:
    return s.split(",", 1)[1] if s.startswith("data:") and "," in s else s


def mask_from_b64(s: str) -> np.ndarray:
    arr = Image.open(io.BytesIO(base64.b64decode(_b64_payload(s)))).convert("L")
    return np.asarray(arr) > 127


def img_from_b64(s: str) -> np.ndarray:
    arr = Image.open(io.BytesIO(base64.b64decode(_b64_payload(s)))).convert("RGB")
    return np.asarray(arr)


# ------------------------------------------------------------------ 图元
def mesh_overlay_image(rgb, verts_cam, faces, focal, out_size):
    """深度排序三角面片，半透明青色覆盖到原图上。"""
    h, w = out_size[1], out_size[0]
    z = np.maximum(verts_cam[:, 2], 1e-6)
    uv = np.stack([focal * verts_cam[:, 0] / z + w / 2.0,
                   focal * verts_cam[:, 1] / z + h / 2.0], axis=1)

    tris = faces[(verts_cam[faces, 2] > 1e-3).all(axis=1)]
    if len(tris) == 0:
        return rgb, 0
    order = np.argsort(-z[tris].mean(axis=1))  # 远 -> 近

    tri3 = verts_cam[tris]
    nrm = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    light = np.array([0.35, -0.55, -0.75])
    light /= np.linalg.norm(light)
    shade = 0.40 + 0.60 * np.abs(nrm @ light)

    import cv2

    layer = np.zeros((h, w, 4), dtype=np.uint8)  # cv2 顺序 BGRA
    for idx in order:
        pts = np.rint(uv[tris[idx]]).astype(np.int32)
        s = float(shade[idx])
        cv2.fillPoly(layer, [pts], (int(225 * s), int(200 * s), int(60 * s), 90),
                     lineType=cv2.LINE_8)
    for idx in order[::6]:
        pts = np.rint(uv[tris[idx]]).astype(np.int32)
        cv2.polylines(layer, [pts], True, (140, 110, 30, 60), 1, cv2.LINE_8)

    base = Image.fromarray(rgb).convert("RGBA")
    over = Image.fromarray(layer[:, :, [2, 1, 0, 3]], mode="RGBA")  # BGRA -> RGBA
    return np.asarray(Image.alpha_composite(base, over).convert("RGB")), int(len(tris))


def draw_camera(ax, cam, rot, w, h, focal, length=0.55):
    def world_dir(u, v):
        d = np.array([(u - w / 2.0) / focal, (v - h / 2.0) / focal, 1.0])
        d /= np.linalg.norm(d)
        return rot @ d

    corners = [cam + length * world_dir(u, v)
               for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    for c in corners:
        ax.plot([cam[0], c[0]], [cam[1], c[1]], [cam[2], c[2]],
                color="#222222", lw=0.8, zorder=6)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="#222222", lw=0.9, zorder=6)
    ax.scatter([cam[0]], [cam[1]], [cam[2]], s=42, c="#111111",
               marker="^", depthshade=False, zorder=7)


def ground_grid_camera(ax_img, rot, tvec, focal, w, h, depth,
                       gx=(-2.0, 2.0), gy=(-0.5, 4.5), stride=0.5):
    """把世界系 z=0 网格投到图像上，并用 DepthPro 深度剔除画到墙上的部分。"""
    xs = np.arange(gx[0], gx[1] + 1e-6, stride)
    ys = np.arange(gy[0], gy[1] + 1e-6, stride)
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
        keep = ins & np.isfinite(dd) & (np.abs(dd - zc) < 0.15)
        ax_img.plot(np.where(keep, uu, np.nan), np.where(keep, vv, np.nan),
                    color=C_PERSON, lw=0.8, alpha=0.6)


# ------------------------------------------------------------------ 主渲染
def render_panel(g: dict, p: dict, image_path: str, stem: str, out_path: str,
                 dpi: int = 130) -> dict:
    world = p["world"]
    rot = np.asarray(world["rotation"], dtype=np.float64)
    tvec = np.asarray(world["camera_position"], dtype=np.float64)
    snap = world.get("ground_snap") or {}
    cam_h = float(world.get("camera_height_m") or 0.0)

    depth = np.load(io.BytesIO(base64.b64decode(_b64_payload(g["depth_b64"])))).astype(np.float32)
    h, w = depth.shape
    focal = float(g["focal_px"])
    person_mask = mask_from_b64(g["masks"]["person"])

    # DepthPro 反投影 -> 世界系点云
    step = max(1, int(np.sqrt(h * w / 90000.0)))
    vs, us = np.mgrid[0:h:step, 0:w:step]
    z = depth[vs, us]
    valid = np.isfinite(z) & (z > 0)
    pc = np.stack([(us - w / 2.0) * z / focal,
                   (vs - h / 2.0) * z / focal, z], axis=-1)
    pw = pc[valid] @ rot.T + tvec
    is_person = person_mask[vs, us][valid]
    scene, person = pw[~is_person], pw[is_person]

    # SAM-3D 网格：世界系 -> 相机系（用于 2D 投影）
    mesh = world.get("mesh") or {}
    mv_world = np.asarray(mesh.get("vertices") or [], dtype=np.float64)
    mf = np.asarray(mesh.get("faces") or [], dtype=np.int64)
    mv_cam = (mv_world - tvec) @ rot if len(mv_world) else mv_world
    f_sam = float((p.get("camera") or {}).get("focal_px") or focal)
    base_rgb = np.asarray(Image.open(to_local_path(image_path)).convert("RGB"))
    mesh_img, ntri = mesh_overlay_image(base_rgb, mv_cam, mf, f_sam, (w, h))

    fig = plt.figure(figsize=(22.0, 8.8), dpi=dpi)
    ax_img = fig.add_axes([0.002, 0.105, 0.140, 0.740])
    ax_msh = fig.add_axes([0.146, 0.105, 0.140, 0.740])
    ax_pc = fig.add_axes([0.292, 0.050, 0.360, 0.845], projection="3d")
    ax_d = fig.add_axes([0.668, 0.105, 0.245, 0.740])
    ax_cb = fig.add_axes([0.921, 0.115, 0.010, 0.720])

    # 1) 相机视角
    ax_img.imshow(img_from_b64(p["overlay_jpeg_b64"]))
    ground_grid_camera(ax_img, rot, tvec, focal, w, h, depth)
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)
    ax_img.axis("off")
    ax_img.set_title(f"camera view | {stem}\nimage + COCO17 skeleton + ground grid",
                     fontsize=8.5)

    # 2) SAM-3D 网格叠加原图
    ax_msh.imshow(mesh_img)
    ax_msh.set_xlim(0, w)
    ax_msh.set_ylim(h, 0)
    ax_msh.axis("off")
    ax_msh.set_title(f"SAM-3D mesh on image\nfocal={f_sam:.0f} px, {ntri} faces",
                     fontsize=8.5)

    # 3) 世界系点云 + 网格
    ax_pc.computed_zorder = False
    ax_pc.set_proj_type("persp", focal_length=0.9)
    scene_keep = scene[(scene[:, 2] > -0.3) & (scene[:, 2] < 2.6)]
    ax_pc.scatter(scene_keep[:, 0], scene_keep[:, 1], scene_keep[:, 2],
                  s=0.7, c=C_SCENE, alpha=0.10, depthshade=False, zorder=2)
    ax_pc.scatter(person[:, 0], person[:, 1], person[:, 2],
                  s=1.8, c=C_PERSON, alpha=0.55, depthshade=False, zorder=4)
    if len(mv_world) and len(mf):
        ax_pc.add_collection3d(Poly3DCollection(
            mv_world[mf], facecolors=C_MESH, edgecolors="none", alpha=0.20, zorder=5))
        ax_pc.scatter(mv_world[:, 0], mv_world[:, 1], mv_world[:, 2],
                      s=0.6, c="#0fb6cf", alpha=0.35, depthshade=False, zorder=5)

    gx0, gx1, gy0, gy1 = -2.0, 2.0, -0.5, 4.5
    ax_pc.plot_surface(np.array([[gx0, gx1], [gx0, gx1]]),
                       np.array([[gy0, gy0], [gy1, gy1]]),
                       np.zeros((2, 2)), color="#c9d6c9", alpha=0.30,
                       linewidth=0, shade=False, zorder=1)
    for x in np.arange(gx0, gx1 + 0.01, 0.5):
        ax_pc.plot([x, x], [gy0, gy1], [0, 0], color=C_GROUND, lw=0.5, zorder=1)
    for y in np.arange(gy0, gy1 + 0.01, 0.5):
        ax_pc.plot([gx0, gx1], [y, y], [0, 0], color=C_GROUND, lw=0.5, zorder=1)
    draw_camera(ax_pc, tvec, rot, w, h, focal)

    ref = np.vstack([person, mv_world]) if len(mv_world) else person
    xlim = (min(np.percentile(ref[:, 0], 1), tvec[0]) - 0.15,
            max(np.percentile(ref[:, 0], 99), tvec[0]) + 0.15)
    ylim = (min(np.percentile(ref[:, 1], 1), tvec[1]) - 0.15,
            max(np.percentile(ref[:, 1], 99), tvec[1]) + 0.15)
    ztop = float(min(np.percentile(ref[:, 2], 99) + 0.25, 2.4))
    ax_pc.set_xlim(*xlim)
    ax_pc.set_ylim(*ylim)
    ax_pc.set_zlim(-0.02, ztop)
    ax_pc.set_xlabel("world x (m)", fontsize=8)
    ax_pc.set_ylabel("world y (m)", fontsize=8)
    ax_pc.set_zlabel("height z (m)", fontsize=8)
    ax_pc.tick_params(labelsize=7)
    for axis in (ax_pc.xaxis, ax_pc.yaxis, ax_pc.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax_pc.view_init(elev=16, azim=-64)
    ax_pc.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], ztop * 1.25))
    ax_pc.set_title("point cloud + SAM-3D mesh in world frame (ground_world_v1)\n"
                    f"DepthPro {len(scene)} scene + {len(person)} person pts | "
                    f"SAM-3D mesh {len(mv_world)} verts / {len(mf)} faces",
                    fontsize=8.5, pad=2)

    # 4) 米制深度
    p1, p99 = np.percentile(depth, [1, 99])
    im = ax_d.imshow(depth, cmap="turbo", vmin=p1, vmax=p99)
    ax_d.contour(person_mask.astype(np.float32), levels=[0.5], colors="white", linewidths=1.1)
    ax_d.axis("off")
    ax_d.set_title(f"DepthPro metric depth ({h}x{w})\n"
                   f"focal={focal:.1f} px, range={p1:.2f}-{p99:.2f} m "
                   "(person contour in white)", fontsize=8.5)
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("depth (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_SCENE,
               markersize=5, label="DepthPro scene cloud"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PERSON,
               markersize=5, label="DepthPro person cloud (person mask)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_MESH,
               markersize=5, label="SAM-3D Body mesh"),
        Line2D([0], [0], color=C_GROUND, lw=1.2, label="ground plane z=0 (0.5 m grid)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#111111",
               markersize=5, label="camera + frustum"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(
        f"{stem}  |  ground_snap={snap.get('status')} / source={snap.get('source')} "
        f"/ offset={float(snap.get('offset_m') or 0.0):+.4f} m  |  camera_height={cam_h:.3f} m  |  "
        f"SAM-3D focal={f_sam:.0f} px vs DepthPro focal={focal:.0f} px; "
        "mesh 2D projected with SAM-3D camera; world 3D placement not aligned to cloud",
        fontsize=9.5, y=0.985,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    stats = {
        "stem": stem,
        "image_hw": [h, w],
        "focal_depthpro": focal,
        "focal_sam3d": f_sam,
        "scene_points": int(len(scene)),
        "person_points": int(len(person)),
        "mesh_verts": int(len(mv_world)),
        "mesh_faces": int(len(mf)),
        "mesh_faces_drawn": ntri,
        "ground_snap": snap.get("status"),
        "ground_offset_m": snap.get("offset_m"),
        "camera_height_m": cam_h,
        "floor_pixels": g.get("floor_pixels"),
        "plane_inliers": g.get("plane_inliers"),
        "contact_pixels": g.get("contact_pixels"),
        "out": out_path,
    }
    return stats


def cam_display(pts: np.ndarray) -> np.ndarray:
    """camera_v1 (x 右 / y 下 / z 前) -> 绘制坐标 (x, z, -y)。

    第三维作为 matplotlib 竖直轴，所以把 -y（向上）放在竖直轴上，
    人物才会以站立的轮廓出现，而不是沿相机轴被压扁。
    """
    return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)


def render_camera_panel(g: dict, p: dict, image_path: str, stem: str, out_path: str,
                        dpi: int = 130, warns: list[tuple[str, str]] | None = None) -> dict:
    """地面不可用时的回退：只画相机锚定（camera_v1）的 SAM-3D 数据。"""
    warns = warns or []
    cs = p.get("camera_space") or {}
    verts = np.asarray(cs.get("vertices") or [], dtype=np.float64)
    faces = np.asarray(cs.get("faces") or [], dtype=np.int64)
    kp = np.asarray(cs.get("keypoints_3d_coco17") or [], dtype=np.float64)
    f_sam = float((p.get("camera") or {}).get("focal_px") or 0.0)

    depth = np.load(io.BytesIO(base64.b64decode(_b64_payload(g["depth_b64"])))).astype(np.float32)
    h, w = depth.shape
    focal = float(g["focal_px"])
    person_mask = mask_from_b64(g["masks"]["person"])

    base_rgb = np.asarray(Image.open(to_local_path(image_path)).convert("RGB"))
    mesh_img, ntri = mesh_overlay_image(base_rgb, verts, faces, f_sam, (w, h))

    fig = plt.figure(figsize=(22.0, 8.8), dpi=dpi)
    ax_img = fig.add_axes([0.002, 0.105, 0.140, 0.740])
    ax_msh = fig.add_axes([0.146, 0.105, 0.140, 0.740])
    ax_pc = fig.add_axes([0.292, 0.050, 0.360, 0.845], projection="3d")
    ax_d = fig.add_axes([0.668, 0.105, 0.245, 0.740])
    ax_cb = fig.add_axes([0.921, 0.115, 0.010, 0.720])

    # 1) 相机视角（无地面网格）
    ax_img.imshow(img_from_b64(p["overlay_jpeg_b64"]))
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)
    ax_img.axis("off")
    ax_img.set_title(f"camera view | {stem}\nimage + COCO17 skeleton (no ground grid)",
                     fontsize=8.5)

    # 2) SAM-3D 网格叠加原图
    ax_msh.imshow(mesh_img)
    ax_msh.set_xlim(0, w)
    ax_msh.set_ylim(h, 0)
    ax_msh.axis("off")
    ax_msh.set_title(f"SAM-3D mesh on image\nfocal={f_sam:.0f} px, {ntri} faces",
                     fontsize=8.5)

    # 3) 相机锚定 3D：只画 SAM-3D 网格 + 骨架
    ax_pc.computed_zorder = False
    ax_pc.set_proj_type("persp", focal_length=0.9)
    if len(verts) and len(faces):
        disp = cam_display(verts)
        ax_pc.add_collection3d(Poly3DCollection(
            disp[faces], facecolors=C_MESH, edgecolors="none", alpha=0.24, zorder=5))
        ax_pc.scatter(disp[:, 0], disp[:, 1], disp[:, 2],
                      s=0.6, c="#0fb6cf", alpha=0.40, depthshade=False, zorder=5)
    if len(kp) > 16:
        kd = cam_display(kp)
        for a, b in COCO17_EDGES:
            ax_pc.plot([kd[a, 0], kd[b, 0]], [kd[a, 1], kd[b, 1]], [kd[a, 2], kd[b, 2]],
                       color=C_PERSON, lw=1.6, zorder=6)
        ax_pc.scatter(kd[:, 0], kd[:, 1], kd[:, 2],
                      s=12, c="#0b2e33", depthshade=False, zorder=7)

    # 取景只包围人物本身：把人放到 z=0 附近会把视角拉远压扁人物
    ref = cam_display(verts) if len(verts) else np.zeros((1, 3))
    def _lim(vals, pad=0.15):
        return (float(np.percentile(vals, 1)) - pad,
                float(np.percentile(vals, 99)) + pad)
    xlim = _lim(ref[:, 0])
    ylim = _lim(ref[:, 1])
    zlim = _lim(ref[:, 2])
    ax_pc.set_xlim(*xlim)
    ax_pc.set_ylim(*ylim)
    ax_pc.set_zlim(*zlim)
    ax_pc.set_xlabel("x right (m)", fontsize=8)
    ax_pc.set_ylabel("z forward (m)", fontsize=8)
    ax_pc.set_zlabel("-y up (m)", fontsize=8)
    ax_pc.tick_params(labelsize=7)
    for axis in (ax_pc.xaxis, ax_pc.yaxis, ax_pc.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    # 视线沿 +z 看过去，-y 为竖直向上，人物呈站立正面
    ax_pc.view_init(elev=10, azim=-80)
    ax_pc.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax_pc.set_title("camera-anchored SAM-3D only (camera_v1)\n"
                    f"x right / y down / z forward, camera at origin | "
                    f"mesh {len(verts)} verts / {len(faces)} faces | {len(kp)} keypoints",
                    fontsize=8.5, pad=2)

    # 4) 米制深度（原始 DepthPro 输出，仅作参考）
    p1, p99 = np.percentile(depth, [1, 99])
    im = ax_d.imshow(depth, cmap="turbo", vmin=p1, vmax=p99)
    ax_d.contour(person_mask.astype(np.float32), levels=[0.5], colors="white", linewidths=1.1)
    ax_d.axis("off")
    ax_d.set_title(f"DepthPro metric depth ({h}x{w}) - reference only\n"
                   f"focal={focal:.1f} px, range={p1:.2f}-{p99:.2f} m",
                   fontsize=8.5)
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("depth (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_MESH,
               markersize=5, label="SAM-3D Body mesh (camera_v1)"),
        Line2D([0], [0], color=C_PERSON, lw=1.6, label="COCO17 3D skeleton (camera_v1)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    warn_txt = " ; ".join(code for code, _ in warns)
    fig.suptitle(
        f"{stem}  |  GROUND REJECTED -> camera-anchored SAM-3D only, no ground-plane transform"
        f"  |  reason: {warn_txt}",
        fontsize=9.5, y=0.985,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "stem": stem,
        "mode": "camera_v1",
        "image_hw": [h, w],
        "focal_depthpro": focal,
        "focal_sam3d": f_sam,
        "mesh_verts": int(len(verts)),
        "mesh_faces": int(len(faces)),
        "mesh_faces_drawn": ntri,
        "keypoints_3d": int(len(kp)),
        "ground_warnings": [msg for _, msg in warns],
        "out": out_path,
    }


def ground_quality_warnings(g: dict) -> list[tuple[str, str]]:
    """地面/焦距可靠性提示：不改变结果，只提示该图的世界系不可直接采信。

    返回 (ascii_code, message)，ascii_code 用于图上文字（matplotlib 无中文字体），
    message 用于控制台。
    """
    warns = []
    cam_h = float(g.get("camera_height_m") or 0.0)
    contact = int(g.get("contact_pixels") or 0)
    inliers = int(g.get("plane_inliers") or 0)
    if cam_h and not (0.35 <= cam_h <= 2.6):
        warns.append((f"camera_height={cam_h:.2f}m out of [0.35,2.6]",
                      f"camera_height={cam_h:.3f} m 超出常见范围 [0.35, 2.6]"))
    if contact and contact < 500:
        warns.append((f"contact_px={contact} too few",
                      f"contact_pixels={contact} 过少，人物与地面接触证据不足"))
    if inliers and inliers < 80000:
        warns.append((f"plane_inliers={inliers} too few",
                      f"plane_inliers={inliers} 过少，平面拟合可能失败"))
    normal = g.get("plane_normal") or []
    if len(normal) == 3 and max(abs(float(v)) for v in normal) < 0.9:
        norm_txt = "[" + ",".join(f"{float(v):.2f}" for v in normal) + "]"
        warns.append((f"plane_normal={norm_txt} unstable",
                      f"plane_normal={normal} 无主轴分量（max|n|<0.9），地面法向不稳定"))
    return warns


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="姿势推理 + DepthPro 四面板可视化（相机视角 / SAM-3D 网格叠图 / 世界系点云+网格 / 深度图）"
    )
    ap.add_argument("images", nargs="+", help="输入图片路径（Windows 或 WSL 路径均可）")
    ap.add_argument("--base", default=DEFAULT_BASE, help="服务地址，默认 %(default)s")
    ap.add_argument("--outdir", default="images", help="输出目录，默认 %(default)s")
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX, help="输出文件名后缀，默认 %(default)s")
    ap.add_argument("--dpi", type=int, default=130, help="输出 DPI，默认 %(default)s")
    ap.add_argument("--no-cache", action="store_true", help="禁用响应缓存，强制重新推理")
    args = ap.parse_args(argv)

    check_health(args.base)
    failed = []
    for path in args.images:
        local = to_local_path(path)
        if not os.path.exists(local):
            print(f"[skip] 找不到文件：{path}")
            failed.append(path)
            continue
        stem = pathlib.Path(local).stem
        print(f"[{stem}] 推理中 ...")
        use_cache = not args.no_cache
        try:
            g, img, tag = fetch_ground(args.base, local, use_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"[{stem}] /v1/ground 请求失败：{exc}")
            failed.append(stem)
            continue
        if "depth_b64" not in g:
            print(f"[{stem}] /v1/ground 未返回 depth_b64")
            failed.append(stem)
            continue

        warns = ground_quality_warnings(g)
        for _, msg in warns:
            print(f"[{stem}][warn] {msg}")
        want_world = not warns  # 地面不可信 -> 直接走相机锚定，不做地面平面转换
        out_path = os.path.join(args.outdir, f"{stem}{args.suffix}.png")
        try:
            p = fetch_pose(args.base, img, tag, use_cache, want_world=want_world)
            if want_world and (p.get("status") != "success"
                               or (p.get("world") or {}).get("status") != "ok"):
                warns.append(("world unavailable", "world.status 不可用，回退相机锚定"))
                want_world = False
                p = fetch_pose(args.base, img, tag, use_cache, want_world=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[{stem}] /v1/pose/single 请求失败：{exc}")
            failed.append(stem)
            continue

        if want_world:
            if p.get("status") != "success" or not p.get("world"):
                print(f"[{stem}] 结果不完整：status={p.get('status')}")
                failed.append(stem)
                continue
            stats = render_panel(g, p, local, stem, out_path, dpi=args.dpi)
        else:
            if not (p.get("camera_space") or {}).get("vertices"):
                print(f"[{stem}] 服务未返回 camera_space（需重启服务加载新版本）")
                failed.append(stem)
                continue
            stats = render_camera_panel(g, p, local, stem, out_path,
                                        dpi=args.dpi, warns=warns)
        print(f"[{stem}] saved {out_path} | {json.dumps(stats, ensure_ascii=False)}")

    if failed:
        print(f"失败 {len(failed)} 张：{failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
