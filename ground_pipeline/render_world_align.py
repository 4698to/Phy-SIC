"""世界空间 SAM-3D 网格 × MoGe-2 点云 对齐验收面板。

limb 档（``POSEEXTRACTOR_SAM3D_SCALE_FIT=limb``）且门控通过时切到 3x4 对比版面：
  上排：camera before / camera after / front before / front after
  中排：side before / side after / top before / top after
  下排：3D 透视(after) / 指标对比表 / 残差直方图(before vs after)
  「before」网格取自同一张图的 torso 档缓存（``_scalefit_torso``），
  它是 limb 修正前的原样（同一 uniform factor、同一 cam_t），只读缓存、不改服务端。
  「after」各面板里被修正的区域按逐顶点位移量由青色渐变为红色（>1cm 起染、>=3cm 全红），
  可直接看出哪些肢体真的被改动、哪些几乎没动。
其余档位（off/moge/torso）与 limb 门控未通过时仍用下面的 2x4 版面，输出不变。

C 方案（``POSEEXTRACTOR_SAM3D_FRONT_SNAP=1``，默认关闭）：以 torso 档缓存为基线，
把每个顶点沿自己的相机射线推到 MoGe-2 实测前表面（见 ``mesh_front_snap``），
before = torso、after = 前表面贴合，同样出 3x4 对比版面；关掉 env 即回滚。

2x4 版面：
  上排：
    1. 相机视角：原图 + 半透明 SAM-3D 网格 + MoGe-2 点云（同一针孔投影）
    2. front  x-z
    3. side   y-z
    4. top    x-y
    点云 = person mask 内的 MoGe-2 反投影点（按相机深度着色，小点）
    网格 = SAM-3D 半透明实体三角面
    地面 = z=0 参考线（ground=rejected 时为脚底参考线）
  下排：
    5. 3D 透视叠加：点云 + 半透明网格 + 红色网格地面 + 相机视锥
    6. 点到网格最近距离直方图
    7. 对齐指标卡（最低点 z、最近距离中位数/P90、bbox 三轴比例、贴合率）

相机视角面板用 ``camera_space``（未做地面变换的 camera_v1 几何）投影，才能与
原图成像严格重合；世界系几何反解回相机会整体平移 40~55 px。

世界系一致性：世界系 z=0 就是 MoGe-2 深度拟合出的地面（地面变换满足
``plane_d_camera = -translation[2]``），所以 ``p @ rot.T + tvec`` 之后的点云
已天然落地，**不能**再叠加 ``ground_snap.offset_m``；该落地平移只用于把
SAM-3D 网格的最低点拉到 z=0。ground=rejected 时两者都留在 camera_v1。

尺度一致性：``/v1/pose/single`` 现在接收 MoGe 实测内参（``cam_int``），SAM-3D
不再退回默认 FOV 先验，``f_sam == f_moge``，相机系与世界系的深度可直接比较。

用法（WSL）：
  python ground_pipeline/render_world_align.py [IMG ...] \
      [--outdir DIR] [--dpi 150] [--no-cache]

默认输出到 ``<图片目录>/moge2_visualization/world_align/``：
  <stem>_world_align.png + world_align_summary.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_scene_panel as rsp  # noqa: E402
import mesh_front_snap as fsnap  # noqa: E402
import bone_depth_snap as bsnap  # noqa: E402
import skeleton_fbx as skelfbx  # noqa: E402
import mhr_mesh_fbx as mhrmesh  # noqa: E402
import mhr_pose_fbx as mhrpose  # noqa: E402

# ------------------------------------------------------------------ 解码
def decode_depth(g: dict) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(rsp._b64_payload(g["depth_b64"])))).astype(np.float32)


def decode_person_mask(g: dict):
    blocks = g.get("masks") or {}
    return rsp.mask_from_b64(blocks["person"]) if blocks.get("person") else None


# ------------------------------------------------------------------ 点云 / 指标
def person_camera_points(depth: np.ndarray, focal: float, mask: np.ndarray,
                         max_pts: int = 15000):
    """person 掩码内的 MoGe-2 反投影点云（相机系），返回 (点, 相机深度)。"""
    dh, dw = depth.shape
    m = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize((dw, dh), Image.NEAREST)
    ) > 127
    n = int(m.sum())
    step = max(1, int(np.sqrt(max(n, 1) / float(max_pts))))
    vs, us = np.mgrid[0:dh:step, 0:dw:step]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0.3) & (z < 8.0) & m[vs, us]
    zz = z[ok].astype(float)
    uu = us[ok].astype(float)
    vv = vs[ok].astype(float)
    pc = np.stack([(uu - dw / 2.0) * zz / focal,
                   (vv - dh / 2.0) * zz / focal, zz], axis=-1)
    return pc, pc[:, 2]


def person_point_cloud(depth: np.ndarray, focal: float, mask: np.ndarray,
                       grounded: bool, rot, tvec,
                       max_pts: int = 15000):
    """person 掩码内的 MoGe-2 反投影点云，返回 (绘图坐标, 相机深度)。"""
    pc, zcam = person_camera_points(depth, focal, mask, max_pts=max_pts)
    if not grounded:
        return rsp.to_display(pc, False), zcam
    return rsp.to_display(pc @ np.asarray(rot).T + np.asarray(tvec), True), zcam


def nn_distance(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """分块计算 a 中每点到 b 的最近距离（b 为网格顶点代理，避免额外依赖）。"""
    out = np.empty(len(a), dtype=np.float64)
    for i in range(0, len(a), chunk):
        d = a[i:i + chunk, None, :] - b[None, :, :]
        out[i:i + chunk] = np.sqrt((d * d).sum(-1)).min(1)
    return out


def align_metrics(mesh_disp: np.ndarray, pts_disp: np.ndarray,
                  max_query: int = 2500, max_ref: int = 8000) -> dict:
    """点云-网格对齐指标（全部在绘图坐标系内计算，grounded 时为世界系）。"""
    if not len(mesh_disp) or not len(pts_disp):
        return {}
    ref = mesh_disp
    if len(ref) > max_ref:
        ref = ref[np.linspace(0, len(ref) - 1, max_ref).astype(int)]
    qry = pts_disp
    if len(qry) > max_query:
        qry = qry[np.linspace(0, len(qry) - 1, max_query).astype(int)]
    dist = nn_distance(qry, ref)

    ext_pc = pts_disp.max(0) - pts_disp.min(0)
    ext_mesh = mesh_disp.max(0) - mesh_disp.min(0)
    ratio = ext_pc / np.maximum(ext_mesh, 1e-6)
    return {
        "pc_min_z_m": float(np.percentile(pts_disp[:, 2], 1.0)),
        "mesh_min_z_m": float(np.percentile(mesh_disp[:, 2], 0.5)),
        "nn_median_m": float(np.median(dist)),
        "nn_p90_m": float(np.percentile(dist, 90.0)),
        "surface_ratio_3cm": float((dist < 0.03).mean()),
        "bbox_ratio_xyz": [float(v) for v in ratio],
        "pc_points": int(len(pts_disp)),
    }


# ------------------------------------------------------------------ 绘制
C_ADJUSTED_RGB = np.array([0.86, 0.24, 0.19])   # 被 limb 修正的顶点/面：红
SHIFT_LO, SHIFT_HI = 0.01, 0.03                 # <1cm 视作未调整，>=3cm 全红


def shift_face_rgb(faces: np.ndarray, shift: np.ndarray | None) -> np.ndarray:
    """逐面位移 → 青色到红色的插值基色；shift 为 None 时返回统一青色。"""
    base = np.asarray(rsp.C_MESH_RGB, dtype=float).reshape(1, 3)
    if shift is None:
        return base
    t = np.clip((np.asarray(shift, dtype=float).reshape(-1)[faces].mean(axis=1)
                 - SHIFT_LO) / (SHIFT_HI - SHIFT_LO), 0.0, 1.0)
    return base * (1.0 - t)[:, None] + C_ADJUSTED_RGB.reshape(1, 3) * t[:, None]


def ortho_overlay(ax, mesh_disp, faces, pts_disp, zcam, ix, iy, window,
                  grounded: bool, z_plane: float, title, xlabel, ylabel,
                  shift: np.ndarray | None = None,
                  joints: np.ndarray | None = None,
                  joints_shift: np.ndarray | None = None) -> None:
    ax.set_facecolor("white")
    if grounded and iy == 2:
        ax.axhline(z_plane, color=rsp.C_GROUND_RED, lw=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
    if len(pts_disp):
        ax.scatter(pts_disp[:, ix], pts_disp[:, iy], c=zcam, cmap="viridis",
                   s=2.0, alpha=0.80, linewidths=0, zorder=2)
    if len(faces):
        tri = mesh_disp[faces]
        rgba = rsp.shade_faces(mesh_disp, faces, shift_face_rgb(faces, shift),
                               alpha=0.34)
        depth_axis = 3 - ix - iy
        order = np.argsort(-tri[:, :, depth_axis].mean(axis=1))
        coll = PolyCollection(tri[order][:, :, [ix, iy]], facecolors=rgba[order],
                              edgecolors=(0.06, 0.35, 0.45, 0.10), linewidths=0.08,
                              rasterized=True, zorder=3)
        ax.add_collection(coll)
    if joints is not None and len(joints):
        rsp.draw_bones(ax, np.asarray(joints)[:, [ix, iy]], lw=1.3,
                       color="#1f2d3d")
        ax.scatter(joints[:, ix], joints[:, iy], s=9, c="#1f2d3d",
                   edgecolors="white", linewidths=0.4, zorder=8)
        if joints_shift is not None and len(joints_shift):
            hot = np.asarray(joints_shift) > SHIFT_LO
            if np.any(hot):
                ax.scatter(joints[hot, ix], joints[hot, iy], s=26,
                           facecolors="none", edgecolors=C_ADJUSTED_RGB,
                           linewidths=1.1, zorder=9)
    ax.set_xlim(*window[0])
    ax.set_ylim(*window[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(which="major", color="#c9d0da", lw=0.7, alpha=0.9, zorder=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def camera_view_panel(ax, base_rgb, pc_cam, zcam, verts_cam, faces, focal,
                      title, shift: np.ndarray | None = None,
                      joints_cam: np.ndarray | None = None,
                      joints_shift: np.ndarray | None = None) -> int:
    """相机视角：原图 + 半透明 SAM-3D 网格 + MoGe-2 点云（同一针孔投影）。

    ``shift`` 给定时，被修正的网格区域按位移量渐变为红色。
    ``joints_cam`` 给定时叠加相机系骨架（图像像素坐标），用于核对骨骼修正。
    """
    ih, iw = base_rgb.shape[:2]
    img, ntri = base_rgb, 0
    if len(verts_cam) and len(faces):
        img, ntri = rsp.mesh_overlay_image(base_rgb, verts_cam, faces, focal, (iw, ih),
                                          shift_m=shift,
                                          shift_lo=SHIFT_LO, shift_hi=SHIFT_HI)
    ax.imshow(img)
    if len(pc_cam):
        z = np.maximum(pc_cam[:, 2], 1e-6)
        u = focal * pc_cam[:, 0] / z + iw / 2.0
        v = focal * pc_cam[:, 1] / z + ih / 2.0
        keep = (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
        ax.scatter(u[keep], v[keep], c=zcam[keep], cmap="viridis", s=1.6,
                   alpha=0.55, linewidths=0, zorder=3)
    if joints_cam is not None and len(joints_cam):
        jz = np.maximum(np.asarray(joints_cam)[:, 2], 1e-6)
        ju = focal * joints_cam[:, 0] / jz + iw / 2.0
        jv = focal * joints_cam[:, 1] / jz + ih / 2.0
        jp = np.stack([ju, jv], axis=1)
        rsp.draw_bones(ax, jp, lw=1.4, color="#1f2d3d")
        ax.scatter(jp[:, 0], jp[:, 1], s=11, c="#1f2d3d", edgecolors="white",
                   linewidths=0.5, zorder=6)
        if joints_shift is not None and len(joints_shift):
            hot = np.asarray(joints_shift) > SHIFT_LO
            if np.any(hot):
                ax.scatter(jp[hot, 0], jp[hot, 1], s=30, facecolors="none",
                           edgecolors=C_ADJUSTED_RGB, linewidths=1.2, zorder=7)
    ax.set_xlim(0, iw)
    ax.set_ylim(ih, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    return ntri


def overlay3d(ax, mesh_disp, faces, pts_disp, zcam, cam_disp, dir_matrix,
              w, h, focal, z_plane, win3, title,
              shift: np.ndarray | None = None,
              joints: np.ndarray | None = None,
              joints_shift: np.ndarray | None = None) -> None:
    ax.computed_zorder = False
    ax.set_proj_type("persp", focal_length=rsp.PERSP_FOCAL)
    rsp.style_3d_axes(ax)
    rsp.red_ground_plane(ax, (win3[0][0], win3[0][1]), (win3[1][0], win3[1][1]), z_plane)

    if len(pts_disp):
        sc = ax.scatter(pts_disp[:, 0], pts_disp[:, 1], pts_disp[:, 2], c=zcam,
                        cmap="viridis", s=1.4, alpha=0.9, depthshade=False, zorder=3)
        cb = plt.colorbar(sc, ax=ax, fraction=0.032, pad=0.012)
        cb.set_label("camera depth (m)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    if len(faces):
        rgba = rsp.shade_faces(mesh_disp, faces, shift_face_rgb(faces, shift),
                               alpha=0.30)
        coll = Poly3DCollection(mesh_disp[faces], facecolors=rgba,
                                edgecolors=(0.05, 0.26, 0.34, 0.18), linewidths=0.08,
                                zsort="average")
        coll.set_zorder(5)
        ax.add_collection3d(coll)
    if joints is not None and len(joints):
        rsp.draw_bones(ax, np.asarray(joints), lw=1.6, three_d=True, color="#1f2d3d")
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=14, c="#1f2d3d",
                   edgecolors="white", linewidths=0.5, depthshade=False, zorder=9)
        if joints_shift is not None and len(joints_shift):
            hot = np.asarray(joints_shift) > SHIFT_LO
            if np.any(hot):
                ax.scatter(joints[hot, 0], joints[hot, 1], joints[hot, 2], s=34,
                           facecolors="none", edgecolors=C_ADJUSTED_RGB,
                           linewidths=1.2, depthshade=False, zorder=10)
    rsp.draw_camera(ax, cam_disp, dir_matrix, w, h, focal)

    rsp._style_3d_view(ax, win3)
    ax.view_init(elev=18, azim=-64)
    ax.set_title(title, fontsize=9, pad=2)


def hist_panel(ax, dist: np.ndarray, metrics: dict) -> None:
    if len(dist):
        ax.hist(dist, bins=60, range=(0.0, max(0.20, float(np.percentile(dist, 99)))),
                color="#21918c", alpha=0.88)
    ax.axvline(0.03, color="#b3402f", lw=1.2, linestyle=(0, (4, 3)),
               label="3 cm 验收线")
    ax.set_xlabel("point → mesh nearest distance (m)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e6eaf0", lw=0.6, alpha=0.9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("alignment residual (sampled)", fontsize=9)


def metrics_panel(ax, m: dict, grounded: bool, landing_m: float) -> None:
    ax.axis("off")
    if not m:
        ax.text(0.02, 0.5, "no person points", fontsize=9, va="center")
        return
    r = m["bbox_ratio_xyz"]
    lines = [
        ("frame", "ground_world_v1" if grounded else "camera_v1"),
        ("mesh landing offset (info)", f"{landing_m:+.4f} m"),
        ("person points", f"{m['pc_points']}"),
        ("pc min z (P1)", f"{m['pc_min_z_m']:+.4f} m"),
        ("mesh min z (P0.5)", f"{m['mesh_min_z_m']:+.4f} m"),
        ("nn median", f"{m['nn_median_m'] * 100:.2f} cm"),
        ("nn P90", f"{m['nn_p90_m'] * 100:.2f} cm"),
        ("surface < 3 cm", f"{m['surface_ratio_3cm'] * 100:.1f} %"),
        ("bbox ratio x/y/z", f"{r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}"),
    ]
    ax.set_title("alignment metrics", fontsize=9)
    for i, (k, v) in enumerate(lines):
        y = 0.92 - i * 0.098
        ax.text(0.02, y, k, fontsize=8.4, color="#4a5563", va="center")
        ax.text(0.98, y, v, fontsize=8.8, color="#1d2733", va="center", ha="right",
                fontfamily="monospace")


def load_pose_variant(tag: str, suffix: str):
    """只读其它尺度档的缓存（不触发服务请求），用于取 limb 修正前的网格。"""
    path = rsp.CACHE_DIR / f"{tag}_pose_vitpose{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_front_snap(g: dict, p: dict, tag: str):
    """C 方案：以 torso 档缓存为基线，把网格推到 MoGe 实测前表面。

    返回 ``(新的 pose, 基线 pose, snap detail)``；缺少 torso 缓存时原样返回 p。
    只写回 camera_space/world.mesh 的顶点，faces 与 scale_fit 其余字段保持原样，
    所以关掉 env 即回滚，且不和 off/moge/torso/limb 各档结果混在一起。
    """
    # 旋转优先互斥：旧 FRONT_SNAP 开关和直接调用都不得覆盖旋转输出。
    if rsp.pose_cache_suffix().startswith("_rotationfit") or p.get("rotation_fit") is not None:
        return p, None, {"enabled": False, "applied": False,
                         "reason": "rotation_only_preserves_native_pose"}
    torso = load_pose_variant(tag, "_scalefit_torso")
    if torso is None:
        return p, None, {"enabled": True, "applied": False,
                         "reason": "no_torso_cache"}
    base_cam = np.asarray((torso.get("camera_space") or {}).get("vertices") or [],
                          dtype=np.float64).reshape(-1, 3)
    world = torso.get("world") or {}
    base_world = np.asarray(((world.get("mesh") or {}).get("vertices")) or [],
                            dtype=np.float64).reshape(-1, 3)

    depth = decode_depth(g)
    mask = decode_person_mask(g)
    new_cam, info = fsnap.front_snap_vertices(base_cam, depth, float(g["focal_px"]), mask)
    if not info.get("applied"):
        return p, torso, info

    pose = copy.deepcopy(p)
    cam_blk = pose.setdefault("camera_space", {})
    cam_blk["vertices"] = new_cam.tolist()
    # 世界系只叠加相机系位移（rot 旋转后），沿用基线自己的平移/落地量
    if (len(base_world) == len(base_cam) and len(base_cam)
            and (pose.get("world") or {}).get("status") == "ok"):
        rot = np.asarray((pose["world"].get("rotation") or
                          (torso.get("world") or {}).get("rotation")), dtype=np.float64)
        pose["world"].setdefault("mesh", {})["vertices"] = (
            base_world + (new_cam - base_cam) @ rot.T).tolist()
    fit = dict(pose.get("scale_fit") or {})
    fit["snap"] = info
    pose["scale_fit"] = fit
    return pose, torso, info


def apply_bone_snap(g: dict, p: dict, tag: str):
    """C2 方案：以 torso 档缓存为基线，修正 COCO-17 关节，网格经蒙皮跟随。

    返回 ``(新的 pose, 基线 pose, bone snap detail)``；缺少 torso 缓存时原样返回 p。
    只写回 camera_space/world 的 vertices 与 keypoints_3d_coco17，faces 与
    scale_fit 其余字段保持原样，所以关掉 env 即回滚。
    """
    if rsp.pose_cache_suffix().startswith("_rotationfit") or p.get("rotation_fit") is not None:
        return p, None, {"enabled": False, "applied": False,
                         "reason": "rotation_only_preserves_native_pose"}
    torso = load_pose_variant(tag, "_scalefit_torso")
    if torso is None:
        return p, None, {"enabled": True, "applied": False,
                         "reason": "no_torso_cache"}
    return bsnap.apply_pose_bone_snap(
        p, torso, decode_depth(g), decode_person_mask(g), float(g["focal_px"]))


def hist_compare_panel(ax, dist_before: np.ndarray, dist_after: np.ndarray) -> None:
    """修正前/后残差直方图叠放，直观看出整体左移（更贴合）。"""
    both = np.concatenate([d for d in (dist_before, dist_after) if len(d)])
    hi = max(0.06, float(np.percentile(both, 99))) if len(both) else 0.20
    bins = np.linspace(0.0, hi, 60)
    if len(dist_before):
        ax.hist(dist_before, bins=bins, color="#8a94a6", alpha=0.62,
                label=f"before  med {np.median(dist_before)*100:.2f} cm")
    if len(dist_after):
        ax.hist(dist_after, bins=bins, color="#21918c", alpha=0.62,
                label=f"after   med {np.median(dist_after)*100:.2f} cm")
    ax.axvline(0.03, color="#b3402f", lw=1.2, linestyle=(0, (4, 3)),
               label="3 cm 验收线")
    ax.set_xlabel("point → mesh nearest distance (m)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e6eaf0", lw=0.6, alpha=0.9)
    ax.legend(fontsize=7.2, frameon=False, loc="upper right")
    ax.set_title("alignment residual: before vs after", fontsize=9)


def metrics_compare_panel(ax, m_before: dict, m_after: dict, grounded: bool,
                          landing_m: float, adj: dict,
                          shift_m: np.ndarray | None = None,
                          dist_before: np.ndarray | None = None,
                          dist_after: np.ndarray | None = None) -> None:
    """修正前后指标并排 + 差值（nn 越小越好、3cm 内表面越大越好）。"""
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    mode = str(adj.get("mode") or "limb")
    ax.set_title(f"alignment metrics  BEFORE (torso) → AFTER ({mode})", fontsize=9)
    if not m_before or not m_after:
        ax.text(0.02, 0.5, "before/after metrics unavailable", fontsize=9, va="center")
        return
    xb, xa, xd = 0.44, 0.76, 1.00
    ax.text(xb, 0.995, "before", fontsize=8.0, color="#6b7280", ha="right", va="center")
    ax.text(xa, 0.995, "after", fontsize=8.0, color="#1d2733", ha="right", va="center")
    ax.text(xd, 0.995, "delta", fontsize=8.0, color="#4a5563", ha="right", va="center")

    def row(i, label, before, after, delta="", color="#4a5563"):
        y = 0.905 - i * 0.098
        ax.text(0.02, y, label, fontsize=8.2, color="#4a5563", va="center")
        ax.text(xb, y, before, fontsize=8.5, color="#6b7280", va="center",
                ha="right", fontfamily="monospace")
        ax.text(xa, y, after, fontsize=8.8, color="#1d2733", va="center",
                ha="right", fontfamily="monospace")
        if delta:
            ax.text(1.0, y, delta, fontsize=8.3, color=color, va="center",
                    ha="right", fontfamily="monospace")

    rb, ra = m_before["bbox_ratio_xyz"], m_after["bbox_ratio_xyz"]
    d_nn = (m_after["nn_median_m"] - m_before["nn_median_m"]) * 100.0
    d_p90 = (m_after["nn_p90_m"] - m_before["nn_p90_m"]) * 100.0
    d_surf = (m_after["surface_ratio_3cm"] - m_before["surface_ratio_3cm"]) * 100.0
    d_z = m_after["mesh_min_z_m"] - m_before["mesh_min_z_m"]

    def good(value, lower_better=True):
        if abs(value) < 1e-6:
            return "#4a5563"
        return "#1c7c54" if (value < 0) == lower_better else "#b3402f"

    rows = [
        ("frame", "ground_world_v1" if grounded else "camera_v1",
         "ground_world_v1" if grounded else "camera_v1", "", "#4a5563"),
        ("person points", f"{m_before['pc_points']}", f"{m_after['pc_points']}", "", "#4a5563"),
        ("mesh landing offset (info)", f"{landing_m:+.4f}", f"{landing_m:+.4f}", "", "#4a5563"),
        ("mesh min z (P0.5)", f"{m_before['mesh_min_z_m']:+.4f}",
         f"{m_after['mesh_min_z_m']:+.4f} m", f"{d_z*100:+.2f} cm", "#4a5563"),
        ("nn median", f"{m_before['nn_median_m']*100:.2f} cm",
         f"{m_after['nn_median_m']*100:.2f} cm", f"{d_nn:+.2f}", good(d_nn)),
        ("nn P90", f"{m_before['nn_p90_m']*100:.2f} cm",
         f"{m_after['nn_p90_m']*100:.2f} cm", f"{d_p90:+.2f}", good(d_p90)),
        ("surface < 3 cm", f"{m_before['surface_ratio_3cm']*100:.1f} %",
         f"{m_after['surface_ratio_3cm']*100:.1f} %", f"{d_surf:+.1f} pp",
         good(d_surf, lower_better=False)),
        ("bbox ratio x/y/z",
         f"{rb[0]:.3f}/{rb[1]:.3f}/{rb[2]:.3f}",
         f"{ra[0]:.3f}/{ra[1]:.3f}/{ra[2]:.3f}", "", "#4a5563"),
    ]
    if m_before.get("joint_nn_median_m") is not None and \
            m_after.get("joint_nn_median_m") is not None:
        d_j = (m_after["joint_nn_median_m"] - m_before["joint_nn_median_m"]) * 100.0
        rows.insert(7, ("joints → pc median",
                        f"{m_before['joint_nn_median_m']*100:.2f} cm",
                        f"{m_after['joint_nn_median_m']*100:.2f} cm",
                        f"{d_j:+.2f}", good(d_j)))
    for i, item in enumerate(rows):
        row(i, *item)

    if adj:
        kmin, kmax = adj.get("k_min"), adj.get("k_max")
        parts = [(f"{mode} k={float(kmin):.3f}~{float(kmax):.3f}  "
                  f"verts={int(adj.get('weighted_vertices') or 0)}  "
                  f"max_shift={float(adj.get('max_shift_m') or 0.0)*100:.2f} cm"
                  if kmin is not None else f"{mode} skipped")]
        if shift_m is not None and len(shift_m):
            parts.append(f"adjusted>{SHIFT_LO*100:.0f}cm "
                         f"{np.mean(np.asarray(shift_m) > SHIFT_LO)*100:.0f}%")
        if (dist_before is not None and dist_after is not None
                and len(dist_before) and len(dist_after)):
            closer = np.mean(np.asarray(dist_after) < np.asarray(dist_before) - 1e-4) * 100.0
            parts.append(f"pts closer {closer:.0f}% / farther {100.0-closer:.0f}%")
        ax.text(0.02, 0.035, "  |  ".join(parts), fontsize=7.8, color="#1d2733",
                va="center")


# ------------------------------------------------------------------ 主渲染
def raw_joints(pose: dict | None, grounded: bool, rot: np.ndarray,
               tvec: np.ndarray) -> np.ndarray:
    """取该 pose 的 COCO-17 关节（世界系；无地面帧为相机系），不做绘图坐标变换。"""
    if not pose:
        return np.zeros((0, 3))
    world = pose.get("world") or {}
    kp = np.asarray((world.get("mesh") or {}).get("keypoints_3d_coco17") or [],
                    dtype=np.float64).reshape(-1, 3)
    if grounded and len(kp):
        return kp
    cam = np.asarray((pose.get("camera_space") or {}).get("keypoints_3d_coco17") or [],
                     dtype=np.float64).reshape(-1, 3)
    if not len(cam):
        return kp
    return cam @ rot.T + tvec if grounded else cam


def render(g: dict, p: dict, image_path: str, stem: str, out_path: str,
           dpi: int = 150, before_pose: dict | None = None,
           skeleton_fbx: str | None = None,
           mesh_rig_fbx: str | None = None,
           mhr_pose_fbx: str | None = None) -> dict:
    world = p.get("world") or {}
    grounded = (world.get("status") == "ok"
                and bool((world.get("mesh") or {}).get("vertices")))
    mesh_block = (world.get("mesh") or {}) if grounded else (p.get("camera_space") or {})
    verts = np.asarray(mesh_block.get("vertices") or [], dtype=np.float64)
    faces = np.asarray(mesh_block.get("faces") or [], dtype=np.int64)

    depth = decode_depth(g)
    dh, dw = depth.shape
    focal = float(g["focal_px"])
    cam_meta = p.get("camera") or {}

    landing_m = 0.0
    if grounded:
        rot = np.asarray(world["rotation"], dtype=np.float64)
        tvec = np.asarray(world["camera_position"], dtype=np.float64)
        landing_m = float((world.get("ground_snap") or {}).get("offset_m") or 0.0)
        mesh_disp = rsp.to_display(verts, True)
        cam_disp = tvec
        dir_matrix = rot
        z_plane = 0.0
    else:
        rot = np.eye(3)
        tvec = np.zeros(3)
        mesh_disp = rsp.to_display(verts, False)
        cam_disp = np.zeros(3)
        dir_matrix = rsp.M_CAM
        z_plane = float(np.percentile(mesh_disp[:, 2], 0.5)) if len(mesh_disp) else 0.0

    mask = decode_person_mask(g)
    if mask is not None:
        pts_disp, zcam = person_point_cloud(depth, focal, mask, grounded, rot, tvec)
        # 相机视角抽稀一点，避免点云把原图与网格边界完全盖住
        pc_cam, zcam_cam = person_camera_points(depth, focal, mask, max_pts=8000)
    else:
        pts_disp, pc_cam, zcam, zcam_cam = (np.zeros((0, 3)), np.zeros((0, 3)),
                                            np.zeros(0), np.zeros(0))

    metrics = align_metrics(mesh_disp, pts_disp)

    # limb / snap / bone 修正前网格：torso 档缓存，与当前结果同 factor、同 cam_t，
    # 仅缺肢体修正本身。用当前世界的 rot/tvec 投到同一坐标系，保证前后只差修正量。
    limb = (p.get("scale_fit") or {}).get("limb") or {}
    snap = (p.get("scale_fit") or {}).get("snap") or {}
    bone = (p.get("scale_fit") or {}).get("bonesnap") or {}
    adj = (bone if bone.get("applied") else
           snap if snap.get("applied") else
           limb if limb.get("gate") == "ok" else {})
    adj_mode = str(adj.get("mode") or "limb")
    before_cam = np.zeros((0, 3))
    if before_pose is not None and adj:
        cand = np.asarray((before_pose.get("camera_space") or {}).get("vertices") or [],
                          dtype=np.float64).reshape(-1, 3)
        if len(cand) and len(cand) == len(verts):
            before_cam = cand
    compare = len(before_cam) > 0

    # 骨架：after 用当前 pose，before 用 torso 基线；统一转到绘图坐标
    def _joints_disp(pose: dict, block: dict) -> np.ndarray:
        kp_cam = np.asarray((pose.get("camera_space") or {}).get("keypoints_3d_coco17")
                            or [], dtype=np.float64).reshape(-1, 3)
        kp = np.asarray(block.get("keypoints_3d_coco17") or [],
                        dtype=np.float64).reshape(-1, 3)
        if grounded:
            if not len(kp) and len(kp_cam):
                kp = kp_cam @ rot.T + tvec
            return rsp.to_display(kp, True) if len(kp) else kp
        return rsp.to_display(kp_cam, False) if len(kp_cam) else kp_cam

    joints_after = (_joints_disp(p, mesh_block) if bone.get("applied")
                    else np.zeros((0, 3)))
    joints_before = (_joints_disp(before_pose, mesh_block)
                     if before_pose is not None and compare and bone.get("applied")
                     else np.zeros((0, 3)))
    joints_shift = np.zeros(0)
    if len(joints_before) and len(joints_before) == len(joints_after):
        joints_shift = np.linalg.norm(joints_after - joints_before, axis=1)

    # 可选：把骨架导出成 ASCII FBX（Blender / Maya / 3ds Max 可旋转检视）。
    # after = 当前（含骨骼吸附）结果；before = torso 基线，仅吸附生效时输出。
    if skeleton_fbx:
        fb_dir = pathlib.Path(skeleton_fbx)
        frame_name = "ground_world_v1" if grounded else "camera_v1"
        after_kp = raw_joints(p, grounded, rot, tvec)
        if len(after_kp) >= skelfbx.COCO17_MIN:
            got = skelfbx.write_skeleton(fb_dir / f"{stem}_skeleton.fbx", after_kp,
                                         object_name="SAM3D_COCO17_after",
                                         note=f"{frame_name}, bone snap after",
                                         frame=frame_name)
            print(f"  skeleton fbx: {os.path.basename(got['path'])} "
                  f"frame={frame_name} bones={got['bones']} "
                  f"height={got['height_cm']:.1f} cm feet_z={got['feet_z_cm']:.1f} cm")
        if bone.get("applied") and before_pose is not None:
            before_kp = raw_joints(before_pose, grounded, rot, tvec)
            if len(before_kp) >= skelfbx.COCO17_MIN:
                got = skelfbx.write_skeleton(fb_dir / f"{stem}_skeleton_before.fbx",
                                             before_kp,
                                             object_name="SAM3D_COCO17_before",
                                             note=f"{frame_name}, torso baseline",
                                             frame=frame_name)
                print(f"  skeleton fbx: {os.path.basename(got['path'])} "
                      f"bones={got['bones']} height={got['height_cm']:.1f} cm")

    # 可选：SAM-3D 网格 + MHR-lod5 骨架（当前姿势）写进单个 FBX；
    # 或把同一份姿势写回官方 MHR-lod5.tpose.fbx（保留其原模型 / 蒙皮结构 / 形态键）。
    mesh_rig = None
    mhr_pose = None
    if mesh_rig_fbx or mhr_pose_fbx:
        # 骨骼吸附时导出的网格来自 torso 基线的世界顶点，关节也必须取同一基线，
        # 否则骨架会沿用当前 pose 的世界尺度，与网格整体错开十几厘米。
        src_world = (world if not bone.get("applied") or before_pose is None
                     else (before_pose.get("world") or world))
        gp = src_world.get("generic_pose") or {}
        joint_coords = np.asarray(gp.get("pred_joint_coords") or [],
                                  dtype=np.float64).reshape(-1, 3)
        if not len(joint_coords) and src_world is not world:
            joint_coords = np.asarray((world.get("generic_pose") or {})
                                      .get("pred_joint_coords") or [],
                                      dtype=np.float64).reshape(-1, 3)
        if not grounded or not len(joint_coords):
            print("  [warn] mesh rig fbx 跳过：需要 ground_world_v1 的 world.generic_pose")
        else:
            # 骨骼吸附只改 COCO-17 关节；这里用同一份关节位移场把 MHR-127 关节
            # 一起带动，得到「吸附调整后」的骨架姿势。
            # 网格刻意保持吸附前的原始顶点（静态、不做任何空间变形），骨架单独
            # 摆到调整后的位置，导入 3ds Max 后即可直接看出骨骼相对网格挪了多少。
            note = ("ground_world_v1, static SAM mesh (pre-snap) + "
                    "MHR-lod5 skeleton (snapped pose)")
            j_out = joint_coords
            mesh_out = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
            base_w = np.asarray(((before_pose or {}).get("world") or {})
                                .get("keypoints_3d_coco17") or [],
                                dtype=np.float64).reshape(-1, 3)
            new_w = np.asarray(world.get("keypoints_3d_coco17") or [],
                               dtype=np.float64).reshape(-1, 3)
            if bone.get("applied") and len(base_w) == len(new_w) and len(base_w):
                dz = float(bone.get("landing_fix_m") or 0.0)
                # 静态网格换成吸附前的基线顶点；落地修正的整体平移照样补上，
                # 保证网格与调整后的骨架处在同一高度。
                base_verts = np.asarray(
                    (((before_pose or {}).get("world") or {}).get("mesh") or {})
                    .get("vertices") or [], dtype=np.float64).reshape(-1, 3)
                if len(base_verts) == len(mesh_out):
                    mesh_out = base_verts + (np.array([0.0, 0.0, dz]) if dz else 0.0)
                if dz:
                    new_w = new_w - np.array([0.0, 0.0, dz])
                j_out = bsnap.deform_points(joint_coords, base_w, new_w)
                if dz:
                    j_out = j_out + np.array([0.0, 0.0, dz])
                # DQS 混合对相邻关节权重不同，等价于各关节套了不同仿射：短骨会被
                # 放大/压扁（近相机缩小、远相机放大）。按原始骨长逐骨回填方向，
                # 只保长度、不改姿态方向。
                j_out = mhrmesh.preserve_bone_lengths(
                    j_out, mhrmesh.get_rig(), reference=joint_coords)
                note = ("ground_world_v1, static SAM mesh (pre-snap) + "
                        "MHR-lod5 skeleton (snapped pose, bone lengths restored)")
            if mesh_rig_fbx:
                got = mhrmesh.write_mesh_rig(
                    pathlib.Path(mesh_rig_fbx) / f"{stem}_mesh_rig.fbx",
                    mesh_out, faces, j_out,
                    object_name=f"SAM3D_MHR_{stem}", note=note)
                print(f"  mesh rig fbx: {os.path.basename(got['path'])} "
                      f"bones={got['bones']} verts={got['verts']} "
                      f"height={got['height_cm']:.1f} cm {got['bytes']/1e6:.1f} MB")
                mesh_rig = {k: got[k] for k in
                            ("path", "bones", "verts", "faces", "height_cm", "bytes")}
                mesh_rig["note"] = note
            if mhr_pose_fbx:
                # 官方资产原地改姿势：只写 126 个 LimbNode 的局部变换，
                # Geometry / Deformer / BlendShape / 连接关系全部原样保留。
                got = mhrpose.export_pose(
                    mhrpose.DEFAULT_TEMPLATE,
                    pathlib.Path(mhr_pose_fbx) / f"{stem}_mhr_pose.fbx",
                    j_out, rig=mhrmesh.get_rig(), note=note,
                    targets_json=pathlib.Path(mhr_pose_fbx) / f"{stem}_pose_targets.json")
                print(f"  mhr pose fbx: {os.path.basename(got['path'])} "
                      f"bones={got['bones']} scale={got['scale']:.3f} "
                      f"err={got['verify_max_err_cm']*10:.3f} mm "
                      f"{got['bytes']/1e6:.1f} MB")
                mhr_pose = {k: got[k] for k in
                            ("path", "bones", "bytes", "scale", "offset_cm",
                             "verify_max_err_cm", "verify_median_err_cm")}
    cam_kp_after = (np.asarray((p.get("camera_space") or {}).get("keypoints_3d_coco17")
                               or [], dtype=np.float64).reshape(-1, 3)
                    if bone.get("applied") else np.zeros((0, 3)))
    cam_kp_before = (np.asarray((before_pose.get("camera_space") or {})
                                .get("keypoints_3d_coco17") or [],
                                dtype=np.float64).reshape(-1, 3)
                     if before_pose is not None and compare and bone.get("applied")
                     else np.zeros((0, 3)))

    before_shift = 0.0
    if compare:
        before_disp = before_cam @ rot.T + tvec if grounded else rsp.to_display(before_cam, False)
        if grounded and bone.get("applied"):
            # 骨架修正模式：before 直接用基线自身世界顶点（已含落地量），
            # 否则会比 after 少一个落地偏移，前后差被高估。
            bw = np.asarray(((before_pose.get("world") or {}).get("mesh") or {})
                            .get("vertices") or [], dtype=np.float64).reshape(-1, 3)
            if len(bw) == len(verts):
                before_disp = rsp.to_display(bw, True)
        m_before = align_metrics(before_disp, pts_disp)
        shift_disp = np.linalg.norm(before_disp - mesh_disp, axis=1)
        before_shift = float(shift_disp.max())
    else:
        before_disp, m_before, shift_disp = np.zeros((0, 3)), {}, np.zeros(0)

    ref = np.vstack([a for a in (mesh_disp, pts_disp) if len(a)] + [cam_disp[None, :]])
    # 三视图窗口同时纳入点云，避免点云被裁到画面外
    wins = rsp._ortho_windows(mesh_disp, pts_disp, grounded=grounded,
                              z_plane=z_plane)

    # 相机视角用未做地面变换的 camera_v1 几何，才能与原图成像严格重合
    base_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    ih, iw = base_rgb.shape[:2]
    f_img = float(cam_meta.get("focal_px") or focal) * iw / float(cam_meta.get("image_width") or iw)
    cam_blk = p.get("camera_space") or {}
    ov_verts = np.asarray(cam_blk.get("vertices") or [], dtype=np.float64)
    ov_faces = np.asarray(cam_blk.get("faces") or [], dtype=np.int64)
    if not len(ov_verts):
        ov_verts, ov_faces = (verts - tvec) @ rot if grounded else verts, faces
    # 相机系位移：用于相机视角把被修正的区域涂红（与模型空间位移等价）
    shift_cam = (np.linalg.norm(ov_verts - before_cam, axis=1)
                 if compare and len(ov_verts) == len(before_cam) else shift_disp)

    if grounded:
        labels = (("world x-z (front)", "x right (m)", "z up (m)"),
                  ("world y-z (side)", "y depth (m)", "z up (m)"),
                  ("world x-y (top)", "x right (m)", "y depth (m)"))
    else:
        labels = (("camera x-up (front)", "x right (m)", "-y up (m)"),
                  ("camera depth-up (side)", "z depth (m)", "-y up (m)"),
                  ("camera x-depth (top)", "x right (m)", "z depth (m)"))

    frame_txt = "world" if grounded else "camera"
    win3 = rsp._fit3(ref, pad=0.15)
    win3[2] = (min(win3[2][0], z_plane - 0.05), win3[2][1])

    def _sub(arr, n):
        return arr if len(arr) <= n else arr[np.linspace(0, len(arr) - 1, n).astype(int)]

    if compare:
        # 3x4 对比版面：同一窗口、同一采样点云，只有网格换 before/after
        fig = plt.figure(figsize=(22.0, 15.2), dpi=dpi)
        gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.02],
                              left=0.030, right=0.982, top=0.878, bottom=0.056,
                              width_ratios=[1.15, 1.0, 1.0, 1.0],
                              wspace=0.14, hspace=0.20)
        ax_cb = fig.add_subplot(gs[0, 0])
        ax_ca = fig.add_subplot(gs[0, 1])
        ax_fb = fig.add_subplot(gs[0, 2])
        ax_fa = fig.add_subplot(gs[0, 3])
        ax_sb = fig.add_subplot(gs[1, 0])
        ax_sa = fig.add_subplot(gs[1, 1])
        ax_tb = fig.add_subplot(gs[1, 2])
        ax_ta = fig.add_subplot(gs[1, 3])
        ax_3d = fig.add_subplot(gs[2, 0], projection="3d")
        ax_m = fig.add_subplot(gs[2, 1:3])
        ax_h = fig.add_subplot(gs[2, 3])

        k_txt = (f"{adj_mode} k={float(adj['k_min']):.3f}~{float(adj['k_max']):.3f}"
                 if adj.get("k_min") is not None else f"{adj_mode} n/a")
        camera_view_panel(ax_cb, base_rgb, pc_cam, zcam_cam, before_cam, ov_faces, f_img,
                          f"BEFORE · camera view (torso fit)\n"
                          f"{len(ov_faces)} faces, focal={f_img:.0f} px",
                          joints_cam=cam_kp_before)
        camera_view_panel(ax_ca, base_rgb, pc_cam, zcam_cam, ov_verts, ov_faces, f_img,
                          f"AFTER · camera view ({adj_mode} corrected)\n{k_txt}"
                          f"  |  red = adjusted > {SHIFT_LO*100:.0f} cm",
                          shift=shift_cam, joints_cam=cam_kp_after,
                          joints_shift=joints_shift)
        for (ax_b, ax_a), ((ix, iy), win, lab) in zip(
                ((ax_fb, ax_fa), (ax_sb, ax_sa), (ax_tb, ax_ta)),
                (((0, 2), wins[0], labels[0]), ((1, 2), wins[1], labels[1]),
                 ((0, 1), wins[2], labels[2]))):
            ortho_overlay(ax_b, before_disp, faces, pts_disp, zcam, ix, iy, win,
                          grounded, z_plane, f"BEFORE · {lab[0]}", lab[1], lab[2],
                          joints=joints_before)
            ortho_overlay(ax_a, mesh_disp, faces, pts_disp, zcam, ix, iy, win,
                          grounded, z_plane, f"AFTER · {lab[0]}", lab[1], lab[2],
                          shift=shift_disp, joints=joints_after,
                          joints_shift=joints_shift)

        overlay3d(ax_3d, mesh_disp, faces, pts_disp, zcam, cam_disp, dir_matrix,
                  int(cam_meta.get("image_width") or dw),
                  int(cam_meta.get("image_height") or dh), f_img, z_plane, win3,
                  f"{frame_txt} view (after): point cloud + translucent SAM-3D mesh",
                  shift=shift_disp, joints=joints_after, joints_shift=joints_shift)

        qry = _sub(pts_disp, 2500)
        dist_after = nn_distance(qry, _sub(mesh_disp, 8000)) if len(qry) else np.zeros(0)
        dist_before = nn_distance(qry, _sub(before_disp, 8000)) if len(qry) else np.zeros(0)
        pc_ref = _sub(pts_disp, 8000)
        if len(joints_after) and len(pc_ref):
            metrics["joint_nn_median_m"] = float(np.median(nn_distance(joints_after, pc_ref)))
        if len(joints_before) and len(pc_ref):
            m_before["joint_nn_median_m"] = float(np.median(nn_distance(joints_before, pc_ref)))
        hist_compare_panel(ax_h, dist_before, dist_after)
        metrics_compare_panel(ax_m, m_before, metrics, grounded, landing_m, adj,
                              shift_m=shift_disp, dist_before=dist_before,
                              dist_after=dist_after)
    else:
        fig = plt.figure(figsize=(22.0, 11.0), dpi=dpi)
        gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05],
                              left=0.030, right=0.982, top=0.902, bottom=0.070,
                              width_ratios=[1.15, 1.0, 1.0, 1.0],
                              wspace=0.14, hspace=0.16)
        ax_c = fig.add_subplot(gs[0, 0])
        ax_f = fig.add_subplot(gs[0, 1])
        ax_s = fig.add_subplot(gs[0, 2])
        ax_t = fig.add_subplot(gs[0, 3])
        ax_3d = fig.add_subplot(gs[1, 0], projection="3d")
        ax_h = fig.add_subplot(gs[1, 1:3])
        ax_m = fig.add_subplot(gs[1, 3])

        cam_title = (f"camera view: image + MoGe-2 points + SAM-3D mesh\n"
                     f"{len(ov_faces)} faces, focal={f_img:.0f} px")
        camera_view_panel(ax_c, base_rgb, pc_cam, zcam_cam, ov_verts, ov_faces, f_img,
                          cam_title, joints_cam=cam_kp_after)

        for ax, (ix, iy), win, lab in zip((ax_f, ax_s, ax_t), ((0, 2), (1, 2), (0, 1)),
                                          wins, labels):
            ortho_overlay(ax, mesh_disp, faces, pts_disp, zcam, ix, iy, win,
                          grounded, z_plane, lab[0], lab[1], lab[2],
                          joints=joints_after)

        overlay3d(ax_3d, mesh_disp, faces, pts_disp, zcam, cam_disp, dir_matrix,
                  int(cam_meta.get("image_width") or dw),
                  int(cam_meta.get("image_height") or dh), f_img, z_plane, win3,
                  f"{frame_txt} view: point cloud + translucent SAM-3D mesh",
                  joints=joints_after)

        qry = _sub(pts_disp, 2500)
        dist = nn_distance(qry, _sub(mesh_disp, 8000)) if len(qry) else np.zeros(0)
        hist_panel(ax_h, dist, metrics)
        metrics_panel(ax_m, metrics, grounded, landing_m)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#21918c",
               markersize=5, label="MoGe-2 person point cloud (camera-depth colored)"),
        Patch(facecolor=(0.31, 0.765, 0.91, 0.34), edgecolor=(0.05, 0.26, 0.34),
              label="SAM-3D mesh (translucent)"),
        Line2D([0], [0], color="red", lw=2.0,
               label="ground plane z=0 (red grid)" if grounded
               else "foot-level reference plane (red grid)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#111111",
               markersize=6, label="camera + frustum"),
    ]
    if compare:
        handles.insert(2, Patch(facecolor=(0.86, 0.24, 0.19, 0.55),
                                edgecolor=(0.45, 0.10, 0.08),
                                label=f"{adj_mode}-adjusted region (shift > "
                                      f"{SHIFT_LO*100:.0f} cm, red)"))
    if len(joints_after):
        handles.append(Line2D([0], [0], marker="o", color="#1f2d3d", lw=1.6,
                              markersize=5,
                              label="COCO-17 skeleton (after adjustment)"))
        if len(joints_shift) and np.any(joints_shift > SHIFT_LO):
            handles.append(Line2D([0], [0], marker="o", color="none",
                                  markerfacecolor="none",
                                  markeredgecolor=C_ADJUSTED_RGB, markersize=7,
                                  label="joint moved by depth fix (red ring)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8.6)
    if compare:
        if adj_mode == "limb":
            after_txt = "limb depth corrected (same uniform factor & cam_t)"
        elif adj_mode == "bone snap":
            after_txt = ("COCO-17 joints pushed to MoGe depth, mesh skinned to bones (LBS)")
        else:
            after_txt = f"{adj_mode}: vertices pushed along camera rays to MoGe depth"
        fig.text(0.5, 0.031,
                 f"BEFORE = torso fit (uncorrected, from _scalefit_torso cache)   |   "
                 f"AFTER = {after_txt}",
                 ha="center", fontsize=8.4, color="#4a5566")

    ground_tag = (f"ground=ok ({world.get('source') or 'inferred'}), "
                  f"mesh landing={landing_m:+.4f} m") if grounded else "ground=rejected"
    if metrics and m_before:
        nn_txt = (f"nn median {m_before['nn_median_m']*100:.2f} -> "
                  f"{metrics['nn_median_m']*100:.2f} cm  "
                  f"<3cm {m_before['surface_ratio_3cm']*100:.1f} -> "
                  f"{metrics['surface_ratio_3cm']*100:.1f} %")
    elif metrics:
        nn_txt = f"nn median={metrics['nn_median_m']*100:.2f} cm"
    else:
        nn_txt = "nn n/a"
    # 对比版面多一行指标摘要，标题区整体上移、绘图区下压，避免与面板标题重叠
    y_title, y_fit, y_limb = ((0.9865, 0.9545, 0.9375) if compare
                              else (0.968, 0.9325, 0.9155))
    fig.suptitle(
        f"{stem}  |  {'frame=ground_world_v1' if grounded else 'frame=camera_v1'}  |  "
        f"{ground_tag}  |  SAM-3D {len(verts)} verts  |  {nn_txt}  |  "
        f"image {int(cam_meta.get('image_width') or dw)}x{int(cam_meta.get('image_height') or dh)}",
        fontsize=9.5, y=y_title,
    )
    if not grounded and world.get("error"):
        fig.text(0.985, y_fit, f"ground rejected: {world['error']}",
                 fontsize=8.5, color="#b3402f", ha="right")

    fit = p.get("scale_fit") or {}
    if str(fit.get("mode") or "off") != "off":
        parts = [f"scale_fit={fit.get('mode')}  k={float(fit.get('factor') or 1.0):.4f}",
                 f"selected n={int(fit.get('samples') or 0)}"]
        for key in ("torso", "limb", "full"):
            if fit.get(f"ratio_{key}") is not None:
                parts.append(f"{key} {fit[f'ratio_{key}']:.4f} (n={int(fit.get(f'samples_{key}') or 0)})")
        if fit.get("torso_fallback"):
            parts.append(f"fallback={fit['torso_fallback']}")
        fig.text(0.030, y_fit, "  |  ".join(parts), fontsize=8.6, color="#1d2733")

    snap = fit.get("snap") or {}
    limb = fit.get("limb") or {}
    bonesnap = fit.get("bonesnap") or {}
    if bonesnap:
        parts = [f"bone_snap={'applied' if bonesnap.get('applied') else 'skipped'}"]
        if bonesnap.get("k_min") is not None:
            parts.append(f"k={float(bonesnap['k_min']):.3f}~{float(bonesnap['k_max']):.3f}")
        if bonesnap.get("moved_joints") is not None:
            parts.append(f"joints_moved={int(bonesnap['moved_joints'])}")
        if bonesnap.get("joint_shift_max_m") is not None:
            parts.append(f"joint_shift={float(bonesnap['joint_shift_max_m'])*100:.2f} cm")
        if bonesnap.get("weighted_vertices"):
            parts.append(f"verts={int(bonesnap['weighted_vertices'])}")
        if bonesnap.get("max_shift_m") is not None:
            parts.append(f"max_shift={float(bonesnap['max_shift_m'])*100:.2f} cm")
        if bonesnap.get("landing_fix_m"):
            parts.append(f"landing_fix=+{float(bonesnap['landing_fix_m'])*100:.2f} cm")
        if bonesnap.get("leg_damp") is not None:
            parts.append(f"leg_damp={float(bonesnap['leg_damp']):.2f}")
        if bonesnap.get("reason"):
            parts.append(f"skip={bonesnap['reason']}")
        fig.text(0.030, y_limb, "  |  ".join(parts), fontsize=7.6, color="#1d2733")
    elif snap:
        parts = [f"front_snap={'applied' if snap.get('applied') else 'skipped'}"]
        if snap.get("k_min") is not None:
            parts.append(f"k={float(snap['k_min']):.3f}~{float(snap['k_max']):.3f}")
        if snap.get("weighted_vertices"):
            parts.append(f"verts={int(snap['weighted_vertices'])}")
        if snap.get("max_shift_m") is not None:
            parts.append(f"max_shift={float(snap['max_shift_m'])*100:.2f} cm")
        if snap.get("trusted_pct") is not None:
            parts.append(f"trusted_px={float(snap['trusted_pct'])*100:.0f}%")
        if snap.get("reason"):
            parts.append(f"skip={snap['reason']}")
        fig.text(0.030, y_limb, "  |  ".join(parts), fontsize=7.6, color="#1d2733")
    elif limb:
        spread = limb.get("k_spread")
        spread_txt = ("n/a" if spread is None
                      else f"{float(spread):.3f}<={float(limb.get('k_spread_limit') or 0.0):.2f}")
        parts = [f"limb gate={limb.get('gate')} kspread={spread_txt}"
                 f" cover={float(limb.get('gate_iou') or 0.0):.3f}"]
        for bone in limb.get("bones") or []:
            name = str(bone.get("name") or "?")
            n = int(bone.get("n") or 0)
            parts.append(f"{name} {float(bone.get('k') or 1.0):.3f}(n={n})" if bone.get("valid")
                         else f"{name} n/a(n={n})")
        if limb.get("k_min") is not None:
            parts.append(f"k={float(limb['k_min']):.3f}~{float(limb['k_max']):.3f}")
        if limb.get("max_shift_m") is not None:
            parts.append(f"max_shift={float(limb['max_shift_m'])*100:.2f} cm")
        if limb.get("reason"):
            parts.append(f"skip={limb['reason']}")
        fig.text(0.030, y_limb, "  |  ".join(parts), fontsize=7.6, color="#1d2733")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "stem": stem,
        "frame": "ground_world_v1" if grounded else "camera_v1",
        "ground_status": world.get("status"),
        "ground_error": world.get("error"),
        "mesh_landing_offset_m": landing_m,
        "mesh_verts": int(len(verts)),
        "mesh_faces": int(len(faces)),
        "focal_moge2_px": focal,
        "focal_sam3d_px": float(cam_meta.get("focal_px") or 0.0),
        "image_hw": [int(dh), int(dw)],
        "scale_fit": p.get("scale_fit"),
        "limb_gate": limb.get("gate"),
        "front_snap": snap or None,
        "bone_snap": bonesnap or None,
        "mesh_rig": mesh_rig,
        "mhr_pose": mhr_pose,
        "joint_shift_cm": (float(np.max(joints_shift)) * 100
                           if len(joints_shift) else None),
        "before_metrics": m_before or None,
        "limb_delta": (
            {"nn_median_cm": [m_before["nn_median_m"] * 100, metrics["nn_median_m"] * 100],
             "nn_p90_cm": [m_before["nn_p90_m"] * 100, metrics["nn_p90_m"] * 100],
             "surface_3cm_pct": [m_before["surface_ratio_3cm"] * 100,
                                 metrics["surface_ratio_3cm"] * 100],
             "mesh_min_z_m": [m_before["mesh_min_z_m"], metrics["mesh_min_z_m"]],
             "max_vertex_shift_cm": before_shift * 100}
            if compare and m_before and metrics else None),
        "out": out_path,
        **metrics,
    }


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="世界空间网格 × 点云对齐验收面板")
    ap.add_argument("images", nargs="*", help="图片路径，默认 test_image 下 4 张 jpg")
    ap.add_argument("--base", default=rsp.DEFAULT_BASE)
    ap.add_argument("--outdir", default=None,
                    help="默认 <图片目录>/moge2_visualization/world_align")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skeleton-fbx", default=None,
                    help="可选：导出 COCO-17 骨架 FBX 的目录")
    ap.add_argument("--mesh-rig-fbx", default=None,
                    help="可选：导出「SAM-3D 网格 + MHR-lod5 骨架」FBX 的目录")
    ap.add_argument("--mhr-pose-fbx", default=None,
                    help="可选：把调整后的姿势写回 MHR-lod5.tpose.fbx 的输出目录"
                         "（保留原模型，只改骨骼）")
    args = ap.parse_args(argv)

    rsp._load_font()
    rsp.check_health(args.base)

    images = args.images or [
        str(p) for p in sorted(pathlib.Path(rsp.DEFAULT_ROOT).glob("*.jpg"))
        if not p.name.startswith("pose_overlay")]
    stats = []
    for path in images:
        local = rsp.to_local_path(path)
        if not os.path.exists(local):
            print(f"[skip] {path}")
            continue
        stem = pathlib.Path(local).stem
        outdir = args.outdir or os.path.join(os.path.dirname(local),
                                             "moge2_visualization", "world_align")
        os.makedirs(outdir, exist_ok=True)
        print(f"[{stem}] ...")
        g, p, tag = rsp.fetch_case(args.base, local, not args.no_cache)
        rotation_only = rsp.pose_cache_suffix().startswith("_rotationfit") or p.get("rotation_fit") is not None
        if rotation_only:
            # 不读取旧 torso/limb 基线，也不执行逐顶点前表面贴合。
            before = None
        elif bsnap.bone_snap_enabled():
            # C2 方案：以 torso 缓存为基线修正 COCO-17 关节，网格经骨骼蒙皮跟随
            p, before, info = apply_bone_snap(g, p, tag)
            if before is None:
                print(f"  [warn] {stem}: 缺少 torso 档缓存，bone snap 跳过")
            elif not info.get("applied"):
                print(f"  [warn] {stem}: bone snap 未生效（{info.get('reason')}）")
            else:
                print(f"  bone snap: joints_moved={info.get('moved_joints')} "
                      f"joint_shift={float(info.get('joint_shift_max_m') or 0.0)*100:.2f} cm "
                      f"verts={info['weighted_vertices']} "
                      f"max_shift={info['max_shift_m']*100:.2f} cm")
        elif fsnap.front_snap_enabled():
            # C 方案：以 torso 缓存为基线做前表面贴合，before = torso
            p, before, info = apply_front_snap(g, p, tag)
            if before is None:
                print(f"  [warn] {stem}: 缺少 torso 档缓存，front snap 跳过")
            elif not info.get("applied"):
                print(f"  [warn] {stem}: front snap 未生效（{info.get('reason')}）")
            else:
                print(f"  front snap: verts={info['weighted_vertices']} "
                      f"max_shift={info['max_shift_m']*100:.2f} cm")
        else:
            limb = (p.get("scale_fit") or {}).get("limb") or {}
            before = None
            if limb.get("gate") == "ok":
                before = load_pose_variant(tag, "_scalefit_torso")
                if before is None:
                    print(f"  [warn] {stem}: 缺少 torso 档缓存（{tag}_pose_vitpose_scalefit_torso.json），"
                          f"本次只输出修正后单版版面")
        out = os.path.join(outdir, f"{stem}_world_align.png")
        st = render(g, p, local, stem, out, dpi=args.dpi, before_pose=before,
                    skeleton_fbx=args.skeleton_fbx,
                    mesh_rig_fbx=args.mesh_rig_fbx,
                    mhr_pose_fbx=args.mhr_pose_fbx)
        stats.append(st)
        print(f"[{stem}] {json.dumps(st, ensure_ascii=False)}")

    if stats:
        summary = os.path.join(os.path.dirname(os.path.abspath(stats[0]["out"])),
                               "world_align_summary.json")
        pathlib.Path(summary).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
