"""C2 骨骼吸附（BONE_SNAP legacy）前后对比可视化 —— 离线、可复跑、不导出 FBX。

读 ``/tmp/phy_sic_moge2_scene_cache`` 里已缓存的 ``ground`` / ``pose`` 响应，在本地重跑
``render_world_align.apply_bone_snap``，把「吸附前（torso 档基线）」与「吸附后」的
COCO-17 骨架同时画到原图与点云上，用于人工核对吸附方向与幅度。全程不请求服务，
也不需要导任何 FBX。

为什么图里只有 17 个关节、看不到手掌/脚掌：
  COCO-17 只定义 17 个关键点 —— 鼻/双眼/双耳/双肩/双肘/双腕/双髋/双膝/双踝。
  骨架因此止于**手腕**（9/10）和**脚踝**（15/16），本身不含掌骨、指骨、跖骨、趾骨。
  本脚本额外叠加 MHR-127 细骨架（来自 ``assets/mhr`` 的 rest profile 父子表），
  其中手指（thumb/index/middle/ring/pinky）与足部（foot/talocrural/subtalar/
  transversetarsal/ball）链条用橙色高亮，即可看到 17 关节之外的手掌与脚掌骨骼。
  注意：COCO-17 吸附只改 17 个关节（射线缩放→LBS 蒙皮），掌指/跖趾顶点只持有
  腕/踝权重，只能整体刚性平移。加 ``--fine-snap`` 才会启用 ``fine_bone_snap``
  的细骨补充级（默认关），此时细骨架画的是吸附后的 ``keypoints_3d_fine``。

版面（5 行 × 3 列）：
  第 1 行：原图 + 骨架（BEFORE 红 / AFTER 青绿 / 叠加 + 位移箭头）
  第 2 行：正视图点云 + 骨架（grounded: world x-z；否则 camera x-up）
  第 3 行：侧视图点云 + 骨架（grounded: world y-z；否则 camera z-up）
  第 4 行：**网格 × 点云**（6 个面板：正视 BEFORE/AFTER、侧视 BEFORE/AFTER、
          顶视 BEFORE/AFTER）—— 半透明 SAM-3D 三角面叠在同一份 MoGe-2 点云上。
          AFTER 按逐顶点位移由青色渐变到红色（>=3cm 全红）；位移颜色不是贴合误差。
  第 5 行：网格贴回原图（BEFORE 压暗底图 / AFTER 位移着色 / AFTER + 骨架）
  地面被判 rejected 的图片自动退回 camera_v1，并在地面线处不画 z=0 参考线。

被贴合收益闸门拒绝（``no_fit_gain``）的图片默认仍会把「候选吸附结果」画出来，
标题标注 ``GATE REJECTED``；加 ``--no-candidate`` 则改为 before == after。

用法（WSL，需 matplotlib/cv2/PIL，解释器如 ~/envs/blade310/bin/python）：
  python ground_pipeline/render_bone_snap_compare.py [IMG ...] \
      [--outdir DIR] [--dpi 150] [--no-candidate] [--no-fine] [--fine-snap]

默认处理 ``images/test_image/*.jpg``，输出到 ``<图片目录>/_wr_skel_compare/``：
  <stem>_skeleton_snap_compare.png + skeleton_snap_compare.json

``--refresh-quality`` 本地运行 ViTPose 更新小缓存；之后离线复用真实坐标和分数。
标签 c 属于 ViTPose 2D 检测（紫色叉号），不代表 SAM3D 3D 精度。
v~ 独立估计原始相机视角下邻近网格表面的自遮挡，不是 COCO 标注真值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("POSEEXTRACTOR_SAM3D_BONE_SNAP", "1")
os.environ.setdefault("POSEEXTRACTOR_SAM3D_BONE_SNAP_MODE", "legacy")
os.environ.setdefault("POSEEXTRACTOR_SAM3D_SCALE_FIT", "torso")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_scene_panel as rsp          # noqa: E402
import render_world_align as rwa          # noqa: E402
import fine_bone_snap as fbs              # noqa: E402
import mhr_mesh_fbx as mhrmesh            # noqa: E402
import joint_quality as jq              # noqa: E402

C_BEFORE = "#d1495b"
C_AFTER = "#1a9c8c"
C_HOT = "#b3402f"
C_FINE = "#8d99ae"
C_FINE_EXT = "#e08a2e"
HOT_M = 0.01

# 细骨架里要跳过的非解剖骨（辅助/扭曲骨、牙齿舌头眼球等）
FINE_SKIP = ("_proc", "_null", "twist", "teeth", "tongue", "eye")
# 手掌/脚掌链条关键字（用于橙色高亮）
FINE_EXT = ("wrist", "pinky", "ring", "middle", "index", "thumb",
            "foot", "talocrural", "subtalar", "transversetarsal", "ball")


# ------------------------------------------------------------------ MHR-127 细骨架
def fine_bone_edges() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """MHR-127 的父子骨边（关节索引对）：(全部解剖骨, 手掌/脚掌骨)。"""
    rig = mhrmesh.load_rig_profile()
    idx = rig["index"]
    all_e: list[tuple[int, int]] = []
    ext_e: list[tuple[int, int]] = []
    for name in rig["bone_order"]:
        if any(t in name for t in FINE_SKIP):
            continue
        parent = rig["parents"][name]
        if parent is None or parent not in idx:
            continue
        edge = (idx[parent], idx[name])
        all_e.append(edge)
        if any(t in name for t in FINE_EXT):
            ext_e.append(edge)
    return all_e, ext_e


def fine_joints(pose: dict, base: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(camera 系 127 关节, world 系 127 关节)``。

    优先取 ``apply_bone_snap`` 写回的 ``keypoints_3d_fine``（已做手掌/脚掌吸附的结果，
    见 ``fine_bone_snap``）；没有该字段时回退到原始位姿的坐标系换算。
    """
    cam = np.asarray((pose.get("camera_space") or {}).get("keypoints_3d_fine") or [],
                     dtype=np.float64).reshape(-1, 3)
    wld = np.asarray((pose.get("world") or {}).get("keypoints_3d_fine") or [],
                     dtype=np.float64).reshape(-1, 3)
    if cam.shape[0] >= 127:
        return cam, wld if wld.shape[0] >= 127 else np.zeros((0, 3))
    return fbs.fine_camera_joints(pose, base)


def draw_fine(ax, pts: np.ndarray, edges, ext, ix: int | None = None,
              iy: int | None = None) -> None:
    """细骨架：灰色全部解剖骨 + 橙色手掌/脚掌骨；ix/iy 给定时按轴取二维坐标。"""
    if not len(pts):
        return
    p = pts if ix is None else pts[:, [ix, iy]]

    def seg(edge):
        a, b = edge
        if np.all(np.abs(p[a]) < 1e-9) or np.all(np.abs(p[b]) < 1e-9):
            return
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=C_FINE, lw=0.55,
                alpha=0.55, solid_capstyle="round", zorder=4)
    for e in edges:
        seg(e)
    for a, b in ext:
        if np.all(np.abs(p[a]) < 1e-9) or np.all(np.abs(p[b]) < 1e-9):
            continue
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=C_FINE_EXT, lw=0.95,
                alpha=0.85, solid_capstyle="round", zorder=5)


# ------------------------------------------------------------------ COCO-17 骨架
def kp_of(pose: dict | None, key: str) -> np.ndarray:
    if not pose:
        return np.zeros((0, 3))
    blk = pose.get(key) or {}
    return np.asarray(blk.get("keypoints_3d_coco17") or [], dtype=np.float64).reshape(-1, 3)


def bone_lines(ax, pts: np.ndarray, color: str, lw: float, ls: str, alpha: float = 1.0,
               zorder: int = 6) -> None:
    if not len(pts):
        return
    zero = np.all(np.abs(pts) < 1e-9, axis=1)
    for edges, _ in rsp.BONE_GROUPS:
        for a, b in edges:
            if zero[a] or zero[b]:
                continue
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color,
                    lw=lw, ls=ls, alpha=alpha, solid_capstyle="round", zorder=zorder)


def joints_dots(ax, pts: np.ndarray, color: str, size: float, zorder: int = 8) -> None:
    if not len(pts):
        return
    zero = np.all(np.abs(pts) < 1e-9, axis=1)
    p = pts[~zero]
    if len(p):
        ax.scatter(p[:, 0], p[:, 1], s=size, c=color, edgecolors="white",
                   linewidths=0.5, zorder=zorder)


COCO_NAMES = ("nose", "L-eye", "R-eye", "L-ear", "R-ear", "L-shoulder",
              "R-shoulder", "L-elbow", "R-elbow", "L-wrist", "R-wrist",
              "L-hip", "R-hip", "L-knee", "R-knee", "L-ankle", "R-ankle")


def joint_metadata(pose: dict, visibility: list[dict]) -> list[dict]:
    """真实 ViTPose 坐标/分数与独立几何可见性。"""
    rows = jq.build_joint_annotations(pose.get("keypoints_2d_vitpose_coco17"),
                                      pose.get("vitpose_info") or {}, visibility)
    for row, name in zip(rows, COCO_NAMES):
        row["name"] = name
    return rows


def label_joints(ax, pts: np.ndarray, metadata: list[dict]) -> None:
    """分列排布逐关节标签，使用引线避免头部密集关节文字互相覆盖。"""
    if not len(pts):
        return
    center = np.median(pts[:, 0])
    for left in (True, False):
        ids = [i for i in range(min(len(pts), 17)) if np.isfinite(pts[i]).all()
               and (pts[i, 0] <= center) == left]
        ids.sort(key=lambda i: ax.transData.transform(pts[i])[1], reverse=True)
        for i, y in zip(ids, np.linspace(0.94, 0.06, len(ids))):
            item = metadata[i]
            score = item["confidence"]
            c = "?" if score is None else f"{score:.2f}"
            if item["confidence_source"] == "fixed_placeholder" and score is not None:
                c += "*"
            visibility = item.get("visibility")
            v = "?" if visibility is None else str(visibility)
            ax.annotate(f"{i} {item['name']}\nv~{v}  c={c}", xy=pts[i],
                        xytext=(0.015 if left else 0.985, y), textcoords="axes fraction",
                        ha="left" if left else "right", va="center", fontsize=5.5,
                        color="#17202a", zorder=15,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="#616a6b", lw=0.45, alpha=0.7))


def project_cam(kp_cam: np.ndarray, focal: float, iw: int, ih: int) -> np.ndarray:
    if not len(kp_cam):
        return np.zeros((0, 2))
    z = np.maximum(kp_cam[:, 2], 1e-6)
    return np.stack([focal * kp_cam[:, 0] / z + iw / 2.0,
                     focal * kp_cam[:, 1] / z + ih / 2.0], axis=1)


def arrow_hot(ax, p0: np.ndarray, p1: np.ndarray, thr: float = HOT_M) -> int:
    if not len(p0) or len(p0) != len(p1):
        return 0
    d = p1 - p0
    hot = ((np.linalg.norm(d, axis=1) > thr)
           & (np.linalg.norm(p0, axis=1) > 0) & (np.linalg.norm(p1, axis=1) > 0))
    if not np.any(hot):
        return 0
    ax.quiver(p0[hot, 0], p0[hot, 1], d[hot, 0], d[hot, 1], angles="xy",
              scale_units="xy", scale=1.0, color=C_HOT, width=0.0018,
              headwidth=4.5, headlength=5.0, alpha=0.95, zorder=10)
    return int(hot.sum())


def win_of(arrs, ix: int, iy: int, pad: float = 0.08):
    pts = np.vstack([a for a in arrs if len(a)])
    lo, hi = pts[:, [ix, iy]].min(0), pts[:, [ix, iy]].max(0)
    span = np.maximum(hi - lo, 1e-3)
    return ((lo[0] - pad * span[0], hi[0] + pad * span[0]),
            (lo[1] - pad * span[1], hi[1] + pad * span[1]))


def snap_ignoring_gate(g: dict, p: dict, tag: str):
    """重跑吸附，但把贴合收益闸门降级为只记录（不动 IK / 旋转开关）。

    只为把「被闸门拒绝的候选姿势」画出来：否则 no_fit_gain 的图 before 与 after
    完全重合，看不出吸附到底把关节推到了哪里。
    """
    orig_enabled, orig_ik = rwa.lik.limb_ik_enabled, rwa.lik.apply_leg_ik
    rwa.lik.limb_ik_enabled = lambda: True
    rwa.lik.apply_leg_ik = lambda *a, **k: (None, {"applied": False,
                                                  "reason": "display_only"})
    try:
        return rwa.apply_bone_snap(g, p, tag)
    finally:
        rwa.lik.limb_ik_enabled, rwa.lik.apply_leg_ik = orig_enabled, orig_ik


def render_case(stem: str, img_path: pathlib.Path, g: dict, p: dict, tag: str,
                out_dir: pathlib.Path, dpi: int = 150, show_candidate: bool = True,
                show_fine: bool = True, fine=None) -> dict:
    after, before, info = rwa.apply_bone_snap(g, p, tag)
    rejected = not info.get("applied")
    if rejected:
        cand, cand_base, cand_info = snap_ignoring_gate(g, p, tag) if show_candidate else (None, None, {})
        if cand_info.get("applied") and cand_base is not None:
            after, before = cand, cand_base
            info = dict(cand_info)
            info.update({"applied": False, "candidate": True, "reason": "no_fit_gain"})
        else:
            after, before = p, rwa.load_pose_variant(tag, "_scalefit_torso")

    world = after.get("world") or {}
    grounded = world.get("status") == "ok"
    rot = np.asarray(world.get("rotation") or np.eye(3), dtype=np.float64)
    tvec = np.asarray(world.get("camera_position") or np.zeros(3), dtype=np.float64)

    depth = rwa.decode_depth(g)
    mask = rwa.decode_person_mask(g)
    focal = float(g["focal_px"])
    pts_disp, zcam = rwa.person_point_cloud(depth, focal, mask, grounded, rot, tvec)
    pc_cam, _ = rwa.person_camera_points(depth, focal, mask)
    if len(pts_disp) > 12000:
        sel = np.linspace(0, len(pts_disp) - 1, 12000).astype(int)
        pts_disp, zcam = pts_disp[sel], zcam[sel]
        pc_cam = pc_cam[sel]   # 与点云同序，供网格面板按同一 focal 投影

    kp_cam_b = kp_of(before if before is not None else after, "camera_space")
    kp_cam_a = kp_of(after, "camera_space")
    kp_w_b = kp_of(before if before is not None else after, "world")
    kp_w_a = kp_of(after, "world")
    if not len(kp_w_b) and len(kp_cam_b) and grounded:
        kp_w_b = kp_cam_b @ rot.T + tvec
    if not len(kp_w_a) and len(kp_cam_a) and grounded:
        kp_w_a = kp_cam_a @ rot.T + tvec

    fine_cam, fine_world = (fine_joints(after, p) if show_fine
                            else (np.zeros((0, 3)),) * 2)
    all_edges, ext_edges = fine or ([], [])

    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    ih, iw = rgb.shape[:2]
    cam_meta = after.get("camera") or {}
    f_img = float(cam_meta.get("focal_px") or focal) * iw / float(cam_meta.get("image_width") or iw)

    uv_b = project_cam(kp_cam_b, f_img, iw, ih)
    uv_a = project_cam(kp_cam_a, f_img, iw, ih)
    shift_a = (np.linalg.norm(kp_cam_a - kp_cam_b, axis=1)
               if len(kp_cam_a) == len(kp_cam_b) and len(kp_cam_a) else np.zeros(0))

    # ---- SAM-3D 网格（camera_space，未做地面变换）：吸附前后逐顶点对比
    base_mesh = (before if before is not None else after).get("camera_space") or {}
    after_mesh = after.get("camera_space") or {}
    faces = np.asarray(base_mesh.get("faces") or after_mesh.get("faces") or [],
                       dtype=np.int64).reshape(-1, 3)
    v_b = np.asarray(base_mesh.get("vertices") or [], dtype=np.float64).reshape(-1, 3)
    v_a = np.asarray(after_mesh.get("vertices") or [], dtype=np.float64).reshape(-1, 3)
    if not len(faces) or not len(v_a):
        v_a = np.zeros((0, 3))
    if not len(v_b):
        v_b = v_a
    v_shift = (np.linalg.norm(v_a - v_b, axis=1)
               if len(v_a) and v_a.shape == v_b.shape else None)

    metadata_b = joint_metadata(p, jq.estimate_visibility(
        kp_cam_b, v_b, faces, f_img, (ih, iw), depth=depth, mask=mask))
    metadata_a = joint_metadata(p, jq.estimate_visibility(
        kp_cam_a, v_a, faces, f_img, (ih, iw), depth=depth, mask=mask))

    fig = plt.figure(figsize=(19.0, 30.6), dpi=dpi)
    gs = fig.add_gridspec(5, 3, left=0.02, right=0.985, top=0.950, bottom=0.058,
                          wspace=0.06, hspace=0.20)
    cand = bool(info.get("candidate"))
    tag_txt = ("  ·  GATE REJECTED (no_fit_gain) → showing rejected candidate"
               if cand else "")
    after_lbl = "AFTER · candidate (rejected)" if cand else "AFTER · bone snap to MoGe-2 depth"
    fig.suptitle(
        f"{stem}  ·  bone snap {str(info.get('mode') or 'bonesnap')}"
        f"  ·  frame={'ground_world_v1' if grounded else 'camera_v1'}"
        f"  ·  joints moved {info.get('moved_joints', 0)}/17"
        f"  ·  max {float(info.get('joint_shift_max_m') or 0.0) * 100:.2f} cm"
        f"  ·  fit {float(info.get('fit_before_cm') or float('nan')):.2f}"
        f" → {float(info.get('fit_after_cm') or float('nan')):.2f} cm{tag_txt}",
        fontsize=12.5)

    # ---- 第一行：原图 + 骨架（camera view）
    for col, (title, mode_) in enumerate(
            (("BEFORE · torso fit (uncorrected)", "before"),
             (after_lbl, "after"),
             ("BEFORE vs AFTER (arrow = joint shift)", "both"))):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(rgb)
        if len(pc_cam):
            z = np.maximum(pc_cam[:, 2], 1e-6)
            u = f_img * pc_cam[:, 0] / z + iw / 2.0
            v = f_img * pc_cam[:, 1] / z + ih / 2.0
            keep = (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
            ax.scatter(u[keep], v[keep], c="#5f6b7a", s=0.7, alpha=0.30,
                       linewidths=0, zorder=2)
        if len(fine_cam):
            draw_fine(ax, project_cam(fine_cam, f_img, iw, ih), all_edges, ext_edges)
        if mode_ in ("before", "both"):
            bone_lines(ax, uv_b, C_BEFORE, 1.6, "--" if mode_ == "both" else "-",
                       alpha=0.95 if mode_ == "both" else 1.0, zorder=6)
            joints_dots(ax, uv_b, C_BEFORE, 14 if mode_ == "both" else 20, zorder=7)
        if mode_ in ("after", "both"):
            bone_lines(ax, uv_a, C_AFTER, 1.8 if mode_ == "both" else 2.2, "-", zorder=7)
            joints_dots(ax, uv_a, C_AFTER, 14 if mode_ == "both" else 20, zorder=8)
        if mode_ == "both":
            arrow_hot(ax, uv_b, uv_a)
        ax.set_xlim(0, iw)
        ax.set_ylim(ih, 0)
        ax.axis("off")
        ax.set_title(title, fontsize=9.5)
        if p.get("keypoints_2d_vitpose_coco17") is not None:
            ax.scatter(np.asarray(p["keypoints_2d_vitpose_coco17"])[:, 0],
                       np.asarray(p["keypoints_2d_vitpose_coco17"])[:, 1],
                       marker="x", s=16, c="#7639a5", linewidths=0.8, zorder=9)
        label_joints(ax, uv_b if mode_ == "before" else uv_a,
                     metadata_b if mode_ == "before" else metadata_a)

    # ---- 第二/三行：点云 + 骨架（正视图 / 侧视图）
    if grounded:
        rows = (((0, 2), "world x-z (front)", "x right (m)", "z up (m)"),
                ((1, 2), "world y-z (side)", "y depth (m)", "z up (m)"))
    else:
        # 相机系：to_display 后为 (x 右, z 深度, -y 向上)
        rows = (((0, 2), "camera x-up", "x right (m)", "up = -y (m)"),
                ((1, 2), "camera z-up", "z depth (m)", "up = -y (m)"))
    fine_rows = fine_world if grounded else fine_cam
    for r, (ix, iy), name, xl, yl in [(1, rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
                                      (2, rows[1][0], rows[1][1], rows[1][2], rows[1][3])]:
        kpb = kp_w_b if grounded else rwa.rsp.to_display(kp_cam_b, False)
        kpa = kp_w_a if grounded else rwa.rsp.to_display(kp_cam_a, False)
        fpts = rwa.rsp.to_display(fine_rows, grounded)
        win = win_of([pts_disp, kpb, kpa, fpts], ix, iy)
        for col, mode_ in enumerate(("before", "after", "both")):
            ax = fig.add_subplot(gs[r, col])
            ax.set_facecolor("white")
            if len(pts_disp):
                ax.scatter(pts_disp[:, ix], pts_disp[:, iy], c=zcam, cmap="viridis",
                           s=2.2, alpha=0.55, linewidths=0, zorder=2)
            if grounded and iy == 2:
                ax.axhline(0.0, color="#c0392b", lw=1.0, ls=(0, (4, 3)), zorder=1)
            if len(fpts):
                draw_fine(ax, fpts, all_edges, ext_edges, ix=ix, iy=iy)
            if mode_ in ("before", "both"):
                bone_lines(ax, kpb[:, [ix, iy]], C_BEFORE, 1.8,
                           "--" if mode_ == "both" else "-", alpha=0.95, zorder=6)
                joints_dots(ax, kpb[:, [ix, iy]], C_BEFORE, 16, zorder=7)
            if mode_ in ("after", "both"):
                bone_lines(ax, kpa[:, [ix, iy]], C_AFTER, 2.0, "-", zorder=7)
                joints_dots(ax, kpa[:, [ix, iy]], C_AFTER, 16, zorder=8)
            if mode_ == "both":
                arrow_hot(ax, kpb[:, [ix, iy]], kpa[:, [ix, iy]])
            ax.set_xlim(*win[0])
            ax.set_ylim(*win[1])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="#dfe4ea", lw=0.6, alpha=0.9, zorder=0)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            sub = {"before": "BEFORE (torso fit)",
                   "after": "AFTER (candidate, rejected)" if cand else "AFTER (bone snap)",
                   "both": "BEFORE vs AFTER"}[mode_]
            ax.set_title(f"{sub} · {name}", fontsize=9.5)
            label_joints(ax, (kpb if mode_ == "before" else kpa)[:, [ix, iy]],
                         metadata_b if mode_ == "before" else metadata_a)

    # ---- 第四行：网格 × 点云（正视图 / 侧视图，吸附前 vs 吸附后）
    # 这一行才是「吸附是否让网格更贴合点云」的直接证据：半透明三角面 + 同一份点云，
    # 后两张按逐顶点位移由青色渐变到红色，能看出哪些表面被推向点云、哪些被推离。
    mesh_cloud_axes = []
    if len(v_a) and len(faces):
        if grounded:
            mb_d, ma_d = v_b @ rot.T + tvec, v_a @ rot.T + tvec
            views = ((0, 2, "front", "x right (m)", "z up (m)"),
                     (1, 2, "side", "y depth (m)", "z up (m)"),
                     (0, 1, "top", "x right (m)", "y depth (m)"))
        else:
            mb_d, ma_d = rsp.to_display(v_b, False), rsp.to_display(v_a, False)
            views = ((0, 2, "front", "x right (m)", "up = -y (m)"),
                     (1, 2, "side", "depth (m)", "up = -y (m)"),
                     (0, 1, "top", "x right (m)", "depth (m)"))
        gs_m = gs[3, :].subgridspec(1, 6, wspace=0.32)
        fit_b = float(info.get("fit_before_cm") or float("nan"))
        fit_a = float(info.get("fit_after_cm") or float("nan"))
        for c, (ix, iy, nm, xl, yl) in enumerate(views):
            win = win_of([pts_disp, mb_d, ma_d], ix, iy)
            for k, (mesh_d, sh, lbl, fit) in enumerate(
                    ((mb_d, None, "BEFORE", fit_b), (ma_d, v_shift, "AFTER", fit_a))):
                ax = fig.add_subplot(gs_m[0, c * 2 + k])
                mesh_cloud_axes.append(ax)
                rwa.ortho_overlay(
                    ax, mesh_d, faces, np.zeros((0, 3)), np.zeros(0), ix, iy, win, grounded, 0.0,
                    f"{nm.upper()} · {lbl}{' (rejected)' if cand and k else ''}\nmesh × cloud · fit {fit:.2f} cm", xl, yl, shift=sh)
                if len(pts_disp):
                    # 保持同一份点云可见；二维叠加不代表三维内外关系。
                    ax.scatter(pts_disp[:, ix], pts_disp[:, iy], c=zcam, cmap="viridis",
                               s=1.6, alpha=0.75, linewidths=0, zorder=6)

    # ---- 第五行：网格贴到原图上（形变落在人体表面的位置）
    # 前两张压暗原图，让青色/红色面片更容易分辨；第三张保留原图以核对真实贴合。
    rgb_dim = (rgb.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    mesh_panels = (
        ("MESH OVER PHOTO · BEFORE (torso fit)", rgb_dim, v_b, None, None, None),
        ("MESH OVER PHOTO · AFTER (blue → red = vertex shift)", rgb_dim, v_a, v_shift, None, None),
        ("MESH OVER PHOTO · AFTER + skeleton", rgb, v_a, v_shift, kp_cam_a, shift_a),
    )
    mesh_tris = 0
    for col, (title, base_img, verts, shift, jc, js) in enumerate(mesh_panels):
        ax = fig.add_subplot(gs[4, col])
        # 不叠点云：这一行只看网格本身，点云在 2/3/4 行已有
        n = rwa.camera_view_panel(
            ax, base_img, np.zeros((0, 3)), np.zeros(0), verts, faces, f_img,
            title, shift=shift, joints_cam=jc, joints_shift=js)
        mesh_tris = max(mesh_tris, int(n))
        if jc is not None:
            label_joints(ax, project_cam(jc, f_img, iw, ih), metadata_a)

    fig.text(0.5, 0.028,
             "c = real ViTPose heatmap score (purple x = its 2D location), NOT 3D accuracy. "
             "v~ = estimated local surface self-visibility: 2 visible / 1 occluded / 0 outside / ? uncertain.\n"
             "Visibility is measured from the original camera, NOT each plotted viewpoint; external occlusion is not classified.",
             ha="center", fontsize=9, color="#8a4b08")
    fine_info = info.get("fine") or {}
    fine_lbl = ("MHR-127 hands/feet bones (fine-snapped)"
                if fine_info.get("applied") else "MHR-127 hands/feet bones (not snapped)")
    handles = [
        Line2D([0], [0], color=C_BEFORE, lw=2.0, ls="--", label="BEFORE skeleton (torso fit)"),
        Line2D([0], [0], color=C_AFTER, lw=2.0, label="AFTER skeleton (bone snap)"),
        Line2D([0], [0], color=C_HOT, lw=1.6, label=f"joint shift > {HOT_M*100:.0f} cm"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#5f6b7a",
               markersize=4, label="MoGe-2 person point cloud"),
        Line2D([0], [0], color=C_FINE_EXT, lw=1.2, label=fine_lbl),
        Patch(facecolor=rsp.C_MESH, alpha=0.55,
              edgecolor="none", label="SAM-3D mesh (unmoved)"),
        Patch(facecolor="#db3d30", alpha=0.55, edgecolor="none",
              label=f"mesh moved > {rwa.SHIFT_HI*100:.0f} cm"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.5, 0.042))
    extra = (f"k = {float(info.get('k_min')):.3f}~{float(info.get('k_max')):.3f}"
             if info.get("k_min") is not None else "k = n/a")
    nz = shift_a[shift_a > 1e-6] if len(shift_a) else shift_a
    foot = (f"joint shift (camera)  med {np.median(nz) * 100:.2f} cm over "
            f"{len(nz)}/{len(shift_a)} moved  ·  max {np.max(shift_a) * 100:.2f} cm  ·  {extra}"
            if len(nz) else f"joints unchanged  ·  {extra}")
    if fine_info.get("applied"):
        ok_g = [g for g in fine_info.get("groups") or [] if g.get("ok")]
        k_txt = "/".join(f"{g['name']} {g['k']:.3f}" for g in ok_g) or "n/a"
        mv = float(fine_info.get("max_vertex_shift_m") or 0.0) * 100
        foot += f"  ·  fine hands/feet {'/'.join(g['name'] for g in ok_g) or 'none'}: {k_txt}"
        foot += f"  ·  max vertex {mv:.2f} cm"
    if v_shift is not None and len(v_shift):
        foot += (f"  ·  mesh vertex shift med {np.median(v_shift) * 100:.2f}"
                 f" / max {np.max(v_shift) * 100:.2f} cm")
    fig.text(0.5, 0.010, foot, ha="center", fontsize=9, color="#3c4654")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_skeleton_snap_compare.png"
    fig.savefig(out, dpi=dpi)
    mesh_cloud_out = None
    if mesh_cloud_axes:
        from matplotlib.transforms import Bbox
        renderer = fig.canvas.get_renderer()
        bounds = Bbox.union([ax.get_tightbbox(renderer) for ax in mesh_cloud_axes])
        bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.08)
        mesh_cloud_out = out_dir / f"{stem}_mesh_cloud_snap_compare.png"
        fig.savefig(mesh_cloud_out, dpi=dpi, bbox_inches=bounds)
    plt.close(fig)
    return {
        "joint_annotations": metadata_a,
        "joint_annotations_before": metadata_b,
        "vitpose_info": p.get("vitpose_info"),
        "mesh_cloud_out": str(mesh_cloud_out) if mesh_cloud_out else None,
        "stem": stem, "out": str(out), "applied": bool(info.get("applied")),
        "shown_as": "rejected_candidate" if cand else "applied",
        "reason": info.get("reason"), "k": extra,
        "moved_joints": info.get("moved_joints"),
        "joint_shift_max_cm": (float(info.get("joint_shift_max_m") or 0.0) * 100
                               if info.get("joint_shift_max_m") is not None else None),
        "fit_before_cm": info.get("fit_before_cm"),
        "fit_after_cm": info.get("fit_after_cm"),
        "landing_fix_mm": float(info.get("landing_fix_m") or 0.0) * 1000,
        "fine_joints_drawn": int(len(fine_world) if grounded else len(fine_cam)),
        "mesh_faces": mesh_tris,
        "mesh_vertices": int(len(v_a)),
        "mesh_vertex_shift_max_cm": (float(np.max(v_shift)) * 100
                                     if v_shift is not None and len(v_shift) else None),
        "fine": info.get("fine"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C2 骨骼吸附前后对比可视化（离线，读缓存重算）")
    ap.add_argument("images", nargs="*", help="图片路径；缺省为 images/test_image/*.jpg")
    ap.add_argument("--outdir", default=None, help="输出目录；缺省 <图片目录>/_wr_skel_compare")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-candidate", action="store_true",
                    help="闸门拒绝时不画候选吸附（before==after）")
    ap.add_argument("--no-fine", action="store_true", help="不画 MHR-127 手掌/脚掌细骨架")
    ap.add_argument("--fine-snap", action="store_true",
                    help="启用手掌/脚掌细骨吸附（POSEEXTRACTOR_SAM3D_BONE_SNAP_FINE=1）")
    ap.add_argument("--refresh-quality", action="store_true",
                    help="本地运行 ViTPose 更新真实坐标/分数小缓存；不重跑 SAM3D")
    args = ap.parse_args(argv)

    if args.fine_snap:
        os.environ["POSEEXTRACTOR_SAM3D_BONE_SNAP_FINE"] = "1"

    rsp._load_font()
    root = pathlib.Path(rsp.DEFAULT_ROOT)
    imgs = ([pathlib.Path(p) for p in args.images] if args.images
            else [p for p in sorted(root.glob("*.jpg")) if not p.name.startswith("pose_overlay")])
    out_dir = pathlib.Path(args.outdir) if args.outdir else root / "_wr_skel_compare"
    fine = fine_bone_edges() if not args.no_fine else None

    stats = []
    for img in imgs:
        tag = hashlib.sha1(img.read_bytes()).hexdigest()[:16]
        g = json.loads((rsp.CACHE_DIR / f"{tag}_ground.json").read_text(encoding="utf-8")) \
            if (rsp.CACHE_DIR / f"{tag}_ground.json").exists() else None
        p = rwa.load_pose_variant(tag, "_scalefit_torso")
        if g is None or p is None:
            print(f"[skip] {img.name}: 缓存缺失 tag={tag}")
            continue
        quality_cache = rsp.CACHE_DIR / f"{tag}_joint_quality_vitpose.json"
        if args.refresh_quality:
            resources = pathlib.Path(os.environ.get("POSEEXTRACTOR_RESOURCES") or
                                     "/mnt/g/UGit/QingGeTools/ImagePose_Resources")
            sys.path.insert(0, str(resources))
            os.environ.setdefault("POSEEXTRACTOR_RESOURCES", str(resources))
            points, vitpose_info = jq.detect_vitpose(np.asarray(Image.open(img).convert("RGB")))
            quality = {"keypoints_2d_vitpose_coco17": points.tolist(),
                       "vitpose_info": vitpose_info}
            quality_cache.write_text(json.dumps(quality, allow_nan=False), encoding="utf-8")
            p = dict(p, **quality)
        elif quality_cache.exists():
            p = dict(p, **json.loads(quality_cache.read_text(encoding="utf-8")))
        print(f"[{img.stem}] tag={tag} ...")
        st = render_case(img.stem, img, g, p, tag, out_dir, dpi=args.dpi,
                         show_candidate=not args.no_candidate,
                         show_fine=not args.no_fine, fine=fine)
        stats.append(st)
        print("  " + json.dumps(st, ensure_ascii=False))
    if stats:
        out = out_dir / "skeleton_snap_compare.json"
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
