"""SAM 3D Body adapter used by the ground pipeline."""
from __future__ import annotations

import base64
import io
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

# 推理服务配置（集中放在文件头部，不读取环境变量）。
# True 使用通过贴合检查的 AFTER 姿势，False 保持原输出。
SAM3D_OUTPUT_AFTER = False
# full 包含身体和左右手；body 仅解码身体。
SAM3D_INFERENCE_TYPE = "full"
# 推理前放大系数（1–3），提高原图手部采样分辨率。
SAM3D_UPSCALE = 1.0
# 深度尺度拟合：off / moge / torso / limb（AFTER=True 时强制 torso）。
SAM3D_SCALE_FIT = "limb"
# 逐骨 k 离散度上限（仅 limb 档使用），超限跳过肢体修正。
SAM3D_LIMB_K_SPREAD = 0.12
LIMB_K_CLAMP = (0.75, 1.35)

# MHR-70 → COCO-17 索引（与 ground_retarget.MHR70_COCO17 逐项一致）。
MHR70_TO_COCO17 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
# COCO-17 四肢线段（肩肘腕、髋膝踝），躯干锚用它把四肢像素排除在统计之外。
COCO17_LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))
# 逐肢体骨表（名称, 起点 COCO-17, 终点 COCO-17），limb 档按骨求深度缩比。
COCO17_LIMB_BONES = (
    ("l_upper_arm", 5, 7),
    ("l_forearm", 7, 9),
    ("r_upper_arm", 6, 8),
    ("r_forearm", 8, 10),
    ("l_thigh", 11, 13),
    ("l_shank", 13, 15),
    ("r_thigh", 12, 14),
    ("r_shank", 14, 16),
)


def _pose_overlay_data_uri(rgb: np.ndarray, keypoints: np.ndarray) -> str:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    limb_colors = (
        (255, 90, 90), (255, 90, 90), (255, 90, 90), (255, 90, 90),
        (255, 190, 40), (255, 190, 40), (255, 120, 40), (60, 210, 255), (40, 150, 255),
        (190, 100, 255), (160, 70, 230), (230, 230, 70),
        (70, 230, 130), (30, 190, 100), (70, 180, 255), (40, 120, 230),
    )
    height, width = rgb.shape[:2]
    line_width = max(6, min(12, round(min(width, height) / 85)))
    joint_radius = max(7, min(13, round(min(width, height) / 75)))
    points = np.asarray(keypoints, dtype=np.float32)[list(MHR70_TO_COCO17)]
    visible = np.isfinite(points).all(axis=1)
    visible &= (points[:, 0] >= 0) & (points[:, 0] < width)
    visible &= (points[:, 1] >= 0) & (points[:, 1] < height)
    for (a, b), color in zip(edges, limb_colors):
        if visible[a] and visible[b]:
            start = tuple(np.rint(points[a, :2]).astype(int))
            end = tuple(np.rint(points[b, :2]).astype(int))
            draw.line((start, end), fill=(0, 0, 0), width=line_width + 4)
            draw.line((start, end), fill=color, width=line_width)
    for index, (point, is_visible) in enumerate(zip(points, visible)):
        if is_visible:
            x, y = np.rint(point[:2]).astype(int)
            radius = joint_radius + 2 if index in (5, 6, 9, 10, 11, 12, 15, 16) else joint_radius
            draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=(0, 0, 0))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 220, 40), outline=(255, 255, 255), width=2)
    title = "SAM3D BODY 2D POSE"
    title_font_size = max(24, min(46, round(min(width, height) / 22)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
    except OSError:
        font = ImageFont.load_default()
    title_box = draw.textbbox((0, 0), title, font=font)
    title_width = title_box[2] - title_box[0]
    title_height = title_box[3] - title_box[1]
    title_pad = max(10, title_font_size // 2)
    draw.rounded_rectangle((8, 8, title_width + title_pad * 2, title_height + title_pad * 2), radius=6, fill=(12, 18, 28))
    draw.text((8 + title_pad, 8 + title_pad), title, font=font, fill=(255, 255, 255), stroke_width=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pose_mesh_data_uri(vertices: np.ndarray, faces: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("SAM3D returned invalid mesh vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("SAM3D returned invalid mesh faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("SAM3D mesh faces reference invalid vertices")

    display_vertices = vertices.copy()
    display_vertices[:, 1:3] *= -1
    triangles = display_vertices[faces]
    z_values = triangles[:, :, 2].mean(axis=1)
    z_min, z_max = float(z_values.min()), float(z_values.max())
    normalized = (z_values - z_min) / max(z_max - z_min, 1e-6)
    face_colors = plt.get_cmap("viridis")(normalized)

    center = (display_vertices.min(axis=0) + display_vertices.max(axis=0)) * 0.5
    radius = max(float((display_vertices.max(axis=0) - display_vertices.min(axis=0)).max()) * 0.58, 1e-3)
    fig = plt.figure(figsize=(15, 5.5), dpi=100, facecolor="#101722")
    for index, (title, azimuth) in enumerate((("Front", -90), ("Side", 0), ("Back", 90)), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d", facecolor="#101722")
        collection = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors=(0.12, 0.16, 0.21, 0.32),
            linewidths=0.08,
            antialiased=True,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=8, azim=azimuth)
        ax.set_title(title, color="white", fontsize=13, pad=4)
        ax.set_axis_off()

    fig.suptitle("SAM3D BODY | PREDICTED MESH · MODEL COORDS", color="white", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def sam3d_inference_type() -> str:
    value = str(SAM3D_INFERENCE_TYPE).strip().lower()
    return "body" if value == "body" else "full"


def sam3d_upscale() -> float:
    try:
        value = float(SAM3D_UPSCALE)
    except (TypeError, ValueError):
        value = 1.0
    return min(max(value, 1.0), 3.0)


def sam3d_scale_fit_mode() -> str:
    value = str(SAM3D_SCALE_FIT).strip().lower()
    if value in ("limb", "per_limb", "limbs", "local"):
        return "limb"
    if value in ("torso", "body", "center"):
        return "torso"
    if value in ("moge", "depth", "on", "1", "true"):
        return "moge"
    return "off"


def limb_k_spread_limit() -> float:
    # 门控用「逐骨 k 离散度 max-min」而非 2D 轮廓覆盖率：
    # 覆盖率随衣物宽松度剧烈变化（w6d1j 宽松外套 sil/mask=0.47 → 0.518；90dc 四肢裸露 → 0.805），
    # 且对沿相机射线的深度误差完全不可见（qco6j 覆盖率 0.699、未覆盖像素 p90=1px 却会把点云改差）。
    # 7 张实测离散度：qco6j 0.226 / 006dpl 0.113 / 5306 0.092 / trvxj 0.084 / w6d1j 0.080 /
    # z459j 0.075 / 90dc 0.056，取 0.12 只挡 qco6j（其逐骨 k 0.776~1.002、位移 55cm，
    # 修正后 nn 3.20→4.02cm、3cm 内表面 45.6%→39.5%）。
    try:
        value = float(SAM3D_LIMB_K_SPREAD)
    except (TypeError, ValueError):
        value = 0.12
    return max(value, 0.0)


def sam3d_output_after() -> bool:
    return SAM3D_OUTPUT_AFTER


RAW_POSE_FIELDS = (
    "pred_joint_coords",
    "mhr_joint_coords",
    "pred_global_rots",
    "mhr_joint_global_rots",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "global_rot",
    "pred_vertices",
    "focal_length",
    "scale_params",
    "shape_params",
    # inference_type="full" 才有：手部解码实际使用的左右手框（原图像素坐标）
    "lhand_bbox",
    "rhand_bbox",
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def mask_to_bbox(person_mask: np.ndarray) -> np.ndarray:
    if person_mask.ndim != 2 or not person_mask.size:
        raise ValueError("Expected a non-empty 2D person mask")
    ys, xs = np.nonzero(person_mask)
    if not len(xs):
        raise ValueError("Person mask is empty")

    height, width = person_mask.shape
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    pad_x = max(20.0, (x2 - x1) * 0.2)
    pad_y = max(20.0, (y2 - y1) * 0.15)
    return np.array([[
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width - 1), x2 + pad_x),
        min(float(height - 1), y2 + pad_y),
    ]], dtype=np.float32)


# 参与等比缩放的 3D 量：都乘同一系数，保证 2D 投影 f·x/z 不变。
SCALE_KEYS = ("pred_vertices", "pred_keypoints_3d", "pred_joint_coords", "mhr_joint_coords")


def torso_region_mask(person_mask, keypoints_2d, shape) -> dict | None:
    """人物掩码内的躯干像素（去掉四肢带与轮廓边缘）。

    四肢是 SAM-3D 网格与 MoGe-2 深度最容易分歧的部位，轮廓边缘又常混入背景深度；
    把两者剔掉后剩下的身体中心像素更适合充当尺度锚点。
    """
    mask = np.asarray(person_mask)
    if mask.ndim != 2 or mask.shape != tuple(shape):
        return None
    keypoints = np.asarray(keypoints_2d, dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[0] <= max(max(p) for p in COCO17_LIMBS):
        return None
    coco = keypoints[list(MHR70_TO_COCO17), :2]
    if not np.isfinite(coco[[5, 6, 9, 10, 11, 12, 13, 14, 15, 16]]).all():
        return None
    import cv2

    mask = mask.astype(bool)
    edge = max(3, int(round(0.008 * min(shape))))
    inner = cv2.erode(
        mask.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    limb = np.zeros(mask.shape, np.uint8)
    pts = np.rint(coco).astype(np.int32)
    for a, b in COCO17_LIMBS:
        half = max(0.20 * float(np.hypot(*(coco[b] - coco[a]))), 8.0)
        cv2.line(limb, tuple(pts[a]), tuple(pts[b]), 1, int(round(2.0 * half)))
    return {"torso": inner & (limb == 0), "limb": inner & (limb > 0)}


def fit_scale_to_depth(
    vertices,
    cam_t,
    depth,
    cam_int,
    person_mask=None,
    *,
    keypoints_2d=None,
    min_samples: int = 200,
    mode: str | None = None,
) -> dict[str, Any] | None:
    """用深度后端的实测深度把网格整体尺度拟合到场景。

    网格顶点投到像素后建最小 z 缓冲（只保留可见前表面），与实测深度逐像素求比值，
    取中位数作为沿相机射线的等比系数。单目只约束 f/z，等比缩放不改变 2D 投影。
    mode=torso 时只用躯干区的中位数（四肢与轮廓边缘不参与）。
    """
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3) + np.asarray(
        cam_t, dtype=np.float64
    ).reshape(1, 3)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        return None
    z = verts[:, 2]
    valid = np.isfinite(z) & (z > 1e-3)
    if not np.any(valid):
        return None
    height, width = depth.shape
    fx, fy = float(cam_int[0, 0]), float(cam_int[1, 1])
    cx, cy = float(cam_int[0, 2]), float(cam_int[1, 2])
    zsafe = np.where(valid, z, 1.0)
    u = np.rint(fx * verts[:, 0] / zsafe + cx).astype(np.int64)
    v = np.rint(fy * verts[:, 1] / zsafe + cy).astype(np.int64)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if person_mask is not None and np.asarray(person_mask).shape == depth.shape:
        valid &= np.asarray(person_mask, dtype=bool)[v.clip(0, height - 1), u.clip(0, width - 1)]
    if np.count_nonzero(valid) < min_samples:
        return None
    u, v, z = u[valid], v[valid], z[valid]
    zbuf = np.full(depth.shape, np.inf)  # 只保留可见表面，避免背面顶点把比值拉偏
    np.minimum.at(zbuf, (v, u), z)
    ok = np.isfinite(zbuf) & np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(ok) < min_samples:
        return None
    pv, pu = np.nonzero(ok)
    ratio = depth[pv, pu] / zbuf[pv, pu]
    keep = np.isfinite(ratio) & (ratio > 0)
    pv, pu, ratio = pv[keep], pu[keep], ratio[keep]
    if ratio.size < min_samples:
        return None

    mode = sam3d_scale_fit_mode() if mode is None else mode
    detail: dict[str, Any] = {}
    select = np.ones(ratio.size, dtype=bool)
    regions = None
    if mode in ("torso", "limb") and person_mask is not None:
        regions = torso_region_mask(person_mask, keypoints_2d, depth.shape)
    if regions is None and mode in ("torso", "limb"):
        # 关键点/掩码不可用时不静默降级，直接记录并退回全图统计
        detail["torso_fallback"] = "keypoints_or_mask_unavailable"
    if regions is not None:
        detail["ratio_full"] = float(np.median(ratio))
        detail["samples_full"] = int(ratio.size)
        torso_sel = regions["torso"][pv, pu]
        limb_sel = regions["limb"][pv, pu]
        if np.count_nonzero(torso_sel) >= min_samples:
            select = torso_sel
        else:
            detail["torso_fallback"] = "too_few_torso_samples"
        for name, sel in (("torso", torso_sel), ("limb", limb_sel)):
            if np.count_nonzero(sel) >= min_samples:
                detail[f"ratio_{name}"] = float(np.median(ratio[sel]))
                detail[f"samples_{name}"] = int(np.count_nonzero(sel))
    picked = ratio[select]
    median = float(np.median(picked))
    return {
        "method": "moge_depth_surface_ratio",
        "mode": mode,
        "factor": float(min(max(median, 0.6), 1.6)),
        "ratio_median": median,
        "ratio_p25": float(np.percentile(picked, 25)),
        "ratio_p75": float(np.percentile(picked, 75)),
        "samples": int(picked.size),
        **detail,
    }


def apply_uniform_scale(person: dict, factor: float) -> dict:
    """沿相机射线等比缩放（3D 点与 cam_t 同乘 k），2D 投影 f·x/z 保持不变。"""
    person = dict(person)
    for key in SCALE_KEYS:
        if person.get(key) is not None:
            person[key] = np.asarray(person[key], dtype=np.float32) * factor
    if person.get("pred_cam_t") is not None:
        person["pred_cam_t"] = np.asarray(person["pred_cam_t"], dtype=np.float32) * factor
    return person


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点到线段 a-b 的距离（points 支持 2D/3D，按末轴维度广播）。"""
    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    proj = np.asarray(a, dtype=np.float64) + t[..., None] * ab
    return np.linalg.norm(points - proj, axis=-1)


def limb_scale_adjust(
    vertices,
    cam_t,
    keypoints_3d,
    keypoints_2d,
    depth,
    cam_int,
    person_mask=None,
    *,
    faces=None,
    min_samples: int = 120,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """按可见点云逐肢体骨修正顶点深度（沿各自相机射线缩放，2D 投影不变）。

    SAM-3D 不暴露蒙皮权重，因此这里自己建立顶点→骨骼归属：
    先用 2D 最近骨把「人物掩码内的四肢像素」分配给 8 根骨，得到逐骨深度比中位数 k_b；
    再用「顶点到骨的 3D 距离高斯权重」把 k_b 融合成逐顶点系数 k(v)=1+Σw(k_b-1)/Σw；
    最后 v'=(v+cam_t)·k(v)-cam_t。cam_t 与 joints/keypoints 均不动，只改网格。
    逐骨 k 的离散度（max-min）超过阈值时判定该图深度不自洽，整段跳过。
    """
    cam_t = np.asarray(cam_t, dtype=np.float64).reshape(3)
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    kp3 = None if keypoints_3d is None else np.asarray(keypoints_3d, dtype=np.float64)
    kp2 = None if keypoints_2d is None else np.asarray(keypoints_2d, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    need_row = max(max(pair[1], pair[2]) for pair in COCO17_LIMB_BONES)
    detail: dict[str, Any] = {"enabled": True}
    if (
        depth.ndim != 2 or person_mask is None or cam_int is None
        or kp3 is None or kp2 is None
        or kp3.ndim != 2 or kp3.shape[0] <= need_row or kp3.shape[-1] != 3
        or kp2.ndim != 2 or kp2.shape[0] != kp3.shape[0] or kp2.shape[-1] < 2
    ):
        return None, {**detail, "gate": "skipped", "reason": "bad_inputs"}

    height, width = depth.shape
    fx, fy = float(cam_int[0][0]), float(cam_int[1][1])
    cx, cy = float(cam_int[0][2]), float(cam_int[1][2])
    vcam = verts + cam_t.reshape(1, 3)
    z = vcam[:, 2]
    front = np.isfinite(z) & (z > 1e-3)
    valid = front.copy()
    if np.count_nonzero(valid) < min_samples:
        return None, {**detail, "gate": "skipped", "reason": "too_few_vertices"}
    zsafe = np.where(valid, z, 1.0)
    u = np.rint(fx * vcam[:, 0] / zsafe + cx).astype(np.int64)
    v = np.rint(fy * vcam[:, 1] / zsafe + cy).astype(np.int64)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    mask = np.asarray(person_mask, dtype=bool)
    if mask.shape == depth.shape:
        valid &= mask[v.clip(0, height - 1), u.clip(0, width - 1)]
    if np.count_nonzero(valid) < min_samples:
        return None, {**detail, "gate": "skipped", "reason": "too_few_projected"}

    # 网格前表面 z 缓冲 → 图像平面上的四肢带覆盖度（即 x-z 前视轮廓吻合程度的代理）
    zbuf = np.full(depth.shape, np.inf)
    np.minimum.at(zbuf, (v[valid], u[valid]), z[valid])
    regions = torso_region_mask(mask, kp2, depth.shape)
    if regions is None:
        return None, {**detail, "gate": "skipped", "reason": "regions_unavailable"}
    band = regions["limb"]
    band_px = int(np.count_nonzero(band))
    if band_px < min_samples:
        return None, {**detail, "gate": "skipped", "reason": "limb_band_too_small"}
    # 轮廓：优先按三角面填充（与渲染一致），无面表时退回顶点足迹膨胀
    import cv2

    if faces is not None:
        face_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        keep_face = front[face_arr].all(axis=1)
        triangles = np.stack([u[face_arr], v[face_arr]], axis=-1).astype(np.int32)
        silhouette = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(silhouette, list(triangles[keep_face]), 1)
    else:
        footprint = np.zeros(depth.shape, np.uint8)
        footprint[v[valid], u[valid]] = 1
        silhouette = cv2.dilate(footprint, np.ones((5, 5), np.uint8))
    coverage = float(np.count_nonzero(silhouette.astype(bool) & band) / band_px)

    ok = np.isfinite(zbuf) & np.isfinite(depth) & (depth > 0)
    pv, pu = np.nonzero(ok)
    ratio = depth[pv, pu] / zbuf[pv, pu]
    keep = np.isfinite(ratio) & (ratio > 0)
    pv, pu, ratio = pv[keep], pu[keep], ratio[keep]

    coco2 = kp2[list(MHR70_TO_COCO17), :2]
    coco3 = kp3[list(MHR70_TO_COCO17), :3] + cam_t.reshape(1, 3)
    if not np.isfinite(coco2).all() or not np.isfinite(coco3).all():
        return None, {**detail, "gate": "skipped", "gate_iou": coverage,
                      "reason": "keypoints_not_finite"}

    # 1) 像素 → 最近骨（2D，仅取骨带内的像素）
    pixels = np.stack([pu, pv], axis=1).astype(np.float64)
    bone_ids = np.full(pixels.shape[0], -1, dtype=np.int64)
    bone_dist = np.full(pixels.shape[0], np.inf)
    for index, (_name, a, b) in enumerate(COCO17_LIMB_BONES):
        dist = _segment_distance(pixels, coco2[a], coco2[b])
        half = max(0.20 * float(np.hypot(*(coco2[b] - coco2[a]))), 8.0)
        take = (dist <= half) & (dist < bone_dist)
        bone_ids[take] = index
        bone_dist[take] = dist[take]

    # 2) 逐骨取深度比中位数
    bones: list[dict[str, Any]] = []
    kb = np.ones(len(COCO17_LIMB_BONES))
    valid_bone = np.zeros(len(COCO17_LIMB_BONES), dtype=bool)
    for index, (name, _a, _b) in enumerate(COCO17_LIMB_BONES):
        sel = bone_ids == index
        n = int(np.count_nonzero(sel))
        if n < min_samples:
            bones.append({"name": name, "k": 1.0, "n": n, "valid": False})
            continue
        value = float(np.median(ratio[sel]))
        kb[index] = min(max(value, LIMB_K_CLAMP[0]), LIMB_K_CLAMP[1])
        valid_bone[index] = True
        bones.append({"name": name, "k": float(kb[index]), "n": n, "valid": True})
    detail["bones"] = bones
    detail["gate_iou"] = coverage  # 仅记录 2D 轮廓覆盖率，不参与判定
    valid_ks = [float(b["k"]) for b in bones if b["valid"]]
    spread = (max(valid_ks) - min(valid_ks)) if len(valid_ks) >= 2 else float("inf")
    detail["k_spread"] = spread if np.isfinite(spread) else None
    detail["k_spread_limit"] = limb_k_spread_limit()
    detail["gate"] = "ok" if np.isfinite(spread) and spread <= detail["k_spread_limit"] else "skipped"
    if not valid_bone.any():
        detail["reason"] = "no_bone_samples"
        return None, detail
    if detail["gate"] != "ok":
        detail["reason"] = "limb_k_inconsistent"
        return None, detail

    # 3) 顶点 → 骨（3D 高斯权重）→ 逐顶点系数 → 沿相机射线缩放
    weight = np.zeros((verts.shape[0], len(COCO17_LIMB_BONES)))
    for index, (_name, a, b) in enumerate(COCO17_LIMB_BONES):
        if not valid_bone[index]:
            continue
        start, end = coco3[a], coco3[b]
        sigma = max(0.30 * float(np.linalg.norm(end - start)), 0.045)
        weight[:, index] = np.exp(-0.5 * (_segment_distance(vcam, start, end) / sigma) ** 2)
    total = weight.sum(axis=1)
    k = np.ones(verts.shape[0])
    touched = total > 1e-6
    k[touched] = 1.0 + (weight[touched] @ (kb - 1.0)) / total[touched]
    new_verts = (vcam * k[:, None] - cam_t.reshape(1, 3)).astype(np.float32)
    detail["k_min"] = float(k.min())
    detail["k_max"] = float(k.max())
    detail["weighted_vertices"] = int(np.count_nonzero(touched))
    detail["max_shift_m"] = float(np.max(np.linalg.norm(new_verts - verts, axis=1)))
    return new_verts, detail


def apply_after_pose(person, faces, depth, person_mask, cam_int, image_hw, ground_rotation=None):
    """Apply the shared visualization AFTER before generic-pose/retarget assembly."""
    from .bone_depth_snap import apply_pose_bone_snap, _rotation_from_to
    from .ground_retarget import FIGURE_P
    from .mhr_mesh_fbx import get_rig, _topo_order

    report = {"enabled": True, "applied": False, "source": "legacy_bone_snap_after"}
    if person.get("rotation_fit") is not None:
        return person, {**report, "reason": "rotation_fit_takes_priority"}
    if depth is None or person_mask is None or cam_int is None:
        return person, {**report, "reason": "missing_depth_mask_or_intrinsics"}
    height, width = image_hw
    dh, dw = np.asarray(depth).shape
    k = np.asarray(cam_int, dtype=np.float64).copy()
    k[0] *= dw / width
    k[1] *= dh / height
    # The legacy rasterizer uses a centered, square-pixel pinhole camera.
    if not np.allclose([k[0, 0], k[0, 2], k[1, 2]], [k[1, 1], dw / 2, dh / 2], atol=1.0):
        return person, {**report, "reason": "unsupported_legacy_intrinsics"}
    cam_t = np.asarray(person["pred_cam_t"], dtype=np.float64).reshape(3)
    verts = np.asarray(person["pred_vertices"], dtype=np.float64) + cam_t
    kp = np.asarray(person["pred_keypoints_3d"], dtype=np.float64) + cam_t
    joints = np.asarray(person["pred_joint_coords"], dtype=np.float64) + cam_t
    coco = kp[list(MHR70_TO_COCO17)]
    baseline = {"camera_space": {"vertices": verts.tolist(), "faces": np.asarray(faces).tolist(),
                                "keypoints_3d_coco17": coco.tolist()},
                "generic_pose": {"pred_joint_coords": ((joints - cam_t) @ FIGURE_P.T).tolist(),
                                 "pred_keypoints_3d": ((kp - cam_t) @ FIGURE_P.T).tolist(),
                                 "pred_cam_t": cam_t.tolist()}}
    if ground_rotation is not None:
        rot = np.asarray(ground_rotation, dtype=np.float64).reshape(3, 3)
        baseline["world"] = {"status": "ok", "rotation": rot.tolist(), "translation": [0, 0, 0],
                             "mesh": {"vertices": (verts @ rot.T).tolist()},
                             "keypoints_3d_coco17": (coco @ rot.T).tolist(),
                             "generic_pose": {"pred_joint_coords": (joints @ rot.T).tolist()}}
    adjusted, _, detail = apply_pose_bone_snap(
        baseline, baseline, depth, person_mask, float(k[0, 0]))
    report.update(detail)
    if not report.get("applied"):
        return person, report
    block = adjusted["camera_space"]
    after_joints = np.asarray(block["keypoints_3d_fine"], dtype=np.float64)
    if after_joints.shape != joints.shape:
        return person, {**report, "applied": False, "reason": "missing_after_mhr_joints"}
    after_kp = np.asarray(block["keypoints_3d_coco17"], dtype=np.float64)
    from .bone_depth_snap import deform_points

    kp_new = deform_points(kp, coco, after_kp)
    kp_new[list(MHR70_TO_COCO17)] = after_kp
    rig = get_rig()
    index, parents = rig["index"], rig["parents"]
    old_rots = np.asarray(person["pred_global_rots"], dtype=np.float64)
    new_rots = old_rots.copy()
    children = {name: [] for name in rig["bone_order"]}
    for name, parent in parents.items():
        if parent is not None:
            children[parent].append(name)
    sign = np.diag([1.0, -1.0, -1.0])
    for name in _topo_order(rig):
        i = index[name]
        candidates = [c for c in children[name] if "twist" not in c and "_null" not in c]
        if candidates:
            child = max(candidates, key=lambda c: np.linalg.norm(joints[index[c]] - joints[i]))
            a = joints[index[child]] - joints[i]
            b = after_joints[index[child]] - after_joints[i]
            if min(np.linalg.norm(a), np.linalg.norm(b)) > 1e-8:
                delta = _rotation_from_to(a / np.linalg.norm(a), b / np.linalg.norm(b))
                new_rots[i] = sign @ delta @ sign @ old_rots[i]
        elif parents[name] is not None:
            parent = index[parents[name]]
            new_rots[i] = new_rots[parent] @ old_rots[parent].T @ old_rots[i]
    result = dict(person)
    result["pred_vertices"] = (np.asarray(block["vertices"]) - cam_t).astype(np.float32)
    result["pred_keypoints_3d"] = (kp_new - cam_t).astype(np.float32)
    result["pred_joint_coords"] = (after_joints - cam_t).astype(np.float32)
    result["pred_global_rots"] = new_rots.astype(np.float32)
    for alias, source in (("mhr_joint_coords", "pred_joint_coords"),
                          ("mhr_joint_global_rots", "pred_global_rots")):
        if alias in person:
            result[alias] = result[source]
    projected = float(np.asarray(person["focal_length"]).reshape(-1)[0]) * kp_new[:, :2] / kp_new[:, 2:3]
    result["pred_keypoints_2d"] = (projected + [width / 2, height / 2]).astype(np.float32)
    report["rotation_source"] = "after_joint_swing_preserving_input_twist"
    report["native_parameters_match_output"] = False
    return result, report


class GroundPoseEstimator:
    def __init__(self, resources: str | Path):
        self.resources = Path(resources).expanduser().resolve()
        vendor_dir = self.resources / "vendor" / "sam-3d-body"
        model_dir = self.resources / "models" / "sam-3d-body-vith"
        checkpoint = model_dir / "model.ckpt"
        mhr_model = model_dir / "assets" / "mhr_model.pt"
        if not vendor_dir.is_dir():
            raise FileNotFoundError(f"SAM3D source not found: {vendor_dir}")
        if not checkpoint.is_file() or not mhr_model.is_file():
            raise FileNotFoundError(f"SAM3D weights not found: {model_dir}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3D Body requires CUDA")

        resource_path = str(self.resources)
        if resource_path not in sys.path:
            sys.path.insert(0, resource_path)
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        model, cfg = load_sam_3d_body(
            checkpoint_path=str(checkpoint),
            mhr_path=str(mhr_model),
            device="cuda",
            offload_mode="cpu",
        )
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self._lock = threading.Lock()

    def infer(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray | None = None,
        want_overlay: bool = False,
        want_mesh: bool = False,
        cam_int: np.ndarray | None = None,
        include_raw: bool = False,
        bbox: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        ground_rotation: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB HxWx3 image")
        if bbox is None:
            if person_mask is None or person_mask.shape != rgb.shape[:2]:
                raise ValueError("Person mask dimensions do not match the image")
            bbox = mask_to_bbox(person_mask)
        else:
            bbox = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        height, width = rgb.shape[:2]

        mode = sam3d_inference_type()
        scale = sam3d_upscale()
        source = rgb
        model_bbox = bbox
        model_cam_int = cam_int
        if scale > 1.001:
            import cv2

            source = cv2.resize(
                np.ascontiguousarray(rgb), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            model_bbox = bbox * scale
            if cam_int is not None:
                model_cam_int = np.array(cam_int, dtype=np.float32, copy=True)
                model_cam_int[0, :] *= scale
                model_cam_int[1, :] *= scale

        with self._lock, torch.inference_mode():
            outputs = self.estimator.process_one_image(
                source,
                bboxes=model_bbox,
                use_mask=False,
                cam_int=(
                    torch.as_tensor(model_cam_int, dtype=torch.float32).unsqueeze(0)
                    if model_cam_int is not None else None
                ),
                inference_type=mode,
            )
        if not outputs:
            raise RuntimeError("SAM3D did not return a body pose")

        person = outputs[0]
        if scale > 1.001:
            # 放大只用于推理；2D 关键点与焦距要还原到原图像素坐标，3D/平移不受影响。
            person = dict(person)
            if person.get("pred_keypoints_2d") is not None:
                person["pred_keypoints_2d"] = (
                    np.asarray(person["pred_keypoints_2d"], dtype=np.float32) / scale
                )
            if person.get("focal_length") is not None:
                person["focal_length"] = (
                    np.asarray(person["focal_length"], dtype=np.float32) / scale
                )
            for key in ("lhand_bbox", "rhand_bbox"):
                if person.get(key) is not None:
                    person[key] = np.asarray(person[key], dtype=np.float32) / scale
        from .mhr_rotation_fit import (fit_native_rotations, rotation_fit_enabled,
                                       synchronize_native_person)

        rotation_enabled = rotation_fit_enabled()
        # Replay both baseline and fitted final hand/body state without changing vendor code.
        with self._lock, torch.inference_mode():
            model = self.estimator.model
            inner = getattr(model, "inner", model)
            head = inner.head_pose
            original_device = head.scale_mean.device
            try:
                head.to(self.estimator.device)
                person = synchronize_native_person(person, head)
                if rotation_enabled:
                    person = fit_native_rotations(
                        person, head, depth, person_mask, cam_int, (height, width)
                    )
            finally:
                head.to(original_device)
                if original_device.type == "cpu":
                    torch.cuda.empty_cache()
        # 旋转模式冻结尺度及相机参数，禁止再进入等比缩放或逐顶点 limb 修正。
        scale_fit: dict[str, Any] | None = None
        output_after = sam3d_output_after()
        fit_mode = "torso" if output_after and not rotation_enabled else sam3d_scale_fit_mode()
        if not rotation_enabled and fit_mode != "off" and depth is not None and cam_int is not None:
            scale_fit = fit_scale_to_depth(
                person["pred_vertices"], person["pred_cam_t"], depth, cam_int, person_mask,
                keypoints_2d=person.get("pred_keypoints_2d"), mode=fit_mode,
            )
            if scale_fit is not None:
                person = apply_uniform_scale(person, scale_fit["factor"])
        # 可选：在躯干锚之上再按可见点云逐肢体骨修顶点深度（只改网格，不动骨骼/关键点）。
        if fit_mode == "limb" and scale_fit is not None:
            new_verts, limb_detail = limb_scale_adjust(
                person["pred_vertices"], person["pred_cam_t"],
                person.get("pred_keypoints_3d"), person.get("pred_keypoints_2d"),
                depth, cam_int, person_mask, faces=self.estimator.faces,
            )
            scale_fit["limb"] = limb_detail
            if new_verts is not None:
                person = dict(person)
                person["pred_vertices"] = new_verts
        after_report = {"enabled": False, "applied": False, "reason": "disabled"}
        if output_after:
            person, after_report = apply_after_pose(
                person, self.estimator.faces, depth, person_mask, cam_int,
                (height, width), ground_rotation,
            )
            if scale_fit is not None:
                scale_fit["bonesnap"] = after_report
        from ai_vitpose.core.normalize import camera_points_to_figure_frame, skeleton24

        joints = camera_points_to_figure_frame(skeleton24(person["pred_joint_coords"]))
        if joints.shape != (24, 3) or not np.isfinite(joints).all():
            raise ValueError("SAM3D returned invalid SMPL-24 joints")
        camera = {
            "focal_px": float(np.asarray(person["focal_length"]).reshape(-1)[0]),
            "translation": np.asarray(person["pred_cam_t"]).reshape(-1).tolist(),
            "image_width": width,
            "image_height": height,
            "calibration": (
                "measured_intrinsics" if cam_int is not None
                else "model_default_fov_not_measured"
            ),
        }
        if not np.isfinite(camera["focal_px"]) or not np.isfinite(camera["translation"]).all():
            raise ValueError("SAM3D returned invalid camera parameters")
        result = {
            "status": "success",
            "engine": "sam3d_body_vith",
            "skeleton_source": "MHR127_direct",
            "joints_3d": joints.tolist(),
            "joints_3d_coordinate_system": "figure",
            "joints_3d_unit": "m",
            "orientation_mode": "fixed_camera_to_figure_v6",
            "bbox_2d_xyxy": bbox[0].tolist(),
            "camera": camera,
            "sam3d_inference_type": mode,
            "sam3d_upscale": scale,
            "pose_adjustment": after_report,
            "output_pose_stage": "after" if after_report.get("applied") else "before",
            "scale_fit": scale_fit if scale_fit is not None else {"mode": fit_mode, "factor": 1.0},
        }
        if rotation_enabled:
            result["rotation_fit"] = person["rotation_fit"]
            result["scale_fit"] = {"mode": "off", "factor": 1.0,
                                   "reason": "rotation_only_freezes_scale", "requested_mode": fit_mode}
        if include_raw:
            from .mhr_rotation_fit import ROTATION_RAW_FIELDS

            raw: dict[str, Any] = {}
            for key in RAW_POSE_FIELDS + ROTATION_RAW_FIELDS:
                value = person.get(key)
                if value is None:
                    continue
                try:
                    raw[key] = _to_jsonable(value)
                except Exception as exc:
                    raw[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
            raw["sam3d_inference_type"] = mode
            raw["sam3d_upscale"] = scale
            raw["pose_adjustment"] = after_report
            result["raw"] = raw
        if want_overlay:
            try:
                keypoints_2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
                if keypoints_2d.shape != (70, 2) or not np.isfinite(keypoints_2d).all():
                    raise ValueError("SAM3D returned invalid MHR-70 2D keypoints")
                result["overlay_png_data_uri"] = _pose_overlay_data_uri(rgb, keypoints_2d)
            except Exception as exc:
                result["overlay_error"] = f"{type(exc).__name__}: {exc}"
        if want_mesh:
            try:
                result["mesh_png_data_uri"] = _pose_mesh_data_uri(
                    person["pred_vertices"], self.estimator.faces
                )
                result["mesh_vertices_model"] = np.asarray(person["pred_vertices"], dtype=np.float32)
                result["mesh_faces"] = np.asarray(self.estimator.faces, dtype=np.int32)
                result["mesh_camera_translation"] = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(-1)
                result["mesh_focal_px"] = float(np.asarray(person["focal_length"]).reshape(-1)[0])
                # 配准用临时字段：MHR-70 关键点的原图像素坐标与相机系 3D 坐标。
                keypoints_2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
                keypoints_3d = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
                if keypoints_2d.shape == (70, 2) and keypoints_3d.shape == (70, 3):
                    result["align_keypoints_2d"] = keypoints_2d
                    result["align_keypoints_3d_cam"] = keypoints_3d + np.asarray(
                        person["pred_cam_t"], dtype=np.float32
                    ).reshape(1, 3)
            except Exception as exc:
                result["mesh_error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def refine_rigid_to_scene(
        vertices,
        keypoints,
        *,
        person_points=None,
        contact_points=None,
        keypoints_2d=None,
        focal_px=None,
        image_size=None,
        world_rotation=None,
        plane_d=0.0,
        iterations=120,
        lr=5e-3,
        contact_weight=200.0,
        penetration_weight=400.0,
        chamfer_weight=10.0,
        projection_weight=150.0,
        rotation_reg_weight=30.0,
        translation_reg_weight=15.0,
        max_rotation_deg=0.0,
        max_vertex_shift_m=0.15,
    ) -> dict[str, Any]:
        """刚体收缩精修：只优化整体平移（默认也允许整体旋转，由 ``max_rotation_deg`` 控制），
        姿势与尺度完全冻结。

        网格内部几何逐顶点保持不变，场景损失只能靠整体搬动网格降低，
        因此不会改写 SAM3D 预测的姿势。整体旋转默认关闭：实测绕接触区质心的
        整体转角会改变角色在图像中的朝向，反而恶化 2D 关键点一致性，
        而地面接触只需要沿重力轴平移即可达成。
        """
        report: dict[str, Any] = {"status": "skipped", "method": "rigid_refine+scene_losses"}
        base_np = np.asarray(vertices, dtype=np.float32)
        if base_np.ndim != 2 or base_np.shape[1] != 3 or len(base_np) < 100:
            report["reason"] = "网格形状异常"
            return {"report": report}
        kp_np = None
        if keypoints is not None:
            candidate = np.asarray(keypoints, dtype=np.float32)
            if candidate.ndim == 2 and candidate.shape[1] == 3 and len(candidate):
                kp_np = candidate

        device = torch.device("cpu")
        base = torch.as_tensor(base_np, device=device)
        if world_rotation is not None:
            world_z = torch.as_tensor(
                np.asarray(world_rotation, dtype=np.float32)[2, :], device=device
            )
        else:
            world_z = torch.tensor([0.0, 0.0, 1.0], device=device)

        with torch.no_grad():
            z_base = (base @ world_z) - plane_d
        z_np = z_base.numpy()
        n_verts = int(base.shape[0])
        # 只用最底部的极小区域（≈0.3% 顶点）定义接触高度，避免为压平整个贴地曲面而强行旋转。
        low_idx = np.argsort(z_np)[: min(60, n_verts)]
        rand_idx = np.random.default_rng(0).choice(n_verts, min(2000, n_verts), replace=False)
        sample_np = np.unique(np.concatenate([low_idx, rand_idx]))
        sample_idx = torch.as_tensor(sample_np, dtype=torch.long)
        low_idx_t = torch.as_tensor(low_idx, dtype=torch.long)
        pivot = base[low_idx_t].mean(dim=0, keepdim=True)
        base_centered = base - pivot
        base_sample = base[sample_idx]

        contact_target = 0.0
        contact_target_source = "ground_plane"
        if contact_points is not None and len(contact_points):
            rotation = (
                np.asarray(world_rotation, dtype=np.float32)
                if world_rotation is not None
                else np.eye(3, dtype=np.float32)
            )
            contact_z = np.asarray(contact_points, dtype=np.float32) @ rotation[2, :] - plane_d
            # 接触点本身应落在平面上；深度噪声会让中位数略高于 0，目标值不能为正。
            contact_target = float(np.clip(np.median(contact_z), -0.05, 0.0))
            contact_target_source = "contact_points_median"

        person_np = None
        nearest_t = None
        if person_points is not None and len(person_points) >= 100:
            import scipy.spatial

            person_np = np.asarray(person_points, dtype=np.float32)
            tree = scipy.spatial.cKDTree(person_np.astype(np.float64))
            _, nearest_index = tree.query(base_sample.numpy().astype(np.float64))
            nearest_t = torch.as_tensor(person_np[nearest_index], device=device)

        kp_cam = torch.as_tensor(kp_np, device=device) if kp_np is not None else None
        kp2d = None
        center = None
        focal = 1.0
        if (
            kp_np is not None
            and keypoints_2d is not None
            and focal_px is not None
            and image_size is not None
        ):
            kp2d_np = np.asarray(keypoints_2d, dtype=np.float32)
            if kp2d_np.shape == (len(kp_np), 2):
                valid = np.isfinite(kp2d_np).all(axis=1)
                if int(valid.sum()) >= 6:
                    kp_cam = torch.as_tensor(kp_np[valid], device=device)
                    kp2d = torch.as_tensor(kp2d_np[valid], device=device)
                    center = torch.tensor(
                        [float(image_size[1]) / 2.0, float(image_size[0]) / 2.0], device=device
                    )
                    focal = float(focal_px)

        max_rot_rad = float(np.deg2rad(max_rotation_deg))
        use_rotation = max_rot_rad > 0.0
        drot = torch.zeros(1, 3, device=device, requires_grad=use_rotation)
        dtrans = torch.zeros(1, 3, device=device, requires_grad=True)
        optimizer = torch.optim.Adam(
            [drot, dtrans] if use_rotation else [dtrans], lr=lr
        )

        def rigid(points):
            # 平移只沿地面法向（世界 z）施加：横向位置沿用前面的配准结果，
            # 避免整体横向漂移改变角色在画面中的位置与 2D 关键点一致性。
            shift = (dtrans @ world_z).unsqueeze(0) * world_z
            if not use_rotation:
                return points + shift
            theta = drot.norm(dim=1, keepdim=True).clamp_min(1e-8)
            axis = drot / theta
            kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
            zero = torch.zeros_like(kx)
            skew = torch.stack(
                [
                    torch.stack([zero, -kz, ky], dim=1),
                    torch.stack([kz, zero, -kx], dim=1),
                    torch.stack([-ky, kx, zero], dim=1),
                ],
                dim=1,
            )
            identity = torch.eye(3, device=device).unsqueeze(0)
            rot = (
                identity
                + torch.sin(theta).unsqueeze(2) * skew
                + (1.0 - torch.cos(theta)).unsqueeze(2) * (skew @ skew)
            )
            return points @ rot[0].T + shift

        metrics: dict[str, float] = {}
        for step in range(iterations):
            cam = rigid(base_centered) + pivot
            z = (cam @ world_z) - plane_d
            loss_contact = ((z[low_idx_t] - contact_target) ** 2).mean()
            loss_penetration = (torch.relu(-z[sample_idx] - 0.01) ** 2).mean()
            loss = contact_weight * loss_contact + penetration_weight * loss_penetration

            loss_chamfer = torch.zeros((), device=device)
            if nearest_t is not None:
                loss_chamfer = ((cam[sample_idx] - nearest_t) ** 2).sum(dim=1).mean()
                loss = loss + chamfer_weight * loss_chamfer

            loss_projection = torch.zeros((), device=device)
            if kp2d is not None:
                kp_now = rigid(kp_cam - pivot) + pivot
                kp_z = kp_now[:, 2]
                safe_z = torch.where(kp_z > 1e-3, kp_z, torch.ones_like(kp_z))
                uv = kp_now[:, :2] * (focal / safe_z).unsqueeze(1) + center
                error_px = ((uv - kp2d) ** 2).sum(dim=1)
                loss_projection = (error_px * (safe_z / focal) ** 2).mean()
                loss = loss + projection_weight * loss_projection

            loss_rotation = (drot ** 2).mean()
            loss_translation = (dtrans ** 2).mean()
            loss = (
                loss
                + rotation_reg_weight * loss_rotation
                + translation_reg_weight * loss_translation
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                angle = float(drot.norm())
                if angle > max_rot_rad:
                    drot.mul_(max_rot_rad / angle)

            if person_np is not None and step % 10 == 0:
                import scipy.spatial

                with torch.no_grad():
                    tree = scipy.spatial.cKDTree(person_np.astype(np.float64))
                    _, nearest_index = tree.query(
                        cam[sample_idx].numpy().astype(np.float64)
                    )
                    nearest_t = torch.as_tensor(person_np[nearest_index], device=device)

            if step in (0, iterations - 1):
                metrics = {
                    "contact_m2": round(float(loss_contact.detach()), 8),
                    "penetration_m2": round(float(loss_penetration.detach()), 8),
                    "chamfer_m2": round(float(loss_chamfer.detach()), 8),
                    "projection_m2": round(float(loss_projection.detach()), 8),
                    "rotation_reg": round(float(loss_rotation.detach()), 8),
                    "translation_reg": round(float(loss_translation.detach()), 8),
                    "total": round(float(loss.detach()), 8),
                }

        with torch.no_grad():
            cam = rigid(base_centered) + pivot
            kp_now = (rigid(kp_cam - pivot) + pivot) if kp_cam is not None else None
            z = (cam @ world_z) - plane_d
            z_final = z.numpy()
            final_vertices = cam.numpy().astype(np.float32)
            final_keypoints = kp_now.numpy().astype(np.float32) if kp_now is not None else None
            shift = float(torch.linalg.norm(cam - base, dim=1).max())
            delta_rot = float(torch.rad2deg(drot.detach().norm()))
            delta_trans = ((dtrans.detach() @ world_z) * world_z).numpy()
            penetration_after = float((z < -0.01).float().mean())
            penetration_before = float((z_base < -0.01).float().mean())
            contact_before = float(z_np[low_idx].mean())
            contact_after = float(z_final[low_idx].mean())

        report.update(
            {
                "iterations": iterations,
                "delta_rot_deg": round(delta_rot, 3),
                "delta_trans_m": np.round(delta_trans, 4).astype(float).tolist(),
                "rotation_limit_deg": max_rotation_deg,
                "contact_target_m": round(contact_target, 4),
                "contact_target_source": contact_target_source,
                "contact_height_before_m": round(contact_before, 4),
                "contact_height_after_m": round(contact_after, 4),
                "penetration_ratio_before": round(penetration_before, 4),
                "penetration_ratio_after": round(penetration_after, 4),
                "max_vertex_shift_m": round(shift, 4),
                "losses": metrics,
            }
        )
        if shift > max_vertex_shift_m:
            report["status"] = "rejected"
            report["reason"] = f"顶点位移 {shift:.3f} m 超过上限 {max_vertex_shift_m:.2f} m"
            return {"report": report}
        report["status"] = "applied"
        return {"vertices": final_vertices, "keypoints": final_keypoints, "report": report}
