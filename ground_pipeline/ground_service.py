"""
ground_pipeline 地面提取服务

- 一次启动加载 LangSAM（GroundingDINO + SAM2）与 DepthPro 模型
- 接口输入图片（Base64 data URI / HTTP URL / 本地路径），输出地面平面参数与点云采样

用法：
    python ground_service.py --port 8001 --host 0.0.0.0
"""

import argparse
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_serializer

import depth_pro
from lang_sam import LangSAM

from ground_pipeline import extract_ground

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path("/mnt/g/UGit/Phy-SIC")
DEFAULT_SAM_CKPT = ROOT / "data/sam2.1_hiera_small_langsam.pt"
DEFAULT_FLOOR_PROMPT = "floor, wooden floor."


class GroundRequest(BaseModel):
    """地面提取请求"""

    image: str = Field(..., description="Base64 data URI / HTTP URL / 本地文件路径")
    floor_prompt: Optional[str] = Field(DEFAULT_FLOOR_PROMPT, description="地面分割文本提示")
    person_prompt: Optional[str] = Field("person.", description="人物分割文本提示；传空串则跳过接触区域检测")
    contact_threshold: Optional[float] = Field(0.05, gt=0, description="接触区域判定距离阈值（米）：人物 mask 反投影点与拟合地面平面的距离小于该值即视为接触")
    distance_threshold: Optional[float] = Field(0.02, gt=0, description="平面拟合距离阈值（米）")
    iterations: Optional[int] = Field(256, ge=1, description="RANSAC 迭代次数")
    grid_steps: Optional[int] = Field(100, ge=2, description="平面网格分辨率")
    return_data: Optional[bool] = Field(False, description="为 true 时额外返回 masks（base64 PNG）、人物点云采样与全量深度图（.npy base64）")
    person_points_limit: Optional[int] = Field(2000, ge=1, le=100000, description="return_data 时人物点云采样上限")
    visualize: Optional[bool] = Field(False, description="返回可视化图片（overlay 覆盖图 + cloud 3D 点云图）")
    pose_inference: Optional[bool] = Field(False, description="对人物 mask 的单人 2D bbox 运行 SAM3D Body 姿势推理（不运行 ViTPose）")
    pose_want_overlay: Optional[bool] = Field(False, description="在 visualize=true 时额外返回 SAM3D 2D 姿势叠加图")
    pose_want_mesh: Optional[bool] = Field(False, description="在 visualize=true 时额外返回 SAM3D Body 3D 网格多视图图像")


class GroundResult(BaseModel):
    """地面提取结果"""

    ok: bool
    plane_normal: List[float]
    plane_d: float
    camera_height_m: float
    focal_px: float
    depth_range_m: List[float]
    image_hw: List[int]
    floor_pixels: int
    plane_inliers: int
    plane_points: List[List[float]]
    inlier_points: List[List[float]]
    person_pixels: Optional[int] = None
    person_bbox_3d_m: Optional[List[float]] = None
    person_centroid_m: Optional[List[float]] = None
    contact_pixels: Optional[int] = None
    contact_centroid_m: Optional[List[float]] = None
    contact_bbox: Optional[List[int]] = None
    masks: Optional[Dict[str, Optional[str]]] = None
    person_points: Optional[List[List[float]]] = None
    depth_b64: Optional[str] = None
    visualization: Optional[Dict[str, str]] = None
    pose: Optional[Dict[str, Any]] = None
    # 相机系 / 世界系分组输出（顶层旧字段保留，向后兼容）
    camera: Optional[Dict[str, Any]] = None
    world: Optional[Dict[str, Any]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        """默认关闭字段：masks / person_points / depth_b64 为 None 时不出现在 JSON 响应中（仅 return_data=true 时返回）。"""
        data = handler(self)
        for _field in ("masks", "person_points", "depth_b64", "pose"):
            if data.get(_field) is None:
                data.pop(_field, None)
        return data


def _load_image(source: str):
    """加载图片：支持 Base64 data URI、HTTP URL、本地文件路径，返回 HxWx3 RGB ndarray。"""
    if source.startswith("data:image"):
        header, encoded = source.split(",", 1)
        image_data = base64.b64decode(encoded)
        return np.asarray(__import__("PIL").Image.open(io.BytesIO(image_data)).convert("RGB"))
    elif source.startswith("http://") or source.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(source) as resp:
            data = resp.read()
        return np.asarray(__import__("PIL").Image.open(io.BytesIO(data)).convert("RGB"))
    else:
        return np.asarray(__import__("PIL").Image.open(source).convert("RGB"))


def _to_data_uri(buf: io.BytesIO) -> str:
    """BytesIO PNG -> base64 data URI，便于下游直接 <img src> 使用。"""
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _mask_png(mask: np.ndarray) -> str:
    """二值 mask -> 8-bit 灰度 PNG base64 data URI（0/255，可用 PIL/OpenCV 直接还原为布尔数组）。"""
    from PIL import Image

    im = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return _to_data_uri(buf)


def _npy_b64(arr: np.ndarray) -> str:
    """ndarray -> np.save 字节流 base64（float32 .npy，下游 np.load(io.BytesIO(base64.b64decode(s))) 还原）。"""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.float32))
    return base64.b64encode(buf.getvalue()).decode()


def _person_mask(segmenter, image_pil, person_prompt: str) -> Optional[np.ndarray]:
    """用 LangSAM 分割人物 mask：GroundingDINO 文本检测 + SAM 分割，多实例取并集。

    提示词为空或未检出人物时返回 None，不抛错。
    """
    if person_prompt is None or not person_prompt.strip():
        return None
    segmentation = segmenter.predict([image_pil], [person_prompt])[0]
    masks = np.asarray(segmentation["masks"], dtype=bool)
    if masks.ndim == 2:
        return masks
    if masks.ndim == 3 and masks.shape[0] > 0:
        return masks.any(axis=0)
    return None


def _contact_region(
    person_mask: Optional[np.ndarray],
    points: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    threshold: float,
) -> tuple[int, Optional[np.ndarray]]:
    """人物 mask 与地面平面的几何接触区域。

    判据：人物 mask 像素反投影到相机系 3D 点后，与拟合地面平面 |n·p + d| <= threshold，
    即该部位贴在地面上。返回 (接触像素数, 接触 mask)；无接触时 mask 为 None。
    """
    if person_mask is None:
        return 0, None
    distance = np.abs(points @ plane_normal + plane_d)
    valid = np.isfinite(points).all(axis=-1)
    contact = person_mask & valid & (distance <= threshold)
    count = int(contact.sum())
    return (count, contact if count > 0 else None)


def _overlay_png(rgb, floor_mask, inlier_mask, person_mask=None, contact_mask=None) -> str:
    """原图 + 地面 mask（绿）+ 平面内点 mask（蓝）+ 人物 mask（黄）+ 接触区域（红）覆盖图。"""
    from PIL import Image

    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    green = Image.new("RGBA", base.size, (46, 204, 113, 70))
    overlay.paste(green, (0, 0), Image.fromarray((floor_mask.astype(np.uint8)) * 255))
    blue = Image.new("RGBA", base.size, (41, 128, 185, 110))
    overlay.paste(blue, (0, 0), Image.fromarray((inlier_mask.astype(np.uint8)) * 255))
    if person_mask is not None:
        yellow = Image.new("RGBA", base.size, (241, 196, 15, 50))
        overlay.paste(yellow, (0, 0), Image.fromarray((person_mask.astype(np.uint8)) * 255))
    if contact_mask is not None:
        red = Image.new("RGBA", base.size, (231, 76, 60, 160))
        overlay.paste(red, (0, 0), Image.fromarray((contact_mask.astype(np.uint8)) * 255))
    merged = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return _to_data_uri(buf)


def rotate_to_up(n, up=(0.0, 0.0, -1.0)):
    """Rodrigues 旋转：把法向量 n 对齐到 up。"""
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


def _world_frame_data(
    *,
    plane_normal,
    plane_d: float,
    plane_points,
    inlier_points,
    person_points=None,
    contact_points=None,
) -> dict:
    """相机系数据 -> 世界系（地面 z=0、法向朝上），供接口 world 分组输出。

    返回字典（值为 ndarray / float）：
      rotation / plane_normal / plane_d(=0) / plane_points / inlier_points
      camera_position / camera_forward / camera_height_m
      person_points / person_bbox_3d_m / person_centroid_m / contact_centroid_m
    """
    R = rotate_to_up(np.asarray(plane_normal, dtype=float))
    d = float(plane_d)
    w_plane = np.asarray(plane_points, dtype=float) @ R.T
    w_plane[:, 2] -= d
    w_in = np.asarray(inlier_points, dtype=float) @ R.T
    w_in[:, 2] -= d
    out = {
        "rotation": R,
        "plane_normal": (R @ np.asarray(plane_normal, dtype=float)).astype(float),
        "plane_d": 0.0,
        "plane_points": w_plane,
        "inlier_points": w_in,
        "camera_position": np.array([0.0, 0.0, -d], dtype=float),
        "camera_forward": (R @ np.array([0.0, 0.0, 1.0])).astype(float),
        "camera_height_m": abs(d),
    }
    if person_points is not None and len(person_points):
        wp = np.asarray(person_points, dtype=float) @ R.T
        wp[:, 2] -= d
        mn, mx = wp.min(axis=0), wp.max(axis=0)
        out["person_points"] = wp
        out["person_bbox_3d_m"] = np.array([mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]], dtype=float)
        out["person_centroid_m"] = wp.mean(axis=0).astype(float)
    if contact_points is not None and len(contact_points):
        wc = np.asarray(contact_points, dtype=float) @ R.T
        wc[:, 2] -= d
        out["contact_centroid_m"] = wc.mean(axis=0).astype(float)
    return out


def _prepare_pose_mesh_for_views(pose):
    """取出 SAM3D 网格数据，返回 (相机系模型网格, 面片, 原始模型网格)。"""
    vertices = np.asarray(pose.pop("mesh_vertices_model"), dtype=np.float32)
    faces = np.asarray(pose.pop("mesh_faces"), dtype=np.int32)
    translation = np.asarray(pose.pop("mesh_camera_translation"), dtype=np.float32).reshape(1, 3)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("SAM3D returned invalid mesh vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("SAM3D returned invalid mesh faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("SAM3D mesh faces reference invalid vertices")
    return vertices + translation, faces, vertices


def _normalize_weights(weights, count):
    """把权重归一化为和为 1；非法输入退化为均匀权重。"""
    if weights is None:
        return np.full(count, 1.0 / count)
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    if len(w) != count or w.sum() <= 1e-12:
        return np.full(count, 1.0 / count)
    return w / w.sum()


def _umeyama_similarity(src, dst, weights=None):
    """加权最小二乘相似变换，求 dst ≈ scale * R @ src + t。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    w = _normalize_weights(weights, len(src))
    src_mean = (w[:, None] * src).sum(axis=0)
    dst_mean = (w[:, None] * dst).sum(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (w[:, None] * dst_centered).T @ src_centered
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = float((w[:, None] * src_centered ** 2).sum())
    if variance <= 1e-12:
        raise ValueError("关键点退化，无法估计相似变换")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def _rigid_fit(src, dst, weights=None):
    """尺度固定为 1 的加权刚体拟合，求 dst ≈ R @ src + t。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    w = _normalize_weights(weights, len(src))
    src_mean = (w[:, None] * src).sum(axis=0)
    dst_mean = (w[:, None] * dst).sum(axis=0)
    covariance = (w[:, None] * (dst - dst_mean)).T @ (src - src_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    translation = dst_mean - rotation @ src_mean
    return rotation, translation


def _keypoints_depth_camera(keypoints_2d, depth, intrinsics, person_mask):
    """在关键点像素位置采样 DepthPro 深度，反投影为相机系点并给出置信度权重。"""
    height, width = depth.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    radius = 4
    points = []
    indices = []
    weights = []
    for index, (u, v) in enumerate(np.asarray(keypoints_2d, dtype=float)):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < width and 0 <= vi < height):
            continue
        y0, y1 = max(0, vi - radius), min(height, vi + radius + 1)
        x0, x1 = max(0, ui - radius), min(width, ui + radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid = np.isfinite(patch) & (patch > 0)
        if person_mask is not None:
            valid &= person_mask[y0:y1, x0:x1]
        if int(valid.sum()) < 3:
            continue
        values = patch[valid]
        z = float(np.median(values))
        # 邻域深度离散度：越大说明边缘/遮挡，权重越低
        spread = float(np.median(np.abs(values - z)))
        # 人物 mask 占比：越低说明采到了背景/地面
        mask_fraction = float(valid.sum()) / float(patch.size)
        weight = mask_fraction / (1.0 + (spread / 0.04) ** 2)
        points.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
        indices.append(index)
        weights.append(weight)
    return (
        np.asarray(points, dtype=float),
        np.asarray(indices, dtype=int),
        np.asarray(weights, dtype=float),
    )


def _trimmed_icp(
    source,
    target,
    iterations=12,
    trim_ratio=0.7,
    max_shift_m=0.20,
    max_rotation_deg=5.0,
    allow_rotation=False,
):
    """固定尺度的 trimmed ICP：把 source 贴合到 target 表面。

    人体侧面近似柱面，纯表面 ICP 的偏航/滚转不可辨，容易越转越偏；
    因此默认只做平移（姿态朝向交给关键点 Umeyama），并对累计位移设上限。
    """
    from scipy.spatial import cKDTree

    points = np.asarray(source, dtype=float).copy()
    reference = np.asarray(target, dtype=float)
    reference = reference[np.isfinite(reference).all(axis=1)]
    if len(reference) < 100 or len(points) < 100:
        return points, np.eye(3), np.zeros(3), {"status": "skipped", "reason": "点云数量不足"}
    tree = cKDTree(reference)
    rotation = np.eye(3)
    translation = np.zeros(3)
    trim = float(np.clip(trim_ratio, 0.1, 1.0))
    keep = int(max(100, round(len(points) * trim)))
    limited = False
    for _ in range(int(iterations)):
        distances, indices = tree.query(points, k=1, workers=-1)
        order = np.argsort(distances)[:keep]
        if allow_rotation:
            step_rotation, step_translation = _rigid_fit(points[order], reference[indices[order]])
        else:
            step_rotation = np.eye(3)
            step_translation = reference[indices[order]].mean(axis=0) - points[order].mean(axis=0)
        candidate_rotation = step_rotation @ rotation
        candidate_translation = step_rotation @ translation + step_translation
        angle = np.degrees(
            np.arccos(np.clip((np.trace(candidate_rotation) - 1.0) / 2.0, -1.0, 1.0))
        )
        if angle > max_rotation_deg or np.linalg.norm(candidate_translation) > max_shift_m:
            limited = True
            break
        candidate = points @ step_rotation.T + step_translation
        shift = float(np.linalg.norm(candidate.mean(axis=0) - points.mean(axis=0)))
        points = candidate
        rotation = candidate_rotation
        translation = candidate_translation
        if shift < 1e-4:
            break
    distances = tree.query(points, k=1, workers=-1)[0]
    trimmed = np.sort(distances)[:keep]
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    return points, rotation, translation, {
        "status": "success",
        "mode": "rigid" if allow_rotation else "translation",
        "trim_ratio": round(trim, 3),
        "correspondence_mean_m": round(float(distances.mean()), 4),
        "trimmed_rmse_m": round(float(np.sqrt(np.mean(trimmed ** 2))), 4),
        "rotation_deg": round(float(angle), 2),
        "rotation_limit_deg": max_rotation_deg,
        "translation_m": np.round(translation, 4).astype(float).tolist(),
        "shift_m": round(float(np.linalg.norm(translation)), 4),
        "shift_limit_m": max_shift_m,
        "limited": limited,
    }


def _ground_snap_offset(points, plane_normal, plane_d, quantile=0.01, limit_m=0.30):
    """沿重力轴 1-DOF 平移量，使最低分位高度贴合地面（世界 z=0）。"""
    rotation = rotate_to_up(np.asarray(plane_normal, dtype=float))
    heights = (np.asarray(points, dtype=float) @ rotation.T)[:, 2] - float(plane_d)
    contact = float(np.quantile(heights, quantile))
    applied = float(np.clip(-contact, -limit_m, limit_m))
    return applied, contact, -contact


def _contact_snap_offset(
    mesh_points,
    contact_points,
    plane_normal,
    plane_d,
    radius_m=0.15,
    limit_m=0.30,
    sink_tolerance_m=0.03,
    min_candidates=30,
    cell_m=0.10,
):
    """按检测到的身体-地面接触区域决定网格沿重力轴的落位。

    只看网格自身最低分位时，只有最低的那一小块会贴地；这里改为：把接触点在世界系
    按水平位置聚成若干簇（臀部、手、脚等），要求每簇附近的网格表面贴到 z=0，
    取各簇所需平移的中位数，并限制任何顶点沉入地面不超过 sink_tolerance_m。

    返回 (offset, report)；接触点过少或附近无网格时返回 None，由调用方回退。
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    rotation = rotate_to_up(np.asarray(plane_normal, dtype=float))
    mesh_world = np.asarray(mesh_points, dtype=float) @ rotation.T
    mesh_world[:, 2] -= float(plane_d)
    contact_world = np.asarray(contact_points, dtype=float) @ rotation.T
    contact_world[:, 2] -= float(plane_d)
    if len(mesh_world) == 0 or len(contact_world) < 20:
        return None

    xy = contact_world[:, :2]
    cell = float(cell_m)
    origin = xy.min(axis=0) - cell
    cells = np.floor((xy - origin) / cell).astype(int)
    raster = np.zeros(cells.max(axis=0) + 1, dtype=bool)
    raster[cells[:, 0], cells[:, 1]] = True
    # 先膨胀再连通标记，避免同一接触部位被打散成很多碎簇
    raster = ndimage.binary_dilation(raster, np.ones((3, 3), dtype=bool), iterations=2)
    labels, cluster_total = ndimage.label(raster, structure=np.ones((3, 3), dtype=bool))
    point_cluster = labels[cells[:, 0], cells[:, 1]]

    tree = cKDTree(xy)
    distances, nearest = tree.query(mesh_world[:, :2], k=1, workers=-1)
    candidate = distances <= radius_m
    if int(candidate.sum()) < min_candidates:
        return None

    cluster_ids = point_cluster[nearest[candidate]]
    candidate_height = mesh_world[candidate, 2]
    clusters = []
    for cluster in np.unique(cluster_ids):
        heights = candidate_height[cluster_ids == cluster]
        if heights.size < 3:
            continue
        clusters.append(
            {
                "cluster": int(cluster),
                "contact_points": int((point_cluster == cluster).sum()),
                "mesh_vertices": int(heights.size),
                "lowest_before_m": round(float(heights.min()), 4),
            }
        )
    if not clusters:
        return None

    lowest = float(mesh_world[:, 2].min())
    desired = -float(np.median([item["lowest_before_m"] for item in clusters]))
    # 不允许把网格整体压到地面以下超过 sink_tolerance_m
    lower_bound = max(-limit_m, -sink_tolerance_m - lowest)
    offset = float(np.clip(desired, lower_bound, limit_m))
    for item in clusters:
        item["gap_after_m"] = round(float(item["lowest_before_m"]) + offset, 4)

    report = {
        "status": "applied" if abs(offset) >= 0.01 else "skipped",
        "source": "contact_footprint",
        "offset_m": round(offset, 4),
        "contact_points": int(len(contact_world)),
        "contact_clusters": int(cluster_total),
        "clusters_with_mesh": len(clusters),
        "clusters": clusters,
        "radius_m": radius_m,
        "lowest_vertex_before_m": round(lowest, 4),
        "lowest_vertex_after_m": round(lowest + offset, 4),
        "limit_m": limit_m,
        "sink_tolerance_m": sink_tolerance_m,
        "limited": bool(abs(offset - desired) > 1e-6),
        "axis_world": "z",
    }
    if report["status"] == "skipped":
        report["reason"] = "接触落位位移小于 1cm"
    return offset, report


def _contact_plane_rotation(
    mesh_points,
    contact_points,
    plane_normal,
    plane_d,
    radius_m=0.15,
    max_angle_deg=5.0,
    min_points=30,
    low_quantile=0.3,
):
    """拟合网格接触面的法向，求把它转平到世界水平面的小角度旋转。

    地面只约束 3 个自由度，但 Umeyama 的旋转可能让身体相对地面翘起；这里用接触区域
    附近的网格低点拟合接触面，把该面转平（转角上限 max_angle_deg），再由垂直落位贴地。

    返回 (rotation_cam, translation_cam, report)；接触点不足或接触面已足够平时返回 None。
    """
    from scipy.spatial import cKDTree

    rotation_world = rotate_to_up(np.asarray(plane_normal, dtype=float))
    mesh_world = np.asarray(mesh_points, dtype=float) @ rotation_world.T
    mesh_world[:, 2] -= float(plane_d)
    contact_world = np.asarray(contact_points, dtype=float) @ rotation_world.T
    contact_world[:, 2] -= float(plane_d)
    if len(contact_world) < 20 or len(mesh_world) < min_points:
        return None

    tree = cKDTree(contact_world[:, :2])
    distances, _ = tree.query(mesh_world[:, :2], k=1, workers=-1)
    selected = mesh_world[distances <= radius_m]
    if len(selected) < min_points:
        return None

    heights = selected[:, 2]
    low = selected[heights <= float(np.quantile(heights, low_quantile))]
    if len(low) < min_points:
        return None

    normal = np.array([0.0, 0.0, 1.0])
    centroid = low.mean(axis=0)
    for _ in range(3):
        centered = low - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[2]
        if normal[2] < 0:
            normal = -normal
        distance = np.abs(centered @ normal)
        spread = float(np.median(np.abs(distance - np.median(distance))))
        keep = distance <= max(float(np.median(distance) + 3.0 * 1.4826 * spread), 0.005)
        if keep.all() or int(keep.sum()) < min_points:
            break
        low = low[keep]
        centroid = low.mean(axis=0)

    angle = float(np.degrees(np.arccos(np.clip(float(normal[2]), -1.0, 1.0))))
    applied = min(angle, float(max_angle_deg))
    report = {
        "status": "skipped",
        "angle_deg": round(angle, 2),
        "applied_deg": round(applied, 2),
        "max_angle_deg": max_angle_deg,
        "points": int(len(low)),
        "normal_world": np.round(normal, 4).astype(float).tolist(),
    }
    if applied < 0.3:
        report["reason"] = "接触面已接近水平"
        return None

    axis = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        report["reason"] = "接触面法向退化"
        return None
    axis = axis / length
    theta = np.radians(applied)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation_up = np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)

    report["status"] = "applied"
    report["limited"] = bool(angle > max_angle_deg)
    report["pivot_world"] = np.round(centroid, 4).astype(float).tolist()
    rotation_cam = rotation_world.T @ rotation_up @ rotation_world
    translation_cam = rotation_world.T @ (
        (np.eye(3) - rotation_up) @ (centroid + float(plane_d) * np.array([0.0, 0.0, 1.0]))
    )
    return rotation_cam, translation_cam, report


def _align_pose_mesh_by_ground(
    mesh_points,
    plane_normal,
    plane_d,
    max_tilt_deg=10.0,
    low_quantile=0.01,
    snap_quantile=0.005,
    limit_m=0.30,
    min_points=200,
    min_contact_points=40,
):
    """相机系地面锚定：只做刚体变换（不缩放），把 SAM3D 自带接触面转平并落位。

    SAM3D 的原始输出在相机空间通常已经站在它自己感知到的地面上，与 DepthPro 检测到的
    相机系地面之间只差「接触面倾角」和「几厘米的高度」。因此这里取网格最低分位顶点
    拟合接触面，把该面法向旋到地面法向（转角上限 max_tilt_deg），再沿地面法向平移
    使最低分位高度归零。尺度保持 1.0，姿势完全冻结。

    返回 (rotation_cam, translation_cam, report)，变换为 p' = p @ rotation_cam.T + translation_cam；
    不需要/不能锚定时 rotation 为单位阵、translation 为零，report["status"] == "skipped"。
    """
    points = np.asarray(mesh_points, dtype=float)
    report = {"status": "skipped", "method": "camera_ground_anchor", "scale": 1.0}
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < min_points:
        report["reason"] = "网格顶点不足"
        return np.eye(3), np.zeros(3), report
    normal = np.asarray(plane_normal, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        report["reason"] = "地面法向退化"
        return np.eye(3), np.zeros(3), report
    normal = normal / norm
    up = -normal  # 世界 +z 在相机系中的方向

    heights = points @ up - float(plane_d)
    low_count = max(min_contact_points, int(max(low_quantile, 0.01) * len(points)))
    low = points[np.argsort(heights)[:low_count]]

    # 稳健拟合接触面法向（截断 3 轮）
    contact_normal = up.copy()
    centroid = low.mean(axis=0)
    for _ in range(3):
        centered = low - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        contact_normal = vt[-1]
        if float(contact_normal @ up) < 0.0:
            contact_normal = -contact_normal
        distance = np.abs(centered @ contact_normal)
        spread = float(np.median(np.abs(distance - np.median(distance))))
        keep = distance <= max(float(np.median(distance) + 3.0 * 1.4826 * spread), 0.005)
        if keep.all() or int(keep.sum()) < min_contact_points:
            break
        low = low[keep]
        centroid = low.mean(axis=0)

    tilt = float(np.degrees(np.arccos(np.clip(float(contact_normal @ up), -1.0, 1.0))))
    report.update(
        {
            "contact_points": int(len(low)),
            "contact_normal_cam": np.round(contact_normal, 4).astype(float).tolist(),
            "tilt_deg": round(tilt, 2),
            "max_tilt_deg": float(max_tilt_deg),
        }
    )
    if tilt > float(max_tilt_deg):
        report["reason"] = "接触面倾角超过上限，回退关键点配准"
        return np.eye(3), np.zeros(3), report

    applied_tilt = tilt if tilt >= 0.3 else 0.0
    rotation = np.eye(3)
    axis = np.cross(contact_normal, up)
    axis_norm = float(np.linalg.norm(axis))
    if applied_tilt > 0.0 and axis_norm > 1e-9:
        axis = axis / axis_norm
        theta = np.radians(applied_tilt)
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        rotation = np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    translation = centroid - centroid @ rotation.T

    contact_before = float(np.quantile(heights, snap_quantile))
    rotated = points @ rotation.T + translation
    contact_rotated = float(np.quantile(rotated @ up - float(plane_d), snap_quantile))
    shift = float(np.clip(-contact_rotated, -limit_m, limit_m))
    translation = translation + shift * up

    report.update(
        {
            "applied_tilt_deg": round(applied_tilt, 2),
            "pivot_cam": np.round(centroid, 4).astype(float).tolist(),
            "axis_cam": np.round(axis, 4).astype(float).tolist() if axis_norm > 1e-9 else [0.0, 0.0, 0.0],
            "contact_quantile": float(snap_quantile),
            "contact_height_before_m": round(contact_before, 4),
            "contact_height_after_m": round(contact_rotated + shift, 4),
            "shift_m": round(shift, 4),
            "shift_limit_m": float(limit_m),
            "lowest_before_m": round(float(heights.min()), 4),
            "lowest_after_m": round(float((rotated @ up - float(plane_d)).min() + shift), 4),
        }
    )
    if applied_tilt < 0.3 and abs(shift) < 0.005:
        report["reason"] = "网格已贴合检测地面，无需搬动"
        return np.eye(3), np.zeros(3), report
    report["status"] = "applied"
    return rotation, translation, report


def _keypoint_diagnostics(source, target, residual, indices, weights, inliers, top_k=12):
    """按残差降序给出最差关键点的诊断信息。"""
    records = []
    for position, index in enumerate(np.asarray(indices)):
        if not bool(inliers[position]):
            continue
        records.append(
            {
                "index": int(index),
                "residual_m": round(float(residual[position]), 4),
                "weight": round(float(weights[position]), 4),
                "sam3d_cam": np.round(source[position], 4).astype(float).tolist(),
                "depth_cam": np.round(target[position], 4).astype(float).tolist(),
            }
        )
    records.sort(key=lambda item: item["residual_m"], reverse=True)
    return records[: int(top_k)]


def _align_pose_mesh_to_depth(
    pose,
    vertices,
    depth,
    intrinsics,
    person_mask,
    person_points=None,
    plane_normal=None,
    plane_d=0.0,
    contact_points=None,
    pose_estimator=None,
):
    """加权 Umeyama → trimmed ICP → 地面垂直吸附 → 刚体（6-DOF）精修。"""
    keypoints_2d = pose.pop("align_keypoints_2d", None)
    keypoints_3d = pose.pop("align_keypoints_3d_cam", None)
    focal_px = pose.pop("mesh_focal_px", None)
    if keypoints_2d is None or keypoints_3d is None:
        return vertices, {"status": "skipped", "reason": "SAM3D 未返回 MHR-70 关键点"}
    keypoints_2d = np.asarray(keypoints_2d, dtype=float)
    keypoints_3d = np.asarray(keypoints_3d, dtype=float)
    if keypoints_2d.shape != (70, 2) or keypoints_3d.shape != (70, 3):
        return vertices, {"status": "skipped", "reason": "关键点形状异常"}
    target, valid_indices, weights = _keypoints_depth_camera(
        keypoints_2d, depth, intrinsics, person_mask
    )
    if len(valid_indices) < 6:
        return vertices, {
            "status": "skipped",
            "reason": "有效深度关键点不足",
            "keypoints_used": int(len(valid_indices)),
        }
    source = keypoints_3d[valid_indices]
    inliers = np.ones(len(source), dtype=bool)

    # 诊断：SAM3D 原始网格（相机系）与检测到的相机系地面平面的关系。
    # 用于判断「模型在相机空间是否已经踩在检测地面上」。
    camera_ground_check = None
    raw_vertices = np.asarray(vertices, dtype=float)
    if plane_normal is not None and len(raw_vertices) >= 100:
        n_plane = np.asarray(plane_normal, dtype=float)
        n_plane = n_plane / max(float(np.linalg.norm(n_plane)), 1e-12)
        # 与世界系一致：height = p·R[2] - plane_d，正值表示在地面之上。
        heights = (raw_vertices @ rotate_to_up(n_plane).T)[:, 2] - float(plane_d)
        low_count = max(20, int(0.01 * len(raw_vertices)))
        low_pts = raw_vertices[np.argsort(heights)[:low_count]]
        centered = low_pts - low_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        n_low = vt[-1]
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(n_low @ n_plane)), 0.0, 1.0))))
        camera_ground_check = {
            "plane_normal": np.round(n_plane, 4).astype(float).tolist(),
            "plane_d_m": round(float(plane_d), 4),
            "mesh_height_m": {
                "min": round(float(heights.min()), 4),
                "p1": round(float(np.percentile(heights, 1)), 4),
                "p5": round(float(np.percentile(heights, 5)), 4),
                "p50": round(float(np.percentile(heights, 50)), 4),
                "max": round(float(heights.max()), 4),
            },
            "mesh_low_plane_tilt_deg": round(tilt, 2),
        }
    # A. 首选：相机系地面锚定（刚体、不缩放）。SAM3D 在相机空间本就站在自己感知到的
    #    地面上，与 DepthPro 地面的差异只有接触面倾角与几厘米高度；关键点相似变换的
    #    0.79 尺度正是姿势走样的根因，因此锚定成功时完全跳过它。
    ground_anchor = {"status": "skipped", "reason": "未提供地面平面"}
    if plane_normal is not None:
        anchor_rotation, anchor_translation, ground_anchor = _align_pose_mesh_by_ground(
            np.asarray(vertices, dtype=float), plane_normal, plane_d
        )
    else:
        anchor_rotation, anchor_translation = np.eye(3), np.zeros(3)

    if ground_anchor.get("status") == "applied":
        scale, rotation, translation = 1.0, anchor_rotation, anchor_translation
        residual = np.linalg.norm(
            source @ (scale * rotation).T + translation - target, axis=1
        )
        deviation = np.median(np.abs(residual - np.median(residual)))
        threshold = max(float(np.median(residual) + 3.0 * 1.4826 * deviation), 0.02)
        inliers = residual <= threshold
        if int(inliers.sum()) < 6:
            inliers = np.ones(len(source), dtype=bool)
        aligned = scale * (np.asarray(vertices, dtype=float) @ rotation.T) + translation
        keypoint_cam = source @ (scale * rotation).T + translation
        report = {
            "status": "success",
            "method": "camera_ground_anchor",
            "keypoints_used": int(inliers.sum()),
            "keypoints_total": int(len(source)),
            "scale": 1.0,
            "rmse_m": round(float(np.sqrt(np.mean(residual[inliers] ** 2))), 4),
            "rmse_all_m": round(float(np.sqrt(np.mean(residual ** 2))), 4),
            "rotation_deg": ground_anchor.get("applied_tilt_deg", 0.0),
            "translation_m": np.round(translation, 4).astype(float).tolist(),
            "keypoint_weight": round(float(weights[inliers].sum()), 4),
            "camera_ground_check": camera_ground_check,
            "ground_anchor": ground_anchor,
            "keypoint_diagnostics": _keypoint_diagnostics(
                source, target, residual, valid_indices, weights, inliers
            ),
        }
    else:
        scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
        for _ in range(3):
            try:
                scale, rotation, translation = _umeyama_similarity(
                    source[inliers], target[inliers], weights[inliers]
                )
            except ValueError as exc:
                return vertices, {"status": "skipped", "reason": str(exc)}
            residual = np.linalg.norm(source @ (scale * rotation).T + translation - target, axis=1)
            deviation = np.median(np.abs(residual - np.median(residual)))
            threshold = max(float(np.median(residual) + 3.0 * 1.4826 * deviation), 0.02)
            updated = residual <= threshold
            if int(updated.sum()) < 6 or np.array_equal(updated, inliers):
                break
            inliers = updated
        if not (0.5 <= scale <= 2.0):
            return vertices, {
                "status": "skipped",
                "reason": "相似变换尺度异常",
                "scale": round(float(scale), 4),
                "keypoints_used": int(inliers.sum()),
            }
        residual = np.linalg.norm(source @ (scale * rotation).T + translation - target, axis=1)
        aligned = scale * (np.asarray(vertices, dtype=float) @ rotation.T) + translation
        keypoint_cam = source @ (scale * rotation).T + translation

        report = {
            "status": "success",
            "method": "weighted_keypoint_umeyama+depthpro",
            "keypoints_used": int(inliers.sum()),
            "keypoints_total": int(len(source)),
            "scale": round(float(scale), 4),
            "rmse_m": round(float(np.sqrt(np.mean(residual[inliers] ** 2))), 4),
            "rmse_all_m": round(float(np.sqrt(np.mean(residual ** 2))), 4),
            "translation_m": np.round(translation, 4).astype(float).tolist(),
            "keypoint_weight": round(float(weights[inliers].sum()), 4),
            "camera_ground_check": camera_ground_check,
            "ground_anchor": ground_anchor,
            "keypoint_diagnostics": _keypoint_diagnostics(
                source, target, residual, valid_indices, weights, inliers
            ),
        }

    # B. 固定尺度的表面 trimmed ICP（只修旋转与平移，不动姿态尺度）
    icp = {"status": "skipped", "reason": "未提供人物点云"}
    if report["method"] == "camera_ground_anchor":
        icp = {"status": "skipped", "reason": "地面锚定路径不做表面 ICP，避免改变已贴地的刚性"}
    elif person_points is not None and len(person_points) >= 100:
        try:
            aligned, icp_rotation, icp_translation, icp = _trimmed_icp(aligned, person_points)
            keypoint_cam = keypoint_cam @ icp_rotation.T + icp_translation
        except Exception as exc:
            icp = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            aligned = scale * (np.asarray(vertices, dtype=float) @ rotation.T) + translation
            keypoint_cam = source @ (scale * rotation).T + translation
    report["icp"] = icp

    # C2. 接触面小角度转平：用检测到的接触区域修正网格接触面的俯仰/滚转（≤5°）
    contact_plane = {"status": "skipped", "reason": "未提供接触点或地面平面"}
    if report["method"] == "camera_ground_anchor":
        contact_plane = {
            "status": "skipped",
            "reason": "地面锚定已完成接触面转平",
            "angle_deg": ground_anchor.get("tilt_deg", 0.0),
            "applied_deg": ground_anchor.get("applied_tilt_deg", 0.0),
        }
    elif contact_points is not None and plane_normal is not None:
        plane_correction = _contact_plane_rotation(aligned, contact_points, plane_normal, plane_d)
        if plane_correction is None:
            contact_plane = {
                "status": "skipped",
                "reason": "接触面点不足或无需修正",
                "angle_deg": 0.0,
                "applied_deg": 0.0,
            }
        else:
            plane_rotation, plane_translation, contact_plane = plane_correction
            aligned = aligned @ plane_rotation.T + plane_translation
            keypoint_cam = keypoint_cam @ plane_rotation.T + plane_translation
    report["contact_plane"] = contact_plane

    # D. 地面落位：优先用检测到的身体-地面接触区域，其次回退到网格最低分位
    ground_snap = {"status": "skipped", "reason": "未提供地面平面"}
    if plane_normal is not None:
        world_rotation = rotate_to_up(np.asarray(plane_normal, dtype=float))
        contact_snap = None
        if contact_points is not None:
            contact_snap = _contact_snap_offset(
                aligned, contact_points, plane_normal, plane_d
            )
        if contact_snap is not None:
            offset, ground_snap = contact_snap
        else:
            offset, contact_before, _ = _ground_snap_offset(aligned, plane_normal, plane_d)
            if abs(offset) < 0.02:
                ground_snap = {
                    "status": "skipped",
                    "source": "mesh_low_quantile",
                    "reason": "接触高度差小于 2cm",
                    "contact_height_before_m": round(contact_before, 4),
                }
            else:
                ground_snap = {
                    "status": "applied",
                    "source": "mesh_low_quantile",
                    "contact_height_before_m": round(contact_before, 4),
                    "contact_height_after_m": round(contact_before + offset, 4),
                    "offset_m": round(float(offset), 4),
                    "axis_world": "z",
                }
        if ground_snap.get("status") == "applied":
            shift = float(offset) * world_rotation[2, :]
            aligned = aligned + shift
            keypoint_cam = keypoint_cam + shift
    report["ground_snap"] = ground_snap

    # E. 刚体精修：只优化整体旋转/平移，姿势与尺度完全冻结，
    #    让接触区贴地并抑制穿透，2D 关键点与初值正则把整体搬动限制在毫米级。
    residual_before_refine = np.linalg.norm(keypoint_cam - target, axis=1)
    refine = {"status": "skipped", "reason": "未提供估计器或地面平面"}
    if pose_estimator is not None and plane_normal is not None:
        try:
            outcome = pose_estimator.refine_rigid_to_scene(
                aligned,
                keypoint_cam,
                person_points=person_points,
                contact_points=contact_points,
                keypoints_2d=keypoints_2d[valid_indices],
                focal_px=focal_px,
                image_size=(
                    tuple(int(v) for v in np.asarray(person_mask).shape)
                    if person_mask is not None
                    else None
                ),
                world_rotation=rotate_to_up(np.asarray(plane_normal, dtype=float)),
                plane_d=float(plane_d),
            )
            refine = outcome.get("report") or {"status": "skipped"}
            refined_vertices = outcome.get("vertices")
            if refined_vertices is not None and len(refined_vertices) == len(aligned):
                aligned = refined_vertices
            refined_keypoints = outcome.get("keypoints")
            if refined_keypoints is not None and len(refined_keypoints) == len(keypoint_cam):
                keypoint_cam = refined_keypoints
        except Exception as exc:
            refine = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    report["refine"] = refine

    residual_final = np.linalg.norm(keypoint_cam - target, axis=1)
    report["rmse_before_refine_m"] = round(
        float(np.sqrt(np.mean(residual_before_refine[inliers] ** 2))), 4
    )
    report["rmse_final_m"] = round(float(np.sqrt(np.mean(residual_final[inliers] ** 2))), 4)
    return aligned.astype(np.float32), report


def _pose_world_view_png(depth, intrinsics, plane_normal, plane_d, person_points, mesh_points, mesh_faces, mesh_note=None) -> str:
    """绘制地面世界坐标系中的场景点云、人物点云与 SAM3D 网格。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    h, w = depth.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    valid = np.isfinite(depth) & (depth > 0)
    indices = np.flatnonzero(valid.ravel())
    if len(indices) > 100_000:
        indices = indices[np.random.default_rng(1).choice(len(indices), 100_000, replace=False)]
    rows, cols = indices // w, indices % w
    z = depth.ravel()[indices]
    scene = np.stack([(cols - cx) * z / fx, (rows - cy) * z / fy, z], axis=-1)

    rotation = rotate_to_up(np.asarray(plane_normal, dtype=float))
    world_scene = scene @ rotation.T
    world_scene[:, 2] -= plane_d
    world_person = person_points @ rotation.T if person_points is not None else None
    if world_person is not None:
        world_person[:, 2] -= plane_d
    world_mesh = mesh_points @ rotation.T
    world_mesh[:, 2] -= plane_d

    focus_points = [world_mesh]
    if world_person is not None and len(world_person):
        focus_points.append(world_person)
    focus = np.concatenate(focus_points, axis=0)
    lower, upper = np.nanpercentile(focus, [1, 99], axis=0)
    mesh_lower, mesh_upper = world_mesh.min(axis=0), world_mesh.max(axis=0)
    lower = np.minimum(lower, mesh_lower)
    upper = np.maximum(upper, mesh_upper)
    ranges = np.maximum(upper - lower, 0.1)
    center = (lower + upper) / 2
    half = max(float(ranges.max()) * 0.7, 0.6)
    limits = np.array([[center[0] - half, center[0] + half],
                       [center[1] - half, center[1] + half],
                       [min(0.0, center[2] - half), max(0.0, center[2] + half)]])
    scene_visible = np.all((world_scene >= limits[:, 0]) & (world_scene <= limits[:, 1]), axis=1)
    visible_scene = world_scene[scene_visible]
    if len(visible_scene) > 40_000:
        visible_scene = visible_scene[
            np.random.default_rng(1).choice(len(visible_scene), 40_000, replace=False)
        ]

    fig = plt.figure(figsize=(16.5, 6.4), dpi=120)
    views = [
        ("front view (world x-z)", 0, -90),
        ("side view (world y-z)", 0, 0),
        ("top view (world x-y)", 88, -90),
    ]
    for index, (view_title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        ax.scatter(visible_scene[:, 0], visible_scene[:, 1], visible_scene[:, 2],
                   c="#a7b0bb", s=0.5, alpha=0.26, depthshade=False)
        if world_person is not None and len(world_person):
            ax.scatter(world_person[:, 0], world_person[:, 1], world_person[:, 2],
                       c="#ffbd3f", s=1.0, alpha=0.45, depthshade=False)
        mesh = Poly3DCollection(
            world_mesh[mesh_faces],
            facecolor=(0.20, 0.04, 0.40, 0.20),
            edgecolor=(0.0, 0.92, 1.0, 0.30),
            linewidth=0.07,
            alpha=1.0,
        )
        mesh.set_zorder(10)
        ax.add_collection3d(mesh)
        vertex_stride = max(len(world_mesh) // 4000, 1)
        sampled_vertices = world_mesh[::vertex_stride]
        ax.scatter(sampled_vertices[:, 0], sampled_vertices[:, 1], sampled_vertices[:, 2],
                   c="#22d3ee", s=0.5, alpha=0.35, depthshade=False)

        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect(tuple((limits[:, 1] - limits[:, 0]) / float((limits[:, 1] - limits[:, 0]).max())))
        ax.set_proj_type("ortho")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("world x (m)", fontsize=8)
        ax.set_ylabel("world y (m)", fontsize=8)
        ax.set_zlabel("height (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(view_title, fontsize=10)
        ax.plot([lower[0], upper[0]], [lower[1], lower[1]], [0, 0], color="#ff453a", alpha=0.7, lw=1.2)
        ax.plot([upper[0], upper[0]], [lower[1], upper[1]], [0, 0], color="#ff453a", alpha=0.7, lw=1.2)
        ax.plot([upper[0], lower[0]], [upper[1], upper[1]], [0, 0], color="#ff453a", alpha=0.7, lw=1.2)
        ax.plot([lower[0], lower[0]], [upper[1], lower[1]], [0, 0], color="#ff453a", alpha=0.7, lw=1.2)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#a7b0bb", markersize=5, label="DepthPro scene cloud"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffbd3f", markersize=5, label="person cloud"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#22d3ee", markersize=5, label="SAM3D Body mesh"),
        Line2D([0], [0], color="#ff453a", lw=1.5, label="ground plane (z=0)"),
    ]
    if world_person is None or not len(world_person):
        handles = [handle for handle in handles if handle.get_label() != "person cloud"]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "World 3-view | SAM3D mesh + DepthPro point cloud\n"
        + (mesh_note or "(mesh placement is visual only; not metrically calibrated)"),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    return _to_data_uri(buf)


def _ground_views_png(depth, intrinsics, inlier_points, plane_grid, plane_normal, plane_d, person_points=None, grid_steps=100, person_bbox_cam=None, pose_mesh_points=None, pose_mesh_faces=None, pose_mesh_aligned=False) -> str:
    """相机视角 + 世界 3D 视角拼合对比图（单张 PNG，左右两面板）。

    - 左：相机坐标系（x 右、y 下、z 前），场景点云按深度着色，相机位于原点（红星）
    - 右：世界坐标系（地面 z=0、法向量朝上），场景点云按离地高度着色，相机（红星）+ 法向箭头
    - 两面板都叠加：地面内点（绿）、拟合地面平面（红网格）、人物点云（黄）
    - person_bbox_cam：人物 3D 轴对齐 bbox（相机系 [xmin,ymin,zmin,xmax,ymax,zmax]），画品红线框
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fx, cx = float(intrinsics[0, 0]), float(intrinsics[0, 2])
    fy, cy = float(intrinsics[1, 1]), float(intrinsics[1, 2])

    # 重建平面网格：范围覆盖内点区域并外扩到"图像四角射线与拟合平面的交点"，
    # 使渲染平面铺满相机视野内的地面，避免网格只覆盖内点中心区域而与点云地面"不重合"。
    n_pl = np.asarray(plane_normal, dtype=float)
    d_pl = float(plane_d)
    basis_seed = np.eye(3)[np.argmin(np.abs(n_pl))]
    axis_u = np.cross(n_pl, basis_seed)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(n_pl, axis_u)
    in_pts0 = np.asarray(inlier_points, dtype=float)
    ctr0 = in_pts0.mean(axis=0)
    h0, w0 = float(depth.shape[0]), float(depth.shape[1])
    corners = np.array([
        [(0.0 - cx) / fx, (0.0 - cy) / fy, 1.0],
        [(w0 - cx) / fx, (0.0 - cy) / fy, 1.0],
        [(w0 - cx) / fx, (h0 - cy) / fy, 1.0],
        [(0.0 - cx) / fx, (h0 - cy) / fy, 1.0],
    ])
    t_c = -d_pl / np.maximum(corners @ n_pl, 1e-9)  # 平面 n·p+d=0，射线交点 t=-d/(n·dir)
    # 只保留与地面真正相交且距离合理的角点（t 极大≈视线与地面平行，属远处地平线，忽略）
    keep = np.isfinite(t_c) & (t_c > 0) & (t_c <= 25.0)
    corner_pts = corners[keep] * t_c[keep][:, None]
    all_pts = np.vstack([in_pts0, corner_pts]) if len(corner_pts) else in_pts0
    # uv 必须以质心 ctr0 为基准（网格展开 plane_grid = ctr0 + uu·axis_u + vv·axis_v），
    # 若直接用 all_pts @ axis_u（相对光心原点），uv 区间会整体平移 ctr0 在平面上的投影，
    # 导致网格与点云地面错位约 1.8m（渲染"不在同一高度"的根因）。
    all_uv = np.column_stack(((all_pts - ctr0) @ axis_u, (all_pts - ctr0) @ axis_v))
    u0, u1 = all_uv[:, 0].min(), all_uv[:, 0].max()
    v0, v1 = all_uv[:, 1].min(), all_uv[:, 1].max()
    u_values = np.linspace(u0, u1, grid_steps)
    v_values = np.linspace(v0, v1, grid_steps)
    uu, vv = np.meshgrid(u_values, v_values)
    plane_grid = ctr0 + uu[..., None] * axis_u + vv[..., None] * axis_v

    # 采样场景点（相机坐标系，不生成全量 HxWx3 数组）
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.ravel())
    n_scene = min(150_000, len(idx))
    if len(idx) > n_scene:
        idx = np.random.default_rng(1).choice(idx, n_scene, replace=False)
    rows = idx // w
    cols = idx % w
    z = depth.ravel()[idx]
    scene = np.stack([(cols - cx) * z / fx, (rows - cy) * z / fy, z], axis=-1)

    # 内点 / 人物点云采样
    rng = np.random.default_rng(1)
    in_pts = np.asarray(inlier_points, dtype=float)
    if len(in_pts) > 30_000:
        in_pts = in_pts[rng.choice(len(in_pts), 30_000, replace=False)]
    per_pts = None
    if person_points is not None and len(person_points):
        per_pts = np.asarray(person_points, dtype=float)
        if len(per_pts) > 20_000:
            per_pts = per_pts[rng.choice(len(per_pts), 20_000, replace=False)]

    # 视觉对齐：把距拟合平面很近的点云直接投影到平面上（p' = p - n·(n·p+d)），
    # 消除深度噪声造成的"点云地面与渲染平面不在同一高度"观感。
    # 内点容差 2cm，此处放宽到 12cm：诊断显示约 3% 的地面像素深度噪声在 5~25cm，
    # 5cm 阈值压不掉这些"毛边"，12cm 可覆盖大部分深度噪声，又不会把沙发脚/低矮物体压平。
    align_thresh = 0.12
    for pts in (scene, in_pts):
        signed = pts @ n_pl + d_pl
        mask = np.abs(signed) <= align_thresh
        if mask.any():
            pts[mask] = pts[mask] - n_pl * signed[mask][:, None]

    # 相机系 -> 世界系（地面对齐到 z=0）
    R = rotate_to_up(np.asarray(plane_normal, dtype=float))
    wscene = scene @ R.T
    win = in_pts @ R.T
    wscene[:, 2] -= plane_d
    win[:, 2] -= plane_d
    # 世界系地面网格：X/Y 范围直接取"点云地面带"(|z|<=0.2m)的范围，
    # 避免 uv 轴对齐矩形转世界系后四角外扩出点云地面一大圈（用户反馈线框 X/Y 与点云不匹配）。
    gm = np.abs(wscene[:, 2]) <= 0.20
    if gm.sum() >= 4:
        pad = 0.02  # 2% 边距让线框清晰可见且紧贴点云
        gx0, gx1 = wscene[gm, 0].min(), wscene[gm, 0].max()
        gy0, gy1 = wscene[gm, 1].min(), wscene[gm, 1].max()
        dx, dy = (gx1 - gx0) * pad, (gy1 - gy0) * pad
        gxx, gyy = np.meshgrid(np.linspace(gx0 - dx, gx1 + dx, grid_steps),
                               np.linspace(gy0 - dy, gy1 + dy, grid_steps))
        wground = np.stack([gxx, gyy, np.zeros_like(gxx)], axis=-1)
    else:  # 兜底：无地面带时退回 uv 矩形
        wground = np.asarray(plane_grid, dtype=float) @ R.T
        wground[:, 2] -= plane_d
    cam_world = np.array([0.0, 0.0, -plane_d], dtype=float)
    wper = per_pts @ R.T if per_pts is not None else None
    if wper is not None:
        wper[:, 2] -= plane_d
    wmesh = pose_mesh_points @ R.T if pose_mesh_points is not None else None
    if wmesh is not None:
        wmesh[:, 2] -= plane_d

    zmin, zmax = np.percentile(wscene[:, 2], [2, 98])
    cam_height = abs(float(plane_d))

    # 人物 bbox：12 条边（8 角点索引对）。左面板用相机系轴对齐 bbox；
    # 右面板在世界系下对人物点云重新取 min/max 做轴对齐 bbox（方向与地面网格一致）。
    box_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # 底面
        (4, 5), (5, 7), (7, 6), (6, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖棱
    ]
    box_corners = None
    if person_bbox_cam is not None:
        x0, y0, z0, x1, y1, z1 = map(float, person_bbox_cam)
        box_corners = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=float)

    # 相机视锥（表示相机位置与观察角度）：相机系 +z 为观察方向，按内参把图像四角
    # 投影到 frustum_z 处作为近平面，转世界系后从相机位置向四个角拉棱线。
    fz = 0.9  # 视锥长度（米）
    hh, ww = float(depth.shape[0]), float(depth.shape[1])
    fc_cam = np.array([
        [(0 - cx) * fz / fx, (0 - cy) * fz / fy, fz],
        [(ww - cx) * fz / fx, (0 - cy) * fz / fy, fz],
        [(ww - cx) * fz / fx, (hh - cy) * fz / fy, fz],
        [(0 - cx) * fz / fx, (hh - cy) * fz / fy, fz],
    ])
    # fc_cam 是相对光心的偏移；转世界系后需要与点云一致地做 z 平移（地面 z=0）
    fc_world = fc_cam @ R.T
    fc_world[:, 2] -= plane_d
    fwd_world = R @ np.array([0.0, 0.0, 1.0])  # 世界系下的相机观察方向（原 +z）

    fig = plt.figure(figsize=(16, 7.2), dpi=100)

    # 左：相机视角
    ax_cam = fig.add_subplot(121, projection="3d")
    ax_cam.scatter(scene[:, 0], scene[:, 1], scene[:, 2], c=scene[:, 2], cmap="viridis", s=0.4, alpha=0.3)
    ax_cam.scatter(in_pts[:, 0], in_pts[:, 1], in_pts[:, 2], c="#2ef05c", s=0.7, alpha=0.5)
    if per_pts is not None:
        ax_cam.scatter(per_pts[:, 0], per_pts[:, 1], per_pts[:, 2], c="#f7c948", s=1.0, alpha=0.7)
    cam_grid = np.asarray(plane_grid, dtype=float)
    if pose_mesh_points is not None and pose_mesh_faces is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = Poly3DCollection(
            pose_mesh_points[pose_mesh_faces],
            facecolor=(0.28, 0.08, 0.48, 0.48),
            edgecolor=(0.08, 0.88, 1.0, 0.98),
            linewidth=0.28,
            alpha=0.56,
        )
        mesh.set_zorder(10)
        ax_cam.add_collection3d(mesh)
    # 与右面板一致：低透明度红色面 + 红色网格线框
    ax_cam.plot_surface(cam_grid[:, :, 0], cam_grid[:, :, 1], cam_grid[:, :, 2], color="red", alpha=0.16)
    ax_cam.plot_wireframe(cam_grid[::5, ::5, 0], cam_grid[::5, ::5, 1], cam_grid[::5, ::5, 2],
                          color="red", alpha=0.85, lw=0.5)
    # 人物 3D bbox 线框（相机系）
    if box_corners is not None:
        for a, b in box_edges:
            ax_cam.plot([box_corners[a, 0], box_corners[b, 0]],
                        [box_corners[a, 1], box_corners[b, 1]],
                        [box_corners[a, 2], box_corners[b, 2]],
                        color="#ff00ff", lw=1.6)
    # 相机视锥：顶点在原点，近平面为图像四角
    for j in range(4):
        ax_cam.plot([0.0, fc_cam[j, 0]], [0.0, fc_cam[j, 1]], [0.0, fc_cam[j, 2]], color="#00e5ff", lw=1.0)
    for j in range(4):
        k = (j + 1) % 4
        ax_cam.plot([fc_cam[j, 0], fc_cam[k, 0]], [fc_cam[j, 1], fc_cam[k, 1]], [fc_cam[j, 2], fc_cam[k, 2]], color="#00e5ff", lw=1.2)
    ax_cam.scatter([0.0], [0.0], [0.0], c="red", s=40, marker="*")
    ax_cam.set_xlabel("x (m)")
    ax_cam.set_ylabel("y (m)")
    ax_cam.set_zlabel("z (m)")
    ax_cam.set_title("camera view | cam at origin, %.2f m above plane" % cam_height, fontsize=11)
    # 相机视角渲染：观察者位于相机（原点）后方沿光轴 +z 向前看，屏幕右 = +x、屏幕下 = +y
    ax_cam.set_proj_type("persp", focal_length=0.55)
    ax_cam.view_init(elev=-90, azim=-90, roll=0)

    # 右：世界 3D
    ax_w = fig.add_subplot(122, projection="3d")
    # 场景点云按高度着色。地面带（|z|<=0.2m）是用户关心的区域，保持高亮；
    # 高处物体（沙发/墙/桌，占约 73%）若同样用黄绿色会淹没红色平面、造成"平面与地面
    # 不重合"的错觉，因此降为灰色低透明度背景。
    wz = wscene[:, 2]
    ground_mask = np.abs(wz) <= 0.20
    if ground_mask.any() and (~ground_mask).any():
        ax_w.scatter(wscene[~ground_mask, 0], wscene[~ground_mask, 1], wscene[~ground_mask, 2],
                     c="#9a9a9a", s=0.3, alpha=0.18)
        ax_w.scatter(wscene[ground_mask, 0], wscene[ground_mask, 1], wscene[ground_mask, 2],
                     c=wz[ground_mask], cmap="viridis", s=0.5, alpha=0.55, vmin=zmin, vmax=zmax)
    else:
        ax_w.scatter(wscene[:, 0], wscene[:, 1], wscene[:, 2], c=wz, cmap="viridis",
                     s=0.4, alpha=0.35, vmin=zmin, vmax=zmax)
    ax_w.scatter(win[:, 0], win[:, 1], win[:, 2], c="#2ef05c", s=0.6, alpha=0.55)
    if wper is not None:
        ax_w.scatter(wper[:, 0], wper[:, 1], wper[:, 2], c="#f7c948", s=1.0, alpha=0.7)
    if wmesh is not None and pose_mesh_faces is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh_world = Poly3DCollection(
            wmesh[pose_mesh_faces],
            facecolor=(0.28, 0.08, 0.48, 0.48),
            edgecolor=(0.08, 0.88, 1.0, 0.98),
            linewidth=0.28,
            alpha=0.56,
        )
        mesh_world.set_zorder(10)
        ax_w.add_collection3d(mesh_world)
    # 人物 3D bbox：世界系下对人物点云重新做轴对齐包围盒（方向与地面网格一致），
    # 而非旋转相机系 bbox（会变成斜盒且范围失真）。
    if wper is not None and len(wper) >= 8:
        wx0, wy0, wz0 = wper[:, 0].min(), wper[:, 1].min(), wper[:, 2].min()
        wx1, wy1, wz1 = wper[:, 0].max(), wper[:, 1].max(), wper[:, 2].max()
        wbc = np.array([[x, y, z] for z in (wz0, wz1) for y in (wy0, wy1) for x in (wx0, wx1)], dtype=float)
        for a, b in box_edges:
            ax_w.plot([wbc[a, 0], wbc[b, 0]], [wbc[a, 1], wbc[b, 1]], [wbc[a, 2], wbc[b, 2]],
                      color="#ff00ff", lw=1.6)
    elif box_corners is not None:  # 兜底：无世界系人物点云时退回相机系 bbox 旋转版
        wbox = box_corners @ R.T
        wbox[:, 2] -= plane_d
        for a, b in box_edges:
            ax_w.plot([wbox[a, 0], wbox[b, 0]], [wbox[a, 1], wbox[b, 1]], [wbox[a, 2], wbox[b, 2]],
                      color="#ff00ff", lw=1.6)
    # 平面：低透明度面 + 红色网格线框，线框让"拟合平面"的范围与贴合关系一目了然
    ax_w.plot_surface(wground[:, :, 0], wground[:, :, 1], wground[:, :, 2], color="red", alpha=0.16)
    ax_w.plot_wireframe(wground[::5, ::5, 0], wground[::5, ::5, 1], wground[::5, ::5, 2],
                        color="red", alpha=0.85, lw=0.5)
    center = win.mean(axis=0)
    ax_w.quiver(center[0], center[1], center[2], 0, 0, 1, length=1.5, color="k", arrow_length_ratio=0.15)
    # 相机视锥：顶点在世界系相机位置，棱线沿图像四角张开，蓝箭头为观察方向
    for j in range(4):
        ax_w.plot([cam_world[0], fc_world[j, 0]], [cam_world[1], fc_world[j, 1]], [cam_world[2], fc_world[j, 2]], color="#00e5ff", lw=1.0)
    for j in range(4):
        k = (j + 1) % 4
        ax_w.plot([fc_world[j, 0], fc_world[k, 0]], [fc_world[j, 1], fc_world[k, 1]], [fc_world[j, 2], fc_world[k, 2]], color="#00e5ff", lw=1.2)
    ax_w.quiver(cam_world[0], cam_world[1], cam_world[2], fwd_world[0], fwd_world[1], fwd_world[2],
                length=0.9, color="#0066ff", arrow_length_ratio=0.25, lw=2.0)
    ax_w.scatter([cam_world[0]], [cam_world[1]], [cam_world[2]], c="red", s=40, marker="*")
    ax_w.set_xlabel("world x (m)")
    ax_w.set_ylabel("world y (m)")
    ax_w.set_zlabel("height above ground (m)")
    ax_w.set_title("world 3D | ground at z=0, normal -> up", fontsize=11)
    ax_w.set_box_aspect((1, 1, 1))
    # 透视投影：focal 越小视场角越大、近大远小越明显（默认 1 仅约 90° 视场）
    ax_w.set_proj_type("persp", focal_length=0.55)
    ax_w.view_init(elev=25, azim=-60)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ef05c", markersize=6, label="ground inliers"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f7c948", markersize=6, label="person cloud"),
        Line2D([0], [0], color="#00e5ff", lw=2, label="camera (view frustum)"),
        Line2D([0], [0], color="red", lw=2, label="fitted plane"),
    ]
    if pose_mesh_points is not None and pose_mesh_faces is not None:
        handles.append(Patch(facecolor=(0.28, 0.08, 0.48, 0.48), edgecolor=(0.08, 0.88, 1.0, 0.98), label="SAM3D mesh" + (" (aligned)" if pose_mesh_aligned else "")))
    if per_pts is None:
        handles = [h for h in handles if "person cloud" not in h.get_label()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    return _to_data_uri(buf)


def create_ground_app(
    *,
    device: str = "auto",
    sam_type: str = "sam2.1_hiera_small",
    sam_ckpt: str | Path = DEFAULT_SAM_CKPT,
    floor_prompt: str = DEFAULT_FLOOR_PROMPT,
    person_prompt: str = "person.",
    sam3d_resources: Optional[str | Path] = None,
) -> FastAPI:
    state: Dict[str, Any] = {"segmenter": None, "depth": None, "transform": None, "device": torch.device("cpu"), "pose_estimator": None}
    sam3d_resources = sam3d_resources or os.environ.get("SAM3D_RESOURCES")
    state["sam3d_resources"] = sam3d_resources

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = torch.device(device)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("启动地面提取服务，加载模型 (device=%s)...", state["device"])
        # 注意加载顺序：必须先加载 DepthPro 初始化 CUDA 上下文，再加载 LangSAM。
        # 若 SAM2 作为第一个 CUDA 模型加载，GroundingDINO 前向传播会导致 WSL VM 崩溃。
        model, transform = depth_pro.create_model_and_transforms(device=state["device"])
        model.eval()
        state["depth"], state["transform"] = model, transform
        logger.info("DepthPro 加载完成")
        state["segmenter"] = LangSAM(sam_type=sam_type, ckpt_path=str(sam_ckpt))
        logger.info("LangSAM 加载完成")
        if state["sam3d_resources"]:
            try:
                from ground_pose import GroundPoseEstimator

                state["pose_estimator"] = GroundPoseEstimator(state["sam3d_resources"])
                logger.info("SAM3D Body 姿势模块加载完成")
            except Exception:
                logger.exception("SAM3D Body 加载失败；地面服务继续运行，姿势功能不可用")
        else:
            logger.info("未配置 SAM3D_RESOURCES，姿势模块保持禁用")
        logger.info("服务就绪")
        yield
        logger.info("关闭服务")

    app = FastAPI(
        title="ground_pipeline Ground-plane API",
        description="RGB -> LangSAM 地面分割 -> DepthPro 深度 -> 地面平面拟合",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {
            "ok": state["segmenter"] is not None and state["depth"] is not None,
            "device": str(state["device"]),
        }

    @app.post("/v1/ground", response_model=GroundResult)
    async def ground(request: GroundRequest):
        if state["segmenter"] is None or state["depth"] is None:
            raise HTTPException(status_code=503, detail="模型未就绪（可能加载失败），请检查服务日志")

        rgb = _load_image(request.image)
        image_pil = __import__("PIL").Image.fromarray(rgb)
        if request.floor_prompt is None:
            request.floor_prompt = floor_prompt
        if len(request.floor_prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="floor_prompt 不能为空")

        try:
            segmentation = state["segmenter"].predict([image_pil], [request.floor_prompt])[0]
        except Exception as e:
            logger.exception("LangSAM 分割失败")
            raise HTTPException(status_code=500, detail=f"分割失败: {e}")
        masks = np.asarray(segmentation["masks"], dtype=bool)
        if masks.ndim == 2:
            floor_mask = masks
        elif masks.ndim == 3 and masks.shape[0] > 0:
            floor_mask = masks.any(axis=0)
        else:
            raise HTTPException(status_code=404, detail="未检出地面区域")

        # 人物 mask：提示词为空则跳过接触区域检测，人物分割失败不影响地面结果。
        if request.person_prompt is None:
            request.person_prompt = person_prompt
        person_mask = None
        if request.person_prompt.strip():
            try:
                person_mask = _person_mask(state["segmenter"], image_pil, request.person_prompt)
            except Exception as e:
                logger.warning("人物分割失败，跳过接触区域检测: %s", e)
                person_mask = None

        try:
            model, transform, device = state["depth"], state["transform"], state["device"]
            tensor = transform(rgb).to(device)
            with torch.inference_mode():
                pred = model.infer(tensor)
            depth = pred["depth"].squeeze().detach().cpu().numpy()
            f_px = float(pred["focallength_px"].detach().cpu().item())
        except Exception as e:
            logger.exception("DepthPro 推理失败")
            raise HTTPException(status_code=500, detail=f"深度推理失败: {e}")

        height, width = rgb.shape[:2]
        intrinsics = np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1]], dtype=np.float32)
        try:
            result = extract_ground(
                rgb,
                depth,
                intrinsics,
                floor_mask,
                distance_threshold=request.distance_threshold,
                iterations=request.iterations,
                grid_steps=request.grid_steps,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        pose = None
        if request.pose_inference:
            if person_mask is None or not person_mask.any():
                pose = {"status": "skipped", "engine": "sam3d_body_vith", "error": "人物 mask 不存在或为空"}
            elif state["pose_estimator"] is None:
                pose = {"status": "error", "engine": "sam3d_body_vith", "error": "SAM3D 未启用或未能加载；请配置 SAM3D_RESOURCES 并检查启动日志"}
            else:
                pose_started = time.perf_counter()
                try:
                    pose = state["pose_estimator"].infer(
                        rgb,
                        person_mask,
                        want_overlay=bool(request.visualize and request.pose_want_overlay),
                        want_mesh=bool(request.visualize and request.pose_want_mesh),
                        cam_int=intrinsics,
                    )
                    pose["time_ms"] = round((time.perf_counter() - pose_started) * 1000.0, 1)
                except Exception as e:
                    logger.exception("SAM3D 姿势推理失败")
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    pose = {
                        "status": "error",
                        "engine": "sam3d_body_vith",
                        "error": f"{type(e).__name__}: {e}",
                        "time_ms": round((time.perf_counter() - pose_started) * 1000.0, 1),
                    }

        # 人物 3D 点云 bbox（相机坐标系轴对齐）：取人物 mask 中有效反投影点的 x/y/z min/max。
        person_bbox_3d_m = None
        person_centroid_m = None
        if person_mask is not None and person_mask.any():
            pts = result["points"][person_mask]
            valid = np.isfinite(pts).all(axis=-1)
            if valid.any():
                pts = pts[valid]
                mn = pts.min(axis=0)
                mx = pts.max(axis=0)
                person_bbox_3d_m = [float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2])]
                person_centroid_m = pts.mean(axis=0).astype(float).tolist()

        # 接触区域：人物 mask 反投影点中，与拟合地面平面距离 <= contact_threshold 的像素。
        contact_pixels, contact_mask = _contact_region(
            person_mask,
            result["points"],
            result["plane_normal"],
            float(result["plane_d"]),
            request.contact_threshold,
        )
        contact_centroid_m = None
        contact_bbox = None
        if contact_mask is not None:
            ys, xs = np.nonzero(contact_mask)
            pts = result["points"][contact_mask]
            contact_centroid_m = pts.mean(axis=0).astype(float).tolist()
            contact_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        rng = np.random.default_rng(0)
        inliers_arr = result["plane_inlier_points"]
        inliers_arr = inliers_arr[rng.choice(len(inliers_arr), size=min(2000, len(inliers_arr)), replace=False)]
        grid = result["plane_points"].reshape(request.grid_steps, request.grid_steps, 3)

        # return_data=true 时，返回全量像素级数据：四张 mask PNG、人物点云采样、全量深度图（.npy base64）。
        masks = None
        person_points = None
        depth_b64 = None
        if request.return_data:
            masks = {
                "floor": _mask_png(floor_mask),
                "inlier": _mask_png(result["plane_inlier_mask"]),
                "person": _mask_png(person_mask) if person_mask is not None else None,
                "contact": _mask_png(contact_mask) if contact_mask is not None else None,
            }
            if person_mask is not None and person_mask.any():
                ppts = result["points"][person_mask]
                ppts = ppts[np.isfinite(ppts).all(axis=-1)]
                if len(ppts) > request.person_points_limit:
                    ppts = ppts[rng.choice(len(ppts), request.person_points_limit, replace=False)]
                person_points = ppts.astype(float).tolist()
            depth_b64 = _npy_b64(depth)

        visualization = None
        if request.visualize:
            person_pts_viz = None
            if person_mask is not None and person_mask.any():
                person_pts_viz = result["points"][person_mask]
                person_pts_viz = person_pts_viz[np.isfinite(person_pts_viz).all(axis=-1)]
                if len(person_pts_viz) > 20_000:
                    person_pts_viz = person_pts_viz[np.random.default_rng(0).choice(len(person_pts_viz), 20_000, replace=False)]
            contact_pts_viz = None
            if contact_mask is not None:
                contact_pts_viz = result["points"][contact_mask]
                contact_pts_viz = contact_pts_viz[np.isfinite(contact_pts_viz).all(axis=-1)]
            pose_mesh_points = None
            pose_mesh_faces = None
            pose_mesh_data = None
            mesh_aligned = False
            if pose is not None and pose.get("status") == "success":
                pose_mesh_data = pose.pop("mesh_png_data_uri", None)
                if request.pose_want_mesh and "mesh_vertices_model" in pose:
                    try:
                        pose_mesh_points, pose_mesh_faces, _ = _prepare_pose_mesh_for_views(pose)
                        pose_mesh_points, alignment = _align_pose_mesh_to_depth(
                            pose,
                            pose_mesh_points,
                            depth,
                            intrinsics,
                            person_mask,
                            person_points=person_pts_viz,
                            plane_normal=result["plane_normal"],
                            plane_d=float(result["plane_d"]),
                            contact_points=contact_pts_viz,
                            pose_estimator=state["pose_estimator"],
                        )
                        pose["alignment"] = alignment
                        mesh_aligned = alignment.get("status") == "success"
                    except Exception as exc:
                        pose_mesh_points, pose_mesh_faces = None, None
                        pose.pop("align_keypoints_2d", None)
                        pose.pop("align_keypoints_3d_cam", None)
                        pose.pop("mesh_vertices_model", None)
                        pose.pop("mesh_faces", None)
                        pose.pop("mesh_camera_translation", None)
                        pose.pop("mesh_focal_px", None)
                        pose["mesh_error"] = f"{type(exc).__name__}: {exc}"
            visualization = {
                "overlay": _overlay_png(rgb, floor_mask, result["plane_inlier_mask"], person_mask, contact_mask),
            }
            mesh_included = pose_mesh_points is not None and pose_mesh_faces is not None
            try:
                visualization["views"] = _ground_views_png(
                    depth,
                    intrinsics,
                    result["plane_inlier_points"],
                    grid,
                    result["plane_normal"],
                    float(result["plane_d"]),
                    person_pts_viz,
                    request.grid_steps,
                    person_bbox_3d_m,
                    pose_mesh_points,
                    pose_mesh_faces,
                    pose_mesh_aligned=mesh_aligned,
                )
            except Exception as exc:
                if mesh_included:
                    pose["mesh_error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        visualization["views"] = _ground_views_png(
                            depth,
                            intrinsics,
                            result["plane_inlier_points"],
                            grid,
                            result["plane_normal"],
                            float(result["plane_d"]),
                            person_pts_viz,
                            request.grid_steps,
                            person_bbox_3d_m,
                        )
                    except Exception:
                        logger.exception("地面视图渲染失败（不含 SAM3D 网格）")
                else:
                    logger.exception("地面视图渲染失败")
            if pose is not None:
                pose_overlay = pose.pop("overlay_png_data_uri", None)
                if pose_overlay is not None:
                    visualization["pose_overlay"] = pose_overlay
                if pose_mesh_data is not None:
                    visualization["pose_mesh"] = pose_mesh_data
                if mesh_included:
                    try:
                        alignment = pose.get("alignment") or {}
                        mesh_note = None
                        if alignment.get("status") == "success":
                            icp = alignment.get("icp") or {}
                            snap = alignment.get("ground_snap") or {}
                            note_parts = [
                                "kp RMSE"
                                f" {float(alignment.get('rmse_m', 0.0)) * 100:.1f} cm",
                            ]
                            anchor = alignment.get("ground_anchor") or {}
                            if alignment.get("method") == "camera_ground_anchor":
                                note_parts.insert(0, "camera ground anchor")
                                note_parts.append(f"scale {float(alignment.get('scale', 1.0)):.2f}")
                                if anchor.get("status") == "applied":
                                    note_parts.append(
                                        f"plane tilt {anchor.get('applied_tilt_deg', 0.0):.1f} deg"
                                        f" / lift {anchor.get('shift_m', 0.0) * 100:+.1f} cm"
                                    )
                            else:
                                note_parts.insert(0, "weighted keypoint Umeyama")
                                note_parts.append(f"scale {float(alignment.get('scale', 1.0)):.2f}")
                            if icp.get("status") == "success":
                                note_parts.append(f"trimmed ICP {icp['trimmed_rmse_m'] * 100:.1f} cm")
                            plane = alignment.get("contact_plane") or {}
                            if plane.get("status") == "applied":
                                note_parts.append(f"contact plane {plane['applied_deg']:.1f} deg")
                            if snap.get("status") == "applied":
                                if snap.get("source") == "contact_footprint":
                                    note_parts.append(
                                        f"contact snap {snap['offset_m'] * 100:+.1f} cm"
                                        f" / {snap['clusters_with_mesh']} regions"
                                    )
                                else:
                                    note_parts.append(f"ground snap {snap['offset_m'] * 100:+.1f} cm")
                            refine = alignment.get("refine") or {}
                            if refine.get("status") == "applied":
                                note_parts.append(
                                    f"rigid refine {refine['iterations']} it"
                                    f" / rot {refine['delta_rot_deg']:.1f} deg"
                                    f" / contact {refine['contact_height_after_m'] * 100:.1f} cm"
                                )
                            mesh_note = "(mesh aligned to DepthPro depth via " + ", ".join(note_parts) + ")"
                        visualization["pose_world"] = _pose_world_view_png(
                            depth,
                            intrinsics,
                            result["plane_normal"],
                            float(result["plane_d"]),
                            person_pts_viz,
                            pose_mesh_points,
                            pose_mesh_faces,
                            mesh_note,
                        )
                    except Exception as exc:
                        pose["mesh_error"] = f"{type(exc).__name__}: {exc}"
                        logger.exception("SAM3D 世界系网格视图渲染失败")

        # ---- 相机系 / 世界系分组输出（顶层旧字段保留，向后兼容）----

        intrinsics_list = [
            [float(f_px), 0.0, float(width / 2)],
            [0.0, float(f_px), float(height / 2)],
            [0.0, 0.0, 1.0],
        ]
        depth_range_m = [float(np.nanmin(depth)), float(np.nanmax(depth))]
        cam_ground = {
            "plane_normal": result["plane_normal"].astype(float).tolist(),
            "plane_d": float(result["plane_d"]),
            "camera_height_m": abs(float(result["plane_d"])),
            "plane_points": grid.reshape(-1, 3).astype(float).tolist(),
            "inlier_points": inliers_arr.astype(float).tolist(),
        }
        cam_info = {
            "focal_px": f_px,
            "intrinsics": intrinsics_list,
            "depth_range_m": depth_range_m,
            "image_hw": [height, width],
            "height_m": abs(float(result["plane_d"])),
        }
        cam_person = None
        if person_mask is not None and person_mask.any():
            cam_person = {
                "pixels": int(person_mask.sum()),
                "bbox_3d_m": person_bbox_3d_m,
                "centroid_m": person_centroid_m,
                "points": person_points,  # return_data=true 时为相机系采样点云
            }
        cam_contact = None
        if contact_mask is not None:
            cam_contact = {
                "pixels": contact_pixels,
                "centroid_m": contact_centroid_m,
                "bbox": contact_bbox,
            }

        # 世界系：人物/接触用全量有效点变换（bbox/质心更准确），points 输出用 return_data 采样
        full_person_pts = None
        if person_mask is not None and person_mask.any():
            fp = result["points"][person_mask]
            fp = fp[np.isfinite(fp).all(axis=-1)]
            full_person_pts = fp
        contact_full = None
        if contact_mask is not None:
            contact_full = result["points"][contact_mask]
        wf = _world_frame_data(
            plane_normal=result["plane_normal"],
            plane_d=float(result["plane_d"]),
            plane_points=grid.reshape(-1, 3),
            inlier_points=result["plane_inlier_points"],
            person_points=full_person_pts,
            contact_points=contact_full,
        )
        world_ground = {
            "plane_normal": wf["plane_normal"].tolist(),
            "plane_d": wf["plane_d"],
            "camera_height_m": wf["camera_height_m"],
            "plane_points": wf["plane_points"].astype(float).tolist(),
            "inlier_points": wf["inlier_points"].astype(float).tolist(),
        }
        world_cam = {
            "focal_px": f_px,
            "intrinsics": intrinsics_list,
            "depth_range_m": depth_range_m,
            "image_hw": [height, width],
            "height_m": wf["camera_height_m"],
            "position_m": wf["camera_position"].tolist(),
            "forward": wf["camera_forward"].tolist(),
        }
        world_person = None
        if cam_person is not None and "person_points" in wf:
            wp_pts = None
            if person_points is not None:
                wp_pts = np.asarray(person_points, dtype=float) @ wf["rotation"].T
                wp_pts[:, 2] -= float(result["plane_d"])
                wp_pts = wp_pts.astype(float).tolist()
            world_person = {
                "pixels": cam_person["pixels"],
                "bbox_3d_m": wf["person_bbox_3d_m"].tolist(),
                "centroid_m": wf["person_centroid_m"].tolist(),
                "points": wp_pts,
            }
        world_contact = None
        if cam_contact is not None and "contact_centroid_m" in wf:
            world_contact = {
                "pixels": cam_contact["pixels"],
                "centroid_m": wf["contact_centroid_m"].tolist(),
            }

        return GroundResult(
            ok=True,
            plane_normal=result["plane_normal"].astype(float).tolist(),
            plane_d=float(result["plane_d"]),
            camera_height_m=abs(float(result["plane_d"])),
            focal_px=f_px,
            depth_range_m=depth_range_m,
            image_hw=[height, width],
            floor_pixels=int(floor_mask.sum()),
            plane_inliers=int(result["plane_inlier_mask"].sum()),
            plane_points=grid.reshape(-1, 3).astype(float).tolist(),
            inlier_points=inliers_arr.astype(float).tolist(),
            person_pixels=None if person_mask is None else int(person_mask.sum()),
            person_bbox_3d_m=person_bbox_3d_m,
            person_centroid_m=person_centroid_m,
            contact_pixels=contact_pixels,
            contact_centroid_m=contact_centroid_m,
            contact_bbox=contact_bbox,
            masks=masks,
            person_points=person_points,
            depth_b64=depth_b64,
            visualization=visualization,
            pose=pose,
            camera={
                "ground": cam_ground,
                "camera": cam_info,
                "person": cam_person,
                "contact": cam_contact,
            },
            world={
                "ground": world_ground,
                "camera": world_cam,
                "person": world_person,
                "contact": world_contact,
            },
        )

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="ground_pipeline 地面提取服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    parser.add_argument("--sam-type", type=str, default="sam2.1_hiera_small", help="SAM2 模型类型")
    parser.add_argument("--sam-ckpt", type=str, default=str(DEFAULT_SAM_CKPT), help="SAM2 权重路径")
    parser.add_argument("--floor-prompt", type=str, default=DEFAULT_FLOOR_PROMPT, help="默认地面分割提示词")
    parser.add_argument("--person-prompt", type=str, default="person.", help="默认人物分割提示词；传空串禁用接触区域检测")
    parser.add_argument("--sam3d-resources", type=str, default=os.environ.get("SAM3D_RESOURCES"), help="ImagePose_Resources 根目录；配置后加载 SAM3D Body（不加载 ViTPose）")

    args = parser.parse_args()
    app = create_ground_app(
        device=args.device,
        sam_type=args.sam_type,
        sam_ckpt=args.sam_ckpt,
        floor_prompt=args.floor_prompt,
        person_prompt=args.person_prompt,
        sam3d_resources=args.sam3d_resources,
    )
    logger.info("启动 ground_pipeline 服务 %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
