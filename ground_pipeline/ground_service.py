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
    # 相机系 / 世界系分组输出（顶层旧字段保留，向后兼容）
    camera: Optional[Dict[str, Any]] = None
    world: Optional[Dict[str, Any]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        """默认关闭字段：masks / person_points / depth_b64 为 None 时不出现在 JSON 响应中（仅 return_data=true 时返回）。"""
        data = handler(self)
        for _field in ("masks", "person_points", "depth_b64"):
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


def _ground_views_png(depth, intrinsics, inlier_points, plane_grid, plane_normal, plane_d, person_points=None, grid_steps=100, person_bbox_cam=None) -> str:
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

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ef05c", markersize=6, label="ground inliers"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f7c948", markersize=6, label="person cloud"),
        Line2D([0], [0], color="#00e5ff", lw=2, label="camera (view frustum)"),
        Line2D([0], [0], color="red", lw=2, label="fitted plane"),
    ]
    if per_pts is None:
        handles = [h for h in handles if "person cloud" not in h.get_label()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
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
) -> FastAPI:
    state: Dict[str, Any] = {"segmenter": None, "depth": None, "transform": None, "device": torch.device("cpu")}

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
        logger.info("LangSAM 加载完成，服务就绪")
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
            visualization = {
                "overlay": _overlay_png(rgb, floor_mask, result["plane_inlier_mask"], person_mask, contact_mask),
                "views": _ground_views_png(
                    depth,
                    intrinsics,
                    result["plane_inlier_points"],
                    grid,
                    result["plane_normal"],
                    float(result["plane_d"]),
                    person_pts_viz,
                    request.grid_steps,
                    person_bbox_3d_m,
                ),
            }

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

    args = parser.parse_args()
    app = create_ground_app(
        device=args.device,
        sam_type=args.sam_type,
        sam_ckpt=args.sam_ckpt,
        floor_prompt=args.floor_prompt,
        person_prompt=args.person_prompt,
    )
    logger.info("启动 ground_pipeline 服务 %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
