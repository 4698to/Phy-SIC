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
from pydantic import BaseModel, Field

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


def _cloud_png(inlier_points, plane_points, grid_steps, camera_height_m) -> str:
    """地面内点 + 地面平面网格的 3D 视图（相机坐标系：x 右、y 下、z 前）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = inlier_points
    if len(pts) > 8000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 8000, replace=False)]
    fig = plt.figure(figsize=(7, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=pts[:, 2], cmap="viridis", alpha=0.4)
    grid = plane_points.reshape(grid_steps, grid_steps, 3)
    ax.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], color="orange", alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("ground plane (camera height %.2f m)" % camera_height_m)
    ax.view_init(elev=15, azim=-135)
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    return _to_data_uri(buf)


def _world3d_png(depth, intrinsics, inlier_points, plane_grid, plane_normal, plane_d) -> str:
    """世界 3D 视角：地面法向量对齐到 -z，地面 z=0、物体离地高度为正（与 instance_ground_views 一致）。

    - scene 点云按离地高度着色（viridis）
    - 地面内点（绿）、拟合地面平面（红网格）、相机位置（红星）
    - 采样反投影，避免大图全量反投影的内存峰值
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fx, cx = float(intrinsics[0, 0]), float(intrinsics[0, 2])
    fy, cy = float(intrinsics[1, 1]), float(intrinsics[1, 2])

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

    # 采样场景点（不生成全量 HxWx3 数组）
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

    # 相机系 -> 世界系
    R = rotate_to_up(np.asarray(plane_normal, dtype=float))
    wscene = scene @ R.T
    win = np.asarray(inlier_points, dtype=float) @ R.T
    wground = np.asarray(plane_grid, dtype=float) @ R.T
    wscene[:, 2] -= plane_d
    win[:, 2] -= plane_d
    wground[:, 2] -= plane_d
    cam_world = np.array([0.0, 0.0, -plane_d], dtype=float)

    n_in = min(80_000, len(win))
    if len(win) > n_in:
        win = win[np.random.default_rng(1).choice(len(win), n_in, replace=False)]

    zmin, zmax = np.percentile(wscene[:, 2], [2, 98])
    fig = plt.figure(figsize=(10.5, 9), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(wscene[:, 0], wscene[:, 1], wscene[:, 2], c=wscene[:, 2], cmap="viridis",
               s=0.4, alpha=0.35, vmin=zmin, vmax=zmax)
    ax.scatter(win[:, 0], win[:, 1], win[:, 2], c="#2ef05c", s=0.6, alpha=0.55)
    ax.plot_surface(wground[:, :, 0], wground[:, :, 1], wground[:, :, 2], color="red", alpha=0.45)
    center = win.mean(axis=0)
    ax.quiver(center[0], center[1], center[2], 0, 0, 1, length=1.5, color="k", arrow_length_ratio=0.15)
    ax.scatter([cam_world[0]], [cam_world[1]], [cam_world[2]], c="red", s=60, marker="*")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("height above ground (m)")
    ax.set_title("GROUND in world 3D | ground at z=0, normal->up", fontsize=12)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
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
            visualization = {
                "overlay": _overlay_png(rgb, floor_mask, result["plane_inlier_mask"], person_mask, contact_mask),
                "cloud": _cloud_png(inliers_arr, grid, request.grid_steps, abs(float(result["plane_d"]))),
                "world_3d": _world3d_png(
                    depth,
                    intrinsics,
                    result["plane_inlier_points"],
                    grid,
                    result["plane_normal"],
                    float(result["plane_d"]),
                ),
            }

        return GroundResult(
            ok=True,
            plane_normal=result["plane_normal"].astype(float).tolist(),
            plane_d=float(result["plane_d"]),
            camera_height_m=abs(float(result["plane_d"])),
            focal_px=f_px,
            depth_range_m=[float(np.nanmin(depth)), float(np.nanmax(depth))],
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
