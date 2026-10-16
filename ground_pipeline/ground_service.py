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
    distance_threshold: Optional[float] = Field(0.02, gt=0, description="平面拟合距离阈值（米）")
    iterations: Optional[int] = Field(256, ge=1, description="RANSAC 迭代次数")
    grid_steps: Optional[int] = Field(100, ge=2, description="平面网格分辨率")
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


def _overlay_png(rgb, floor_mask, inlier_mask) -> str:
    """原图 + 地面 mask（绿）+ 平面内点 mask（蓝）覆盖图。"""
    from PIL import Image

    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    floor_im = Image.fromarray((floor_mask.astype(np.uint8)) * 255)
    inlier_im = Image.fromarray((inlier_mask.astype(np.uint8)) * 255)
    green = Image.new("RGBA", base.size, (46, 204, 113, 70))
    overlay.paste(green, (0, 0), floor_im)
    blue = Image.new("RGBA", base.size, (41, 128, 185, 110))
    overlay.paste(blue, (0, 0), inlier_im)
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


def create_ground_app(
    *,
    device: str = "auto",
    sam_type: str = "sam2.1_hiera_small",
    sam_ckpt: str | Path = DEFAULT_SAM_CKPT,
    floor_prompt: str = DEFAULT_FLOOR_PROMPT,
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

        rng = np.random.default_rng(0)
        inliers_arr = result["plane_inlier_points"]
        inliers_arr = inliers_arr[rng.choice(len(inliers_arr), size=min(2000, len(inliers_arr)), replace=False)]
        grid = result["plane_points"].reshape(request.grid_steps, request.grid_steps, 3)

        visualization = None
        if request.visualize:
            visualization = {
                "overlay": _overlay_png(rgb, floor_mask, result["plane_inlier_mask"]),
                "cloud": _cloud_png(inliers_arr, grid, request.grid_steps, abs(float(result["plane_d"]))),
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

    args = parser.parse_args()
    app = create_ground_app(
        device=args.device,
        sam_type=args.sam_type,
        sam_ckpt=args.sam_ckpt,
        floor_prompt=args.floor_prompt,
    )
    logger.info("启动 ground_pipeline 服务 %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
