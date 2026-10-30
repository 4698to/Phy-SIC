"""深度后端适配层：把 DepthPro / MoGe-2 统一成 ``infer(rgb) -> (depth, K)``。

约定（``extract_ground`` 与可视化依赖）：

- ``depth``：**原图 H×W** 米制深度，相机系 z 深度（x 右、y 下、z 前）；
- ``K``：**像素 3×3** 内参，主点允许不在图像中心（MoGe-2 会估计主点）。

只要保持这两项，下游点云 / 地面平面 / 世界系 / 可视化全部不用改。
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

logger = logging.getLogger(__name__)

DepthPrediction = Tuple[np.ndarray, np.ndarray]

MOGE_MODEL_DEFAULT = "Ruicheng/moge-2-vitl"


@runtime_checkable
class DepthBackend(Protocol):
    """深度后端最小接口。"""

    name: str
    model_name: str
    use_fp16: bool

    def infer(self, rgb: np.ndarray) -> DepthPrediction:
        """RGB (H, W, 3, uint8) -> (米制 depth (H, W), 像素内参 K (3, 3))。"""
        ...


class DepthProBackend:
    """Apple DepthPro：只输出焦距，主点按图像中心补齐。"""

    name = "depthpro"

    def __init__(self, device: str = "cuda", *, use_fp16: bool = False):
        import depth_pro

        if use_fp16:
            raise ValueError("DepthPro 后端不支持 FP16；请改用 --depth-backend moge2 或关闭 --depth-fp16")
        self.device = torch.device(device)
        self.use_fp16 = False
        self.model_name = "depth-pro"
        self.model, self.transform = depth_pro.create_model_and_transforms(device=self.device)
        self.model.eval()

    def infer(self, rgb: np.ndarray) -> DepthPrediction:
        height, width = rgb.shape[:2]
        tensor = self.transform(rgb).to(self.device)
        with torch.inference_mode():
            pred = self.model.infer(tensor)
        depth = pred["depth"].squeeze().float().detach().cpu().numpy()
        f_px = float(pred["focallength_px"].float().detach().cpu().item())
        intrinsics = np.array(
            [[f_px, 0.0, width / 2.0], [0.0, f_px, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return depth, intrinsics


class MoGe2Backend:
    """MoGe-2：一次前向给出米制深度 + 归一化内参（需自行转像素）。"""

    name = "moge2"

    def __init__(
        self,
        device: str = "cuda",
        *,
        model_name: str = MOGE_MODEL_DEFAULT,
        num_tokens: Optional[int] = None,
        use_fp16: bool = False,
    ):
        try:
            from moge.model.v2 import MoGeModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "当前环境未安装含 v2 的 moge（需要 moge.model.v2）；"
                "请先安装 MoGe-2 后再用 --depth-backend moge2 启动"
            ) from exc
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16)
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.model = MoGeModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def infer(self, rgb: np.ndarray) -> DepthPrediction:
        height, width = rgb.shape[:2]
        image = torch.from_numpy(np.ascontiguousarray(rgb))
        image = image.to(self.device).permute(2, 0, 1).float().div_(255.0)

        amp = self.use_fp16 and self.device.type == "cuda"
        autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            out = (
                self.model.infer(image, use_fp16=amp)
                if self.num_tokens is None
                else self.model.infer(image, num_tokens=self.num_tokens, use_fp16=amp)
            )

        depth = out["depth"].float()
        intrinsics = out["intrinsics"].float().detach().cpu().numpy().reshape(3, 3)

        # MoGe 声明输出与输入同尺寸；若内部降采样后未还原，插值回原图并留痕。
        if tuple(depth.shape) != (height, width):
            logger.warning(
                "MoGe-2 输出深度尺寸 %s 与输入 %s 不一致，插值回原图",
                tuple(depth.shape),
                (height, width),
            )
            depth = torch.nn.functional.interpolate(
                depth[None, None], size=(height, width), mode="bilinear", align_corners=False
            )[0, 0]
        depth = depth.detach().cpu().numpy()

        # valid mask 仅用于日志统计；无效像素保留 inf，由 extract_ground 的 isfinite 过滤。
        mask = out.get("mask")
        if mask is None:
            mask = out.get("valid_mask")
        if mask is not None:
            logger.debug("MoGe-2 valid_mask ratio=%.3f", float(mask.float().mean().item()))

        fx = float(intrinsics[0, 0]) * width
        fy = float(intrinsics[1, 1]) * height
        cx = float(intrinsics[0, 2]) * width
        cy = float(intrinsics[1, 2]) * height
        if not (np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0):
            raise ValueError(f"MoGe-2 内参非法: fx={fx}, fy={fy}")
        intrinsics_px = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return depth, intrinsics_px


def build_depth_backend(
    backend: str = "depthpro",
    device: str = "cuda",
    *,
    moge_model: str = MOGE_MODEL_DEFAULT,
    moge_num_tokens: Optional[int] = None,
    use_fp16: bool = False,
) -> DepthBackend:
    """按名字构造深度后端；名字非法直接报错（不做静默回退）。"""
    name = (backend or "depthpro").strip().lower()
    if name == "depthpro":
        return DepthProBackend(device=device, use_fp16=use_fp16)
    if name == "moge2":
        return MoGe2Backend(
            device=device,
            model_name=moge_model,
            num_tokens=moge_num_tokens,
            use_fp16=use_fp16,
        )
    raise ValueError(f"未知深度后端: {backend}（可选 depthpro / moge2）")
