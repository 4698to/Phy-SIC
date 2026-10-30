"""单张/多张基线图的深度后端基准测试。

示例：
    python -m ground_pipeline.bench_depth --depth-backend depthpro --out depthpro.json
    python -m ground_pipeline.bench_depth --depth-backend moge2 --moge-model Ruicheng/moge-2-vitl --depth-fp16 --out moge2-vitl-fp16.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from ground_pipeline.depth_backend import MOGE_MODEL_DEFAULT, build_depth_backend

logger = logging.getLogger("bench_depth")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = [
    ROOT / "images/test_image/5306c1d3gy1hsktgsqco6j20sg11wq4s.jpg",
    ROOT / "images/test_image/5306c1d3gy1hsktgsz459j20sg0ugmyv.jpg",
    ROOT / "images/yoga_core.jpg",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取基准图片: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _measure_image(
    backend,
    image_path: Path,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    rgb = _load_rgb(image_path)
    for _ in range(warmup):
        backend.infer(rgb)
    _sync(device)

    allocated_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples_ms: List[float] = []
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        depth, intrinsics = backend.infer(rgb)
        _sync(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert depth is not None and intrinsics is not None
    finite_depth = depth[np.isfinite(depth)]
    fast_samples_ms = [value for value in samples_ms if value <= 5000.0]
    if finite_depth.size:
        depth_stats = {
            "median_m": float(np.median(finite_depth)),
            "p5_m": float(np.percentile(finite_depth, 5)),
            "p95_m": float(np.percentile(finite_depth, 95)),
            "depth_range_m": [float(finite_depth.min()), float(finite_depth.max())],
        }
    else:
        depth_stats = {"median_m": None, "p5_m": None, "p95_m": None, "depth_range_m": None}

    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    return {
        "image": str(image_path),
        "image_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "depth_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "depth_ms_samples": [round(value, 2) for value in samples_ms],
        "depth_ms_fast_samples": [round(value, 2) for value in fast_samples_ms],
        "depth_ms_median": round(float(np.median(fast_samples_ms)), 2) if fast_samples_ms else None,
        "depth_ms_median_all": round(float(np.median(samples_ms)), 2),
        "slow_sample_count": len(samples_ms) - len(fast_samples_ms),
        "f_px": float(intrinsics[0, 0]),
        "fx_px": float(intrinsics[0, 0]),
        "fy_px": float(intrinsics[1, 1]),
        "cx_px": float(intrinsics[0, 2]),
        "cy_px": float(intrinsics[1, 2]),
        "valid_ratio": float(np.isfinite(depth).mean()),
        **depth_stats,
        "gpu_allocated_before_mb": round(allocated_before / 1024**2, 2) if allocated_before is not None else None,
        "gpu_peak_allocated_mb": round(peak_allocated / 1024**2, 2) if peak_allocated is not None else None,
        "gpu_peak_allocated_delta_mb": round((peak_allocated - allocated_before) / 1024**2, 2)
        if peak_allocated is not None and allocated_before is not None
        else None,
        "gpu_peak_reserved_mb": round(peak_reserved / 1024**2, 2) if peak_reserved is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthPro / MoGe-2 单模型深度基准")
    parser.add_argument("--depth-backend", choices=("depthpro", "moge2"), default="depthpro")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument("--moge-model", default=MOGE_MODEL_DEFAULT)
    parser.add_argument("--moge-num-tokens", type=int, default=None)
    parser.add_argument("--depth-fp16", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--images", nargs="+", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out", type=Path, default=None, help="结果 JSON 输出路径；默认打印到 stdout")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup 必须 >= 0，--repeats 必须 >= 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("加载 %s 后端 (device=%s)", args.depth_backend, device)
    backend = build_depth_backend(
        args.depth_backend,
        device=str(device),
        moge_model=args.moge_model,
        moge_num_tokens=args.moge_num_tokens,
        use_fp16=args.depth_fp16,
    )
    results = [
        _measure_image(backend, path, device=device, warmup=args.warmup, repeats=args.repeats)
        for path in args.images
    ]
    report = {
        "depth_backend": backend.name,
        "model": backend.model_name,
        "device": str(device),
        "fp16": bool(args.depth_fp16),
        "num_tokens": args.moge_num_tokens if backend.name == "moge2" else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
        logger.info("基准结果写入 %s", args.out)
    else:
        print(output)
    del backend
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
