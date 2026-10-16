from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import extract_ground, save_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a ground plane from RGB, metric depth, camera intrinsics, and a floor mask."
    )
    parser.add_argument("--image", required=True, type=Path, help="RGB image")
    parser.add_argument("--depth", required=True, type=Path, help="Metric depth .npy file")
    parser.add_argument("--mask", required=True, type=Path, help="Floor mask image or .npy file")
    parser.add_argument("--intrinsics", required=True, type=Path, help="3x3 camera intrinsics .npy file")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz path")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    depth = np.load(args.depth)
    if args.mask.suffix.lower() == ".npy":
        mask = np.load(args.mask).astype(bool)
    else:
        mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    intrinsics = np.load(args.intrinsics)

    result = extract_ground(image, depth, intrinsics, mask)
    save_result(result, args.output)
    print(f"Saved ground data to {args.output}")
    print(f"Plane: n={result['plane_normal']}, d={result['plane_d']:.6f}")
    print(f"Ground inliers: {result['plane_inlier_mask'].sum()}")


if __name__ == "__main__":
    main()
