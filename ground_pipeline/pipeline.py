from __future__ import annotations

from pathlib import Path

import numpy as np


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject an HxW depth map into camera-space points (x, y, z)."""
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if not np.isfinite(intrinsics).all() or abs(np.linalg.det(intrinsics)) < 1e-12:
        raise ValueError("intrinsics must be finite and invertible")

    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((cols, rows, np.ones_like(cols)), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    return rays * depth[..., None]


def extract_ground(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    floor_mask: np.ndarray,
    *,
    distance_threshold: float = 0.02,
    iterations: int = 256,
    grid_steps: int = 100,
    random_seed: int = 0,
    min_vertical_component: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """Fit a ground plane from a supplied metric depth map and floor mask.

    Coordinates and thresholds use the same units as ``depth``. The image is
    accepted to enforce that all image-space inputs share a common resolution.
    Segmentation and depth-model inference are deliberately injectable at the
    project boundary; this core does not load human-scene models.

    ``min_vertical_component`` 是竖直面剔除阈值：相机系 y 轴向下，真实地面法向的
    ``|n_y|`` 应接近 1；背板/墙面等竖直面的 ``|n_y|`` 接近 0。RANSAC 只接受
    ``|n_y| >= min_vertical_component`` 的候选平面，避免分割掩码里的大面积背景墙
    抢走最大内点数。
    """
    image_rgb = np.asarray(image_rgb)
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    floor_mask = np.asarray(floor_mask, dtype=bool)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if depth.shape != image_rgb.shape[:2] or floor_mask.shape != depth.shape:
        raise ValueError("image_rgb, depth, and floor_mask must have matching HxW")
    if distance_threshold <= 0 or iterations < 1 or grid_steps < 2:
        raise ValueError("distance_threshold must be positive; iterations >= 1; grid_steps >= 2")
    if not 0.0 <= min_vertical_component <= 1.0:
        raise ValueError("min_vertical_component must be within [0, 1]")

    points = depth_to_points(depth, intrinsics)
    valid_mask = np.isfinite(points).all(axis=-1) & (depth > 0)
    candidates = points[floor_mask & valid_mask]
    if len(candidates) < 3:
        raise ValueError("at least three valid floor points are required")

    rng = np.random.default_rng(random_seed)
    best_inliers = np.zeros(len(candidates), dtype=bool)
    rejected_vertical = 0
    for _ in range(iterations):
        sample = candidates[rng.choice(len(candidates), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if abs(float(normal[1])) < min_vertical_component:
            rejected_vertical += 1
            continue
        offset = -np.dot(normal, sample[0])
        inliers = np.abs(candidates @ normal + offset) <= distance_threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 3:
        raise ValueError(
            "could not fit a horizontal plane from the supplied floor points"
            f"（{rejected_vertical}/{iterations} 个候选因接近竖直面被剔除）"
        )

    plane_inlier_points = candidates[best_inliers]
    center = plane_inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(plane_inlier_points - center, full_matrices=False)
    plane_normal = vh[-1]
    plane_normal /= np.linalg.norm(plane_normal)
    plane_d = float(-np.dot(plane_normal, center))

    # 精化后的平面仍需通过竖直面检查，防止内点混合把法向拉回墙面。
    if abs(float(plane_normal[1])) < min_vertical_component:
        raise ValueError(
            "fitted plane is vertical"
            f" (|n_y|={abs(float(plane_normal[1])):.2f} < {min_vertical_component:.2f})；"
            "地面掩码可能包含墙面/背板"
        )

    # Fix the sign deterministically so repeated runs have stable output.
    dominant_axis = np.argmax(np.abs(plane_normal))
    if plane_normal[dominant_axis] < 0:
        plane_normal = -plane_normal
        plane_d = -plane_d

    # Generate a grid in the fitted plane's own basis (works for any orientation).
    basis_seed = np.eye(3)[np.argmin(np.abs(plane_normal))]
    axis_u = np.cross(plane_normal, basis_seed)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(plane_normal, axis_u)
    centered = plane_inlier_points - center
    uv = np.column_stack((centered @ axis_u, centered @ axis_v))
    u_values = np.linspace(*np.percentile(uv[:, 0], [1, 99]), grid_steps)
    v_values = np.linspace(*np.percentile(uv[:, 1], [1, 99]), grid_steps)
    uu, vv = np.meshgrid(u_values, v_values)
    plane_points = center + uu[..., None] * axis_u + vv[..., None] * axis_v

    return {
        "depth": depth.astype(np.float32),
        "K": intrinsics.astype(np.float32),
        "points": points.astype(np.float32),
        "valid_mask": valid_mask,
        "floor_mask": floor_mask,
        "plane_inlier_mask": _make_inlier_mask(floor_mask, valid_mask, best_inliers),
        "plane_normal": plane_normal.astype(np.float32),
        "plane_d": plane_d,
        "plane_points": plane_points.reshape(-1, 3).astype(np.float32),
        "plane_inlier_points": plane_inlier_points.astype(np.float32),
    }


def _make_inlier_mask(
    floor_mask: np.ndarray, valid_mask: np.ndarray, inliers: np.ndarray
) -> np.ndarray:
    result = np.zeros(floor_mask.shape, dtype=bool)
    candidate_pixels = np.flatnonzero(floor_mask & valid_mask)
    result.flat[candidate_pixels[inliers]] = True
    return result


def save_result(result: dict[str, np.ndarray | float], output_path: str | Path) -> None:
    """Save arrays and plane parameters in a compressed, portable NPZ file."""
    np.savez_compressed(Path(output_path), **result)
