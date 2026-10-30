"""Shared COCO-17 heatmap scores and independent mesh self-visibility estimates."""
from __future__ import annotations

import numpy as np

if __package__:
    from .bone_depth_snap import _raster_zbuf
else:
    from bone_depth_snap import _raster_zbuf

COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO17_NAMES = COCO_NAMES


def detect_vitpose(rgb):
    """Run the external detector without hiding inference or offload failures."""
    import cv2
    from ai_vitpose.core import vitpose_detector

    try:
        points, info = vitpose_detector.detect_keypoints_vitpose(
            cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        )
        points = np.asarray(points)
        if points.shape != (17, 3):
            raise ValueError("ViTPose must return a 17x3 array")
        return points, info
    finally:
        vitpose_detector.offload_to_cpu()


def vitpose_bbox(points, image_hw):
    """Return the external detector's padded tight box, or no reliable box."""
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return None
    valid = np.all(np.isfinite(points[:, :3]), axis=1) & (points[:, 2] >= 0.20)
    if np.count_nonzero(valid) < 5:
        return None
    h, w = image_hw
    x1, y1 = points[valid, :2].min(axis=0)
    x2, y2 = points[valid, :2].max(axis=0)
    pad_x = max(20.0, (x2 - x1) * 0.20)
    pad_y = max(20.0, (y2 - y1) * 0.15)
    return np.array([[
        max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
        min(float(w - 1), x2 + pad_x), min(float(h - 1), y2 + pad_y),
    ]], dtype=np.float32)


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _array(value, columns):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.ndim != 2 or result.shape[1] != columns:
        return None
    return result


def _local_coverage(data, xy, image_hw, person_mask=False):
    """Independent 3x3 observation coverage, in the observation's own pixels."""
    if data is None:
        return None
    try:
        data = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if data.ndim != 2 or not data.size:
        return None
    h, w = image_hw
    dh, dw = data.shape
    x = min(dw - 1, int(np.floor(xy[0] * dw / w)))
    y = min(dh - 1, int(np.floor(xy[1] * dh / h)))
    patch = data[max(0, y - 1):min(dh, y + 2), max(0, x - 1):min(dw, x + 2)]
    valid = np.isfinite(patch)
    if person_mask:
        if not np.any(valid):
            return None
        return float(np.mean(patch[valid] > 0))
    return float(np.mean(valid & (patch > 0)))


def estimate_visibility(joints_cam, verts_cam, faces, focal, image_hw, depth=None, mask=None):
    """Estimate SELF-occlusion from nearby surface samples, never joint depth.

    Values resemble COCO visibility codes but are estimates, not ground truth.
    Scene depth and person mask coverage are diagnostics only.
    """
    thresholds = {
        "visible_fraction_min": 0.6, "occluded_fraction_max": 0.2,
        "surface_radius_extra_m": 0.025, "surface_radius_cap_extra_m": 0.08,
        "minimum_local_samples": 8, "depth_tolerance_base_m": 0.015,
        "depth_tolerance_per_m": 0.005, "raster_long_side_max": 256,
        "raster_sample_window": 3, "observation_sample_window": 3,
    }
    results = [{
        "visibility": None,
        "visibility_source": "estimated_mesh_self_occlusion_not_coco_ground_truth",
        "method": "nearby_3d_surface_vertices_vs_global_raster_zbuf_3x3_max_finite",
        "visible_surface_fraction": None, "local_surface_samples": 0,
        "raster_surface_samples": 0, "nearest_surface_distance_m": None,
        "local_surface_radius_m": None, "joint_projection_xy": None,
        "local_depth_valid_fraction": None, "person_mask_coverage": None,
        "observation_source": "independent_depth_and_person_mask_not_visibility",
        "thresholds": dict(thresholds), "reason": "invalid_or_missing_geometry",
    } for _ in COCO_NAMES]
    joints = _array(joints_cam, 3)
    verts = _array(verts_cam, 3)
    try:
        h, w = map(int, image_hw)
        focal = float(focal)
    except (TypeError, ValueError, OverflowError):
        return results
    if h <= 0 or w <= 0 or not np.isfinite(focal) or focal <= 0 or joints is None:
        return results

    zbuf = None
    surface = None
    scale = min(1.0, 256.0 / max(h, w))
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    sx, sy = rw / w, rh / h
    face_array = _array(faces, 3)
    if verts is not None and len(verts) and face_array is not None and len(face_array):
        valid_faces = np.all(np.isfinite(face_array), axis=1)
        valid_faces &= np.all((face_array >= 0) & (face_array < len(verts)), axis=1)
        valid_faces &= np.all(face_array == np.floor(face_array), axis=1)
        face_indices = face_array[valid_faces].astype(np.int64)
        vertex_valid = np.all(np.isfinite(verts), axis=1) & (verts[:, 2] > 1e-3)
        face_indices = face_indices[np.all(vertex_valid[face_indices], axis=1)]
        if len(face_indices):
            # Match exact resized principal point even when the aspect ratio rounds.
            raster_verts = verts.copy()
            raster_verts[:, 0] *= sx / scale
            raster_verts[:, 1] *= sy / scale
            zbuf = _raster_zbuf(raster_verts, face_indices, focal * scale, rh, rw)
            surface = verts[np.unique(face_indices)]

    for i, joint in enumerate(joints[:17]):
        item = results[i]
        if not np.all(np.isfinite(joint)) or joint[2] <= 1e-3:
            item["reason"] = "invalid_joint_projection"
            continue
        xy = focal * joint[:2] / joint[2] + np.array([w / 2.0, h / 2.0])
        if not np.all(np.isfinite(xy)):
            item["reason"] = "invalid_joint_projection"
            continue
        item["joint_projection_xy"] = xy.tolist()
        if not (0 <= xy[0] < w and 0 <= xy[1] < h):
            item["visibility"] = 0
            item["visibility_source"] = "estimated_out_of_image_projection_not_coco_ground_truth"
            item["reason"] = "out_of_image"
            continue
        item["local_depth_valid_fraction"] = _local_coverage(depth, xy, (h, w))
        item["person_mask_coverage"] = _local_coverage(mask, xy, (h, w), person_mask=True)
        if surface is None or zbuf is None or not np.any(np.isfinite(zbuf)):
            continue
        distances = np.linalg.norm(surface - joint, axis=1)
        nearest = float(distances.min())
        radius = nearest + 0.025
        # Broaden sparse local neighborhoods without reaching unrelated body parts.
        target = min(8, len(surface))
        if np.count_nonzero(distances <= radius) < target:
            radius = min(nearest + 0.08, max(radius, float(np.partition(distances, target - 1)[target - 1])))
        local = surface[distances <= radius]
        item["nearest_surface_distance_m"] = nearest
        item["local_surface_radius_m"] = radius
        item["local_surface_samples"] = len(local)
        uv = (focal * local[:, :2] / local[:, 2:3] + [w / 2.0, h / 2.0]) * [sx, sy]
        in_image = np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < rw)
        in_image &= (uv[:, 1] >= 0) & (uv[:, 1] < rh)
        visible = []
        for point, pixel in zip(local[in_image], uv[in_image]):
            x, y = np.floor(pixel).astype(int)
            patch = zbuf[max(0, y - 1):min(rh, y + 2), max(0, x - 1):min(rw, x + 2)]
            finite = patch[np.isfinite(patch)]
            if not len(finite):
                continue
            tolerance = 0.015 + 0.005 * float(point[2])
            visible.append(bool(point[2] <= float(finite.max()) + tolerance))
        item["raster_surface_samples"] = len(visible)
        if not visible:
            item["reason"] = "no_projected_local_surface_samples"
            continue
        fraction = float(np.mean(visible))
        item["visible_surface_fraction"] = fraction
        item["visibility"] = 2 if fraction >= 0.6 else 1 if fraction <= 0.2 else None
        item["reason"] = "estimated_visible" if fraction >= 0.6 else "estimated_self_occluded" if fraction <= 0.2 else "ambiguous_surface_fraction"
    return _native(results)


def build_joint_annotations(points, info, visibility):
    """Keep raw heatmap scores separate from independent geometric visibility."""
    points = _array(points, 3)
    info = info or {}
    detected = points is not None and info.get("persons", 1) != 0
    annotations = []
    for i, name in enumerate(COCO_NAMES):
        point = points[i] if points is not None and i < len(points) else None
        score = float(point[2]) if detected and point is not None and np.isfinite(point[2]) else None
        diagnostics = visibility[i] if visibility is not None and i < len(visibility) else {"visibility": None}
        entry = dict(diagnostics)
        entry.update({
            "id": i, "name": name, "confidence": score,
            "confidence_source": "vitpose_heatmap" if score is not None else "unavailable",
            "vitpose_xy": point[:2].tolist() if point is not None else None,
        })
        annotations.append(entry)
    return _native(annotations)
