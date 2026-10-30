"""Opt-in rotation-only fitting through the native SAM3D MHR head."""
from __future__ import annotations

import os

import numpy as np
import torch

ROTATION_FIT_ENV = "POSEEXTRACTOR_SAM3D_ROTATION_FIT"
BONE_SNAP_ENV = "POSEEXTRACTOR_SAM3D_BONE_SNAP"
BONE_SNAP_MODE_ENV = "POSEEXTRACTOR_SAM3D_BONE_SNAP_MODE"
ROTATION_RAW_FIELDS = (
    "body_pose_params", "hand_pose_params", "expr_params", "mhr_model_params",
    "native_sync", "rotation_fit",
)


def rotation_fit_enabled():
    true_values = ("1", "on", "true", "yes", "mhr")
    explicit = os.environ.get(ROTATION_FIT_ENV, "").strip().lower() in true_values
    snap = os.environ.get(BONE_SNAP_ENV, "").strip().lower() in true_values
    legacy = os.environ.get(BONE_SNAP_MODE_ENV, "native").strip().lower() == "legacy"
    return explicit or (snap and not legacy)


def synchronize_native_person(person, head):
    """Replay final body/hand parameters, including full-mode vendor rotations."""
    required = ("body_pose_params", "hand_pose_params", "shape_params", "scale_params",
                "expr_params", "mhr_model_params", "global_rot")
    if any(person.get(key) is None for key in required):
        return {**person, "native_sync": {"synchronized": False, "reason": "missing_native_parameters"}}
    params = {key: torch.as_tensor(np.array(person[key], copy=True), device=head.scale_mean.device,
                                  dtype=torch.float32).reshape(1, -1) for key in required}
    with torch.inference_mode():
        outputs = head.mhr_forward(
            global_trans=params["mhr_model_params"][:, :3] / 10.0,
            global_rot=params["global_rot"], body_pose_params=params["body_pose_params"],
            hand_pose_params=params["hand_pose_params"], scale_params=params["scale_params"],
            shape_params=params["shape_params"], expr_params=params["expr_params"],
            return_keypoints=True, return_joint_coords=True, return_model_params=True,
            return_joint_rotations=True)
    verts, kp, joints, model, rots = [x[0].detach().cpu().numpy() for x in outputs]
    sign = np.array([1, -1, -1], dtype=np.float32)
    verts, kp, joints = verts * sign, kp[:70] * sign, joints * sign
    mismatch = float(np.max(np.abs(verts - np.asarray(person["pred_vertices"]))))
    model_delta = float(np.max(np.abs(model - np.asarray(person["mhr_model_params"]))))
    valid = all(np.isfinite(x).all() for x in (verts, kp, joints, model, rots))
    detail = {"synchronized": False, "native_reproduction_max_m": mismatch,
              "model_parameter_max_delta": model_delta}
    if not valid or mismatch > 2e-5 or model_delta != 0:
        return {**person, "native_sync": {**detail, "reason": "native_forward_does_not_reproduce_input"}}
    detail.update({"synchronized": True, "mesh_source": "native_mhr_forward",
                   "input_rotation_max_delta": float(np.max(np.abs(rots - person["pred_global_rots"])))})
    result = {**person, "pred_vertices": verts, "pred_keypoints_3d": kp,
              "pred_joint_coords": joints, "pred_global_rots": rots,
              "mhr_model_params": model, "native_sync": detail}
    for key, value in (("mhr_joint_coords", joints), ("mhr_joint_global_rots", rots)):
        if key in person:
            result[key] = value
    return result


def local_rotation_delta_deg(before_global, after_global, parents):
    """由父子全局旋转还原局部旋转，测量相对旋转的实际角度。"""
    before = np.asarray(before_global, dtype=np.float64).copy()
    after = np.asarray(after_global, dtype=np.float64).copy()
    parents = np.asarray(parents)
    children = np.flatnonzero(parents >= 0)
    before[children] = np.swapaxes(before_global[parents[children]], -1, -2) @ before[children]
    after[children] = np.swapaxes(after_global[parents[children]], -1, -2) @ after[children]
    relative = after @ np.swapaxes(before, -1, -2)
    # atan2 同时利用反对称项与迹，避免 float32 旋转矩阵在零角附近的 acos 误差。
    sine = 0.5 * np.linalg.norm(np.stack((relative[:, 2, 1] - relative[:, 1, 2],
                                        relative[:, 0, 2] - relative[:, 2, 0],
                                        relative[:, 1, 0] - relative[:, 0, 1]), axis=1), axis=1)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arctan2(sine, cosine))


def fit_native_rotations(person, head, depth, person_mask, cam_int, image_hw):
    """Caller owns the estimator lock and head device lifetime; no autograd required."""
    import cv2
    from scipy.spatial import cKDTree

    detail = {"method": "native_mhr_coordinate_search", "accepted": False,
              "max_parameter_delta_deg": 5.0, "steps_deg": [2.0, 1.0],
              "local_joint_angle_limit_enforced": False,
              "regularization_weight_m": 0.003, "projection_weight_m": 0.003,
              "projection_scale_px": 10.0, "evaluations": 0}

    def fallback(reason):
        detail["reason"] = reason
        return {**person, "rotation_fit": detail}

    character = getattr(head.mhr, "character_torch", None)
    parameter_transform = getattr(getattr(character, "parameter_transform", None),
                                  "parameter_transform", None)
    joint_parents = getattr(getattr(character, "skeleton", None), "joint_parents", None)
    detail["native_backend"] = "torchscript" if isinstance(head.mhr, torch.jit.ScriptModule) else "momentum"
    detail["character_torch_metadata_available"] = parameter_transform is not None and joint_parents is not None
    if parameter_transform is None or joint_parents is None:
        return fallback("missing_native_character_torch_metadata")
    if person.get("native_sync", {}).get("synchronized") is False:
        return fallback("native_forward_does_not_reproduce_input")
    required = ("body_pose_params", "hand_pose_params", "shape_params", "scale_params",
                "expr_params", "mhr_model_params", "global_rot", "pred_cam_t")
    if any(person.get(k) is None for k in required):
        return fallback("missing_native_parameters")
    if depth is None or person_mask is None or cam_int is None:
        return fallback("missing_depth_mask_or_intrinsics")
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(person_mask, dtype=np.uint8)
    k = np.asarray(cam_int, dtype=np.float64).copy()
    ih, iw = image_hw
    if depth.ndim != 2 or mask.shape != (ih, iw) or k.shape != (3, 3):
        return fallback("invalid_target_dimensions")
    dh, dw = depth.shape
    # Intrinsics passed to infer are original-image pixels, even for resized depth.
    k[0] *= dw / iw
    k[1] *= dh / ih
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        return fallback("invalid_intrinsics")
    mask = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
    valid = mask & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    if len(u) < 120:
        return fallback("too_few_valid_depth_samples")
    pick = np.linspace(0, len(u) - 1, min(len(u), 2500)).astype(int)
    v, u = v[pick], u[pick]
    z = depth[v, u]
    cloud = np.stack(((u - k[0, 2]) * z / k[0, 0],
                      (v - k[1, 2]) * z / k[1, 1], z), axis=1)
    cloud_tree = cKDTree(cloud)
    device = head.scale_mean.device
    params = {key: torch.as_tensor(np.array(person[key], copy=True), device=device,
                                  dtype=torch.float32).reshape(1, -1) for key in required
              if key != "pred_cam_t"}
    body0 = params["body_pose_params"].clone()
    cam_t = np.asarray(person["pred_cam_t"], dtype=np.float64).reshape(3)
    global_trans = params["mhr_model_params"][:, :3] / 10.0
    # The native transform is authoritative: rotation candidates must not drive
    # local translations or scales. Hands and face remain at their native values.
    parents = joint_parents.detach().cpu().numpy()
    transform = parameter_transform.to_dense().detach().cpu().numpy()
    if transform.ndim != 2 or transform.shape[0] != len(parents) * 7 or len(parents) != 127:
        return fallback("invalid_native_character_metadata")
    transform = transform.reshape(len(parents), 7, -1)
    # 仅搜索既有前 56 个身体参数；列耦合可同时影响多个关节，不等同于逐骨角度限幅。
    indices = [i for i in range(min(56, body0.shape[1]))
               if np.max(np.abs(transform[:, [0, 1, 2, 6], i + 6])) < 1e-8
               and np.max(np.abs(transform[:, 3:6, i + 6])) > 1e-8]
    if not indices:
        return fallback("no_rotation_only_native_columns")
    edges = np.flatnonzero(parents >= 0)
    sign = np.array([1, -1, -1], dtype=np.float32)
    focal = float(np.asarray(person["focal_length"]).reshape(-1)[0])

    def forward(body):
        outputs = head.mhr_forward(
            global_trans=global_trans, global_rot=params["global_rot"],
            body_pose_params=body, hand_pose_params=params["hand_pose_params"],
            scale_params=params["scale_params"], shape_params=params["shape_params"],
            expr_params=params["expr_params"], return_keypoints=True,
            return_joint_coords=True, return_model_params=True, return_joint_rotations=True)
        verts, kp, joints, model, rots = [x[0].detach().cpu().numpy() for x in outputs]
        verts, kp, joints = verts * sign, kp[:70] * sign, joints * sign
        camera_kp = kp + cam_t
        uv = focal * camera_kp[:, :2] / camera_kp[:, 2:3] + np.array([iw / 2, ih / 2])
        return verts, kp, joints, model, rots, uv

    with torch.inference_mode():
        native = forward(body0)
        mismatch = float(np.max(np.abs(native[0] - np.asarray(person["pred_vertices"]))))
        detail["native_reproduction_max_m"] = mismatch
        if not np.isfinite(mismatch) or mismatch > 2e-5:
            return fallback("native_forward_does_not_reproduce_input")
        verts_cam = native[0] + cam_t
        uv = verts_cam @ k.T
        uv = uv[:, :2] / uv[:, 2:3]
        good = np.isfinite(uv).all(axis=1) & (verts_cam[:, 2] > 0)
        pixels = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(int)
        good &= (pixels[:, 0] >= 0) & (pixels[:, 0] < dw)
        good &= (pixels[:, 1] >= 0) & (pixels[:, 1] < dh)
        ids = np.flatnonzero(good)
        ids = ids[valid[pixels[ids, 1], pixels[ids, 0]]]
        zbuf = np.full((dh, dw), np.inf)
        np.minimum.at(zbuf, (pixels[ids, 1], pixels[ids, 0]), verts_cam[ids, 2])
        ids = ids[verts_cam[ids, 2] <= zbuf[pixels[ids, 1], pixels[ids, 0]] + 0.015]
        if len(ids) < 120:
            return fallback("too_few_native_visible_samples")
        ids = ids[np.linspace(0, len(ids) - 1, min(len(ids), 2000)).astype(int)]
        # Both directions retain identical point/vertex IDs for every candidate.
        detail.update({"depth_samples": len(cloud), "visible_vertex_samples": len(ids),
                       "sampling": "fixed_depth_pixels_and_native_visible_vertex_ids",
                       "rotation_parameter_indices": indices,
                       "body_parameter_count": int(body0.shape[1]),
                       "searched_body_parameter_prefix": min(56, int(body0.shape[1]))})
        native_lengths = np.linalg.norm(native[2][edges] - native[2][parents[edges]], axis=1)
        input_joints = np.asarray(person["pred_joint_coords"])
        input_lengths = np.linalg.norm(input_joints[edges] - input_joints[parents[edges]], axis=1)
        detail.update({"validated_bone_count": int(len(edges)), "bone_length_tolerance_m": 2e-6,
                       "joint_parents": parents.tolist(), "before_bone_lengths_m": input_lengths.tolist()})
        if float(np.max(np.abs(native_lengths - input_lengths))) > 2e-6:
            return fallback("native_input_bone_lengths_mismatch")
        frozen_cols = np.ones(len(native[3]), dtype=bool)
        frozen_cols[np.asarray(indices) + 6] = False
        target_uv = np.asarray(person["pred_keypoints_2d"], dtype=np.float64)

        def score(output, body):
            verts, _, joints, model, _, uv = output
            if not all(np.isfinite(x).all() for x in output) or np.any((verts + cam_t)[:, 2] <= 0):
                return None
            if np.any(model[frozen_cols] != native[3][frozen_cols]):
                return None
            lengths = np.linalg.norm(joints[edges] - joints[parents[edges]], axis=1)
            bone_delta = float(np.max(np.abs(lengths - native_lengths)))
            if bone_delta > 2e-6:
                return None
            query = cKDTree(verts + cam_t).query(cloud)[0]
            surface = cloud_tree.query((verts + cam_t)[ids])[0]
            depth_loss = float(np.minimum(query, 0.2).mean() + 0.5 * np.minimum(surface, 0.2).mean())
            proj = float(np.sqrt(np.mean(np.sum((uv - target_uv) ** 2, axis=1))))
            if proj > 10.0 or float(np.max(np.linalg.norm(uv - target_uv, axis=1))) > 25.0:
                return None
            delta = (body - body0)[0, indices].cpu().numpy()
            reg = float(np.mean((delta / np.deg2rad(5.0)) ** 2))
            return {"objective_m": depth_loss + 0.003 * (proj / 10.0) ** 2 + 0.003 * reg,
                    "depth_loss_m": depth_loss, "cloud_mesh_mean_m": float(query.mean()),
                    "cloud_mesh_p90_m": float(np.percentile(query, 90)),
                    "cloud_within_3cm": float(np.mean(query < 0.03)),
                    "projection_rms_px": proj, "bone_length_max_delta_m": bone_delta}

        baseline = score(native, body0)
        if baseline is None:
            return fallback("invalid_native_baseline")
        best_body, best_output, best_score = body0.clone(), native, baseline
        for step in (2.0, 1.0):
            for index in indices:
                base_body = best_body.clone()
                for direction in (-1, 1):
                    candidate = base_body.clone()
                    candidate[0, index] = torch.clamp(
                        candidate[0, index] + direction * np.deg2rad(step),
                        body0[0, index] - np.deg2rad(5), body0[0, index] + np.deg2rad(5))
                    output = forward(candidate)
                    detail["evaluations"] += 1
                    measured = score(output, candidate)
                    if (measured is not None and measured["objective_m"] < best_score["objective_m"] - 1e-7
                            and measured["depth_loss_m"] < best_score["depth_loss_m"]
                            and measured["cloud_mesh_mean_m"] < best_score["cloud_mesh_mean_m"]):
                        best_body, best_output, best_score = candidate, output, measured
        detail["before"], detail["after"] = baseline, best_score
        local_angles = local_rotation_delta_deg(native[4], best_output[4], parents)
        max_joint = int(np.argmax(local_angles))
        detail.update({"local_rotation_delta_deg": local_angles.tolist(),
                       "max_local_rotation_delta_deg": float(local_angles[max_joint]),
                       "max_local_rotation_joint_id": max_joint,
                       "local_rotation_over_5deg_joint_ids": np.flatnonzero(local_angles > 5.0 + 1e-4).tolist()})
        if best_score["objective_m"] >= baseline["objective_m"] - 1e-6:
            return fallback("no_matched_metric_improvement")
        vertices, kp, joints, model, rots, uv = best_output
        delta = (best_body - body0)[0].cpu().numpy()
        frozen_cols = np.ones(len(model), dtype=bool)
        frozen_cols[np.asarray(indices) + 6] = False
        freeze_delta = float(np.max(np.abs(model[frozen_cols] - native[3][frozen_cols])))
        if freeze_delta != 0:
            return fallback("frozen_model_parameters_changed")
        changed = np.flatnonzero(np.abs(delta) > 1e-7)
        rotation_delta = np.linalg.norm(rots - native[4], axis=(1, 2))
        detail.update({"accepted": True, "reason": "matched_metrics_improved",
                       "changed_body_indices": changed.tolist(),
                       "body_delta_deg": np.rad2deg(delta[changed]).tolist(),
                       "frozen_model_parameters_max_delta": freeze_delta,
                       "changed_global_rotation_joint_ids": np.flatnonzero(rotation_delta > 1e-5).tolist(),
                       "before_joint_coords": native[2].tolist(),
                       "before_global_rots": native[4].tolist(), "joint_parents": parents.tolist(),
                       "mesh_max_shift_m": float(np.linalg.norm(vertices - native[0], axis=1).max())})
        result = {**person, "body_pose_params": best_body[0].cpu().numpy(),
                  "pred_vertices": vertices, "pred_keypoints_3d": kp,
                  "pred_keypoints_2d": uv.astype(np.float32), "pred_joint_coords": joints,
                  "pred_global_rots": rots, "mhr_model_params": model, "rotation_fit": detail}
        for key, value in (("mhr_joint_coords", joints), ("mhr_joint_global_rots", rots)):
            if key in person:
                result[key] = value
        return result
