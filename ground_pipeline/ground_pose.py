"""SAM 3D Body adapter used by the ground pipeline."""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _pose_overlay_data_uri(rgb: np.ndarray, keypoints: np.ndarray) -> str:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    limb_colors = (
        (255, 90, 90), (255, 90, 90), (255, 90, 90), (255, 90, 90),
        (255, 190, 40), (255, 190, 40), (255, 120, 40), (60, 210, 255), (40, 150, 255),
        (190, 100, 255), (160, 70, 230), (230, 230, 70),
        (70, 230, 130), (30, 190, 100), (70, 180, 255), (40, 120, 230),
    )
    height, width = rgb.shape[:2]
    line_width = max(6, min(12, round(min(width, height) / 85)))
    joint_radius = max(7, min(13, round(min(width, height) / 75)))
    mhr70_to_coco17 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
    points = np.asarray(keypoints, dtype=np.float32)[list(mhr70_to_coco17)]
    visible = np.isfinite(points).all(axis=1)
    visible &= (points[:, 0] >= 0) & (points[:, 0] < width)
    visible &= (points[:, 1] >= 0) & (points[:, 1] < height)
    for (a, b), color in zip(edges, limb_colors):
        if visible[a] and visible[b]:
            start = tuple(np.rint(points[a, :2]).astype(int))
            end = tuple(np.rint(points[b, :2]).astype(int))
            draw.line((start, end), fill=(0, 0, 0), width=line_width + 4)
            draw.line((start, end), fill=color, width=line_width)
    for index, (point, is_visible) in enumerate(zip(points, visible)):
        if is_visible:
            x, y = np.rint(point[:2]).astype(int)
            radius = joint_radius + 2 if index in (5, 6, 9, 10, 11, 12, 15, 16) else joint_radius
            draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=(0, 0, 0))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 220, 40), outline=(255, 255, 255), width=2)
    title = "SAM3D BODY 2D POSE"
    title_font_size = max(24, min(46, round(min(width, height) / 22)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
    except OSError:
        font = ImageFont.load_default()
    title_box = draw.textbbox((0, 0), title, font=font)
    title_width = title_box[2] - title_box[0]
    title_height = title_box[3] - title_box[1]
    title_pad = max(10, title_font_size // 2)
    draw.rounded_rectangle((8, 8, title_width + title_pad * 2, title_height + title_pad * 2), radius=6, fill=(12, 18, 28))
    draw.text((8 + title_pad, 8 + title_pad), title, font=font, fill=(255, 255, 255), stroke_width=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pose_mesh_data_uri(vertices: np.ndarray, faces: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("SAM3D returned invalid mesh vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("SAM3D returned invalid mesh faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("SAM3D mesh faces reference invalid vertices")

    display_vertices = vertices.copy()
    display_vertices[:, 1:3] *= -1
    triangles = display_vertices[faces]
    z_values = triangles[:, :, 2].mean(axis=1)
    z_min, z_max = float(z_values.min()), float(z_values.max())
    normalized = (z_values - z_min) / max(z_max - z_min, 1e-6)
    face_colors = plt.get_cmap("viridis")(normalized)

    center = (display_vertices.min(axis=0) + display_vertices.max(axis=0)) * 0.5
    radius = max(float((display_vertices.max(axis=0) - display_vertices.min(axis=0)).max()) * 0.58, 1e-3)
    fig = plt.figure(figsize=(15, 5.5), dpi=100, facecolor="#101722")
    for index, (title, azimuth) in enumerate((("Front", -90), ("Side", 0), ("Back", 90)), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d", facecolor="#101722")
        collection = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors=(0.12, 0.16, 0.21, 0.32),
            linewidths=0.08,
            antialiased=True,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=8, azim=azimuth)
        ax.set_title(title, color="white", fontsize=13, pad=4)
        ax.set_axis_off()

    fig.suptitle("SAM3D BODY | PREDICTED MESH · MODEL COORDS", color="white", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# SAM3D 推理类型（与环境变量语义保持一致，见 ai_vitpose/core/sam3d_adapter.py）：
#   full（默认）= 身体解码 + 左右手解码，finger 关节不再是 T-pose，重定向手指链才有数据；
#   body        = 仅身体解码（旧行为），手指停在 T-pose。
SAM3D_INFERENCE_ENV = "POSEEXTRACTOR_SAM3D_INFERENCE"


def sam3d_inference_type() -> str:
    value = os.environ.get(SAM3D_INFERENCE_ENV, "").strip().lower()
    return "body" if value == "body" else "full"


# 推理前放大系数（1–3）。手部 crop 固定 512×512，放大原图可让手部从更多原始像素采样。
SAM3D_UPSCALE_ENV = "POSEEXTRACTOR_SAM3D_UPSCALE"


def sam3d_upscale() -> float:
    try:
        value = float(os.environ.get(SAM3D_UPSCALE_ENV, "1") or 1.0)
    except ValueError:
        value = 1.0
    return min(max(value, 1.0), 3.0)


RAW_POSE_FIELDS = (
    "pred_joint_coords",
    "mhr_joint_coords",
    "pred_global_rots",
    "mhr_joint_global_rots",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "global_rot",
    "pred_vertices",
    "focal_length",
    "scale_params",
    "shape_params",
    # inference_type="full" 才有：手部解码实际使用的左右手框（原图像素坐标）
    "lhand_bbox",
    "rhand_bbox",
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def mask_to_bbox(person_mask: np.ndarray) -> np.ndarray:
    if person_mask.ndim != 2 or not person_mask.size:
        raise ValueError("Expected a non-empty 2D person mask")
    ys, xs = np.nonzero(person_mask)
    if not len(xs):
        raise ValueError("Person mask is empty")

    height, width = person_mask.shape
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    pad_x = max(20.0, (x2 - x1) * 0.2)
    pad_y = max(20.0, (y2 - y1) * 0.15)
    return np.array([[
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width - 1), x2 + pad_x),
        min(float(height - 1), y2 + pad_y),
    ]], dtype=np.float32)


class GroundPoseEstimator:
    def __init__(self, resources: str | Path):
        self.resources = Path(resources).expanduser().resolve()
        vendor_dir = self.resources / "vendor" / "sam-3d-body"
        model_dir = self.resources / "models" / "sam-3d-body-vith"
        checkpoint = model_dir / "model.ckpt"
        mhr_model = model_dir / "assets" / "mhr_model.pt"
        if not vendor_dir.is_dir():
            raise FileNotFoundError(f"SAM3D source not found: {vendor_dir}")
        if not checkpoint.is_file() or not mhr_model.is_file():
            raise FileNotFoundError(f"SAM3D weights not found: {model_dir}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3D Body requires CUDA")

        resource_path = str(self.resources)
        if resource_path not in sys.path:
            sys.path.insert(0, resource_path)
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        model, cfg = load_sam_3d_body(
            checkpoint_path=str(checkpoint),
            mhr_path=str(mhr_model),
            device="cuda",
            offload_mode="cpu",
        )
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self._lock = threading.Lock()

    def infer(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray | None = None,
        want_overlay: bool = False,
        want_mesh: bool = False,
        cam_int: np.ndarray | None = None,
        include_raw: bool = False,
        bbox: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB HxWx3 image")
        if bbox is None:
            if person_mask is None or person_mask.shape != rgb.shape[:2]:
                raise ValueError("Person mask dimensions do not match the image")
            bbox = mask_to_bbox(person_mask)
        else:
            bbox = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        height, width = rgb.shape[:2]

        mode = sam3d_inference_type()
        scale = sam3d_upscale()
        source = rgb
        model_bbox = bbox
        model_cam_int = cam_int
        if scale > 1.001:
            import cv2

            source = cv2.resize(
                np.ascontiguousarray(rgb), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            model_bbox = bbox * scale
            if cam_int is not None:
                model_cam_int = np.array(cam_int, dtype=np.float32, copy=True)
                model_cam_int[0, :] *= scale
                model_cam_int[1, :] *= scale

        with self._lock, torch.inference_mode():
            outputs = self.estimator.process_one_image(
                source,
                bboxes=model_bbox,
                use_mask=False,
                cam_int=(
                    torch.as_tensor(model_cam_int, dtype=torch.float32).unsqueeze(0)
                    if model_cam_int is not None else None
                ),
                inference_type=mode,
            )
        if not outputs:
            raise RuntimeError("SAM3D did not return a body pose")

        person = outputs[0]
        if scale > 1.001:
            # 放大只用于推理；2D 关键点与焦距要还原到原图像素坐标，3D/平移不受影响。
            person = dict(person)
            if person.get("pred_keypoints_2d") is not None:
                person["pred_keypoints_2d"] = (
                    np.asarray(person["pred_keypoints_2d"], dtype=np.float32) / scale
                )
            if person.get("focal_length") is not None:
                person["focal_length"] = (
                    np.asarray(person["focal_length"], dtype=np.float32) / scale
                )
            for key in ("lhand_bbox", "rhand_bbox"):
                if person.get(key) is not None:
                    person[key] = np.asarray(person[key], dtype=np.float32) / scale
        from ai_vitpose.core.normalize import camera_points_to_figure_frame, skeleton24

        joints = camera_points_to_figure_frame(skeleton24(person["pred_joint_coords"]))
        if joints.shape != (24, 3) or not np.isfinite(joints).all():
            raise ValueError("SAM3D returned invalid SMPL-24 joints")
        camera = {
            "focal_px": float(np.asarray(person["focal_length"]).reshape(-1)[0]),
            "translation": np.asarray(person["pred_cam_t"]).reshape(-1).tolist(),
            "image_width": width,
            "image_height": height,
            "calibration": "model_default_fov_not_measured",
        }
        if not np.isfinite(camera["focal_px"]) or not np.isfinite(camera["translation"]).all():
            raise ValueError("SAM3D returned invalid camera parameters")
        result = {
            "status": "success",
            "engine": "sam3d_body_vith",
            "skeleton_source": "MHR127_direct",
            "joints_3d": joints.tolist(),
            "joints_3d_coordinate_system": "figure",
            "joints_3d_unit": "m",
            "orientation_mode": "fixed_camera_to_figure_v6",
            "bbox_2d_xyxy": bbox[0].tolist(),
            "camera": camera,
            "sam3d_inference_type": mode,
            "sam3d_upscale": scale,
        }
        if include_raw:
            raw: dict[str, Any] = {}
            for key in RAW_POSE_FIELDS:
                value = person.get(key)
                if value is None:
                    continue
                try:
                    raw[key] = _to_jsonable(value)
                except Exception as exc:
                    raw[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
            raw["sam3d_inference_type"] = mode
            raw["sam3d_upscale"] = scale
            result["raw"] = raw
        if want_overlay:
            try:
                keypoints_2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
                if keypoints_2d.shape != (70, 2) or not np.isfinite(keypoints_2d).all():
                    raise ValueError("SAM3D returned invalid MHR-70 2D keypoints")
                result["overlay_png_data_uri"] = _pose_overlay_data_uri(rgb, keypoints_2d)
            except Exception as exc:
                result["overlay_error"] = f"{type(exc).__name__}: {exc}"
        if want_mesh:
            try:
                result["mesh_png_data_uri"] = _pose_mesh_data_uri(
                    person["pred_vertices"], self.estimator.faces
                )
                result["mesh_vertices_model"] = np.asarray(person["pred_vertices"], dtype=np.float32)
                result["mesh_faces"] = np.asarray(self.estimator.faces, dtype=np.int32)
                result["mesh_camera_translation"] = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(-1)
                result["mesh_focal_px"] = float(np.asarray(person["focal_length"]).reshape(-1)[0])
                # 配准用临时字段：MHR-70 关键点的原图像素坐标与相机系 3D 坐标。
                keypoints_2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
                keypoints_3d = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
                if keypoints_2d.shape == (70, 2) and keypoints_3d.shape == (70, 3):
                    result["align_keypoints_2d"] = keypoints_2d
                    result["align_keypoints_3d_cam"] = keypoints_3d + np.asarray(
                        person["pred_cam_t"], dtype=np.float32
                    ).reshape(1, 3)
            except Exception as exc:
                result["mesh_error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def refine_rigid_to_scene(
        vertices,
        keypoints,
        *,
        person_points=None,
        contact_points=None,
        keypoints_2d=None,
        focal_px=None,
        image_size=None,
        world_rotation=None,
        plane_d=0.0,
        iterations=120,
        lr=5e-3,
        contact_weight=200.0,
        penetration_weight=400.0,
        chamfer_weight=10.0,
        projection_weight=150.0,
        rotation_reg_weight=30.0,
        translation_reg_weight=15.0,
        max_rotation_deg=0.0,
        max_vertex_shift_m=0.15,
    ) -> dict[str, Any]:
        """刚体收缩精修：只优化整体平移（默认也允许整体旋转，由 ``max_rotation_deg`` 控制），
        姿势与尺度完全冻结。

        网格内部几何逐顶点保持不变，场景损失只能靠整体搬动网格降低，
        因此不会改写 SAM3D 预测的姿势。整体旋转默认关闭：实测绕接触区质心的
        整体转角会改变角色在图像中的朝向，反而恶化 2D 关键点一致性，
        而地面接触只需要沿重力轴平移即可达成。
        """
        report: dict[str, Any] = {"status": "skipped", "method": "rigid_refine+scene_losses"}
        base_np = np.asarray(vertices, dtype=np.float32)
        if base_np.ndim != 2 or base_np.shape[1] != 3 or len(base_np) < 100:
            report["reason"] = "网格形状异常"
            return {"report": report}
        kp_np = None
        if keypoints is not None:
            candidate = np.asarray(keypoints, dtype=np.float32)
            if candidate.ndim == 2 and candidate.shape[1] == 3 and len(candidate):
                kp_np = candidate

        device = torch.device("cpu")
        base = torch.as_tensor(base_np, device=device)
        if world_rotation is not None:
            world_z = torch.as_tensor(
                np.asarray(world_rotation, dtype=np.float32)[2, :], device=device
            )
        else:
            world_z = torch.tensor([0.0, 0.0, 1.0], device=device)

        with torch.no_grad():
            z_base = (base @ world_z) - plane_d
        z_np = z_base.numpy()
        n_verts = int(base.shape[0])
        # 只用最底部的极小区域（≈0.3% 顶点）定义接触高度，避免为压平整个贴地曲面而强行旋转。
        low_idx = np.argsort(z_np)[: min(60, n_verts)]
        rand_idx = np.random.default_rng(0).choice(n_verts, min(2000, n_verts), replace=False)
        sample_np = np.unique(np.concatenate([low_idx, rand_idx]))
        sample_idx = torch.as_tensor(sample_np, dtype=torch.long)
        low_idx_t = torch.as_tensor(low_idx, dtype=torch.long)
        pivot = base[low_idx_t].mean(dim=0, keepdim=True)
        base_centered = base - pivot
        base_sample = base[sample_idx]

        contact_target = 0.0
        contact_target_source = "ground_plane"
        if contact_points is not None and len(contact_points):
            rotation = (
                np.asarray(world_rotation, dtype=np.float32)
                if world_rotation is not None
                else np.eye(3, dtype=np.float32)
            )
            contact_z = np.asarray(contact_points, dtype=np.float32) @ rotation[2, :] - plane_d
            # 接触点本身应落在平面上；深度噪声会让中位数略高于 0，目标值不能为正。
            contact_target = float(np.clip(np.median(contact_z), -0.05, 0.0))
            contact_target_source = "contact_points_median"

        person_np = None
        nearest_t = None
        if person_points is not None and len(person_points) >= 100:
            import scipy.spatial

            person_np = np.asarray(person_points, dtype=np.float32)
            tree = scipy.spatial.cKDTree(person_np.astype(np.float64))
            _, nearest_index = tree.query(base_sample.numpy().astype(np.float64))
            nearest_t = torch.as_tensor(person_np[nearest_index], device=device)

        kp_cam = torch.as_tensor(kp_np, device=device) if kp_np is not None else None
        kp2d = None
        center = None
        focal = 1.0
        if (
            kp_np is not None
            and keypoints_2d is not None
            and focal_px is not None
            and image_size is not None
        ):
            kp2d_np = np.asarray(keypoints_2d, dtype=np.float32)
            if kp2d_np.shape == (len(kp_np), 2):
                valid = np.isfinite(kp2d_np).all(axis=1)
                if int(valid.sum()) >= 6:
                    kp_cam = torch.as_tensor(kp_np[valid], device=device)
                    kp2d = torch.as_tensor(kp2d_np[valid], device=device)
                    center = torch.tensor(
                        [float(image_size[1]) / 2.0, float(image_size[0]) / 2.0], device=device
                    )
                    focal = float(focal_px)

        max_rot_rad = float(np.deg2rad(max_rotation_deg))
        use_rotation = max_rot_rad > 0.0
        drot = torch.zeros(1, 3, device=device, requires_grad=use_rotation)
        dtrans = torch.zeros(1, 3, device=device, requires_grad=True)
        optimizer = torch.optim.Adam(
            [drot, dtrans] if use_rotation else [dtrans], lr=lr
        )

        def rigid(points):
            # 平移只沿地面法向（世界 z）施加：横向位置沿用前面的配准结果，
            # 避免整体横向漂移改变角色在画面中的位置与 2D 关键点一致性。
            shift = (dtrans @ world_z).unsqueeze(0) * world_z
            if not use_rotation:
                return points + shift
            theta = drot.norm(dim=1, keepdim=True).clamp_min(1e-8)
            axis = drot / theta
            kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
            zero = torch.zeros_like(kx)
            skew = torch.stack(
                [
                    torch.stack([zero, -kz, ky], dim=1),
                    torch.stack([kz, zero, -kx], dim=1),
                    torch.stack([-ky, kx, zero], dim=1),
                ],
                dim=1,
            )
            identity = torch.eye(3, device=device).unsqueeze(0)
            rot = (
                identity
                + torch.sin(theta).unsqueeze(2) * skew
                + (1.0 - torch.cos(theta)).unsqueeze(2) * (skew @ skew)
            )
            return points @ rot[0].T + shift

        metrics: dict[str, float] = {}
        for step in range(iterations):
            cam = rigid(base_centered) + pivot
            z = (cam @ world_z) - plane_d
            loss_contact = ((z[low_idx_t] - contact_target) ** 2).mean()
            loss_penetration = (torch.relu(-z[sample_idx] - 0.01) ** 2).mean()
            loss = contact_weight * loss_contact + penetration_weight * loss_penetration

            loss_chamfer = torch.zeros((), device=device)
            if nearest_t is not None:
                loss_chamfer = ((cam[sample_idx] - nearest_t) ** 2).sum(dim=1).mean()
                loss = loss + chamfer_weight * loss_chamfer

            loss_projection = torch.zeros((), device=device)
            if kp2d is not None:
                kp_now = rigid(kp_cam - pivot) + pivot
                kp_z = kp_now[:, 2]
                safe_z = torch.where(kp_z > 1e-3, kp_z, torch.ones_like(kp_z))
                uv = kp_now[:, :2] * (focal / safe_z).unsqueeze(1) + center
                error_px = ((uv - kp2d) ** 2).sum(dim=1)
                loss_projection = (error_px * (safe_z / focal) ** 2).mean()
                loss = loss + projection_weight * loss_projection

            loss_rotation = (drot ** 2).mean()
            loss_translation = (dtrans ** 2).mean()
            loss = (
                loss
                + rotation_reg_weight * loss_rotation
                + translation_reg_weight * loss_translation
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                angle = float(drot.norm())
                if angle > max_rot_rad:
                    drot.mul_(max_rot_rad / angle)

            if person_np is not None and step % 10 == 0:
                import scipy.spatial

                with torch.no_grad():
                    tree = scipy.spatial.cKDTree(person_np.astype(np.float64))
                    _, nearest_index = tree.query(
                        cam[sample_idx].numpy().astype(np.float64)
                    )
                    nearest_t = torch.as_tensor(person_np[nearest_index], device=device)

            if step in (0, iterations - 1):
                metrics = {
                    "contact_m2": round(float(loss_contact.detach()), 8),
                    "penetration_m2": round(float(loss_penetration.detach()), 8),
                    "chamfer_m2": round(float(loss_chamfer.detach()), 8),
                    "projection_m2": round(float(loss_projection.detach()), 8),
                    "rotation_reg": round(float(loss_rotation.detach()), 8),
                    "translation_reg": round(float(loss_translation.detach()), 8),
                    "total": round(float(loss.detach()), 8),
                }

        with torch.no_grad():
            cam = rigid(base_centered) + pivot
            kp_now = (rigid(kp_cam - pivot) + pivot) if kp_cam is not None else None
            z = (cam @ world_z) - plane_d
            z_final = z.numpy()
            final_vertices = cam.numpy().astype(np.float32)
            final_keypoints = kp_now.numpy().astype(np.float32) if kp_now is not None else None
            shift = float(torch.linalg.norm(cam - base, dim=1).max())
            delta_rot = float(torch.rad2deg(drot.detach().norm()))
            delta_trans = ((dtrans.detach() @ world_z) * world_z).numpy()
            penetration_after = float((z < -0.01).float().mean())
            penetration_before = float((z_base < -0.01).float().mean())
            contact_before = float(z_np[low_idx].mean())
            contact_after = float(z_final[low_idx].mean())

        report.update(
            {
                "iterations": iterations,
                "delta_rot_deg": round(delta_rot, 3),
                "delta_trans_m": np.round(delta_trans, 4).astype(float).tolist(),
                "rotation_limit_deg": max_rotation_deg,
                "contact_target_m": round(contact_target, 4),
                "contact_target_source": contact_target_source,
                "contact_height_before_m": round(contact_before, 4),
                "contact_height_after_m": round(contact_after, 4),
                "penetration_ratio_before": round(penetration_before, 4),
                "penetration_ratio_after": round(penetration_after, 4),
                "max_vertex_shift_m": round(shift, 4),
                "losses": metrics,
            }
        )
        if shift > max_vertex_shift_m:
            report["status"] = "rejected"
            report["reason"] = f"顶点位移 {shift:.3f} m 超过上限 {max_vertex_shift_m:.2f} m"
            return {"report": report}
        report["status"] = "applied"
        return {"vertices": final_vertices, "keypoints": final_keypoints, "report": report}
