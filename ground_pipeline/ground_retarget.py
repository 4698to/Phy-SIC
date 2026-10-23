"""地面世界系姿势适配层（P0）。

职责：
- 解析 / 校验请求级 ``target_tpose_bundle`` 并构造 profile（复用 ImagePose_Resources）；
- 内置默认 Biped 目标（客户端不传 bundle 也能得到 ``retarget``）；
- ``generic_pose`` -> ``bip22_pose_v5``（v3 / v4）；
- Character Studio 载荷与 COCO-17 关键点派生；
- ``figure_v6`` -> ``ground_world_v1`` 的刚体换基（位置 + 旋转）。

坐标系（升级计划第 7 节）：
- 相机系：x 右 / y 下 / z 前，地面 ``n·p + d = 0``；
- ``figure_v6``：``p_fig = P·p_cam``，``P = [[-1,0,0],[0,0,1],[0,-1,0]]``，``det(P) = -1``；
- ``ground_world_v1``：``p_world = R·p_cam + (0,0,-d)``，``R = rotate_to_up(n)``，地面 ``z = 0``。

本模块不得反向依赖 ``ground_service``（服务端启动时把 ``rotate_to_up`` 注入）。
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FRAME_FIGURE = "figure_v6"
FRAME_WORLD = "ground_world_v1"
FRAME_CAMERA = "camera_x_right_y_down_z_forward"
UNIT_METER = "meter"

BUNDLE_SCHEMA = "target_tpose_bundle_v1"

# p_fig = P @ p_cam（与 ai_vitpose.core.generic_pose._FIGURE_P 一致）
FIGURE_P = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
# bundle（y_up_right_handed，厘米）-> figure_v6
SOLVER_TO_FIGURE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
FIGURE_TO_SOLVER = SOLVER_TO_FIGURE.T

COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# 与 ai_vitpose.core.sam3d_adapter.MHR70_FALLBACK 逐项一致
MHR70_FALLBACK: Dict[str, int] = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3,
    "right_ear": 4, "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8, "left_wrist": 62,
    "right_wrist": 41, "left_hip": 9, "right_hip": 10,
    "left_knee": 11, "right_knee": 12, "left_ankle": 13,
    "right_ankle": 14,
}
MHR70_COCO17 = tuple(MHR70_FALLBACK[name] for name in COCO17)

DEFAULT_KEYPOINT_SCORE = 0.92

# 默认 Biped 目标（assets/biped_rest_profile.v1.json，66 骨静态参考）
DEFAULT_HIPS_BONE = "Bip001 Pelvis"
DEFAULT_HEAD_BONE = "Bip001 Head"
DEFAULT_SPINE_SLOT = (
    "Bip001 Spine", "Bip001 Spine1", "Bip001 Spine2", "Bip001 Spine3",
)
DEFAULT_NECK_SLOT = ("Bip001 Neck",)
DEFAULT_SLOTS: Dict[str, Any] = {
    "hips": DEFAULT_HIPS_BONE,
    "head": DEFAULT_HEAD_BONE,
    "leftShoulder": "Bip001 L Clavicle",
    "rightShoulder": "Bip001 R Clavicle",
    "leftUpperArm": "Bip001 L UpperArm",
    "leftLowerArm": "Bip001 L Forearm",
    "leftHand": "Bip001 L Hand",
    "rightUpperArm": "Bip001 R UpperArm",
    "rightLowerArm": "Bip001 R Forearm",
    "rightHand": "Bip001 R Hand",
    "leftUpperLeg": "Bip001 L Thigh",
    "leftLowerLeg": "Bip001 L Calf",
    "leftFoot": "Bip001 L Foot",
    "rightUpperLeg": "Bip001 R Thigh",
    "rightLowerLeg": "Bip001 R Calf",
    "rightFoot": "Bip001 R Foot",
    "spine": list(DEFAULT_SPINE_SLOT),
    "neck": list(DEFAULT_NECK_SLOT),
}
DEFAULT_CHAINS: Dict[str, List[str]] = {
    "spine": [
        "Bip001 Pelvis", "Bip001 Spine", "Bip001 Spine1", "Bip001 Spine2",
        "Bip001 Spine3", "Bip001 Neck", "Bip001 Head",
    ],
    "arm_l": ["Bip001 L Clavicle", "Bip001 L UpperArm", "Bip001 L Forearm", "Bip001 L Hand"],
    "arm_r": ["Bip001 R Clavicle", "Bip001 R UpperArm", "Bip001 R Forearm", "Bip001 R Hand"],
    "leg_l": ["Bip001 L Thigh", "Bip001 L Calf", "Bip001 L Foot", "Bip001 L Toe0"],
    "leg_r": ["Bip001 R Thigh", "Bip001 R Calf", "Bip001 R Foot", "Bip001 R Toe0"],
}


class RetargetError(RuntimeError):
    """可映射到 HTTP 状态码的适配层异常。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        stage: str = "retarget",
        target_status: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.stage = str(stage)
        self.target_status = str(target_status)


# ---------------------------------------------------------------------------
# 依赖加载
# ---------------------------------------------------------------------------


def ensure_resource_path(resources: Optional[str | Path]) -> Optional[Path]:
    """把资源根写入环境变量与 ``sys.path``（幂等）。"""
    if not resources:
        return None
    root = Path(resources).expanduser().resolve()
    os.environ["POSEEXTRACTOR_RESOURCES"] = str(root)
    os.environ.setdefault("POSEEXTRACTOR_CS_ASSETS", str(root / "cs_assets"))
    os.environ.setdefault("POSEEXTRACTOR_ASSETS", str(root / "assets"))
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


_CORE: Optional[Dict[str, Any]] = None


def _load_core() -> Dict[str, Any]:
    """导入 ImagePose_Resources 的参考实现（失败只影响 /v1/pose/*）。"""
    global _CORE
    if _CORE is not None:
        return _CORE
    from ai_vitpose.core import (  # noqa: WPS433 - 延迟导入，便于 503 降级
        cs_response,
        generic_pose,
        normalize,
        overlay,
        response_serialize,
        target_bundle,
        target_profile,
        target_retarget,
    )

    _CORE = {
        "cs_response": cs_response,
        "generic_pose": generic_pose,
        "normalize": normalize,
        "overlay": overlay,
        "response_serialize": response_serialize,
        "target_bundle": target_bundle,
        "target_profile": target_profile,
        "target_retarget": target_retarget,
    }
    return _CORE


# ---------------------------------------------------------------------------
# 默认 Biped bundle
# ---------------------------------------------------------------------------


def _mat_to_quat(matrix) -> np.ndarray:
    """3x3 旋转 -> (x, y, z, w)，w >= 0。"""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RetargetError("默认 Biped 骨骼旋转非法", status_code=503, stage="retarget_assets")
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat


def _referenced_bones() -> List[str]:
    names: List[str] = []
    for value in DEFAULT_SLOTS.values():
        names.extend(value if isinstance(value, list) else [value])
    for chain in DEFAULT_CHAINS.values():
        names.extend(chain)
    return sorted(set(names))


def default_biped_bundle(assets_root: Optional[Path] = None) -> Dict[str, Any]:
    """由 ``biped_rest_profile.v1.json`` 合成 ``target_tpose_bundle_v1``。

    不含 ``fingerprint``；调用方用 ``target_bundle.compute_fingerprint`` 补齐。
    """
    root = Path(assets_root) if assets_root else Path(os.environ.get("POSEEXTRACTOR_ASSETS") or ".")
    path = root / "biped_rest_profile.v1.json"
    if not path.is_file():
        raise RetargetError(
            f"默认 Biped 资产不存在: {path}",
            status_code=503,
            stage="retarget_assets",
        )
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            rest = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RetargetError(
            f"默认 Biped 资产不可读: {path}: {exc}",
            status_code=503,
            stage="retarget_assets",
        ) from exc

    source_bones = rest.get("bones")
    matrix = rest.get("object_matrix_world")
    if not isinstance(source_bones, dict) or not source_bones:
        raise RetargetError(
            f"默认 Biped 资产缺少 bones: {path}",
            status_code=503,
            stage="retarget_assets",
        )
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise RetargetError(
            f"默认 Biped 资产缺少 object_matrix_world: {path}",
            status_code=503,
            stage="retarget_assets",
        )
    world_matrix = np.asarray(matrix, dtype=np.float64)

    bones: Dict[str, Any] = {}
    for name, row in source_bones.items():
        if not isinstance(row, dict):
            continue
        head_local = np.asarray(row.get("head_local"), dtype=np.float64).reshape(3)
        head_figure = (world_matrix @ np.array([*head_local, 1.0]))[:3]
        head_solver = 100.0 * (FIGURE_TO_SOLVER @ head_figure)
        rest_rotation = np.asarray(row.get("rest_world_figure"), dtype=np.float64).reshape(3, 3)
        quat = _mat_to_quat(FIGURE_TO_SOLVER @ rest_rotation @ SOLVER_TO_FIGURE)
        parent = row.get("parent") or None
        bones[name] = {
            "parent": parent if isinstance(parent, str) else None,
            "head_world": [round(float(v), 6) for v in head_solver],
            "world_quat": [round(float(v), 6) for v in quat],
            "length": round(float(row.get("length") or 0.0), 6),
        }

    missing = [name for name in _referenced_bones() if name not in bones]
    if missing:
        raise RetargetError(
            "默认 Biped 骨骼缺失: " + "、".join(missing),
            status_code=503,
            stage="retarget_assets",
        )
    hips = bones[DEFAULT_HIPS_BONE]
    return {
        "schema_version": BUNDLE_SCHEMA,
        "skeleton_id": "mhr_biped_standard",
        "rest_id": str(rest.get("source_sha256") or "biped_rest_profile_v1")[:64],
        "unit": "centimeter",
        "coordinate_system": "y_up_right_handed",
        "source": "ground_service_default_biped",
        "created_at": str(rest.get("blender") or ""),
        "object": {"name": "Bip001"},
        "tpose": {
            "mode": "figure",
            "frame": 0.0,
            "hips": {
                "name": DEFAULT_HIPS_BONE,
                "world_pos": list(hips["head_world"]),
                "world_quat": list(hips["world_quat"]),
            },
            "bones": bones,
        },
        "slots": {key: (list(value) if isinstance(value, list) else value) for key, value in DEFAULT_SLOTS.items()},
        "chains": {key: list(value) for key, value in DEFAULT_CHAINS.items()},
        "diagnostics": {"required_slots_ok": True},
    }


# ---------------------------------------------------------------------------
# 世界系变换
# ---------------------------------------------------------------------------


def _rotate_to_up(normal, up=(0.0, 0.0, -1.0)) -> np.ndarray:
    """与 ``ground_service.rotate_to_up`` 同式（注入优先，供独立单测使用）。"""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    target = np.asarray(up, dtype=np.float64)
    target = target / np.linalg.norm(target)
    v = np.cross(n, target)
    s = float(np.linalg.norm(v))
    c = float(np.dot(n, target))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s ** 2)


class GroundTransform:
    """相机 -> 地面世界系刚体变换 ``p_world = p_cam @ rotation.T + translation``。"""

    def __init__(
        self,
        rotation,
        translation,
        *,
        source: str,
        plane_normal_camera,
    ) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)
        self.source = str(source)
        self.plane_normal_camera = np.asarray(plane_normal_camera, dtype=np.float64).reshape(3)

    @property
    def plane_d_camera(self) -> float:
        return -float(self.translation[2])

    @property
    def camera_height_m(self) -> float:
        return abs(self.plane_d_camera)

    def to_world(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return p @ self.rotation.T + self.translation

    def as_dict(self) -> Dict[str, Any]:
        normal_world = self.rotation @ self.plane_normal_camera
        return {
            "frame": FRAME_WORLD,
            "unit": UNIT_METER,
            "source": self.source,
            "camera_frame": FRAME_CAMERA,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "plane_normal_camera": self.plane_normal_camera.tolist(),
            "plane_d_camera": round(self.plane_d_camera, 9),
            "ground_normal_world": [round(float(v), 9) for v in normal_world],
            # 地面法向指向地下，世界 +Z 才是朝上方向（与 /v1/ground 的 world 分组一致）
            "up_axis_world": [round(float(v), 9) for v in (self.rotation @ (-self.plane_normal_camera))],
            "camera_height_m": round(self.camera_height_m, 9),
            "camera_position": self.translation.tolist(),
        }

    @classmethod
    def from_plane(
        cls,
        normal,
        plane_d: float,
        *,
        source: str = "request_plane",
        rotate: Optional[Callable[..., np.ndarray]] = None,
    ) -> "GroundTransform":
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if not math.isfinite(norm) or norm < 1e-9:
            raise RetargetError("ground.plane_normal 必须是有效单位向量", status_code=400, stage="world")
        n = n / norm
        if not math.isfinite(float(plane_d)):
            raise RetargetError("ground.plane_d 必须是有限数值", status_code=400, stage="world")
        rotation = (rotate or _rotate_to_up)(n)
        return cls(
            rotation,
            np.array([0.0, 0.0, -float(plane_d)], dtype=np.float64),
            source=source,
            plane_normal_camera=n,
        )

    @classmethod
    def from_rotation(
        cls,
        rotation,
        translation,
        *,
        source: str,
    ) -> "GroundTransform":
        r = np.asarray(rotation, dtype=np.float64)
        t = np.asarray(translation, dtype=np.float64)
        if r.shape != (3, 3) or t.shape != (3,):
            raise RetargetError(
                "ground.rotation 必须是 3x3、ground.translation 必须是长度 3",
                status_code=400,
                stage="world",
            )
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            raise RetargetError("ground.rotation/translation 必须是有限数值", status_code=400, stage="world")
        det = float(np.linalg.det(r))
        if abs(abs(det) - 1.0) > 1e-3:
            raise RetargetError(f"ground.rotation 不是旋转矩阵 (det={det:.6f})", status_code=400, stage="world")
        return cls(
            r,
            t,
            source=source,
            plane_normal_camera=r.T @ np.array([0.0, 0.0, -1.0]),
        )


def parse_ground_request(
    ground: Any,
    *,
    rotate: Optional[Callable[..., np.ndarray]] = None,
) -> GroundTransform:
    """解析请求 ``ground``：``{plane_normal, plane_d}`` 或 ``{rotation, translation}``。"""
    if not isinstance(ground, dict):
        raise RetargetError("ground 必须是对象", status_code=400, stage="world")
    if ground.get("rotation") is not None or ground.get("translation") is not None:
        if ground.get("rotation") is None or ground.get("translation") is None:
            raise RetargetError(
                "ground.rotation 与 ground.translation 必须同时提供",
                status_code=400,
                stage="world",
            )
        return GroundTransform.from_rotation(
            ground["rotation"], ground["translation"], source="request_transform"
        )
    if ground.get("plane_normal") is None or ground.get("plane_d") is None:
        raise RetargetError(
            "ground 需提供 plane_normal + plane_d 或 rotation + translation",
            status_code=400,
            stage="world",
        )
    try:
        plane_d = float(ground["plane_d"])
    except (TypeError, ValueError) as exc:
        raise RetargetError(f"ground.plane_d 非法: {exc}", status_code=400, stage="world") from exc
    return GroundTransform.from_plane(ground["plane_normal"], plane_d, rotate=rotate)


def skipped_world(status: str, *, ground_mode: str, error: str = "", transform: Optional[GroundTransform] = None) -> Dict[str, Any]:
    """``skipped`` / ``unavailable`` / ``error`` 的最小 world 块。"""
    block: Dict[str, Any] = {
        "status": str(status),
        "frame": FRAME_WORLD,
        "unit": UNIT_METER,
        "source": None,
        "ground_mode": str(ground_mode),
        "camera_frame": FRAME_CAMERA,
        "ground_normal_world": None,
        "up_axis_world": None,
        "ground_snap": {"status": "skipped", "source": None, "offset_m": 0.0},
        "min_z_m": None,
        "error": str(error or ""),
    }
    if transform is not None:
        block.update(transform.as_dict())
        block["status"] = str(status)
        block["ground_mode"] = str(ground_mode)
        block["error"] = str(error or "")
    return block


def _conjugate(matrix, q: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return q @ m @ q.T


def _transform_retarget(payload: Dict[str, Any], q: np.ndarray, offset: np.ndarray) -> Dict[str, Any]:
    """``bip22_pose_v5`` 位置/旋转换基到世界系（global 与 local 都必须共轭）。"""
    out: Dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key not in ("root", "bones", "frame", "position_space")
    }
    out["frame"] = FRAME_WORLD
    out["position_space"] = FRAME_WORLD

    root = payload.get("root")
    if isinstance(root, dict):
        row = dict(root)
        position = row.get("position")
        if position is not None:
            row["position"] = (q @ np.asarray(position, dtype=np.float64) + offset).tolist()
        out["root"] = row

    bones: Dict[str, Any] = {}
    for name, row in (payload.get("bones") or {}).items():
        if not isinstance(row, dict):
            continue
        item = dict(row)
        for key in ("global_rotation", "local_rotation"):
            matrix = row.get(key)
            if matrix is not None:
                item[key] = _conjugate(matrix, q).tolist()
        position = row.get("position")
        if position is not None:
            item["position"] = (q @ np.asarray(position, dtype=np.float64) + offset).tolist()
        bones[name] = item
    out["bones"] = bones
    return out


def _transform_cs_block(
    cs_block: Optional[Dict[str, Any]],
    *,
    q: np.ndarray,
    offset: np.ndarray,
    bip22_world,
) -> Optional[Dict[str, Any]]:
    """``cs`` 载荷（Figure 系，厘米）-> ``ground_world_v1``（厘米）。

    三处来源不同：

    - ``pose_payload.data[0]`` 前 22 行与 ``bip22_payload.data[0]`` 是 L5 对齐后的
      bip22 骨架，直接用世界系 ``world.bip22_joints``（米）覆盖；``pose_payload``
      第 22 行起的躯干派生点、``points_53`` 与相机中心是 figure 绝对坐标，做
      换基 + ``offset``；
    - 方向类字段（头轴、足法线、胸向右、手指轴、``bip22_facing``）只换基不平移；
    - 相机中心在 ``data[6][0]`` 里是 figure **米**，其余槽位不受影响。
    """
    if not isinstance(cs_block, dict):
        return cs_block
    offset_cm = offset * 100.0
    world_positions = (
        np.asarray(bip22_world, dtype=np.float64).reshape(-1, 3)
        if bip22_world is not None and len(bip22_world)
        else None
    )
    world_rows = None if world_positions is None else world_positions * 100.0

    def _point(row) -> List[float]:
        return (q @ np.asarray(row, dtype=np.float64).reshape(3) + offset_cm).tolist()

    def _direction(row) -> List[float]:
        return (q @ np.asarray(row, dtype=np.float64).reshape(3)).tolist()

    def _directions(rows) -> List[Any]:
        return [None if row is None else _direction(row) for row in (rows or [])]

    def _points53_tsv(tsv: Any) -> Any:
        """``points_53_tsv``（figure 系厘米）逐行改写为世界系厘米。

        ``root`` 是骨盆 xy 在 z=0 地面上的投影，世界系同样取 ``world_rows[0]``
        的 xy、z 置 0；缺失点（xyz 全 0）保持原样。
        """
        if not isinstance(tsv, str) or not tsv.strip():
            return tsv
        lines = [line for line in tsv.replace("\r\n", "\n").split("\n") if line.strip()]
        if not lines:
            return tsv
        headers = lines[0].split("\t")
        try:
            key_col = headers.index("source_key")
            vec_cols = (headers.index("x"), headers.index("y"), headers.index("z"))
        except ValueError:
            return tsv
        rewritten = [lines[0]]
        for line in lines[1:]:
            cols = line.split("\t")
            if key_col >= len(cols) or max(vec_cols) >= len(cols):
                rewritten.append(line)
                continue
            try:
                xyz = [float(cols[i]) for i in vec_cols]
            except ValueError:
                rewritten.append(line)
                continue
            if not any(xyz):
                rewritten.append(line)
                continue
            if cols[key_col].strip().lower() == "root" and world_rows is not None:
                xyz = [world_rows[0][0], world_rows[0][1], 0.0]
            else:
                xyz = _point(xyz)
            for col, value in zip(vec_cols, xyz):
                cols[col] = f"{float(value):.6f}"
            rewritten.append("\t".join(cols))
        return "\n".join(rewritten) + "\n"

    def _skeleton(rows) -> List[Any]:
        """L5 对齐的骨架：前 22 行用世界系值，超出部分按绝对坐标换基。"""
        out: List[Any] = []
        for index, row in enumerate(rows or []):
            if row is None:
                out.append(None)
            elif world_rows is not None and index < len(world_rows):
                out.append(world_rows[index].tolist())
            else:
                out.append(_point(row))
        return out

    def _pose_payload(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return payload
        rows = list(data)
        if isinstance(rows[0], list):
            rows[0] = _skeleton(rows[0])
        if len(rows) > 1:
            rows[1] = _directions(rows[1])
        if len(rows) > 2:
            rows[2] = _directions(rows[2])
        if len(rows) > 3 and rows[3] is not None:
            rows[3] = _direction(rows[3])
        if len(rows) > 5 and isinstance(rows[5], list):
            rows[5] = [
                None if hand is None else [_direction(axis) for axis in hand]
                for hand in rows[5]
            ]
        if len(rows) > 6 and isinstance(rows[6], list) and rows[6] and rows[6][0] is not None:
            camera = list(rows[6])
            center = np.asarray(camera[0], dtype=np.float64).reshape(3)
            camera[0] = ((q @ center + offset) * 100.0).tolist()
            rows[6] = camera
        updated = dict(payload)
        updated["data"] = rows
        updated["frame"] = FRAME_WORLD
        updated["unit"] = "centimeter"
        return updated

    def _bip22_payload(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return payload
        rows = list(data)
        positions = rows[0]
        if isinstance(positions, list) and positions:
            if world_rows is not None:
                rows[0] = world_rows[: len(positions)].tolist()
            else:
                rows[0] = [_point(row) for row in positions]
        updated = dict(payload)
        updated["data"] = rows
        updated["frame"] = FRAME_WORLD
        updated["unit"] = "centimeter"
        return updated

    out = dict(cs_block)
    out["frame"] = FRAME_WORLD
    out["unit"] = "centimeter"
    pose_payload = _pose_payload(cs_block.get("pose_payload"))
    out["pose_payload"] = pose_payload
    if isinstance(pose_payload, dict):
        out["pose_data"] = pose_payload.get("data")
    bip22_payload = _bip22_payload(cs_block.get("bip22_payload"))
    out["bip22_payload"] = bip22_payload
    if isinstance(bip22_payload, dict):
        out["bip22_pose_data"] = bip22_payload.get("data")
    if world_positions is not None:
        out["bip22_joints"] = world_positions.tolist()
    facing = cs_block.get("bip22_facing")
    if isinstance(facing, (list, tuple)) and len(facing) == 3:
        out["bip22_facing"] = _direction(facing)

    if "points_53_tsv" in out:
        out["points_53_tsv"] = _points53_tsv(out["points_53_tsv"])
    points = out.get("points_53")
    if isinstance(points, list):
        converted: List[Any] = []
        for point in points:
            if not isinstance(point, dict):
                converted.append(point)
                continue
            item = dict(point)
            x, y, z = point.get("x"), point.get("y"), point.get("z")
            missing = not all(isinstance(v, (int, float)) for v in (x, y, z)) or not (
                x or y or z
            )
            if missing:
                converted.append(item)
                continue
            if str(point.get("source_key") or "").strip().lower() == "root" and world_rows is not None:
                item["x"], item["y"], item["z"] = (
                    float(world_rows[0][0]),
                    float(world_rows[0][1]),
                    0.0,
                )
            else:
                item["x"], item["y"], item["z"] = _point([x, y, z])
            converted.append(item)
        out["points_53"] = converted
    return out


def apply_world_frames(
    body: Dict[str, Any],
    world: Optional[Dict[str, Any]],
    transform: Optional[GroundTransform],
    pred_cam_t=None,
) -> None:
    """把返回给客户端的骨骼字段统一替换为 ``ground_world_v1`` 数值（原地修改）。

    ``world`` 不可用时保持 figure 系原值，``joints_3d_frame`` 等标签仍如实描述。
    """
    if not isinstance(world, dict) or world.get("status") != "ok" or transform is None:
        return
    cam_t = (
        np.zeros(3, dtype=np.float64)
        if pred_cam_t is None
        else np.asarray(pred_cam_t, dtype=np.float64).reshape(3)
    )
    q = transform.rotation @ FIGURE_P.T
    offset = transform.rotation @ cam_t + transform.translation

    joints = world.get("joints_3d")
    if isinstance(joints, list) and joints:
        body["joints_3d"] = joints
        body["joints_3d_frame"] = FRAME_WORLD
    keypoints = world.get("keypoints_3d_coco17")
    if isinstance(keypoints, list) and keypoints:
        body["keypoints_3d_coco17"] = keypoints
        body["keypoints_3d_coco17_frame"] = FRAME_WORLD
    if isinstance(world.get("retarget"), dict):
        body["retarget"] = world["retarget"]
    if isinstance(world.get("generic_pose"), dict):
        body["generic_pose"] = world["generic_pose"]

    cs = _transform_cs_block(
        body.get("cs"),
        q=q,
        offset=offset,
        bip22_world=world.get("bip22_joints"),
    )
    if isinstance(cs, dict):
        body["cs"] = cs
        if isinstance(cs.get("bip22_joints"), list):
            body["bip22_joints"] = cs["bip22_joints"]
        if cs.get("bip22_facing") is not None:
            body["bip22_facing"] = cs["bip22_facing"]


_BIP22_SMPL_SOURCE = re.compile(r"^SMPL\[(\d+)\]$")


def _bip22_anchor(joints_fig, bip22_joints, sources) -> Optional[np.ndarray]:
    """恢复 ``cs`` 组装时施加的 figure 系平移。

    ``bip22_assembler`` 的 L5 会把脚底对齐到 z=0（整体沿 z 平移），
    但其 ``bip22_source`` 仍标注每根骨来自哪个 SMPL24 关节；用这些同名关节
    的差值取中位数即可反算平移，使 ``bip22_joints`` 回到 figure 系绝对坐标。
    """
    if bip22_joints is None or not sources:
        return None
    joints_fig = np.asarray(joints_fig, dtype=np.float64)
    bip22_joints = np.asarray(bip22_joints, dtype=np.float64)
    deltas: List[np.ndarray] = []
    for row, source in enumerate(sources):
        match = _BIP22_SMPL_SOURCE.match(str(source or ""))
        if match is None or row >= len(bip22_joints):
            continue
        index = int(match.group(1))
        if index >= len(joints_fig):
            continue
        deltas.append(joints_fig[index] - bip22_joints[row])
    if len(deltas) < 3:
        return None
    return np.median(np.stack(deltas), axis=0)


def alignment_rotation(transform: GroundTransform) -> Tuple[np.ndarray, float]:
    """Return the proper rotation that maps figure-space ground-up onto +Z."""
    up = FIGURE_P @ np.asarray(transform.rotation, dtype=np.float64).T @ np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)
    cosine = float(np.clip(up[2], -1.0, 1.0))
    angle = math.acos(cosine)
    if cosine < -0.999999:
        raise ValueError("ground up is nearly opposite to figure +Z")
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64), 0.0
    axis = np.cross(up, np.array([0.0, 0.0, 1.0]))
    axis /= np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return rotation, math.degrees(angle)


def apply_gravity_alignment(
    body: Dict[str, Any],
    transform: GroundTransform,
    *,
    pred_cam_t=None,
    mesh_vertices=None,
) -> Dict[str, Any]:
    """Align all consumer pose values to figure +Z, then place the selected geometry on z=0."""
    from ai_vitpose.core.generic_pose import _euler_zyx_to_rotmat, _rotmat_to_euler_zyx

    original_body = body
    body = copy.deepcopy(body)
    rotation, angle_deg = alignment_rotation(transform)
    joints = np.asarray(body.get("joints_3d", []), dtype=np.float64).reshape(-1, 3)
    if not len(joints):
        raise ValueError("pose has no joints for gravity alignment pivot")
    pivot = joints[0].copy()

    def align_points(points, origin, scale=1.0):
        arr = np.asarray(points, dtype=np.float64)
        return ((arr - origin) @ rotation.T + origin).reshape(arr.shape) * scale

    joints = align_points(joints, pivot)
    body["joints_3d"] = joints.tolist()
    body["joints_3d_frame"] = FRAME_FIGURE

    kp = body.get("keypoints_3d_coco17")
    if kp is not None:
        kp_figure = np.asarray(kp, dtype=np.float64).reshape(-1, 3) @ FIGURE_P.T
        body["keypoints_3d_coco17"] = align_points(kp_figure, pivot).tolist()
        body["keypoints_3d_coco17_frame"] = FRAME_FIGURE

    generic = body.get("generic_pose")
    if isinstance(generic, dict):
        generic = dict(generic)
        generic_points = np.asarray(generic.get("pred_joint_coords", [pivot]), dtype=np.float64).reshape(-1, 3)
        generic_pivot = generic_points[1].copy() if len(generic_points) > 1 else pivot
        for key in ("pred_joint_coords", "pred_keypoints_3d"):
            if generic.get(key) is not None:
                generic[key] = align_points(generic[key], generic_pivot).tolist()
        if generic.get("pred_global_rots") is not None:
            generic["pred_global_rots"] = np.stack(
                [rotation @ np.asarray(m, dtype=np.float64).reshape(3, 3)
                 for m in generic["pred_global_rots"]]
            ).tolist()
        if generic.get("global_rot") is not None:
            old = _euler_zyx_to_rotmat(np.asarray(generic["global_rot"], dtype=np.float64).reshape(3))
            generic["global_rot"] = _rotmat_to_euler_zyx(rotation @ old).tolist()
        generic["frame"] = FRAME_FIGURE
        body["generic_pose"] = generic

    retarget = body.get("retarget")
    if isinstance(retarget, dict):
        retarget = dict(retarget)
        bones = {}
        for name, row in (retarget.get("bones") or {}).items():
            if not isinstance(row, dict):
                bones[name] = row
                continue
            item = dict(row)
            if item.get("global_rotation") is not None:
                item["global_rotation"] = (rotation @ np.asarray(item["global_rotation"], dtype=np.float64).reshape(3, 3)).tolist()
            if item.get("position") is not None:
                item["position"] = (rotation @ (np.asarray(item["position"], dtype=np.float64).reshape(3) - pivot) + pivot).tolist()
            bones[name] = item
        retarget["bones"] = bones
        root = retarget.get("root")
        if isinstance(root, dict):
            root = dict(root)
            if root.get("position") is not None:
                root["position"] = (rotation @ (np.asarray(root["position"], dtype=np.float64).reshape(3) - pivot) + pivot).tolist()
            root_bone = root.get("bone")
            if root_bone in bones and isinstance(bones[root_bone], dict) and bones[root_bone].get("position") is not None:
                root["position"] = copy.deepcopy(bones[root_bone]["position"])
            retarget["root"] = root
        retarget["frame"] = FRAME_FIGURE
        retarget["position_space"] = "figure_v6_world"
        body["retarget"] = retarget

    cs = body.get("cs")
    cs = dict(cs) if isinstance(cs, dict) else None
    if cs is not None:
        pelvis_cm = np.asarray(cs.get("bip22_joints", [])[1], dtype=np.float64) if len(cs.get("bip22_joints", [])) > 1 else pivot * 100.0

        def cs_point(point):
            return (rotation @ (np.asarray(point, dtype=np.float64).reshape(3) / 100.0 - pelvis_cm / 100.0) + pelvis_cm / 100.0) * 100.0

        def cs_direction(vector):
            return (rotation @ np.asarray(vector, dtype=np.float64).reshape(3)).tolist()

        bip = cs.get("bip22_joints")
        aligned_bip = None
        if isinstance(bip, list) and bip:
            aligned_bip = np.asarray(bip, dtype=np.float64).reshape(-1, 3)
            aligned_bip = (aligned_bip / 100.0 - pelvis_cm / 100.0) @ rotation.T * 100.0 + pelvis_cm
            if len(aligned_bip) > 1:
                aligned_bip[0] = [aligned_bip[1, 0], aligned_bip[1, 1], 0.0]
            cs["bip22_joints"] = aligned_bip.tolist()
        pelvis_xy_cm = aligned_bip[1, :2].copy() if aligned_bip is not None and len(aligned_bip) > 1 else None
        for payload_key in ("pose_payload", "bip22_payload"):
            payload = cs.get(payload_key)
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or not payload["data"]:
                continue
            payload = dict(payload)
            rows = list(payload["data"])
            if isinstance(rows[0], list):
                values = np.asarray(rows[0], dtype=np.float64).reshape(-1, 3)
                values = (values / 100.0 - pelvis_cm / 100.0) @ rotation.T * 100.0 + pelvis_cm
                if len(values) > 1:
                    values[0] = [values[1, 0], values[1, 1], 0.0]
                rows[0] = values.tolist()
            if payload_key == "pose_payload":
                for row_index in (1, 2):
                    if row_index < len(rows) and isinstance(rows[row_index], list):
                        rows[row_index] = [None if v is None else cs_direction(v) for v in rows[row_index]]
                for row_index in (3,):
                    if row_index < len(rows) and rows[row_index] is not None:
                        rows[row_index] = cs_direction(rows[row_index])
                if len(rows) > 5 and isinstance(rows[5], list):
                    rows[5] = [None if hand is None else [cs_direction(v) for v in hand] for hand in rows[5]]
                if len(rows) > 6 and isinstance(rows[6], list) and rows[6] and rows[6][0] is not None:
                    center_m = np.asarray(rows[6][0], dtype=np.float64)
                    center = (rotation @ (center_m - pivot) + pivot).tolist()
                    rows[6] = [center, *rows[6][1:]]
            payload["data"] = rows
            payload["frame"] = FRAME_FIGURE
            cs[payload_key] = payload
        if isinstance(cs.get("pose_payload"), dict):
            cs["pose_data"] = cs["pose_payload"].get("data")
        if isinstance(cs.get("bip22_payload"), dict):
            cs["bip22_pose_data"] = cs["bip22_payload"].get("data")
        if isinstance(cs.get("bip22_facing"), (list, tuple)):
            cs["bip22_facing"] = cs_direction(cs["bip22_facing"])
        if isinstance(cs.get("points_53_tsv"), str):
            lines = cs["points_53_tsv"].splitlines()
            if lines:
                headers = lines[0].split("\t")
                try:
                    key_i, xyz_i = headers.index("source_key"), [headers.index(k) for k in ("x", "y", "z")]
                    for i in range(1, len(lines)):
                        cols = lines[i].split("\t")
                        if max(xyz_i) < len(cols):
                            vals = np.array([float(cols[j]) for j in xyz_i])
                            if np.any(vals) or cols[key_i].strip().lower() == "root":
                                aligned = cs_point(vals)
                                if cols[key_i].strip().lower() == "root":
                                    if pelvis_xy_cm is not None:
                                        aligned[:2] = pelvis_xy_cm
                                    aligned[2] = 0.0
                                for j, val in zip(xyz_i, aligned): cols[j] = f"{val:.6f}"
                                lines[i] = "\t".join(cols)
                    cs["points_53_tsv"] = "\n".join(lines) + "\n"
                except (ValueError, IndexError):
                    pass
        points = cs.get("points_53")
        if isinstance(points, list):
            aligned_points = []
            for point in points:
                if not isinstance(point, dict):
                    aligned_points.append(point)
                    continue
                item = dict(point)
                xyz = [item.get(key) for key in ("x", "y", "z")]
                if all(isinstance(value, (int, float)) for value in xyz) and (
                    any(xyz) or str(item.get("source_key") or "").strip().lower() == "root"
                ):
                    aligned = cs_point(xyz)
                    if str(item.get("source_key") or "").strip().lower() == "root":
                        if pelvis_xy_cm is not None:
                            aligned[:2] = pelvis_xy_cm
                        aligned[2] = 0.0
                    item.update(zip(("x", "y", "z"), (float(value) for value in aligned)))
                aligned_points.append(item)
            cs["points_53"] = aligned_points
        cs["bip22_facing"] = cs.get("bip22_facing")
        body["cs"] = cs
        for key in ("bip22_joints", "bip22_facing", "bip22_conf", "bip22_source", "bip22_height_m"):
            if key in cs:
                body[key] = cs[key]

    points_for_floor = []
    if mesh_vertices is not None:
        cam_t = np.zeros(3, dtype=np.float64) if pred_cam_t is None else np.asarray(pred_cam_t, dtype=np.float64).reshape(3)
        mesh_fig = (np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3) + cam_t) @ FIGURE_P.T
        points_for_floor.append((mesh_fig - pivot) @ rotation.T + pivot)
        floor_source = "sam3d_mesh"
    else:
        points_for_floor.append(joints)
        if kp is not None:
            points_for_floor.append(np.asarray(body["keypoints_3d_coco17"], dtype=np.float64))
        floor_source = "joints"
    dz = -min(float(np.min(points[:, 2])) for points in points_for_floor)

    def shift_positions(points):
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        arr[:, 2] += dz
        return arr.tolist()

    body["joints_3d"] = shift_positions(body["joints_3d"])
    if body.get("keypoints_3d_coco17") is not None:
        body["keypoints_3d_coco17"] = shift_positions(body["keypoints_3d_coco17"])
    generic = body.get("generic_pose")
    if isinstance(generic, dict):
        for key in ("pred_joint_coords", "pred_keypoints_3d"):
            if generic.get(key) is not None:
                generic[key] = shift_positions(generic[key])
        body["generic_pose"] = generic
    retarget = body.get("retarget")
    if isinstance(retarget, dict):
        for row in (retarget.get("bones") or {}).values():
            if isinstance(row, dict) and row.get("position") is not None:
                row["position"][2] += dz
        if isinstance(retarget.get("root"), dict) and retarget["root"].get("position") is not None:
            retarget["root"]["position"][2] += dz
        body["retarget"] = retarget
    cs = body.get("cs")
    if isinstance(cs, dict):
        cs = dict(cs)
        if isinstance(cs.get("bip22_joints"), list):
            bip = np.asarray(cs["bip22_joints"], dtype=np.float64).reshape(-1, 3)
            bip[:, 2] += dz * 100.0
            if len(bip):
                bip[0, 2] = 0.0
            cs["bip22_joints"] = bip.tolist()
        for payload_key in ("pose_payload", "bip22_payload"):
            payload = cs.get(payload_key)
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or not payload["data"]:
                continue
            payload = dict(payload)
            rows = list(payload["data"])
            if isinstance(rows[0], list):
                arr = np.asarray(rows[0], dtype=np.float64).reshape(-1, 3)
                arr[:, 2] += dz * 100.0
                if len(arr):
                    arr[0, 2] = 0.0
                rows[0] = arr.tolist()
            if payload_key == "pose_payload" and len(rows) > 6 and isinstance(rows[6], list) and rows[6] and rows[6][0] is not None:
                center = np.asarray(rows[6][0], dtype=np.float64)
                center[2] += dz
                rows[6] = [center.tolist(), *rows[6][1:]]
            payload["data"] = rows
            cs[payload_key] = payload
        if isinstance(cs.get("pose_payload"), dict):
            cs["pose_data"] = cs["pose_payload"].get("data")
        if isinstance(cs.get("bip22_payload"), dict):
            cs["bip22_pose_data"] = cs["bip22_payload"].get("data")
        if isinstance(cs.get("points_53_tsv"), str):
            lines = cs["points_53_tsv"].splitlines()
            if lines:
                headers = lines[0].split("\t")
                try:
                    key_i, z_i = headers.index("source_key"), headers.index("z")
                    for i in range(1, len(lines)):
                        cols = lines[i].split("\t")
                        if z_i < len(cols) and cols[key_i].strip().lower() != "root" and cols[z_i].strip():
                            cols[z_i] = f"{float(cols[z_i]) + dz * 100.0:.6f}"
                            lines[i] = "\t".join(cols)
                    cs["points_53_tsv"] = "\n".join(lines) + "\n"
                except (ValueError, IndexError):
                    pass
        points = cs.get("points_53")
        if isinstance(points, list):
            shifted_points = []
            for point in points:
                if not isinstance(point, dict):
                    shifted_points.append(point)
                    continue
                item = dict(point)
                if all(isinstance(item.get(key), (int, float)) for key in ("x", "y", "z")):
                    item["z"] = 0.0 if str(item.get("source_key") or "").strip().lower() == "root" else item["z"] + dz * 100.0
                shifted_points.append(item)
            cs["points_53"] = shifted_points
        body["cs"] = cs
        for key in ("bip22_joints", "bip22_facing", "bip22_conf", "bip22_source", "bip22_height_m"):
            if key in cs:
                body[key] = cs[key]
    original_body.update(body)
    report = {"status": "applied", "angle_deg": round(angle_deg, 6), "dz_m": round(dz, 6), "source": floor_source}
    return report


def landing_shift(
    transform: GroundTransform,
    *,
    mesh_vertices=None,
    joint_coords=None,
    keypoints_3d=None,
    pred_cam_t=None,
) -> Dict[str, Any]:
    """计算落地平移：让 SAM-3D 网格最低点贴到世界地面 ``z=0``。

    返回 ``status`` / ``source`` / ``offset_m``（世界 +Z 位移量）/ ``shift_cam``
    （相机系平移向量，沿 ``-n``）。``shift_cam`` 直接加到 SAM-3D 的模型系坐标
    （``pred_joint_coords`` / ``pred_keypoints_3d`` / ``pred_vertices``）上即可完成
    落地，之后所有下游派生量天然一致，不需要再在世界系二次平移。
    """
    cam_t = (
        np.zeros(3, dtype=np.float64)
        if pred_cam_t is None
        else np.asarray(pred_cam_t, dtype=np.float64).reshape(3)
    )
    rotation = transform.rotation
    translation = transform.translation

    def _min_z(points) -> Optional[float]:
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3 or arr.size == 0:
            return None
        world = (arr[:, :3] + cam_t) @ rotation.T + translation
        value = float(np.min(world[:, 2]))
        return value if math.isfinite(value) else None

    base_z: Optional[float] = None
    source = "none"
    if mesh_vertices is not None:
        base_z = _min_z(mesh_vertices)
        source = "sam3d_mesh"
    if base_z is None:
        candidates = [z for z in (_min_z(joint_coords), _min_z(keypoints_3d)) if z is not None]
        if candidates:
            base_z = min(candidates)
            source = "joints"
    if base_z is None:
        return {
            "status": "skipped",
            "source": "none",
            "offset_m": 0.0,
            "shift_cam": np.zeros(3, dtype=np.float64),
        }

    dz = -base_z
    shift_cam = rotation.T @ np.array([0.0, 0.0, dz], dtype=np.float64)
    return {
        "status": "applied",
        "source": source,
        "offset_m": float(dz),
        "shift_cam": shift_cam,
    }


def build_world_block(
    transform: GroundTransform,
    *,
    ground_mode: str,
    joints_fig,
    keypoints_3d_cam,
    pred_cam_t,
    generic_pose: Optional[Dict[str, Any]],
    retarget_payload: Optional[Dict[str, Any]],
    bip22_joints,
    bip22_anchor=None,
    mesh_vertices=None,
    mesh_faces=None,
    include_mesh: bool = True,
    ground_snap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Figure 结果 -> ``ground_world_v1``（换基 + ``pred_cam_t`` 平移）。

    落地平移已在 SAM-3D 源数据上完成（见 :func:`landing_shift`），因此这里的输入
    已经是落地后的几何；``ground_snap`` 只用于回填落地报告。
    """
    rotation = transform.rotation
    translation = transform.translation
    q = rotation @ FIGURE_P.T  # figure -> world 线性部分，det = -1
    cam_t = np.asarray(pred_cam_t, dtype=np.float64).reshape(3)
    offset = rotation @ cam_t + translation

    joints_world = np.asarray(joints_fig, dtype=np.float64) @ q.T + offset
    kp3_world = (np.asarray(keypoints_3d_cam, dtype=np.float64) + cam_t) @ rotation.T + translation
    bip22_world = None
    if bip22_joints is not None:
        bip22_world = np.asarray(bip22_joints, dtype=np.float64) @ q.T + offset
        if bip22_anchor is not None and len(bip22_world):
            bip22_world = bip22_world + q @ np.asarray(bip22_anchor, dtype=np.float64).reshape(3)

    vertices_world = None
    if mesh_vertices is not None:
        vertices_world = (np.asarray(mesh_vertices, dtype=np.float64) + cam_t) @ rotation.T + translation

    if bip22_world is not None and len(bip22_world) > 1:
        # Bip001 root 语义是「骨盆在 z=0 地面上的水平投影」；换基非纯 z 平移，
        # 因此 xy 必须按世界系骨盆重算，不能沿用变换后的 figure 原点。
        bip22_world[0, 0] = bip22_world[1, 0]
        bip22_world[0, 1] = bip22_world[1, 1]
        bip22_world[0, 2] = 0.0
    elif bip22_world is not None and len(bip22_world):
        bip22_world[0, 2] = 0.0

    generic_world: Optional[Dict[str, Any]] = None
    if isinstance(generic_pose, dict):
        generic_world = {
            "frame": FRAME_WORLD,
            "pred_joint_coords": (
                np.asarray(generic_pose["pred_joint_coords"], dtype=np.float64) @ q.T + offset
            ).tolist(),
            "pred_keypoints_3d": (
                np.asarray(generic_pose["pred_keypoints_3d"], dtype=np.float64) @ q.T + offset
            ).tolist(),
            "pred_global_rots": np.stack(
                [
                    _conjugate(matrix, q)
                    for matrix in np.asarray(generic_pose["pred_global_rots"], dtype=np.float64)
                ]
            ).tolist(),
            "global_rot": _global_rot_world(generic_pose.get("global_rot"), q),
            "global_rot_space": "zyx_ground_world_v1",
            "pred_cam_t_world": offset.tolist(),
        }

    retarget_world = None
    if isinstance(retarget_payload, dict) and retarget_payload.get("bones"):
        retarget_world = _transform_retarget(retarget_payload, q, offset)

    mesh_block = None
    if vertices_world is not None and include_mesh:
        mesh_block = {
            "frame": FRAME_WORLD,
            "unit": UNIT_METER,
            "vertices": vertices_world.tolist(),
            "faces": np.asarray(mesh_faces, dtype=np.int64).tolist() if mesh_faces is not None else [],
        }

    zs: List[float] = [float(np.min(joints_world[:, 2])), float(np.min(kp3_world[:, 2]))]
    if bip22_world is not None and len(bip22_world) > 1:
        # 跳过 Bip001 root（地面投影点，恒为 0）
        zs.append(float(np.min(bip22_world[1:, 2])))
    if retarget_world and retarget_world.get("bones"):
        positions = [
            row["position"]
            for row in retarget_world["bones"].values()
            if isinstance(row, dict) and row.get("position") is not None
        ]
        if positions:
            zs.append(float(np.min(np.asarray(positions, dtype=np.float64)[:, 2])))
    if vertices_world is not None and len(vertices_world):
        zs.append(float(np.min(vertices_world[:, 2])))

    block = transform.as_dict()
    snap = dict(ground_snap) if isinstance(ground_snap, dict) else {}
    block.update(
        {
            "status": "ok",
            "ground_mode": str(ground_mode),
            "ground_snap": {
                "status": snap.get("status", "skipped"),
                "source": snap.get("source"),
                "offset_m": round(float(snap.get("offset_m", 0.0) or 0.0), 6),
            },
            "min_z_m": round(min(zs), 6),
            "joints_3d": joints_world.tolist(),
            "keypoints_3d_coco17": kp3_world.tolist(),
            "bip22_joints": bip22_world.tolist() if bip22_world is not None else [],
            "generic_pose": generic_world,
            "retarget": retarget_world,
            "mesh": mesh_block,
            "error": None,
        }
    )
    return block


def _global_rot_world(global_rot, q: np.ndarray) -> Optional[List[float]]:
    if global_rot is None:
        return None
    from ai_vitpose.core.generic_pose import _euler_zyx_to_rotmat, _rotmat_to_euler_zyx

    matrix = _euler_zyx_to_rotmat(np.asarray(global_rot, dtype=np.float64).reshape(3))
    return [float(v) for v in _rotmat_to_euler_zyx(_conjugate(matrix, q))]


# ---------------------------------------------------------------------------
# XML（<worldPose> 为本项目扩展，参考序列化器不含该元素）
# ---------------------------------------------------------------------------


def _fmt(value) -> str:
    return f"{float(value):.6f}"


def _matrix_attr(matrix) -> str:
    return " ".join(_fmt(v) for v in np.asarray(matrix, dtype=np.float64).reshape(9))


def world_pose_xml(world: Optional[Dict[str, Any]]) -> str:
    if not world:
        return ""
    status = str(world.get("status") or "")
    attrs = (
        f'frame="{world.get("frame") or FRAME_WORLD}" '
        f'unit="{world.get("unit") or UNIT_METER}" '
        f'source="{world.get("source") or ""}" '
        f'status="{status}"'
    )
    parts: List[str] = [f"<worldPose {attrs}>"]
    if status != "ok":
        error = str(world.get("error") or "")
        parts.append(f"  <error>{_xml_escape(error)}</error>")
        parts.append("</worldPose>")
        return "\n".join(parts)

    parts.append(
        '  <transform rotation="{}" translation="{}"/>'.format(
            _matrix_attr(world.get("rotation")),
            " ".join(_fmt(v) for v in np.asarray(world.get("translation"), dtype=np.float64).reshape(3)),
        )
    )
    joints = world.get("joints_3d") or []
    parts.append(f'  <joints count="{len(joints)}">')
    for index, row in enumerate(joints):
        parts.append(
            f'    <joint index="{index}" x="{_fmt(row[0])}" y="{_fmt(row[1])}" z="{_fmt(row[2])}"/>'
        )
    parts.append("  </joints>")

    retarget = world.get("retarget") or {}
    bones = retarget.get("bones") or {}
    if bones:
        parts.append(f'  <bones count="{len(bones)}">')
        for name, row in bones.items():
            parts.append(f'    <bone name="{_xml_escape(str(name))}">')
            for key, tag in (("global_rotation", "global"), ("local_rotation", "local")):
                matrix = row.get(key)
                if matrix is not None:
                    parts.append(f'      <{tag} matrix="{_matrix_attr(matrix)}"/>')
            position = row.get("position")
            if position is not None:
                parts.append(
                    '      <position x="{}" y="{}" z="{}"/>'.format(
                        _fmt(position[0]), _fmt(position[1]), _fmt(position[2])
                    )
                )
            parts.append("    </bone>")
        parts.append("  </bones>")
    parts.append("</worldPose>")
    return "\n".join(parts)


def _xml_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def insert_world_xml(xml: str, world: Optional[Dict[str, Any]]) -> str:
    """把 ``<worldPose>`` 插入 ``poseResponse`` 根节点末尾（``<overlay>`` 是 ``poseResult`` 的子节点）。"""
    element = world_pose_xml(world)
    if not element:
        return xml
    index = xml.rfind("</poseResponse>")
    if index < 0:
        return xml
    return xml[:index] + element + "\n" + xml[index:]


# ---------------------------------------------------------------------------
# 适配器
# ---------------------------------------------------------------------------


class RetargetAdapter:
    """lifespan 常驻：core 导入 + cs_assets 自检 + 默认 Biped profile。"""

    def __init__(
        self,
        resources: Optional[str | Path],
        *,
        cache_root: Optional[str | Path] = None,
        assets_root: Optional[str | Path] = None,
    ) -> None:
        self.ready = False
        self.error = ""
        self.resources = ensure_resource_path(resources) if resources else None
        self.cache_root = Path(cache_root) if cache_root else None
        self.assets_root = Path(assets_root) if assets_root else None
        self.cs_assets = ""
        self._core: Dict[str, Any] = {}
        self.default_profile: Optional[Dict[str, Any]] = None
        self.default_fingerprint = ""
        self.default_diagnostics: Dict[str, Any] = {}
        self.default_error = ""
        self._default_summary: Dict[str, Any] = {"status": "not_provided"}
        self._setup()

    # -- 初始化 ------------------------------------------------------------

    def _setup(self) -> None:
        if self.resources is None:
            self.error = "未配置 SAM3D_RESOURCES，姿势适配层不可用"
            return
        try:
            self._core = _load_core()
        except Exception as exc:  # noqa: BLE001 - 任何导入失败都只让新接口 503
            self.error = f"参考实现导入失败: {type(exc).__name__}: {exc}"
            return
        try:
            self.cs_assets = str(self._core["target_retarget"].cs_asset_root())
            self._core["cs_response"].cs_asset_root()
        except Exception as exc:  # noqa: BLE001
            self.error = f"重定向/CS 资产缺失: {exc}"
            return
        if self.assets_root is None:
            self.assets_root = Path(self._core["target_profile"].default_assets_root())
        self._build_default_target()
        self.ready = True

    def _build_default_target(self) -> None:
        tb = self._core["target_bundle"]
        tp = self._core["target_profile"]
        try:
            bundle = default_biped_bundle(self.assets_root)
            bundle["fingerprint"] = tb.compute_fingerprint(bundle)
            validated = tb.validate_bundle(bundle)
            profile = tp.ensure_profile(
                validated, cache_root=self.cache_root, assets_root=self.assets_root
            )
        except Exception as exc:  # noqa: BLE001 - 失败只退回 not_provided
            self.default_error = f"{type(exc).__name__}: {exc}"
            self._default_summary = {"status": "not_provided", "error": self.default_error}
            return
        self.default_profile = profile
        self.default_fingerprint = str(validated.get("fingerprint") or "")
        self.default_diagnostics = validated.get("diagnostics") or {}
        meta = (profile or {}).get("meta") or {}
        self._default_summary = {
            "status": "default",
            "skeleton_id": validated.get("skeleton_id"),
            "fingerprint": self.default_fingerprint,
            "bone_count": meta.get("bone_count"),
            "mapping_bone_count": meta.get("mapping_bone_count"),
            "chains": meta.get("chains"),
            "warnings": meta.get("warnings"),
            "cache_hit": bool((profile or {}).get("cache_hit")),
        }

    # -- 状态 --------------------------------------------------------------

    @property
    def core(self) -> Dict[str, Any]:
        return self._core

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "resources": str(self.resources or ""),
            "assets_root": str(self.assets_root or ""),
            "cs_assets": self.cs_assets,
            "default_target": dict(self._default_summary),
            "error": self.error,
        }

    # -- 目标解析 ----------------------------------------------------------

    def resolve_target(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """请求 -> ``{profile, schema, target_status, target_fingerprint, ...}``。"""
        tb = self._core["target_bundle"]
        tp = self._core["target_profile"]
        tr = self._core["target_retarget"]

        requested = str(data.get("retarget_schema") or "").strip().lower() or "auto"
        try:
            accepted = tb.parse_target_request(data)
        except tb.TargetBundleTooLarge as exc:
            raise RetargetError(str(exc), status_code=413, stage="bundle", target_status="rejected") from exc
        except tb.TargetBundleError as exc:
            raise RetargetError(str(exc), status_code=400, stage="bundle", target_status="rejected") from exc

        if accepted is None:
            if self.default_profile is None:
                return {
                    "profile": None,
                    "schema": "v1",
                    "target_status": "not_provided",
                    "target_fingerprint": "",
                    "target_diagnostics": {},
                    "skipped_reason": self.default_error or "默认目标不可用",
                }
            profile = self.default_profile
            target_status = "default"
            fingerprint = self.default_fingerprint
            diagnostics = self.default_diagnostics
        else:
            try:
                profile = tp.ensure_profile(
                    accepted, cache_root=self.cache_root, assets_root=self.assets_root
                )
            except tp.TargetProfileError as exc:
                raise RetargetError(
                    str(exc), status_code=400, stage="retarget", target_status="rejected"
                ) from exc
            target_status = "used"
            fingerprint = str(accepted.get("fingerprint") or "")
            diagnostics = accepted.get("diagnostics") or {}

        try:
            schema = tr.resolve_schema(profile, requested)
        except tr.TargetRetargetError as exc:
            raise RetargetError(
                str(exc), status_code=400, stage="retarget", target_status="rejected"
            ) from exc

        return {
            "profile": profile,
            "schema": schema,
            "target_status": target_status,
            "target_fingerprint": fingerprint,
            "target_diagnostics": diagnostics,
            "skipped_reason": "v1_baseline" if schema == "v1" else "",
        }

    def retarget(self, generic_pose: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        """``generic_pose`` -> ``{schema, payload, diagnostics}``（schema=v1 时为空）。"""
        if not target or target.get("schema") in (None, "", "v1"):
            return {}
        tr = self._core["target_retarget"]
        try:
            result = tr.retarget_generic_pose(
                generic_pose,
                target.get("profile"),
                schema=target.get("schema"),
                assets_root=self.assets_root,
            )
        except tr.TargetRetargetError as exc:
            raise RetargetError(
                str(exc), status_code=500, stage="retarget", target_status="rejected"
            ) from exc
        if not result.get("payload"):
            raise RetargetError("重定向未产生有效结果", status_code=500, stage="retarget")
        return result

    # -- 姿势组装 ----------------------------------------------------------

    def assemble_pose(
        self,
        raw: Dict[str, Any],
        camera: Dict[str, Any],
        *,
        want_raw: bool = False,
        require_generic: bool = False,
        bbox: Optional[Sequence[float]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """原始 SAM3D 输出 -> 参考同构 body + 世界系所需的 numpy 中间量。"""
        gp = self._core["generic_pose"]
        nm = self._core["normalize"]
        cs = self._core["cs_response"]

        coords = raw.get("mhr_joint_coords")
        if coords is None:
            coords = raw.get("pred_joint_coords")
        if coords is None:
            raise RetargetError("SAM3D 缺少 127 关节坐标", status_code=500, stage="engine")
        coords_np = np.asarray(coords, dtype=np.float32).reshape(127, 3)

        rots = raw.get("mhr_joint_global_rots")
        if rots is None:
            rots = raw.get("pred_global_rots")
        kp3 = raw.get("pred_keypoints_3d")
        kp2 = raw.get("pred_keypoints_2d")
        cam_t = raw.get("pred_cam_t")
        global_rot = raw.get("global_rot")
        if rots is None or kp3 is None or kp2 is None or cam_t is None or global_rot is None:
            raise RetargetError("SAM3D 原始输出缺少必需的旋转/关键点/相机参数", status_code=500, stage="engine")

        joints_cam = nm.skeleton24(coords_np)
        joints_fig = nm.camera_points_to_figure_frame(joints_cam)
        conf24 = nm.fill_mhr_confidence(np.zeros(24, dtype=np.float32), base=DEFAULT_KEYPOINT_SCORE)

        kp2_np = np.asarray(kp2, dtype=np.float32).reshape(70, -1)[:, :2]
        kp3_np = np.asarray(kp3, dtype=np.float32).reshape(70, 3)
        kp2_coco = np.zeros((17, 3), dtype=np.float32)
        kp3_coco = np.zeros((17, 3), dtype=np.float32)
        for dst, src in enumerate(MHR70_COCO17):
            kp2_coco[dst, :2] = kp2_np[src, :2]
            kp2_coco[dst, 2] = DEFAULT_KEYPOINT_SCORE
            kp3_coco[dst] = kp3_np[src]

        cam = dict(camera or {})
        cam["translation"] = np.asarray(cam.get("translation", cam_t), dtype=np.float32).reshape(-1).tolist()
        cam = nm.enrich_camera(cam, coords_np)

        body: Dict[str, Any] = {
            "status": "success",
            "engine": "sam3d_vitpose",
            "skeleton_source": "MHR127_direct",
            "orientation_mode": "fixed_camera_to_figure_v6",
            "joints_3d": joints_fig.tolist(),
            "joints_3d_frame": FRAME_FIGURE,
            "joints_3d_unit": UNIT_METER,
            "confidence": conf24.tolist(),
            "valid_joints": int((conf24 > 0).sum()),
            "keypoints_2d_coco17": kp2_coco.tolist(),
            "keypoints_3d_coco17": kp3_coco.tolist(),
            "camera": cam,
        }

        cs_block: Optional[Dict[str, Any]] = None
        try:
            cs_block = cs.build_cs_response(
                joints_fig,
                conf24,
                cam,
                {"mhr70_keypoints_3d": kp3_np, "mhr_joint_coords": coords_np},
            )
            body.update(cs_block)
        except Exception as exc:  # noqa: BLE001 - CS 失败不阻断主流程
            body["cs_response_status"] = "error"
            body["cs_response_error"] = f"{type(exc).__name__}: {exc}"

        generic_np: Optional[Dict[str, Any]] = None
        try:
            extras: Dict[str, Any] = {
                "pred_joint_coords": coords_np,
                "pred_global_rots": np.asarray(rots, dtype=np.float32),
                "pred_keypoints_3d": kp3_np,
                "global_rot": np.asarray(global_rot, dtype=np.float32).reshape(3),
                "pred_cam_t": np.asarray(cam_t, dtype=np.float32).reshape(3),
            }
            for key in ("scale_params", "shape_params"):
                if raw.get(key) is not None:
                    extras[key] = np.asarray(raw[key], dtype=np.float32)
            if bbox is not None:
                extras["bbox"] = np.asarray(bbox, dtype=np.float32)
            generic_np = gp.build_generic_pose(extras, camera=cam)
            body["generic_pose"] = gp.to_jsonable(generic_np)
            body["generic_pose_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            body["generic_pose_status"] = "error"
            body["generic_pose_error"] = f"{type(exc).__name__}: {exc}"
            if require_generic:
                raise RetargetError(
                    f"generic_pose 组装失败: {exc}", status_code=500, stage="generic_pose"
                ) from exc

        if want_raw:
            body["raw"] = {
                "bbox": list(bbox) if bbox is not None else raw.get("bbox"),
                "mhr70_keypoints_2d": kp2_np.tolist(),
                "mhr70_keypoints_3d": kp3_np.tolist(),
                "mhr_joint_coords": coords_np.tolist(),
                "mhr_joint_global_rots": np.asarray(rots, dtype=np.float32).tolist(),
                "pred_joint_coords": coords_np.tolist(),
                "pred_global_rots": np.asarray(rots, dtype=np.float32).tolist(),
                "pred_keypoints_3d": kp3_np.tolist(),
                "global_rot": np.asarray(global_rot, dtype=np.float32).reshape(3).tolist(),
                "pred_cam_t": np.asarray(cam_t, dtype=np.float32).reshape(3).tolist(),
                "scale_params": raw.get("scale_params"),
                "shape_params": raw.get("shape_params"),
            }

        bip22_joints = (
            np.asarray(cs_block["bip22_joints"], dtype=np.float32) if cs_block else None
        )
        internal = {
            "joints_fig": joints_fig,
            "keypoints_3d_cam": kp3_coco,
            "pred_cam_t": np.asarray(cam_t, dtype=np.float32).reshape(3),
            "generic": generic_np,
            "bip22_joints": bip22_joints,
            "bip22_anchor": (
                _bip22_anchor(joints_fig, bip22_joints, cs_block.get("bip22_source"))
                if cs_block
                else None
            ),
        }
        return body, internal
