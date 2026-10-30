"""腿部 two-bone IK（解析解）：保骨长，末端落在点云深度射线上。

第一级「沿射线缩放关节」能精确命中点云深度，但代价是骨长被改；这里把两条腿换成
标准两段 IK：髋沿用第一级的射线缩放结果（骨盆/躯干深度已校正），踝作为末端落在
**自身的相机射线 × 点云深度**上（2D 投影不变），膝只由当前姿态的弯曲方向（pole）
决定；骨长取原姿态，全程守恒。

默认关闭，``POSEEXTRACTOR_SAM3D_LIMB_IK=1`` 打开；关掉即回滚到射线缩放。
相机系（相机在原点、+z 向前），长度单位米。
"""

from __future__ import annotations

import math
import os

import numpy as np

# (髋, 膝, 踝) 的 COCO-17 下标
LEGS = {"l": (11, 13, 15), "r": (12, 14, 16)}
LIMB_IK_ENV = "POSEEXTRACTOR_SAM3D_LIMB_IK"
_TRUE = ("1", "on", "true", "yes")
# 可达球壳留一点余量，避免完全伸直/折死时 pole 退化
REACH_EPS = 1e-4


def limb_ik_enabled() -> bool:
    return str(os.environ.get(LIMB_IK_ENV, "")).strip().lower() in _TRUE


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _reach_u(root: np.ndarray, ray: np.ndarray, u0: float, d: float):
    """在射线上求满足 ``|ray*u - root| = d`` 且离 ``u0`` 最近的 u；无解返回 None。"""
    b = float(ray @ root)
    perp2 = float(root @ root) - b * b
    disc = d * d - perp2
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    cand = [u for u in (b - s, b + s) if u > 1e-6]
    return min(cand, key=lambda u: abs(u - u0)) if cand else None


def _pole_dir(root, knee, end, base_root, base_knee, base_end) -> np.ndarray:
    """膝的弯曲方向：当前姿态相对 root→end 轴的垂向分量，退化时用基线姿态。"""
    for r, k, e in ((root, knee, end), (base_root, base_knee, base_end)):
        axis = _unit(e - r)
        if not np.any(axis):
            continue
        v = k - r
        n = v - float(v @ axis) * axis
        if float(np.linalg.norm(n)) > 1e-6:
            return _unit(n)
    axis = _unit(end - root)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(axis, tmp))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.clip((a @ b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def apply_leg_ik(base_kp, new_kp, targets=None, focal: float | None = None):
    """把射线缩放后的两条腿换成保骨长的 two-bone IK。

    ``base_kp`` 只用来取骨长与 pole 基线；``targets`` 是每根骨做骨长约束**之前**的
    纯深度目标（即踝在自身射线上的点云深度位置），缺省时退回 ``new_kp`` 本身。
    """
    base = np.asarray(base_kp, dtype=np.float64).reshape(-1, 3)
    cur = np.asarray(new_kp, dtype=np.float64).reshape(-1, 3).copy()
    tgt = None if targets is None else np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    info: dict = {"enabled": True, "applied": False, "method": "two_bone_ik",
                  "sides": {}}
    if base.shape != cur.shape or len(base) < 17:
        info["reason"] = "bad_inputs"
        return cur, info
    if tgt is None or tgt.shape != cur.shape or not np.isfinite(tgt).all():
        info["target_source"] = "new_kp"
        tgt = cur
    else:
        info["target_source"] = "ray_target"

    total = 0.0
    for side, (h, k, a) in LEGS.items():
        hip, knee0, ankle0 = cur[h], cur[k], cur[a]
        goal = tgt[a]                  # 骨长约束之前的纯深度目标（在踝自身射线上）
        l1 = float(np.linalg.norm(base[k] - base[h]))
        l2 = float(np.linalg.norm(base[a] - base[k]))
        if l1 < 1e-6 or l2 < 1e-6:
            info["sides"][side] = {"ok": False, "reason": "degenerate_bone"}
            continue

        ray = _unit(base[a])           # 踝自身的相机射线（缩放前后同一条）
        d0 = float(np.linalg.norm(goal - hip))
        lo, hi = abs(l1 - l2) + REACH_EPS, l1 + l2 - REACH_EPS
        d = min(max(d0, lo), hi)
        clamped = abs(d - d0) > 1e-9
        target = goal
        if clamped:
            u = _reach_u(hip, ray, float(np.linalg.norm(goal)), d)
            if u is None:                       # 射线上确实不可达：保持原样
                info["sides"][side] = {"ok": False, "reason": "unreachable",
                                       "reach": d0 / max(l1 + l2, 1e-9)}
                continue
            target = ray * u

        axis = _unit(target - hip)
        n = _pole_dir(hip, knee0, target, base[h], base[k], base[a])
        aa = (d * d + l1 * l1 - l2 * l2) / (2.0 * d)
        hh = math.sqrt(max(l1 * l1 - aa * aa, 0.0))
        knee = hip + aa * axis + hh * n
        ankle = hip + d * axis

        def _px(p, q):
            if not focal:
                return None
            zp, zq = max(float(p[2]), 1e-6), max(float(q[2]), 1e-6)
            return float(math.hypot(focal * (p[0] / zp - q[0] / zq),
                                    focal * (p[1] / zp - q[1] / zq)))

        info["sides"][side] = {
            "ok": True,
            "bone_m": [l1, l2],
            "distance_m": float(d),
            "distance_before_m": float(d0),
            "reach": float(d / max(l1 + l2, 1e-9)),
            "clamped": bool(clamped),
            "ankle_shift_px": _px(ankle0, ankle),
            "knee_shift_px": _px(knee0, knee),
            "ankle_target_px": _px(goal, ankle),   # 与点云深度目标的残余像素差
            "upper_angle_deg": _angle_deg(knee0 - hip, knee - hip),
            "lower_angle_deg": _angle_deg(ankle0 - knee0, ankle - knee),
            "ankle_depth_delta_m": float(ankle[2] - goal[2]),
        }
        total = max(total, float(np.linalg.norm(ankle - ankle0)),
                    float(np.linalg.norm(knee - knee0)))
        cur[k], cur[a] = knee, ankle

    info["applied"] = bool(total > 1e-3)
    info["max_joint_shift_m"] = float(total)
    return cur, info
