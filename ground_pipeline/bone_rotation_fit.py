"""旋转参数化深度精修：骨骼吸附第二级（保骨长 · 只解深度 · 2D 偏移受预算约束）。

第一级 ``bone_depth_snap`` 沿相机射线缩放关节（u/v 严格不变），但从「修正后的父关节」
出发重新瞄准时，方向仍取自射线缩放前的目标向量，末端深度会有残余偏差。
这里改为 **绕父关节旋转** 的 FK：

    j_i = j_{p(i)} + R_{p(i)} · R_i · (j_i^0 − j_{p(i)}^0)

``R_i`` 只取 swing（2 DOF：骨方向在切平面内的两个偏转），绕骨自身轴的 twist 固定为 0，
且 ``|j_i − j_{p(i)}|`` 恒等于原始骨长（旋转不改变长度）。每条链（左/右臂、左/右腿）
从不动锚点（肩 5/6、髋 11/12）出发逐级前进，父级累积旋转参与子级 FK。

目标函数（相机系，像素按深度图分辨率）：

    E = Σ_s w_s · Huber( (z_s(ω) − z*_s) / δ )
        + λ · Σ_j ‖π(j_j) − π(j_j^0)‖² / budget²
        + w_reg · Σ_i (θ_i / θ_max)²

- ``z*_s``：深度目标。默认取第一级射线缩放的逐关节目标深度 ``j_z^0 · k``（``k`` 由网格
  z-buffer 与 MoGe-2 表面比得到，已消掉「关节在体内、离表面一个肢体半径」的常数偏置），
  未给定时退化为直接采样 ``D(π(j^0))``；骨中间点的目标由两端线性插值。
- ``w_s``：像素可信度，人物 mask 内、深度有效、局部深度梯度小 → 权重大。
- λ 自适应（初始 1.0）：本轮关节投影偏移超预算 → λ×3，压回预算内；已可行 → λ×0.3
  让深度更到位；每轮保留「可行且深度代价最低」的解，最后取它。
- 闸门：关节投影偏移最大值 > ``max_shift_px``，或加权深度代价没有下降 2% → 回退 ω=0
  （等价「不修」）。

只作用于渲染端已有缓存，不写回服务；``POSEEXTRACTOR_SAM3D_BONE_ROTATION_FIT=1`` 打开，
默认关闭，关闭后代码路径不进入本模块。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

ROTATION_REFINE_ENV = "POSEEXTRACTOR_SAM3D_BONE_ROTATION_FIT"

# 每条链从不动锚点出发：(肩/髋, 中间关节, 末端关节)
CHAINS = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))
BONE_NAMES = ("l_upper_arm", "l_forearm", "r_upper_arm", "r_forearm",
              "l_thigh", "l_shank", "r_thigh", "r_shank")
# (链号, 起点, 终点) —— 顺序与 bone_depth_snap.BONES 一致
EDGES = tuple((ci, chain[k], chain[k + 1])
              for ci, chain in enumerate(CHAINS) for k in range(len(chain) - 1))
# 会移动的关节（每根骨的末端）
MOVED_JOINTS = tuple(edge[2] for edge in EDGES)

MAX_ANGLE_DEG = 30.0
MAX_PROJ_SHIFT_PX = 2.0
HUBER_DELTA_M = 0.01
BONE_INTERIOR = (0.25, 0.5, 0.75)
INTERIOR_WEIGHT = 0.5
LAMBDA_SCHEDULE = (1.0, 0.3, 0.1, 0.03)
REG_WEIGHT = 0.05
DEPTH_RANGE_M = (0.3, 8.0)
MIN_DEPTH_SAMPLES = 40

_TRUE = ("1", "on", "true", "yes")

# (骨下标, 骨轴参数 s, 起点关节, 终点关节, 基础权重)
SAMPLES = tuple(
    [(index, 1.0, a, b, 1.0) for index, (_ci, a, b) in enumerate(EDGES)]
    + [(index, float(f), a, b, INTERIOR_WEIGHT)
       for index, (_ci, a, b) in enumerate(EDGES) for f in BONE_INTERIOR]
)


def rotation_refine_enabled() -> bool:
    return str(os.environ.get(ROTATION_REFINE_ENV, "")).strip().lower() in _TRUE


# ------------------------------------------------------------------ 旋转几何
def _tangent_basis(d0: np.ndarray):
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(d0[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d0, ref)
    n = float(np.linalg.norm(e1))
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    e1 = e1 / n
    return e1, np.cross(d0, e1)


def _rotation_between(d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """d0 → d1 的最小旋转（swing，不含 twist）；角度 <= MAX_ANGLE_DEG，无 180° 分支。"""
    v = np.cross(d0, d1)
    s = float(np.linalg.norm(v))
    c = float(np.clip(d0 @ d1, -1.0, 1.0))
    if s < 1e-12:
        return np.eye(3)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _forward(j0: np.ndarray, params: np.ndarray):
    """旋转 FK：返回 ``(新关节, 各骨局部旋转)``；骨长由旋转天然保持。"""
    j = np.array(j0, dtype=np.float64, copy=True)
    local = np.tile(np.eye(3), (len(EDGES), 1, 1))
    cumulative = [np.eye(3) for _ in CHAINS]
    for index, (_ci, a, b) in enumerate(EDGES):
        v0 = j0[b] - j0[a]
        length = float(np.linalg.norm(v0))
        if length < 1e-9:
            continue
        d0 = v0 / length
        e1, e2 = _tangent_basis(d0)
        d = d0 + params[index, 0] * e1 + params[index, 1] * e2
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        rot = _rotation_between(d0, d / norm)
        chain = EDGES[index][0]
        cumulative[chain] = cumulative[chain] @ rot
        j[b] = j[a] + cumulative[chain] @ v0
        local[index] = rot
    return j, local


def _project(j: np.ndarray, focal: float, dw: int, dh: int):
    z = j[:, 2]
    uv = np.full((len(j), 2), np.nan)
    ok = np.isfinite(z) & (z > 1e-4)
    uv[ok, 0] = focal * j[ok, 0] / z[ok] + dw / 2.0
    uv[ok, 1] = focal * j[ok, 1] / z[ok] + dh / 2.0
    return uv, ok


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    h, w = img.shape
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ok = (np.isfinite(u) & np.isfinite(v)
          & (u >= 0.0) & (u <= w - 1.001) & (v >= 0.0) & (v <= h - 1.001))
    out = np.zeros(u.shape, dtype=np.float64)
    if not np.any(ok):
        return out, ok
    uu, vv = u[ok], v[ok]
    x0 = np.floor(uu).astype(np.int64)
    y0 = np.floor(vv).astype(np.int64)
    fx, fy = uu - x0, vv - y0
    out[ok] = (img[y0, x0] * (1.0 - fx) * (1.0 - fy) + img[y0, x0 + 1] * fx * (1.0 - fy)
               + img[y0 + 1, x0] * (1.0 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return out, ok


def _weight_map(depth: np.ndarray, mask):
    """像素可信度：mask 内 + 深度有效 + 局部梯度小（遮挡边缘/斜面权重大幅下降）。"""
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > DEPTH_RANGE_M[0]) & (d < DEPTH_RANGE_M[1])
    if mask is not None:
        arr = np.asarray(mask)
        if arr.shape != d.shape:
            arr = np.asarray(Image.fromarray((arr.astype(bool).astype(np.uint8)) * 255)
                             .resize((d.shape[1], d.shape[0]), Image.NEAREST)) > 127
        valid &= arr.astype(bool)
    weight = valid.astype(np.float64)
    if np.count_nonzero(valid) < MIN_DEPTH_SAMPLES:
        return weight, valid
    gy, gx = np.gradient(np.where(valid, d, 0.0))
    grad = np.hypot(gx, gy)
    median = float(np.median(grad[valid]))
    if median > 1e-6:
        weight = np.where(valid, 1.0 / (1.0 + (grad / median) ** 2), 0.0)
    return weight, valid


def _huber(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def _cost(params, j0, target_z, focal, dh, dw, depth, weight_map, lam, reg_weight):
    """返回 (总代价, 深度代价, 2D 偏移平方和)。梯度由 scipy 数值差分。"""
    j, local = _forward(j0, params)
    _uv, ok = _project(j, focal, dw, dh)
    su = np.full(len(SAMPLES), np.nan)
    sv = np.full(len(SAMPLES), np.nan)
    sz = np.full(len(SAMPLES), np.nan)
    st = np.zeros(len(SAMPLES))
    sw = np.zeros(len(SAMPLES))
    for k, (index, s, a, b, base) in enumerate(SAMPLES):
        pa, pb = j[a], j[b]
        p = pa + s * (pb - pa)
        if not np.isfinite(p).all() or p[2] <= 1e-4:
            continue
        su[k] = focal * p[0] / p[2] + dw / 2.0
        sv[k] = focal * p[1] / p[2] + dh / 2.0
        sz[k] = p[2]
        st[k] = (1.0 - s) * target_z[a] + s * target_z[b]
        w, w_ok = _bilinear(weight_map, np.array([su[k]]), np.array([sv[k]]))
        sw[k] = float(w[0]) if w_ok[0] else 0.0
        sw[k] *= base
    res = sz - st
    depth_cost = float(np.sum(sw * _huber(res / HUBER_DELTA_M)))
    e2d = 0.0
    for k, (_index, s, a, b, _base) in enumerate(SAMPLES):
        if s < 1.0:
            continue
        if not ok[b]:
            e2d += 1.0e4
            continue
        du = su[k] - (focal * j0[b, 0] / j0[b, 2] + dw / 2.0)
        dv = sv[k] - (focal * j0[b, 1] / j0[b, 2] + dh / 2.0)
        e2d += (du * du + dv * dv) / (MAX_PROJ_SHIFT_PX ** 2)
    angles = np.array([np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
                       for r in local])
    reg = float(np.sum((angles / np.deg2rad(MAX_ANGLE_DEG)) ** 2))
    return depth_cost + lam * e2d + reg_weight * reg, depth_cost, e2d, local


def rotation_refine(joints_cam, depth, focal: float, mask=None, target_z=None, *,
                    max_angle_deg: float = MAX_ANGLE_DEG,
                    max_shift_px: float = MAX_PROJ_SHIFT_PX,
                    lambdas=LAMBDA_SCHEDULE, reg_weight: float = REG_WEIGHT,
                    max_rounds: int = 8):
    """在保骨长约束下把关节深度拉向目标，同时把 2D 投影偏移压在预算内。

    返回 ``(新关节, detail)``；被闸门拒绝时原样返回输入关节。
    """
    detail = {"enabled": True, "applied": False, "method": "rotation_fk_swing",
              "dof_per_bone": 2, "max_angle_deg": float(max_angle_deg),
              "max_shift_px": float(max_shift_px),
              "lambda_schedule": [float(x) for x in lambdas],
              "regularization_weight": float(reg_weight)}
    j0 = np.asarray(joints_cam, dtype=np.float64).reshape(-1, 3)
    if j0.shape[0] <= max(e[2] for e in EDGES) or not np.isfinite(j0).all():
        detail["reason"] = "bad_joints"
        return j0.copy(), detail
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2 or d.size == 0 or not np.isfinite(float(focal)) or float(focal) <= 0:
        detail["reason"] = "bad_depth_or_focal"
        return j0.copy(), detail

    dh, dw = d.shape
    weight_map, valid = _weight_map(d, mask)
    if np.count_nonzero(valid) < MIN_DEPTH_SAMPLES:
        detail["reason"] = "too_few_valid_pixels"
        return j0.copy(), detail

    uv0, ok0 = _project(j0, float(focal), dw, dh)
    if target_z is None:
        sampled, s_ok = _bilinear(d, uv0[:, 0], uv0[:, 1])
        tz = np.where(s_ok & np.isfinite(sampled) & (sampled > 0), sampled, j0[:, 2])
    else:
        tz = np.asarray(target_z, dtype=np.float64).reshape(-1)
        if tz.shape[0] != j0.shape[0]:
            detail["reason"] = "target_z_shape"
            return j0.copy(), detail
        tz = np.where(np.isfinite(tz) & (tz > 1e-4), tz, j0[:, 2])
    detail["depth_target"] = "ray_scale_k" if target_z is not None else "moge_depth"

    try:
        from scipy.optimize import minimize
    except Exception:                                    # pragma: no cover
        detail["reason"] = "scipy_unavailable"
        return j0.copy(), detail

    bound = float(np.tan(np.deg2rad(max_angle_deg)))
    bounds = [(-bound, bound)] * (2 * len(EDGES))
    zeros = np.zeros((len(EDGES), 2))
    base_cost, base_depth, _e2d, _ = _cost(zeros, j0, tz, float(focal), dh, dw,
                                           d, weight_map, 0.0, 0.0)
    detail["depth_cost_before"] = float(base_depth)
    if base_depth <= 1e-12:
        detail["reason"] = "no_depth_signal"
        return j0.copy(), detail

    # λ 自适应：投影超预算 → 加大 λ 压回预算内；已在预算内 → 减小 λ 让深度更到位。
    # 全程记录「可行且深度代价最低」的解，最后用它，避免末轮 λ 过小把投影顶出预算。
    params = zeros.copy()
    trace = []
    lam = float(lambdas[0]) if len(lambdas) else 1.0
    lam_lo, lam_hi = 1e-3, 3.0 * (max(float(x) for x in lambdas) if len(lambdas) else 1.0)
    best = None                                    # (depth_cost, params, shift_max)
    stall = 0
    for _round in range(int(max_rounds)):
        state = {"n": 0}

        def fun(x, lam=lam, state=state):
            state["n"] += 1
            return _cost(x.reshape(len(EDGES), 2), j0, tz, float(focal), dh, dw,
                         d, weight_map, lam, reg_weight)[0]

        res = minimize(fun, params.reshape(-1), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 120, "eps": 1e-3, "ftol": 1e-10, "gtol": 1e-8})
        params = res.x.reshape(len(EDGES), 2)
        cost, depth_cost, e2d, _ = _cost(params, j0, tz, float(focal), dh, dw,
                                         d, weight_map, lam, reg_weight)
        j_try, _ = _forward(j0, params)
        uv_try, _ = _project(j_try, float(focal), dw, dh)
        shift_now = float(np.max(np.linalg.norm(
            uv_try[list(MOVED_JOINTS)] - uv0[list(MOVED_JOINTS)], axis=1)))
        feasible = np.isfinite(shift_now) and shift_now <= float(max_shift_px)
        trace.append({"lambda": float(lam), "evaluations": int(state["n"]),
                      "depth_cost": float(depth_cost), "e2d": float(e2d),
                      "shift_px_max": shift_now, "feasible": bool(feasible),
                      "total": float(cost)})
        improved = False
        if feasible and (best is None or depth_cost < best[0]):
            best = (float(depth_cost), params.copy(), shift_now)
            improved = True
        stall = 0 if improved else stall + 1
        if stall >= 2:
            break                                  # 连续两轮没有更优可行解，收手
        lam = lam * 3.0 if not feasible else lam * 0.3
        if lam > lam_hi or lam < lam_lo:
            break
    if best is not None:
        params = best[1]
    detail["lambda_adaptive"] = True
    detail["lambda_final"] = float(trace[-1]["lambda"]) if trace else float(lam)
    detail["lambda_feasible_rounds"] = int(sum(1 for t in trace if t["feasible"]))

    j1, local = _forward(j0, params)
    uv1, ok1 = _project(j1, float(focal), dw, dh)
    moved = list(MOVED_JOINTS)
    shift = np.linalg.norm(uv1[moved] - uv0[moved], axis=1)
    lengths = np.array([np.linalg.norm(j1[b] - j1[a]) - np.linalg.norm(j0[b] - j0[a])
                        for _ci, a, b in EDGES])
    angles = np.array([np.rad2deg(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)))
                       for r in local])
    _c, depth_after, _e, _l = _cost(params, j0, tz, float(focal), dh, dw,
                                    d, weight_map, 0.0, 0.0)
    detail.update({
        "joint_shift_px_max": float(shift.max()), "joint_shift_px_median": float(np.median(shift)),
        "bone_length_max_error_m": float(np.max(np.abs(lengths))),
        "angle_deg_max": float(angles.max()),
        "angles_deg": [float(x) for x in angles],
        "bones": [{"name": BONE_NAMES[i], "angle_deg": float(angles[i]),
                   "shift_px": float(shift[i])} for i in range(len(EDGES))],
        "depth_cost_after": float(depth_after),
        "depth_gain_pct": float((1.0 - depth_after / max(base_depth, 1e-12)) * 100.0),
        "lambda_trace": trace,
    })
    if not np.isfinite(depth_after) or depth_after >= base_depth * 0.98:
        detail["reason"] = "no_depth_gain"
        return j0.copy(), detail
    if float(shift.max()) > float(max_shift_px):
        detail["reason"] = "projection_budget_exceeded"
        return j0.copy(), detail
    detail["applied"] = True
    detail["reason"] = None
    return j1, detail
