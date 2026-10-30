"""C2 方案：按肢体骨修正 3D 关节位置，网格经线性混合蒙皮跟随骨骼。

与 C 方案（``mesh_front_snap``，逐顶点把网格推到 MoGe 前表面）的区别：
这里真正的修改变量是 **COCO-17 关节**，网格只是骨骼变换的从动结果，
因此骨骼叠图与网格始终自洽，也不会出现逐像素位移噪声造成的顶点撕裂。

流程：
1. 网格前表面与 MoGe-2 depth 的比值估计各骨端深度目标；
2. 深度目标沿射线求得后，保持肩/髋锚点，以目标相对方向恢复 COCO 原始肢体骨长；
3. 各骨旋转 + 平移（无长度缩放），网格和 MHR 关节共享双四元数变形场。
   空间混合权重变化仍可能改变局部距离，不宣称 MHR 细骨严格保长。
   ``POSEEXTRACTOR_SAM3D_BONE_SNAP_RIGID=0`` 可回退到旧的射线缩放 + 相似变换 LBS。

调用分两步，便于上层复用同一份 z-buffer 做「逐级阻尼 + 收益闸门」：
``bone_depth_stats`` 只统计 k（不变形），``bone_depth_apply`` 用 stats 应用。

可调抑制项（都在 apply 侧，默认关闭/宽松，便于按图回调）：
- ``leg_damp``：把大腿/小腿四根骨的 (k-1) 按比例缩小（0 表示该组骨完全不修），
  避免腿部深度比系统性偏小导致整条腿沿射线收缩、脚离地；
- ``shift_cap``：单关节位移上限（默认 0.25 m），远处关节的 |j|·(k-1) 会被放大，
  超过上限视为不可信并钳制。

旧模式仅在 ``POSEEXTRACTOR_SAM3D_BONE_SNAP=1`` 且
``POSEEXTRACTOR_SAM3D_BONE_SNAP_MODE=legacy`` 时启用；默认走原生 MHR 旋转拟合。
显式 ``POSEEXTRACTOR_SAM3D_ROTATION_FIT=1`` 优先，禁止此空间变形。
"""
from __future__ import annotations

import copy
import os

import numpy as np
from PIL import Image

BONE_SNAP_ENV = "POSEEXTRACTOR_SAM3D_BONE_SNAP"
# 默认骨长约束 + 网格/关节共享 DQS；设 0 回退到旧的射线目标 + 相似变换 LBS
RIGID_SNAP_ENV = "POSEEXTRACTOR_SAM3D_BONE_SNAP_RIGID"
BONE_K_CLAMP = (0.75, 1.35)
MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 8.0
MIN_SAMPLES = 120
# 顶点到骨轴的高斯半径倍数：<=1 满影响、>=INFLUENCE_CUT 完全不受影响
INFLUENCE_CUT = 2.5
# 单关节位移上限（米）：k 在远处会被 |j| 放大，超过该值的位移视为不可信
MAX_JOINT_SHIFT_M = 0.45
# BONES 里的大腿/小腿四根骨下标（leg_damp 只作用于它们，避免脚离地）
LOWER_BODY = (4, 5, 6, 7)
# 骨骼修正后允许的世界系最低点抬升上界（米）；超过则阻尼腿部修正以保落地
GROUND_LIFT_TOL = 0.2

# (名称, 起点 COCO-17, 终点 COCO-17) —— 与 ground_pose.COCO17_LIMB_BONES 一致
BONES = (
    ("l_upper_arm", 5, 7),
    ("l_forearm", 7, 9),
    ("r_upper_arm", 6, 8),
    ("r_forearm", 8, 10),
    ("l_thigh", 11, 13),
    ("l_shank", 13, 15),
    ("r_thigh", 12, 14),
    ("r_shank", 14, 16),
)

_TRUE = ("1", "on", "true", "yes")


def bone_snap_enabled() -> bool:
    """Spatial DQS/LBS is only available through the explicit legacy switch."""
    rotation = os.environ.get("POSEEXTRACTOR_SAM3D_ROTATION_FIT", "").strip().lower()
    return (str(os.environ.get(BONE_SNAP_ENV, "")).strip().lower() in _TRUE
            and os.environ.get("POSEEXTRACTOR_SAM3D_BONE_SNAP_MODE", "native").strip().lower() == "legacy"
            and rotation not in ("1", "on", "true", "yes", "mhr"))


def rigid_snap_enabled() -> bool:
    """骨长约束与共享 DQS 默认开启；设 0/off 回退旧的缩放 LBS。"""
    raw = str(os.environ.get(RIGID_SNAP_ENV, "1")).strip().lower()
    return raw not in ("0", "off", "false", "no")


def _seg_dist(points: np.ndarray, a, b) -> np.ndarray:
    """点到线段距离（points: (N,D)，D 由端点维度决定）。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=-1)


def _rotation_from_to(d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """把单位向量 d0 旋到 d1 的最小旋转（Rodrigues）。"""
    v = np.cross(d0, d1)
    c = float(d0 @ d1)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(d0, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(d0, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _raster_zbuf(verts: np.ndarray, faces, focal: float, dh: int, dw: int) -> np.ndarray:
    """三角形栅格化 z-buffer（逐像素重心插值取最近表面）。

    顶点级 z-buffer 只覆盖顶点落点（实测仅 ~8% 像素），会把逐骨深度比拉向 1；
    栅格化后覆盖率 ~85%，比值才反映真实表面差。
    """
    zbuf = np.full((dh, dw), np.inf, dtype=np.float64)
    z = verts[:, 2]
    front = np.isfinite(z) & (z > 1e-3)
    if not np.any(front):
        return zbuf
    zsafe = np.where(front, z, 1.0)
    u = focal * verts[:, 0] / zsafe + dw / 2.0
    w = focal * verts[:, 1] / zsafe + dh / 2.0
    ok = front & np.isfinite(u) & np.isfinite(w)
    F = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    F = F[np.all(ok[F], axis=1)]
    if not len(F):
        return zbuf
    tu, tw, tz = u[F], w[F], z[F]
    for k in range(len(F)):
        us, vs, zs = tu[k], tw[k], tz[k]
        u0 = max(int(np.floor(us.min())), 0)
        u1 = min(int(np.ceil(us.max())), dw - 1)
        v0 = max(int(np.floor(vs.min())), 0)
        v1 = min(int(np.ceil(vs.max())), dh - 1)
        if u0 > u1 or v0 > v1:
            continue
        gu, gv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        x1, y1 = us[0], vs[0]
        x2, y2 = us[1], vs[1]
        x3, y3 = us[2], vs[2]
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(den) < 1e-12:
            continue
        l1 = ((y2 - y3) * (gu - x3) + (x3 - x2) * (gv - y3)) / den
        l2 = ((y3 - y1) * (gu - x3) + (x1 - x3) * (gv - y3)) / den
        l3 = 1.0 - l1 - l2
        ins = (l1 >= -0.02) & (l2 >= -0.02) & (l3 >= -0.02)
        if not ins.any():
            continue
        zz = l1 * zs[0] + l2 * zs[1] + l3 * zs[2]
        np.minimum.at(zbuf, (gv[ins], gu[ins]), zz[ins])
    return zbuf


def bone_depth_stats(verts_cam: np.ndarray, joints_cam: np.ndarray,
                     depth: np.ndarray, focal: float, mask=None, faces=None, *,
                     clamp=BONE_K_CLAMP, min_samples: int = MIN_SAMPLES) -> dict:
    """估计每根骨两端的深度比 k（栅格化 z-buffer），不做任何变形。

    参数均为相机系（相机在原点，+z 向前），``focal`` 为 depth 分辨率下的像素焦距。
    返回 ``{"ok", "reason", "zbuf", "k_a", "k_b", "n_a", "n_b", "n_all", "valid"}``；
    ``k_*`` 为长度 8 的数组（``BONES`` 顺序）。拆成两步是为了让调用方能在不重算
    z-buffer 的前提下，用不同阻尼/钳制重新应用同一组 k。
    """
    v = np.asarray(verts_cam, dtype=np.float64).reshape(-1, 3)
    j = np.asarray(joints_cam, dtype=np.float64).reshape(-1, 3)
    dz = np.asarray(depth, dtype=np.float64)
    use_raster = faces is not None and len(np.asarray(faces).reshape(-1)) > 0
    zbuf_mode = "raster" if use_raster else "vertex"
    need = max(max(a, b) for _n, a, b in BONES)
    empty = np.zeros(len(BONES), dtype=bool)

    def _fail(reason: str) -> dict:
        return {"ok": False, "reason": reason, "zbuf": zbuf_mode,
                "k_a": np.ones(len(BONES)), "k_b": np.ones(len(BONES)),
                "n_a": np.zeros(len(BONES), dtype=np.int64),
                "n_b": np.zeros(len(BONES), dtype=np.int64),
                "n_all": np.zeros(len(BONES), dtype=np.int64), "valid": empty}

    if dz.ndim != 2 or j.ndim != 2 or j.shape[0] <= need or j.shape[1] < 3 or not len(v):
        return _fail("bad_inputs")

    dh, dw = dz.shape
    z = v[:, 2]
    front = np.isfinite(z) & (z > 1e-3)
    if np.count_nonzero(front) < min_samples:
        return _fail("too_few_vertices")
    zsafe = np.where(front, z, 1.0)
    u = focal * v[:, 0] / zsafe + dw / 2.0
    w = focal * v[:, 1] / zsafe + dh / 2.0
    inb = front & (u >= 0) & (u <= dw - 1) & (w >= 0) & (w <= dh - 1)
    if np.count_nonzero(inb) < min_samples:
        return _fail("nothing_projected")

    if use_raster:
        zbuf = _raster_zbuf(v, faces, focal, dh, dw)
    else:
        iu = np.rint(u[inb]).astype(np.int64)
        iv = np.rint(w[inb]).astype(np.int64)
        zbuf = np.full((dh, dw), np.inf, dtype=np.float64)
        np.minimum.at(zbuf, (iv, iu), z[inb])

    m = None
    if mask is not None:
        arr = np.asarray(mask)
        if arr.shape != (dh, dw):
            arr = np.asarray(Image.fromarray((arr.astype(bool).astype(np.uint8)) * 255)
                             .resize((dw, dh), Image.NEAREST)) > 127
        m = arr.astype(bool)

    surf = (np.isfinite(zbuf) & np.isfinite(dz)
            & (dz > MIN_DEPTH_M) & (dz < MAX_DEPTH_M))
    if m is not None:
        surf &= m
    if np.count_nonzero(surf) < min_samples:
        return _fail("no_overlap")

    # 关节按同一针孔模型投到 depth 分辨率，用于「像素 → 骨」归属
    jz = j[:, 2]
    okj = np.isfinite(j).all(axis=1) & (jz > 1e-3)
    ju = np.full(len(j), np.nan)
    jv = np.full(len(j), np.nan)
    ju[okj] = focal * j[okj, 0] / jz[okj] + dw / 2.0
    jv[okj] = focal * j[okj, 1] / jz[okj] + dh / 2.0

    pv, pu = np.nonzero(surf)
    ratio = dz[pv, pu] / zbuf[pv, pu]
    keep = np.isfinite(ratio) & (ratio > 0)
    pv, pu, ratio = pv[keep], pu[keep], ratio[keep]
    if not len(ratio):
        return _fail("no_ratio")

    pixels = np.stack([pu, pv], axis=1).astype(np.float64)
    bone_ids = np.full(len(pixels), -1, dtype=np.int64)
    bone_dist = np.full(len(pixels), np.inf)
    bone_t = np.zeros(len(pixels))
    for index, (_name, a, b) in enumerate(BONES):
        if not (okj[a] and okj[b]):
            continue
        pa = np.array([ju[a], jv[a]])
        pb = np.array([ju[b], jv[b]])
        dist = _seg_dist(pixels, pa, pb)
        half = max(0.20 * float(np.hypot(*(pb - pa))), 8.0)
        take = (dist <= half) & (dist < bone_dist)
        bone_ids[take] = index
        bone_dist[take] = dist[take]
        ab = pb - pa
        denom = float(ab @ ab)
        if denom > 1e-9:
            bone_t[take] = np.clip((pixels[take] - pa) @ ab / denom, 0.0, 1.0)

    # 每骨两端各自估 k：深度比沿骨并非常数（抬臂时肘端可比腕端偏近 5%），
    # 单骨单 k 会把远端关节一起推错，因此取靠近 a 的 1/4 段与靠近 b 的 1/4 段分别取中位数。
    valid = np.zeros(len(BONES), dtype=bool)
    n_a = np.zeros(len(BONES), dtype=np.int64)
    n_b = np.zeros(len(BONES), dtype=np.int64)
    n_all = np.zeros(len(BONES), dtype=np.int64)
    k_a = np.ones(len(BONES))
    k_b = np.ones(len(BONES))
    end_min = max(int(min_samples) // 4, 25)
    for index, (_name, _a, _b) in enumerate(BONES):
        sel = bone_ids == index
        n = int(np.count_nonzero(sel))
        n_all[index] = n
        if n < min_samples:
            continue
        r = ratio[sel]
        t = bone_t[sel]
        near_a = t < 0.25
        near_b = t >= 0.75
        na = int(np.count_nonzero(near_a))
        nb = int(np.count_nonzero(near_b))
        n_a[index] = na
        n_b[index] = nb
        med_all = float(np.median(r))
        ka = float(np.median(r[near_a])) if na >= end_min else med_all
        kbb = float(np.median(r[near_b])) if nb >= end_min else med_all
        k_a[index] = float(np.clip(ka, float(clamp[0]), float(clamp[1])))
        k_b[index] = float(np.clip(kbb, float(clamp[0]), float(clamp[1])))
        valid[index] = True
    if not valid.any():
        return {**_fail("no_bone_samples"), "valid": valid}
    return {"ok": True, "reason": None, "zbuf": zbuf_mode,
            "k_a": k_a, "k_b": k_b, "n_a": n_a, "n_b": n_b,
            "n_all": n_all, "valid": valid}


def _joint_targets(joints_cam: np.ndarray, stats: dict, leg_damp: float = 1.0,
                   shift_cap=MAX_JOINT_SHIFT_M):
    """按 stats 的 k 求关节目标位置（沿相机射线缩放 + 位移上限钳制）。

    返回 ``(new_j, kj, moved_joint, capped, ctx)``；``ctx`` 保存各骨端点 k 供报告。
    """
    j = np.asarray(joints_cam, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(stats["valid"], dtype=bool)
    k_a = np.asarray(stats["k_a"], dtype=np.float64).copy()
    k_b = np.asarray(stats["k_b"], dtype=np.float64).copy()
    n_a = np.asarray(stats["n_a"])
    n_b = np.asarray(stats["n_b"])
    if leg_damp != 1.0:
        damp = float(np.clip(leg_damp, 0.0, 1.0))
        for index in LOWER_BODY:
            k_a[index] = 1.0 + damp * (k_a[index] - 1.0)
            k_b[index] = 1.0 + damp * (k_b[index] - 1.0)

    # 1) 关节：同一关节被多根骨共享时，按各自端点样本数加权平均 k
    num = np.zeros(len(j))
    den = np.zeros(len(j))
    for index, (_name, a, b) in enumerate(BONES):
        if not valid[index]:
            continue
        w_a = float(n_a[index])
        w_b = float(n_b[index])
        num[a] += w_a * (k_a[index] - 1.0)
        num[b] += w_b * (k_b[index] - 1.0)
        den[a] += w_a
        den[b] += w_b
    kj = np.ones(len(j))
    moved_joint = den > 0
    kj[moved_joint] = 1.0 + num[moved_joint] / den[moved_joint]
    new_j = j.copy()
    new_j[moved_joint] = j[moved_joint] * kj[moved_joint, None]
    # 射线缩放的原始目标（位移上限钳制前）：第二级旋转精修用它作逐关节深度目标
    ray_target = new_j.copy()

    # 单关节位移上限：远距离（z 大）时 |j|·(k-1) 会被放大到不可信的程度，需硬钳制
    capped = 0
    if shift_cap is not None and np.any(moved_joint):
        delta = new_j[moved_joint] - j[moved_joint]
        dist = np.linalg.norm(delta, axis=1)
        over = dist > float(shift_cap)
        if np.any(over):
            capped = int(np.count_nonzero(over))
            scale = (float(shift_cap) / dist[over])[:, None]
            delta[over] *= scale
            new_j[moved_joint] = j[moved_joint] + delta
            kj = np.ones(len(j))
            kj[moved_joint] = np.linalg.norm(new_j[moved_joint], axis=1) \
                / np.maximum(np.linalg.norm(j[moved_joint], axis=1), 1e-9)
    if rigid_snap_enabled():
        new_j = _constrain_limb_lengths(j, new_j)
        moved_joint = np.linalg.norm(new_j - j, axis=1) > 1e-12
    ctx = {"k_a": k_a, "k_b": k_b, "n_a": n_a, "n_b": n_b, "valid": valid,
           "ray_target": ray_target}
    return new_j, kj, moved_joint, capped, ctx


def _constrain_limb_lengths(reference, targets):
    """肩/髋留在原位，父到子方向只读未修改的深度目标。"""
    ref = np.asarray(reference, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    out = target.copy()
    out[[5, 6, 11, 12]] = ref[[5, 6, 11, 12]]
    for _name, a, b in BONES:
        delta = target[b] - target[a]
        length = np.linalg.norm(ref[b] - ref[a])
        dist = np.linalg.norm(delta)
        if not np.isfinite(length) or not np.isfinite(dist):
            continue
        direction = delta / dist if dist > 1e-9 else (ref[b] - ref[a]) / max(length, 1e-9)
        out[b] = out[a] + direction * length
    return out


def _quat_mul(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scalar = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    vector = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([scalar, vector], axis=-1)


def _rotation_quat(rot):
    # 最大对角项分支同样覆盖 180 度旋转，使用 wxyz 顺序。
    candidates = np.array([1 + np.trace(rot), 1 + 2 * rot[0, 0] - np.trace(rot),
                           1 + 2 * rot[1, 1] - np.trace(rot), 1 + 2 * rot[2, 2] - np.trace(rot)])
    k = int(np.argmax(candidates))
    q = np.zeros(4)
    q[k] = 0.5 * np.sqrt(max(candidates[k], 0.0))
    den = 4 * q[k]
    if k == 0:
        q[1:] = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / den
    else:
        i = k - 1
        j, l = (i + 1) % 3, (i + 2) % 3
        q[0] = (rot[l, j] - rot[j, l]) / den
        q[j + 1] = (rot[j, i] + rot[i, j]) / den
        q[l + 1] = (rot[l, i] + rot[i, l]) / den
    return q / np.linalg.norm(q)


def _dqs_deform(points, joints_cam, new_joints, okj):
    """无缩放刚性变换的双四元数混合；恒等影响也参与混合而非事后 LBS。"""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    j = np.asarray(joints_cam, dtype=np.float64)
    new_j = np.asarray(new_joints, dtype=np.float64)
    weight = np.zeros((len(p), len(BONES) + 1))
    dn = np.full((len(p), len(BONES)), np.inf)
    real = np.zeros((len(BONES) + 1, 4))
    real[:, 0] = 1.0
    dual = np.zeros_like(real)
    for index, (_name, a, b) in enumerate(BONES):
        if not (okj[a] and okj[b]):
            continue
        d0, d1 = j[b] - j[a], new_j[b] - new_j[a]
        len0, len1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if min(len0, len1) < 1e-6:
            continue
        dn[:, index] = _seg_dist(p, j[a], j[b]) / max(0.30 * len0, 0.045)
        weight[:, index] = np.exp(-0.5 * dn[:, index] ** 2)
        rot = _rotation_from_to(d0 / len0, d1 / len1)
        real[index] = _rotation_quat(rot)
        translation = new_j[a] - rot @ j[a]
        dual[index] = 0.5 * _quat_mul(np.r_[0.0, translation], real[index])
    infl = np.clip((INFLUENCE_CUT - dn.min(axis=1)) / (INFLUENCE_CUT - 1.0), 0.0, 1.0)
    total = weight[:, :-1].sum(axis=1)
    weight[:, :-1] *= (infl / np.maximum(total, 1e-300))[:, None]
    weight[:, -1] = 1.0 - infl
    # 对每个点以最大权重旋转为基准，实部/双部同步翻转到同一半球。
    reference = real[np.argmax(weight, axis=1)]
    signs = np.where(reference @ real.T < 0.0, -1.0, 1.0)
    signed = weight * signs
    qr, qd = signed @ real, signed @ dual
    norm = np.linalg.norm(qr, axis=1, keepdims=True)
    qr, qd = qr / norm, qd / norm
    qd -= qr * np.sum(qr * qd, axis=1, keepdims=True)
    conj = qr * np.array([1.0, -1.0, -1.0, -1.0])
    translated = 2.0 * _quat_mul(qd, conj)[:, 1:]
    rotated = p + 2.0 * np.cross(qr[:, 1:], np.cross(qr[:, 1:], p) + qr[:, :1] * p)
    return rotated + translated, infl


def _lbs_deform(points: np.ndarray, joints_cam: np.ndarray,
                new_joints: np.ndarray, okj: np.ndarray):
    """旧版相似变换 LBS，仅供 env 回退。

    返回 ``(新点, influence)``；远处点按恒等影响处理。
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    j = np.asarray(joints_cam, dtype=np.float64).reshape(-1, 3)
    new_j = np.asarray(new_joints, dtype=np.float64).reshape(-1, 3)
    weight = np.zeros((len(p), len(BONES)))
    dn = np.full((len(p), len(BONES)), np.inf)
    moved = np.tile(p, (len(BONES), 1, 1))   # 默认恒等变换
    for index, (_name, a, b) in enumerate(BONES):
        if not (okj[a] and okj[b]):
            continue
        p0, p1 = j[a], j[b]
        len0 = float(np.linalg.norm(p1 - p0))
        if len0 < 1e-6:
            continue
        sigma = max(0.30 * len0, 0.045)
        dn[:, index] = _seg_dist(p, p0, p1) / sigma
        weight[:, index] = np.exp(-0.5 * dn[:, index] ** 2)
        q0, q1 = new_j[a], new_j[b]
        len1 = float(np.linalg.norm(q1 - q0))
        if len1 < 1e-6:
            continue
        rot = _rotation_from_to((p1 - p0) / len0, (q1 - q0) / len1)
        moved[index] = ((p - p0) @ rot.T) * (len1 / len0) + q0

    total = weight.sum(axis=1)
    blend = p.copy()
    hit = total > 1e-9
    if np.any(hit):
        blend[hit] = (np.einsum("nb,bnc->nc", weight, moved)[hit]
                      / total[hit][:, None])
    dmin = dn.min(axis=1)
    infl = np.clip((INFLUENCE_CUT - dmin) / (INFLUENCE_CUT - 1.0), 0.0, 1.0)
    return p + infl[:, None] * (blend - p), infl


def deform_points(points, joints, new_joints, rigid: bool | None = None):
    """任意点集共享网格变形场；DQS 不含缩放，但空间权重变化仍可能改变细骨长度。"""
    j = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    okj = np.isfinite(j).all(axis=1)
    use_rigid = rigid_snap_enabled() if rigid is None else rigid
    deform = _dqs_deform if use_rigid else _lbs_deform
    new, _infl = deform(points, j, new_joints, okj)
    return new


def bone_depth_apply(verts_cam: np.ndarray, joints_cam: np.ndarray, stats: dict, *,
                     leg_damp: float = 1.0, shift_cap: float = MAX_JOINT_SHIFT_M):
    """用深度目标修正关节（默认保 COCO 骨长）并由共享 DQS 带动网格。

    ``leg_damp`` 把大腿/小腿四根骨 (k-1) 按比例缩小（1=原样、0=腿部不动）。腿部深度比
    一旦系统性偏小，整条腿会沿射线收缩、脚离地，调用方可据此保落地。
    返回 ``(新顶点, 新关节, detail)``；未能生效时前两项为 ``None``。
    """
    detail: dict = {"enabled": True, "mode": "bone snap", "applied": False}
    if not stats or not stats.get("ok"):
        detail["reason"] = (stats or {}).get("reason") or "no_stats"
        detail["zbuf"] = (stats or {}).get("zbuf")
        return None, None, detail

    v = np.asarray(verts_cam, dtype=np.float64).reshape(-1, 3)
    j = np.asarray(joints_cam, dtype=np.float64).reshape(-1, 3)
    new_j, kj, moved_joint, capped, ctx = _joint_targets(j, stats, leg_damp, shift_cap)
    okj = np.isfinite(j).all(axis=1) & (j[:, 2] > 1e-3)
    deform = _dqs_deform if rigid_snap_enabled() else _lbs_deform
    new_v, infl = deform(v, j, new_j, okj)

    shift = np.linalg.norm(new_j - j, axis=1)
    touched_shift = shift[moved_joint] if np.any(moved_joint) else np.zeros(1)
    kj_moved = kj[moved_joint]
    spread = (float(kj_moved.max() - kj_moved.min())
              if len(kj_moved) >= 2 else float("inf"))
    k_a, k_b = ctx["k_a"], ctx["k_b"]
    n_a, n_b, valid = ctx["n_a"], ctx["n_b"], ctx["valid"]
    bones: list[dict] = []
    for index, (name, _a, _b) in enumerate(BONES):
        entry = {"name": name, "k": 0.5 * (float(k_a[index]) + float(k_b[index])),
                 "k_a": float(k_a[index]), "k_b": float(k_b[index]),
                 "n": int(stats["n_all"][index]),
                 "n_a": int(n_a[index]), "n_b": int(n_b[index]),
                 "valid": bool(valid[index])}
        bones.append(entry)
    detail.update({
        "applied": True,
        "zbuf": stats.get("zbuf"),
        "bones": bones,
        "deformation": "rigid_dqs" if rigid_snap_enabled() else "similarity_lbs",
        "coco_bone_length_max_error_m": float(max(
            abs(np.linalg.norm(new_j[b] - new_j[a]) - np.linalg.norm(j[b] - j[a]))
            for _name, a, b in BONES)),
        "k_min": float(kj_moved.min()) if len(kj_moved) else 1.0,
        "k_max": float(kj_moved.max()) if len(kj_moved) else 1.0,
        "k_spread": spread if np.isfinite(spread) else None,
        "joint_k": [float(x) for x in kj],
        "joint_ray_target": [[float(c) for c in pt] for pt in ctx["ray_target"]],
        "weighted_vertices": int(np.count_nonzero(infl > 0.0)),
        "max_shift_m": float(np.max(np.linalg.norm(new_v - v, axis=1))),
        "moved_joints": int(np.count_nonzero(moved_joint)),
        "joint_shift_max_m": float(touched_shift.max()),
        "joint_shift_median_m": float(np.median(touched_shift)),
    })
    if leg_damp != 1.0:
        detail["leg_damp"] = float(np.clip(leg_damp, 0.0, 1.0))
    if capped:
        detail["shift_capped_joints"] = capped
    return new_v, new_j, detail


def bone_depth_snap(verts_cam: np.ndarray, joints_cam: np.ndarray,
                    depth: np.ndarray, focal: float, mask=None, faces=None, *,
                    clamp=BONE_K_CLAMP, min_samples: int = MIN_SAMPLES):
    """``bone_depth_stats`` + ``bone_depth_apply`` 的一次性封装（无阻尼）。"""
    stats = bone_depth_stats(verts_cam, joints_cam, depth, focal, mask, faces,
                             clamp=clamp, min_samples=min_samples)
    return bone_depth_apply(verts_cam, joints_cam, stats)


def _person_camera_points(depth, focal, mask, max_pts=15000):
    dh, dw = depth.shape
    m = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize((dw, dh), Image.NEAREST)
    ) > 127
    step = max(1, int(np.sqrt(max(int(m.sum()), 1) / float(max_pts))))
    vs, us = np.mgrid[0:dh:step, 0:dw:step]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0.3) & (z < 8.0) & m[vs, us]
    zz = z[ok].astype(float)
    uu = us[ok].astype(float)
    vv = vs[ok].astype(float)
    return np.stack([(uu - dw / 2.0) * zz / focal,
                     (vv - dh / 2.0) * zz / focal, zz], axis=-1)


def _fit_median(pts, verts, max_query=2000, max_ref=6000):
    if not len(pts) or not len(verts):
        return float("nan")
    q = pts if len(pts) <= max_query else pts[np.linspace(0, len(pts) - 1, max_query).astype(int)]
    r = verts if len(verts) <= max_ref else verts[np.linspace(0, len(verts) - 1, max_ref).astype(int)]
    dist = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), 256):
        d = q[i:i + 256, None, :] - r[None, :, :]
        dist[i:i + 256] = np.sqrt((d * d).sum(-1)).min(1)
    return float(np.median(dist))


def apply_pose_bone_snap(p, torso, depth, mask, focal):
    """基于 torso 修正相机/世界几何，返回 (pose, torso, detail)。"""
    if __package__:
        from . import fine_bone_snap as finesnap
        from . import bone_rotation_fit as brot
        from . import limb_ik as lik
    else:
        import fine_bone_snap as finesnap
        import bone_rotation_fit as brot
        import limb_ik as lik

    torso_cam = torso.get("camera_space") or {}
    base_cam = np.asarray(torso_cam.get("vertices") or [],
                          dtype=np.float64).reshape(-1, 3)
    base_kp = np.asarray(torso_cam.get("keypoints_3d_coco17") or [],
                         dtype=np.float64).reshape(-1, 3)
    torso_world = torso.get("world") or {}
    base_world = np.asarray(((torso_world.get("mesh") or {}).get("vertices")) or [],
                            dtype=np.float64).reshape(-1, 3)
    base_wkp = np.asarray(torso_world.get("keypoints_3d_coco17") or [],
                          dtype=np.float64).reshape(-1, 3)
    base_faces = np.asarray(torso_cam.get("faces") or [], dtype=np.int64)
    stats = bone_depth_stats(
        base_cam, base_kp, depth, focal, mask, faces=base_faces)
    if not stats.get("ok"):
        return p, torso, {"enabled": True, "applied": False,
                          "reason": stats.get("reason") or "no_bone_samples",
                          "zbuf": stats.get("zbuf")}

    world = p.get("world") or {}
    world_ok = (world.get("status") == "ok"
                and len(base_world) == len(base_cam) and len(base_cam))
    rot = None
    z0 = 0.0
    if world_ok:
        rot = np.asarray(world.get("rotation") or torso_world.get("rotation"),
                         dtype=np.float64)
        z0 = float(np.percentile(base_world[:, 2], 0.5))

    # 深度比里腿部系统性偏小会让整条腿沿射线收缩、脚离地；按 1→0 依次阻尼腿部，
    # 直到世界系最低点抬升 <= GROUND_LIFT_TOL 为止（相机系仍保留手臂等修正）。
    new_cam = new_kp = info = None
    lift = 0.0
    for damp in (1.0, 0.5, 0.25, 0.0):
        new_cam, new_kp, info = bone_depth_apply(
            base_cam, base_kp, stats, leg_damp=damp)
        if new_cam is None or not info.get("applied"):
            return p, torso, info
        if not world_ok:
            break
        moved_world = base_world + (new_cam - base_cam) @ rot.T
        lift = float(np.percentile(moved_world[:, 2], 0.5)) - z0
        if lift <= GROUND_LIFT_TOL:
            break

    # 相机系点云只需算一次，后面 IK / 旋转两级复用同一份做贴合闸门
    pts_cam = None
    d_before = d_after = float("nan")
    if mask is not None and np.any(mask):
        pts_cam = _person_camera_points(depth, focal, mask)
        d_before = _fit_median(pts_cam, base_cam)

    # 贴合收益闸门：蒙皮后用同一份相机系点云复核最近距离中位数，变差超过 2%
    # 说明骨骼修正把人整体推离点云，直接回退基线（可撤回）。开着腿部 IK 时该指标
    # 只记录不门控：射线缩放靠改骨长把模型深度误差藏进网格，指标偏好它并不代表
    # 姿势更对；此时只剩落地闸门在起作用。
    if pts_cam is not None:
        d_after = _fit_median(pts_cam, new_cam)
        info["fit_before_cm"] = float(d_before * 100.0)
        info["fit_after_cm"] = float(d_after * 100.0)
        info["fit_gate"] = "record_only" if lik.limb_ik_enabled() else "enforced"
        if (not lik.limb_ik_enabled() and np.isfinite(d_before)
                and np.isfinite(d_after) and d_after > d_before * 1.02):
            info.update({"applied": False, "reason": "no_fit_gain"})
            return p, torso, info

    # 腿部换成保骨长的 two-bone IK：射线缩放能命中深度但会改骨长，这里让踝落在
    # 自身射线 × 点云深度上、骨长守恒。腿骨正确性优先，网格最近距离只作为参考
    # 指标记录（射线缩放会靠改骨长把模型深度误差藏进网格，指标偏好它并不代表
    # 姿势更对）；只有抬升超过落地容差（脚被抬离地面）才退回射线缩放结果。
    if lik.limb_ik_enabled() and new_kp is not None and info.get("applied"):
        ik_kp, ik_info = lik.apply_leg_ik(
            base_kp, new_kp,
            targets=np.asarray(info.get("joint_ray_target") or [], dtype=np.float64),
            focal=focal)
        if ik_info.get("applied"):
            ik_cam = deform_points(base_cam, base_kp, ik_kp)
            lift_ik = 0.0
            if world_ok:
                moved = base_world + (ik_cam - base_cam) @ rot.T
                lift_ik = float(np.percentile(moved[:, 2], 0.5)) - z0
            if world_ok and lift_ik > GROUND_LIFT_TOL:
                ik_info.update({"applied": False, "reason": "ground_lift"})
            else:
                ik_info["ground_lift_m"] = float(lift_ik)
                if pts_cam is not None:
                    ik_info["fit_before_cm"] = info.get("fit_after_cm")
                    d_after = _fit_median(pts_cam, ik_cam)
                    ik_info["fit_after_cm"] = float(d_after * 100.0)
                    info["fit_after_cm"] = ik_info["fit_after_cm"]
                new_kp, new_cam = ik_kp, ik_cam
        info["limb_ik"] = ik_info

    # 第二级：旋转参数化精修。第一级沿射线缩放会改变 3D 骨长，这里改成绕父关节的
    # swing 旋转（2 DOF/骨，twist=0），骨长天然不变；深度目标沿用第一级的逐关节
    # 射线目标深度，2D 投影偏移按预算（默认 2 px）约束，超预算或深度无改善即回退。
    if brot.rotation_refine_enabled() and info.get("applied"):
        target = np.asarray(info.get("joint_ray_target") or [], dtype=np.float64)
        if target.shape == new_kp.shape and len(target):
            refined, rot_info = brot.rotation_refine(
                new_kp, depth, focal, mask,
                target_z=target[:, 2])
            if rot_info.get("applied"):
                moved_cam = deform_points(new_cam, new_kp, refined)
                if pts_cam is not None and np.isfinite(d_after):
                    d_rot = _fit_median(pts_cam, moved_cam)
                    rot_info["fit_before_cm"] = float(d_after * 100.0)
                    rot_info["fit_after_cm"] = float(d_rot * 100.0)
                    if np.isfinite(d_rot) and d_rot > d_after * 1.02:
                        rot_info.update({"applied": False, "reason": "no_fit_gain"})
                if rot_info.get("applied"):
                    new_cam, new_kp = moved_cam, refined
                    info["fit_after_cm"] = rot_info.get("fit_after_cm", info.get("fit_after_cm"))
            info["rotation"] = rot_info

    # 补充级：手掌/脚掌细骨吸附（默认关闭）。COCO-17 止于腕/踝，网格上的掌指/跖趾
    # 顶点只持有腕/踝权重，只能整体平移；这里用 MHR-127 细骨链把它们再向深度表面
    # 推一小步（纯旋转 + 保骨长，见 fine_bone_snap）。
    fine_cam, fine_w = finesnap.fine_camera_joints(torso)
    if not len(fine_cam):
        fine_cam, fine_w = finesnap.fine_camera_joints(p)
    fine_cam_new = fine_cam
    if not finesnap.fine_snap_enabled() and len(fine_cam):
        fine_cam_new = deform_points(fine_cam, base_kp, new_kp)
    fine_world_new = None
    if finesnap.fine_snap_enabled() and info.get("applied"):
        if len(fine_cam):
            groups = finesnap.fine_groups()
            fstats = finesnap.group_depth_stats(
                base_cam, groups, fine_cam, depth, focal,
                mask, faces=base_faces)
            d_pre = (_fit_median(pts_cam, new_cam)
                     if pts_cam is not None else float("nan"))
            new_cam, fine_cam_new, fine_info = finesnap.apply_fine_snap(
                new_cam, base_kp, new_kp, fine_cam, fstats, groups)
            if pts_cam is not None and np.isfinite(d_pre):
                # 掌/脚只占整图很小面积，全局最近距离几乎不动，这里只做记录不作闸门
                fine_info["fit_before_cm"] = float(d_pre * 100.0)
                fine_info["fit_after_cm"] = float(_fit_median(pts_cam, new_cam) * 100.0)
            if world_ok and len(fine_w):
                fine_world_new = fine_w + (fine_cam_new - fine_cam) @ rot.T
            info["fine"] = fine_info

    pose = copy.deepcopy(p)
    cam_blk = pose.setdefault("camera_space", {})
    cam_blk["vertices"] = new_cam.tolist()
    cam_blk["keypoints_3d_coco17"] = new_kp.tolist()
    cam_blk["keypoints_3d_fine"] = fine_cam_new.tolist()
    world = pose.get("world") or {}
    if world_ok:
        new_world = base_world + (new_cam - base_cam) @ rot.T
        new_wkp_world = None
        if len(base_wkp) == len(base_kp) and len(base_kp):
            new_wkp_world = base_wkp + (new_kp - base_kp) @ rot.T
        # 落地复校：骨骼修正可能把最低点压到落地位置以下（脚略陷地），
        # 仅在变低时沿世界 z 整体抬回，保持落地不穿地。
        # 与既有落地量一致：只改世界系，相机系保持原位（相机视角才能与原图重合）。
        z1 = float(np.percentile(new_world[:, 2], 0.5))
        dz = max(0.0, z0 - z1)
        if dz > 1e-4:
            new_world = new_world + np.array([0.0, 0.0, dz])
            if new_wkp_world is not None:
                new_wkp_world = new_wkp_world + np.array([0.0, 0.0, dz])
            if fine_world_new is not None:
                fine_world_new = fine_world_new + np.array([0.0, 0.0, dz])
        info["landing_fix_m"] = float(dz)
        info["ground_lift_m"] = float(z1 - z0)
        if new_world is not None:
            world.setdefault("mesh", {})["vertices"] = new_world.tolist()
        if new_wkp_world is not None:
            world["keypoints_3d_coco17"] = new_wkp_world.tolist()
        if fine_world_new is not None:
            world["keypoints_3d_fine"] = fine_world_new.tolist()
    fit = dict(pose.get("scale_fit") or {})
    fit["bonesnap"] = info
    pose["scale_fit"] = fit
    return pose, torso, info
