"""C 方案：把网格顶点沿相机射线推到 MoGe-2 实测前表面（非刚性贴合）。

与 limb 档的区别：limb 逐骨取一个深度比 k_b 后缩放整根骨，C 方案不给骨、
不做刚性假设，而是逐像素比较「网格前表面 zbuf」与「MoGe 实测深度 depth」，
得到该射线方向上的深度差 delta，再让该列顶点沿射线平移 delta：

    k = 1 + delta / z,  v' = v * k          （相机系，原点即光心）

delta 的使用有三道闸门：
  1. 只在人物掩码内统计；
  2. ``|delta| > trust_m``（默认 0.25 m）视为遮挡/背景穿透，不采信；
  3. 空缺像素用已测像素归一化卷积扩散补齐，再做高斯平滑（默认 5 px），
     避免逐顶点跳变把网格撕开。
逐顶点 k 仍 clamp 在 0.90~1.10，单点最多修正 10% 的相机深度。

可撤回：默认关闭，仅当 ``POSEEXTRACTOR_SAM3D_FRONT_SNAP=1`` 时启用；
渲染脚本里只多了一个 env 分支，删掉本模块 + 该分支即完全回滚。
off/moge/torso/limb 各档行为与本模块无关，不受影响。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

FRONT_SNAP_ENV = "POSEEXTRACTOR_SAM3D_FRONT_SNAP"
FRONT_SNAP_CLAMP = (0.90, 1.10)
FRONT_SNAP_TRUST_M = 0.25
FRONT_SNAP_SMOOTH_PX = 5.0
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 8.0


def front_snap_enabled() -> bool:
    return os.environ.get(FRONT_SNAP_ENV, "").strip().lower() in (
        "1", "on", "true", "yes")


def _fill_smooth(delta: np.ndarray, ok: np.ndarray, smooth_px: float):
    """归一化卷积：先由已测像素向空缺处扩散，再整体高斯平滑。"""
    val = np.where(ok, delta, 0.0).astype(np.float32)
    wgt = ok.astype(np.float32)
    out = val.copy()
    have = ok.copy()
    for sigma in (2.0, 6.0, 18.0):
        num = cv2.GaussianBlur(val, (0, 0), sigma)
        den = cv2.GaussianBlur(wgt, (0, 0), sigma)
        fill = (~have) & (den > 1e-3)
        if not fill.any():
            continue
        out[fill] = num[fill] / den[fill]
        wgt = np.where(fill, 1.0, wgt).astype(np.float32)
        have |= fill
    w = have.astype(np.float32)
    num = cv2.GaussianBlur(out * w, (0, 0), smooth_px)
    den = cv2.GaussianBlur(w, (0, 0), smooth_px)
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return smooth.astype(np.float32), have


def front_snap_vertices(verts_cam, depth: np.ndarray, focal: float, mask=None, *,
                        trust_m: float = FRONT_SNAP_TRUST_M,
                        clamp=FRONT_SNAP_CLAMP,
                        smooth_px: float = FRONT_SNAP_SMOOTH_PX):
    """把相机系顶点沿射线推到 MoGe 实测前表面。

    返回 ``(新顶点, detail)``；detail 供渲染面板展示与验收记录。
    顶点数与输入一致，未被采信的顶点原样返回。
    """
    v = np.asarray(verts_cam, dtype=np.float64).reshape(-1, 3)
    detail = {"enabled": True, "applied": False, "mode": "front snap",
              "weighted_vertices": 0, "max_shift_m": 0.0,
              "k_min": None, "k_max": None, "trust_m": float(trust_m)}
    if not len(v):
        return v.copy(), {**detail, "reason": "no_vertices"}
    if depth.ndim != 2:
        return v.copy(), {**detail, "reason": "bad_depth"}
    dh, dw = depth.shape
    dz = np.asarray(depth, dtype=np.float32)

    z = v[:, 2]
    front = np.isfinite(z) & (z > 1e-3)
    zsafe = np.where(front, z, 1.0)
    u = focal * v[:, 0] / zsafe + dw / 2.0
    w = focal * v[:, 1] / zsafe + dh / 2.0
    inb = (front & (u >= 0) & (u <= dw - 1) & (w >= 0) & (w <= dh - 1))
    if not inb.any():
        return v.copy(), {**detail, "reason": "nothing_projected"}

    m = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (dh, dw):
            m = cv2.resize(m.astype(np.uint8), (dw, dh),
                           interpolation=cv2.INTER_NEAREST) > 0

    # 网格前表面 z 缓冲（每条射线只保留最近顶点）
    iu = np.clip(np.rint(u), 0, dw - 1).astype(np.int64)
    iv = np.clip(np.rint(w), 0, dh - 1).astype(np.int64)
    zbuf = np.full((dh, dw), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (iv[inb], iu[inb]), z[inb].astype(np.float32))

    surf = (np.isfinite(zbuf) & np.isfinite(dz)
            & (dz > MIN_DEPTH_M) & (dz < MAX_DEPTH_M))
    if m is not None:
        surf &= m
    delta = np.full((dh, dw), np.nan, dtype=np.float32)
    np.subtract(dz, zbuf, out=delta, where=surf)
    trusted = surf & np.isfinite(delta) & (np.abs(delta) <= trust_m)
    if not trusted.any():
        return v.copy(), {**detail, "reason": "no_trusted_pixels"}
    smooth, have = _fill_smooth(delta, trusted, smooth_px)
    if m is not None:
        have &= m

    su = np.where(inb, u, -1.0).astype(np.float32).reshape(-1, 1)
    sv = np.where(inb, w, -1.0).astype(np.float32).reshape(-1, 1)
    samp = cv2.remap(smooth, su, sv, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0).reshape(-1)
    cov = cv2.remap(have.astype(np.float32), su, sv, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0).reshape(-1)

    use = inb & (cov > 0.5) & np.isfinite(samp)
    k = np.ones(len(v), dtype=np.float64)
    k[use] = np.clip(1.0 + samp[use] / z[use], float(clamp[0]), float(clamp[1]))
    moved = use & (np.abs(k - 1.0) > 1e-4)
    if not moved.any():
        return v.copy(), {**detail, "reason": "no_vertex_moved",
                          "coverage": float(use.mean())}

    new = v.copy()
    new[moved] = v[moved] * k[moved, None]
    shift = np.linalg.norm(new - v, axis=1)
    return new, {
        **detail,
        "applied": True,
        "weighted_vertices": int(moved.sum()),
        "max_shift_m": float(shift.max()),
        "k_min": float(k[moved].min()),
        "k_max": float(k[moved].max()),
        "coverage": float(use.mean()),
        "trusted_pct": float(trusted.sum() / max(int(surf.sum()), 1)),
    }
