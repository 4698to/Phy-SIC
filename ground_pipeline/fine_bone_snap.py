"""MHR-127 手掌/脚掌细骨吸附 —— ``BONE_SNAP`` 之上的补充阶段（默认关闭）。

COCO-17 的 BONE_SNAP 只修 8 根四肢骨，骨架止于手腕 9/10 与脚踝 15/16；网格上的
手掌、手指、脚掌、脚趾顶点只持有腕/踝权重，只能跟着腕/踝整体刚性平移，没有各自的
深度修正。本模块在这之后补一级：用 MHR-127 细骨链（wrist→五指、foot→ball）把
掌/指/跖/趾区域向 MoGe 深度表面推一小步。

每只手、每只脚各一组（共 4 组），组内流程：
1. 127 关节从 figure_v6 换回相机系（用 ``camera_points_to_figure_frame`` 的逆变换），
   再用与 COCO 级**同一个** DQS 变形场把细关节带到吸附后的位置 —— 细骨链的起点
   因此始终与吸附后的腕/踝重合，不会与网格脱节；
2. 细骨投到 depth 分辨率，先算组级深度比（骨段邻域像素 ``depth / z-buffer`` 中位数），
   再对每个关节在小窗口内单独取视比 —— 单组一个 k 在保骨长条件下等价于恒等旋转，
   只有逐关节 k 不同才会让指骨/趾骨产生真实的角度修正；
3. 每个关节沿自身射线缩放得到深度目标，逐骨「只取目标方向、保留原始骨长」重建整条
   链（与 ``bone_depth_snap._constrain_limb_lengths`` 同一套思路），骨长天然守恒；
4. 网格顶点按到各骨段的高斯权重混合逐骨刚体旋转（旋转极小，线性混合即可），
   远端按 ``INFLUENCE_CUT`` 渐隐，避免与 COCO 级变形撕裂。

开关 ``POSEEXTRACTOR_SAM3D_BONE_SNAP_FINE=1``（默认 off，关掉即完全回滚）。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

if __package__:
    from . import bone_depth_snap as bsnap
    from . import mhr_mesh_fbx as mhrmesh
else:
    import bone_depth_snap as bsnap
    import mhr_mesh_fbx as mhrmesh

FINE_ENV = "POSEEXTRACTOR_SAM3D_BONE_SNAP_FINE"
# 单组位移上限与深度比钳制：手/脚只有几十像素、深度噪声大，宁可少修
MAX_GROUP_SHIFT_M = 0.08
K_CLAMP = (0.94, 1.06)
MIN_PIXELS = 40
SIGMA_M = 0.045
INFLUENCE_CUT = 2.2

# (组名, 锚点关节名, 可用于配对的 COCO-17 腕/踝)
GROUP_SPEC = (
    ("lhand", "l_wrist", (9, 10)),
    ("rhand", "r_wrist", (9, 10)),
    ("lfoot", "l_foot", (15, 16)),
    ("rfoot", "r_foot", (15, 16)),
)

# figure_v6 → 相机系：camera_points_to_figure_frame 里 fig = (-x, z, -y)，
# 该矩阵正交，逆变换即转置 ⇒ cam = (-fx, -fz, fy)。
FIG_TO_CAM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

_TRUE = ("1", "on", "true", "yes")


def fine_snap_enabled() -> bool:
    return str(os.environ.get(FINE_ENV, "")).strip().lower() in _TRUE


def _skip(name: str) -> bool:
    return "twist" in name or "_null" in name


def fine_groups(rig: dict | None = None) -> list[dict]:
    """4 个肢体组：锚点关节、成员关节（父先子后）、组内父子边。"""
    rig = rig or mhrmesh.load_rig_profile()
    index = rig["index"]
    parents = rig["parents"]
    children: dict[str, list[str]] = {}
    for name in rig["bone_order"]:
        parent = parents.get(name)
        if parent:
            children.setdefault(parent, []).append(name)

    groups: list[dict] = []
    for gname, anchor, coco in GROUP_SPEC:
        if anchor not in index:
            continue
        members, stack = [anchor], [anchor]
        while stack:
            cur = stack.pop()
            for child in children.get(cur, []):
                if _skip(child):
                    continue
                members.append(child)
                stack.append(child)
        joints = [index[m] for m in members]
        edges = []
        for name in members[1:]:
            parent = parents.get(name)
            if parent in index:
                edges.append((index[parent], index[name]))
        groups.append({"name": gname, "anchor": index[anchor], "joints": joints,
                       "edges": edges, "coco": coco})
    return groups


def fine_camera_joints(pose: dict, base: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(camera 系 127 关节, world 系 127 关节)``；坐标系校验不过时返回空数组。

    - grounded（``world.status == "ok"``）：``world.generic_pose`` 已是 ground_world_v1，
      用 17 对 COCO 关节反解 world→camera 的刚体变换（残差 >1cm 视为不同源）。
    - rejected：world 块里没有 127 关节，按 figure_v6 → 相机系的标准逆变换换回，
      并用吸附前的 ``base`` 校验上半身 9 个关键点（鼻/眼/耳/肩/肘，与 MHR-70 同源）。
    """
    empty = np.zeros((0, 3))
    gp = pose.get("generic_pose") or {}
    fig127 = np.asarray(gp.get("pred_joint_coords") or [], dtype=np.float64).reshape(-1, 3)
    if fig127.shape[0] < 127:
        return empty, empty
    world = pose.get("world") or {}
    if world.get("status") == "ok":
        cam17 = np.asarray((pose.get("camera_space") or {}).get("keypoints_3d_coco17") or [],
                           dtype=np.float64).reshape(-1, 3)
        wgp = (world.get("generic_pose") or {}).get("pred_joint_coords")
        w17 = np.asarray(world.get("keypoints_3d_coco17") or [], dtype=np.float64).reshape(-1, 3)
        if cam17.shape != (17, 3) or wgp is None or w17.shape != (17, 3):
            return empty, empty
        rot = np.asarray(world.get("rotation"), dtype=np.float64)
        tvec = np.asarray(world.get("translation"), dtype=np.float64)
        off = w17 - (cam17 @ rot.T + tvec)
        if float(np.abs(off - off.mean(0)).max()) > 0.01:
            return empty, empty
        w127 = np.asarray(wgp, dtype=np.float64).reshape(-1, 3)
        return (w127 - tvec - off.mean(0)) @ rot, w127

    ref = base if base is not None else pose
    fig70 = np.asarray((ref.get("generic_pose") or {}).get("pred_keypoints_3d") or [],
                       dtype=np.float64).reshape(-1, 3)
    cam17_b = np.asarray((ref.get("camera_space") or {}).get("keypoints_3d_coco17") or [],
                         dtype=np.float64).reshape(-1, 3)
    if fig70.shape[0] < 9 or cam17_b.shape != (17, 3):
        return empty, empty
    cam_t = np.asarray(gp.get("pred_cam_t") or np.zeros(3), dtype=np.float64)
    check = fig70[:9] @ FIG_TO_CAM.T + cam_t
    if float(np.median(np.linalg.norm(check - cam17_b[:9], axis=1))) > 0.03:
        return empty, empty
    return fig127 @ FIG_TO_CAM.T + cam_t, empty


def _project(points: np.ndarray, focal: float, dw: int, dh: int) -> np.ndarray:
    """相机系 → depth 分辨率像素（无效点为 nan）。"""
    z = points[:, 2]
    ok = np.isfinite(points).all(axis=1) & (z > 1e-3)
    out = np.full((len(points), 2), np.nan)
    out[ok, 0] = focal * points[ok, 0] / z[ok] + dw / 2.0
    out[ok, 1] = focal * points[ok, 1] / z[ok] + dh / 2.0
    return out


def group_depth_stats(verts_cam: np.ndarray, groups: list[dict], fine_cam: np.ndarray,
                      depth: np.ndarray, focal: float, mask=None, faces=None,
                      min_pixels: int = MIN_PIXELS, clamp=K_CLAMP) -> list[dict]:
    """每组一个深度比 k（骨段邻域像素的 depth / 网格 z-buffer 中位数）。"""
    dz = np.asarray(depth, dtype=np.float64)
    v = np.asarray(verts_cam, dtype=np.float64).reshape(-1, 3)
    fail = {"ok": False, "reason": "bad_inputs", "k": 1.0, "n": 0, "median": 1.0}
    if dz.ndim != 2 or not len(v) or len(fine_cam) < 127:
        return [dict(fail) for _ in groups]

    dh, dw = dz.shape
    z = v[:, 2]
    front = np.isfinite(z) & (z > 1e-3)
    if np.count_nonzero(front) < 16:
        return [dict(fail, reason="too_few_vertices") for _ in groups]
    zsafe = np.where(front, z, 1.0)
    u = focal * v[:, 0] / zsafe + dw / 2.0
    w = focal * v[:, 1] / zsafe + dh / 2.0
    inb = front & (u >= 0) & (u <= dw - 1) & (w >= 0) & (w <= dh - 1)
    if np.count_nonzero(inb) < 16:
        return [dict(fail, reason="nothing_projected") for _ in groups]

    if faces is not None and len(np.asarray(faces).reshape(-1)):
        zbuf = bsnap._raster_zbuf(v, faces, focal, dh, dw)
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
            & (dz > bsnap.MIN_DEPTH_M) & (dz < bsnap.MAX_DEPTH_M))
    if m is not None:
        surf &= m
    if np.count_nonzero(surf) < min_pixels:
        return [dict(fail, reason="no_overlap") for _ in groups]

    pts = _project(np.asarray(fine_cam, dtype=np.float64).reshape(-1, 3), focal, dw, dh)
    out: list[dict] = []
    for group in groups:
        joints = [j for j in group["joints"] if np.isfinite(pts[j]).all()]
        if len(joints) < 2:
            out.append(dict(fail, reason="joints_off_screen"))
            continue
        pix = pts[joints]
        segs = [(pix[joints.index(a)], pix[joints.index(b)])
                for a, b in group["edges"] if a in joints and b in joints]
        if not segs:
            out.append(dict(fail, reason="no_edges"))
            continue
        span = float(np.max(np.linalg.norm(pix[:, None] - pix[None], axis=-1)))
        radius = float(np.clip(0.8 * span, 9.0, 60.0))

        lo = np.floor(pix.min(0) - radius).astype(int)
        hi = np.ceil(pix.max(0) + radius).astype(int)
        u0, v0 = max(lo[0], 0), max(lo[1], 0)
        u1, v1 = min(hi[0], dw - 1), min(hi[1], dh - 1)
        if u0 > u1 or v0 > v1:
            out.append(dict(fail, reason="outside_image"))
            continue
        sv, su = np.nonzero(surf[v0:v1 + 1, u0:u1 + 1])
        if not len(su):
            out.append(dict(fail, reason="no_pixels"))
            continue
        px = np.stack([su + u0, sv + v0], axis=1).astype(np.float64)
        near = np.zeros(len(px), dtype=bool)
        for pa, pb in segs:
            near |= bsnap._seg_dist(px, pa, pb) <= radius
        n = int(np.count_nonzero(near))
        if n < min_pixels:
            out.append(dict(fail, reason="too_few_pixels", n=n, radius_px=radius))
            continue
        ratio = dz[sv[near] + v0, su[near] + u0] / zbuf[sv[near] + v0, su[near] + u0]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if not len(ratio):
            out.append(dict(fail, reason="no_ratio", n=n))
            continue
        med = float(np.median(ratio))
        k_grp = float(np.clip(med, *clamp))
        out.append({"ok": True, "reason": None, "k": k_grp, "median": med,
                    "n": int(len(ratio)), "radius_px": radius,
                    "joint_k": _joint_k(pts, joints, dz, zbuf, surf, k_grp, clamp)})
    return out


JOINT_WIN_PX = 4.0
JOINT_MIN_PIX = 12


def _joint_k(pts: np.ndarray, joints: list[int], dz: np.ndarray, zbuf: np.ndarray,
             surf: np.ndarray, k_group: float, clamp) -> list[list[float]]:
    """逐关节深度比：关节点小窗口内 ``depth / z-buffer`` 中位数，样本不足用组值。"""
    dh, dw = dz.shape
    out: list[list[float]] = []
    for j in joints:
        u, w = pts[j]
        if not (np.isfinite(u) and np.isfinite(w)):
            out.append([int(j), k_group])
            continue
        u0 = max(int(np.floor(u - JOINT_WIN_PX)), 0)
        u1 = min(int(np.ceil(u + JOINT_WIN_PX)), dw - 1)
        v0 = max(int(np.floor(w - JOINT_WIN_PX)), 0)
        v1 = min(int(np.ceil(w + JOINT_WIN_PX)), dh - 1)
        if u0 > u1 or v0 > v1:
            out.append([int(j), k_group])
            continue
        sv, su = np.nonzero(surf[v0:v1 + 1, u0:u1 + 1])
        if len(su) < JOINT_MIN_PIX:
            out.append([int(j), k_group])
            continue
        pu, pv = su + u0, sv + v0
        near = (pu - u) ** 2 + (pv - w) ** 2 <= JOINT_WIN_PX ** 2
        if int(np.count_nonzero(near)) < JOINT_MIN_PIX:
            out.append([int(j), k_group])
            continue
        r = dz[pv[near], pu[near]] / zbuf[pv[near], pu[near]]
        r = r[np.isfinite(r) & (r > 0)]
        out.append([int(j), float(np.clip(np.median(r), *clamp)) if len(r) else k_group])
    return out


def apply_fine_snap(verts_cam: np.ndarray, base_kp: np.ndarray, new_kp: np.ndarray,
                    fine_cam: np.ndarray, stats: list[dict], groups: list[dict],
                    max_shift: float = MAX_GROUP_SHIFT_M):
    """按组把细骨链与掌/脚区域顶点推向深度目标；返回 (新顶点, 新细关节, 报告)。"""
    v = np.asarray(verts_cam, dtype=np.float64).reshape(-1, 3)
    base_kp = np.asarray(base_kp, dtype=np.float64).reshape(-1, 3)
    new_kp = np.asarray(new_kp, dtype=np.float64).reshape(-1, 3)
    fine = np.asarray(fine_cam, dtype=np.float64).reshape(-1, 3)
    report = {"enabled": True, "applied": False, "groups": []}

    if len(fine) < 127 or base_kp.shape != new_kp.shape:
        report["reason"] = "bad_inputs"
        return v, np.zeros((0, 3)), report

    # 先用与 COCO 级完全相同的 DQS 场把细关节带到吸附后的位置：细骨链起点因此与
    # 吸附后的腕/踝重合，后续只在本阶段内做「沿射线收放 + 保骨长」的微调。
    pre = bsnap.deform_points(fine, base_kp, new_kp)
    out = pre.copy()
    applied: list[tuple[dict, dict]] = []
    for group, st in zip(groups, stats):
        entry = {"name": group["name"], "ok": bool(st.get("ok")),
                 "k": float(st.get("k", 1.0)), "n": int(st.get("n", 0)),
                 "reason": st.get("reason")}
        report["groups"].append(entry)
        joints = group["joints"]
        anchor = group["anchor"]
        if not st.get("ok"):
            # 样本不足：整条链保持 COCO 级 DQS 带到的位置，本阶段不改
            continue
        kj = {int(i): float(k) for i, k in (st.get("joint_k") or [])}
        tgt = pre[joints] * np.array([kj.get(j, entry["k"]) for j in joints])[:, None]
        pos = {anchor: pre[anchor]}
        for a, b in group["edges"]:
            if a not in pos:
                pos[a] = pre[a]
            d = tgt[joints.index(b)] - tgt[joints.index(a)]
            n = float(np.linalg.norm(d))
            length = float(np.linalg.norm(pre[b] - pre[a]))
            if n < 1e-9 or length < 1e-9:
                pos[b] = pos[a] + (pre[b] - pre[a])
            else:
                pos[b] = pos[a] + d / n * length
        moved = np.stack([pos[j] for j in joints])
        shift = float(np.max(np.linalg.norm(moved - pre[joints], axis=1)))
        entry.update({"shift_m": shift,
                      "anchor_moved_m": float(np.linalg.norm(pre[anchor] - fine[anchor]))})
        if shift > max_shift:
            entry.update({"ok": False, "reason": "shift_capped"})
            continue
        out[joints] = moved
        applied.append((group, entry))

    if not applied:
        report["reason"] = report.get("reason") or "no_group_applied"
        # 仍返回 pre：至少与 COCO 级吸附后的网格相连，供可视化使用
        return v, out, report

    # 顶点：逐骨刚体（保长）旋转的高斯混合；距离用 pre 骨段（= 网格当前状态），
    # 远端按 INFLUENCE_CUT 渐隐，避免与 COCO 级变形撕裂。
    acc = np.zeros_like(v)
    total = np.zeros(len(v))
    dmin = np.full(len(v), np.inf)
    for group, _entry in applied:
        for a, b in group["edges"]:
            d0 = pre[b] - pre[a]
            len0 = float(np.linalg.norm(d0))
            d1 = out[b] - out[a]
            len1 = float(np.linalg.norm(d1))
            if len0 < 1e-6 or len1 < 1e-6:
                continue
            sigma = max(0.30 * len0, SIGMA_M)
            dn = bsnap._seg_dist(v, pre[a], pre[b]) / sigma
            w = np.exp(-0.5 * dn ** 2)
            rot = bsnap._rotation_from_to(d0 / len0, d1 / len1)
            acc += w[:, None] * ((v - pre[a]) @ rot.T + out[a])
            total += w
            np.minimum(dmin, dn, out=dmin)

    infl = np.clip((INFLUENCE_CUT - dmin) / (INFLUENCE_CUT - 1.0), 0.0, 1.0)
    hit = (total > 1e-9) & (infl > 0.0)
    if np.any(hit):
        blend = acc[hit] / total[hit][:, None]
        v = v.copy()
        v[hit] = v[hit] + infl[hit][:, None] * (blend - v[hit])

    report["applied"] = True
    report["max_vertex_shift_m"] = float(np.max(np.linalg.norm(v - verts_cam, axis=1)))
    return v, out, report
