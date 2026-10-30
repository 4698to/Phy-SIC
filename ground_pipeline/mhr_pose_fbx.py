"""把调整后的 MHR 骨架姿势写回 ``MHR-lod5.tpose.fbx``（保留原文件的模型 / 蒙皮结构 / 形态键）。

与 ``mhr_mesh_fbx`` 的区别：那个模块**新建**一份 ASCII FBX（SAM-3D 静态网格 + 自造骨架）；
本模块**改写官方资产**，只动 126 个 ``LimbNode`` 的 ``Lcl Translation`` / ``Lcl Rotation``，
``Geometry`` / ``Deformer`` / ``BlendShape`` / 连接关系一律原样保留，输出仍是二进制 FBX 7.7。

姿态换算（在 FBX 自身坐标系内，Y-up、厘米）：
- 先对原文件做 FK 得到 rest 世界位置与世界旋转；
- 目标关节 ``joints``（Z-up、米）按 ``s = 各骨长度比中位数`` 缩放，再映射
  ``(x, y, z)_zup → (x, z, -y)_fbx``，平移量取「脚底贴原 rest 最低点、root 水平对齐」；
- 每根骨取「rest 骨方向 → 目标骨方向」的最小旋转 D，``W_new = D · W_rest``；
- 局部旋转 ``L_new = W_parent_newᵀ · W_new``，按 FBX/Blender 实测约定
  ``L = PreRotation · Lcl``（欧拉按 ``Rz·Ry·Rx`` 分解，即 Blender 的 eulerXYZ）
  反解 ``Lcl Rotation = PreRotationᵀ · L`` 写回；
- 局部平移 ``t = W_parent_newᵀ · (p_new - p_parent_new)``（与原文件 FK 逐骨吻合到 1e-14）。

因此**骨骼世界位置严格等于目标关节**；只有 roll（自转）依赖欧拉约定。

注意：``body_mesh`` 的 Cluster 带 ``Indexes``/``Weights``（100 根骨有权重），
所以改骨骼会让网格随之蒙皮形变；其余 Geometry 无权重，只随 Node 平移。
"""

from __future__ import annotations

import json
import math
import pathlib
import struct
import zlib

import numpy as np

import mhr_mesh_fbx as mhr

MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_SCALAR = {"Y": ("<h", 2), "C": ("<B", 1), "I": ("<i", 4), "F": ("<f", 4),
           "D": ("<d", 8), "L": ("<q", 8)}
_ARR = {"f": "<f", "d": "<d", "l": "<q", "i": "<i", "b": "<B"}

_LOCAL_DIR = pathlib.Path(__file__).resolve().parent / "assets" / "mhr"
DEFAULT_TEMPLATE = _LOCAL_DIR / "MHR-lod5.tpose.fbx"

_EULER_ORDER = "ZYX"     # 本文件实测：欧拉按 Rz·Ry·Rx 合成（等同 Blender/FBX 的 eXYZ）


# ------------------------------------------------------------------ 二进制读写
class Node:
    """一个 FBX 记录：属性保留原始字节，便于零损失重写。"""

    __slots__ = ("name", "props", "raw_props", "children", "start", "end",
                 "null_end")

    def __init__(self, name, start):
        self.name = name
        self.props = []
        self.raw_props = b""
        self.children = []
        self.start = start
        self.end = 0
        self.null_end = False


def _header_len(ver: int) -> int:
    return 25 if ver >= 7500 else 13


def parse(buf, off: int, ver: int):
    hsz = _header_len(ver)
    fmt = "<QQQB" if ver >= 7500 else "<IIIB"
    nodes = []
    while off + hsz <= len(buf):
        end, nprops, plen, nlen = struct.unpack_from(fmt, buf, off)
        if end == 0 and nprops == 0 and plen == 0 and nlen == 0:
            return nodes, off + hsz
        p = off + hsz
        name = bytes(buf[p:p + nlen]).decode("utf-8", "replace")
        p += nlen
        node = Node(name, off)
        node.end = end
        p0 = p
        for _ in range(nprops):
            t = chr(buf[p])
            p += 1
            if t in _SCALAR:
                sf, sz = _SCALAR[t]
                node.props.append(struct.unpack_from(sf, buf, p)[0])
                p += sz
            elif t in ("S", "R"):
                n = struct.unpack_from("<I", buf, p)[0]
                p += 4
                raw = bytes(buf[p:p + n])
                node.props.append(raw.decode("utf-8", "replace") if t == "S" else raw)
                p += n
            elif t in _ARR:
                n, enc, clen = struct.unpack_from("<III", buf, p)
                p += 12
                raw = bytes(buf[p:p + clen])
                p += clen
                if enc == 1:
                    raw = zlib.decompress(raw)
                node.props.append(np.frombuffer(raw, dtype=_ARR[t], count=n).copy())
            else:
                raise ValueError(f"未知属性类型 {t!r} in {name!r}")
        node.raw_props = bytes(buf[p0:p])
        if p < end:
            node.children, _ = parse(buf, p, ver)
        # 本文件：有子节点的节点必带 25/13 字节空记录终止符；叶子一般不带
        # （唯一例外 References 这类空容器，用尾部是否全零判定并原样保留）。
        node.null_end = bool(node.children) or (
            end - hsz >= p and bytes(buf[end - hsz:end]) == b"\x00" * hsz)
        nodes.append(node)
        if end == 0:
            return nodes, p
        off = end
    return nodes, off


def load_bin(path):
    """返回 (ver, nodes, tail)。tail 为顶层结束符之后的原始尾部（footer）。"""
    buf = bytearray(pathlib.Path(path).read_bytes())
    if bytes(buf[:len(MAGIC)]) != MAGIC:
        raise ValueError(f"不是二进制 FBX: {path}")
    ver = struct.unpack_from("<I", buf, 23)[0]
    nodes, after = parse(buf, 27, ver)
    return ver, nodes, bytes(buf[after:])


def _enc_prop(v) -> bytes:
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(v, bytes):
        return b"R" + struct.pack("<I", len(v)) + v
    if isinstance(v, (np.ndarray, list, tuple)):
        arr = np.asarray(v)
        if arr.dtype == np.int64 or arr.dtype == np.int32:
            t, dt = "l", "<q"
        else:
            t, dt = "d", "<d"
        data = arr.astype(dt).tobytes()
        return t.encode() + struct.pack("<III", len(arr), 0, len(data)) + data
    if isinstance(v, bool):
        return b"C" + struct.pack("<B", int(v))
    if isinstance(v, int):
        return b"I" + struct.pack("<i", int(v))
    return b"D" + struct.pack("<d", float(v))


def _emit(node: Node, ver: int, out: bytearray) -> None:
    hsz = _header_len(ver)
    fmt = "<QQQB" if ver >= 7500 else "<IIIB"
    name = node.name.encode("utf-8")
    hpos = len(out)
    out += b"\x00" * hsz
    out += name
    out += node.raw_props
    for c in node.children:
        _emit(c, ver, out)
    if node.null_end:
        out += b"\x00" * hsz
    end = len(out)
    struct.pack_into(fmt, out, hpos, end, len(node.props), len(node.raw_props), len(name))


def save_bin(path, ver: int, nodes, tail: bytes) -> int:
    out = bytearray(MAGIC + struct.pack("<I", ver))
    for n in nodes:
        _emit(n, ver, out)
    out += b"\x00" * _header_len(ver)
    out += tail
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return len(out)


# ------------------------------------------------------------------ 骨架结构
def _objects(nodes):
    obj = next(n for n in nodes if n.name == "Objects")
    con = next(n for n in nodes if n.name == "Connections")
    models = {int(c.props[0]): c for c in obj.children if c.name == "Model"}
    names = {k: str(v.props[1]).split("\x00")[0] for k, v in models.items()}
    pairs = [(int(c.props[1]), int(c.props[2])) for c in con.children
             if c.props[0] == "OO"]
    parent = {c: p for c, p in pairs if c in models and p in models}
    limbs = {k: v for k, v in models.items() if v.props[2] == "LimbNode"}
    return models, names, parent, limbs


def _props70(model: Node) -> dict:
    out = {}
    for c in model.children:
        if c.name == "Properties70":
            for p in c.children:
                if p.name == "P" and len(p.props) >= 5:
                    out[p.props[0]] = p
    return out


def _vec(p70: dict, key: str) -> np.ndarray:
    p = p70.get(key)
    if p is None:
        return np.zeros(3)
    return np.asarray(p.props[4:7], dtype=np.float64)


def _set_vec(model: Node, key: str, values, sub: str = "") -> None:
    """改写（或补写）``Properties70`` 里的一个 Vector3D 属性。"""
    p70 = next((c for c in model.children if c.name == "Properties70"), None)
    if p70 is None:
        p70 = Node("Properties70", model.start)
        model.children.append(p70)
    vals = [float(v) for v in np.asarray(values, dtype=np.float64).reshape(3)]
    p = _props70(model).get(key)
    if p is not None:
        p.props[4:7] = vals
        head = [_enc_prop(v) for v in p.props[:4]]
        p.raw_props = b"".join(head + [_enc_prop(v) for v in vals])
        return
    p = Node("P", model.start)
    p.props = [key, key, sub, "A", *vals]
    p.raw_props = b"".join(_enc_prop(v) for v in p.props)
    p70.children.append(p)


# ------------------------------------------------------------------ 旋转工具（列向量约定）
def euler_to_mat(e, order: str = _EULER_ORDER) -> np.ndarray:
    x, y, z = [math.radians(float(v)) for v in e]
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    m = {"X": rx, "Y": ry, "Z": rz}
    out = np.eye(3)
    for c in order:
        out = out @ m[c]
    return out


def mat_to_euler_zyx(r: np.ndarray) -> np.ndarray:
    """R = Rz·Ry·Rx → (rx, ry, rz) 角度。"""
    sb = max(-1.0, min(1.0, -float(r[2, 0])))
    b = math.asin(sb)
    if abs(math.cos(b)) > 1e-8:
        a = math.atan2(float(r[2, 1]), float(r[2, 2]))
        c = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        a, c = 0.0, math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return np.degrees([a, b, c])


def rot_between(a, b) -> np.ndarray:
    """把单位向量 a 转到 b 的最小旋转（列向量约定）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


# ------------------------------------------------------------------ FK
def _fk(limbs, names, parent, euler_order=_EULER_ORDER):
    """按原文件节点值做 FK，返回 rest 世界位置(cm) 与世界旋转（列向量）。"""
    p70 = {k: _props70(v) for k, v in limbs.items()}
    pos, rot = {}, {}

    def visit(k):
        if k in pos:
            return pos[k], rot[k]
        t = _vec(p70[k], "Lcl Translation")
        pre = _vec(p70[k], "PreRotation")
        loc = euler_to_mat(_vec(p70[k], "Lcl Rotation"), euler_order)
        # FBX/Blender 实测：局部旋转 = PreRotation · Lcl Rotation
        local_rot = euler_to_mat(pre, euler_order) @ loc
        pk = parent.get(k)
        if pk is None or pk not in limbs:
            pos[k], rot[k] = t.copy(), local_rot
        else:
            pp, rp = visit(pk)
            # 列向量约定：W = W_parent · L，p = p_parent + R_parent · t
            pos[k] = pp + rp @ t
            rot[k] = rp @ local_rot
        return pos[k], rot[k]

    for k in list(limbs):
        visit(k)
    return pos, rot, p70


def _bind(limbs, nodes) -> dict:
    """Cluster ``TransformLink`` → ``{骨 id: (R 3x3 列向量, p 3)}``（绑定姿态）。

    Blender / 3ds Max 的蒙皮是相对**绑定姿态**做的，而模板文件里 27 根骨的
    ``Lcl Translation/Rotation`` 并不等于绑定姿态；只用节点 rest 摆姿势会让这些骨
    在蒙皮时多转一个「节点 rest ↔ 绑定」的差，网格于是被拧到错误方向。
    """
    obj = next(n for n in nodes if n.name == "Objects")
    con = next(n for n in nodes if n.name == "Connections")
    clusters = {int(c.props[0]): c for c in obj.children
                if c.name == "Deformer" and c.props[2] == "Cluster"}
    owner = {}
    for c in con.children:
        if c.props[0] != "OO":
            continue
        child, par = int(c.props[1]), int(c.props[2])
        if par in clusters:
            owner[par] = child
    out = {}
    for cid, bone in owner.items():
        if bone not in limbs:
            continue
        tl = next((ch for ch in clusters[cid].children
                   if ch.name == "TransformLink"), None)
        if tl is None:
            continue
        # FBX Matrix44 是行主序、平移在第 4 行；列向量约定下旋转取转置
        m = np.asarray(tl.props[0], dtype=np.float64).reshape(4, 4)
        out[bone] = (m[:3, :3].T.copy(), m[3, :3].copy())
    return out


def _topo(names, parent) -> list[int]:
    out, seen = [], set()

    def visit(k):
        if k in seen:
            return
        pk = parent.get(k)
        if pk is not None:
            visit(pk)
        seen.add(k)
        out.append(k)

    for k in names:
        visit(k)
    return out


def _zup_to_fbx(v: np.ndarray) -> np.ndarray:
    """Z-up 米 → Y-up（未乘尺度）: (x, y, z) → (x, z, -y)。"""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


# ------------------------------------------------------------------ 导出
def export_pose(src_fbx, out_fbx, joints, rig: dict | None = None,
                note: str = "", verify: bool = True,
                targets_json=None) -> dict:
    """把 ``joints``（127×3，世界系 Z-up 米）写成 ``src_fbx`` 的骨骼姿势。

    网格 / UV / Deformer / BlendShape 全部原样保留，只改 LimbNode 的局部变换。
    ``out_fbx`` 为空则只做校验不落盘。
    """
    rig = rig or mhr.get_rig()
    joints = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    ver, nodes, tail = load_bin(src_fbx)
    models, names, parent, limbs = _objects(nodes)

    # 骨名一致性：profile 里的每根骨都必须能在 FBX 里找到
    fb_by_name = {names[k]: k for k in limbs}
    order_names = [n for n in rig["bone_order"] if n in fb_by_name]
    missing = [n for n in rig["bone_order"] if n not in fb_by_name]
    if missing:
        raise ValueError(f"FBX 缺少 profile 骨: {missing[:5]}")
    idx = rig["index"]

    order_ids = [fb_by_name[n] for n in order_names]
    pos, rot, p70 = _fk(limbs, names, parent, _EULER_ORDER)
    bind = _bind(limbs, nodes)             # 绑定姿态（蒙皮真正相对的那套变换）
    bind_pos = {k: (bind[k][1] if k in bind else pos[k]) for k in limbs}
    bind_rot = {k: (bind[k][0] if k in bind else rot[k]) for k in limbs}
    topo = [k for k in _topo(order_ids, parent) if k in limbs]

    # 目标：profile rest（Z-up 米）↔ FBX rest（Y-up 厘米）求尺度
    prof = json.loads(pathlib.Path(mhr.profile_paths()[0]).read_text(encoding="utf-8"))["bones"]
    ratios = []
    for k in order_ids:
        name = names[k]
        pk = parent.get(k)
        if pk is None or pk not in fb_by_name:
            continue
        pname = rig["parents"][name]
        if pname is None or pname not in prof:
            continue
        rest_m = np.asarray(prof[name]["head_world_figure"], dtype=np.float64)
        rest_pm = np.asarray(prof[pname]["head_world_figure"], dtype=np.float64)
        lm = float(np.linalg.norm(rest_m - rest_pm))
        lf = float(np.linalg.norm(pos[k] - pos[pk]))
        if lm > 1e-6:
            ratios.append(lf / lm)
    scale = float(np.median(ratios)) if ratios else 100.0

    root_name = rig["root"]
    tgt = {}
    for name in order_names:
        j = joints[idx[name]]
        tgt[name] = scale * _zup_to_fbx(j)
    rest_pts = np.array([pos[k] for k in order_ids])
    tgt_pts = np.array([tgt[n] for n in order_names])
    tz = float(rest_pts[:, 1].min() - tgt_pts[:, 1].min())
    root_id = fb_by_name[root_name]
    off = np.array([pos[root_id][0] - tgt[root_name][0], 0.0,
                    pos[root_id][2] - tgt[root_name][2]])
    for name in order_names:
        tgt[name] = tgt[name] + np.array([off[0], tz, off[2]])
    tgt_id = {fb_by_name[n]: tgt[n] for n in order_names}

    if targets_json:
        # 供 Blender（或其它工具）复用同一份目标：坐标已在 FBX 自身空间
        # （Y-up、厘米），与 Armature 对象空间一致，无需再做轴/尺度换算。
        payload = {
            "frame": "mhr_lod5_fbx",
            "unit": "cm",
            "up": "Y",
            "source": str(src_fbx),
            "scale": scale,
            "root": root_name,
            "bones": order_names,
            "parents": {n: rig["parents"][n] for n in order_names},
            "targets": {n: [float(v) for v in tgt[n]] for n in order_names},
        }
        pathlib.Path(targets_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(targets_json).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 骨方向（指向子关节的平均方向，rest / target 各一份）
    kids = {k: [] for k in limbs}
    for k in limbs:
        pk = parent.get(k)
        if pk in kids:
            kids[pk].append(k)

    def direction(pid, pts):
        """目标骨方向：子关节向量的**和**（保留原长），与 Blender 参考实现一致。"""
        vs = []
        for c in kids.get(pid, []):
            d = pts[c] - pts[pid]
            if np.linalg.norm(d) > 1e-9:
                vs.append(d)
        if not vs:
            return None
        v = np.sum(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    swing = {}
    for k in limbs:
        if k not in tgt_id or k not in bind_rot:
            swing[k] = np.eye(3)
            continue
        # rest 骨方向必须是绑定姿态的 Y 轴（Blender 的 head→tail），
        # 不能用「子骨 bind 头平均方向」——两者在分叉/扭转骨上可差 80°+，
        # 会让 swing 带上错误的 roll（位置仍对，网格却被拧）。
        a = bind_rot[k][:, 1]
        b = direction(k, tgt_id)
        if b is None:
            b = a                       # 叶子骨：与参考一样退化为不转
        swing[k] = rot_between(a, b)

    # 逐骨写回（父先子后）
    new_pos, new_rot = {}, {}
    for k in topo:
        if k not in tgt_id:
            new_pos[k], new_rot[k] = pos[k], rot[k]
            continue
        # 新世界旋转 = 摇摆 D · 绑定姿态旋转（不是节点 rest 旋转）
        w = swing[k] @ bind_rot[k]
        pk = parent.get(k)
        if pk is not None and pk in new_rot:
            local = new_rot[pk].T @ w
            t = new_rot[pk].T @ (tgt_id[k] - new_pos[pk])
        else:
            local, t = w, tgt_id[k].copy()
        new_rot[k] = w
        new_pos[k] = tgt_id[k]
        # 写回：Lcl Rotation = PreRotationᵀ · L_local（欧拉按 ZYX=Rz·Ry·Rx 分解）
        pre = euler_to_mat(_vec(p70[k], "PreRotation"), _EULER_ORDER)
        _set_vec(models[k], "Lcl Rotation", mat_to_euler_zyx(pre.T @ local))
        _set_vec(models[k], "Lcl Translation", t)

    size = save_bin(out_fbx, ver, nodes, tail) if out_fbx else 0

    info = {
        "path": str(out_fbx) if out_fbx else "",
        "bytes": size,
        "bones": len(topo),
        "scale": scale,
        "offset_cm": [float(off[0]), float(tz), float(off[2])],
        "note": note,
    }
    if verify:
        info.update(_verify(out_fbx or src_fbx, tgt_id, names, parent, pos))
    return info


def _verify(path, tgt_id, names, parent, rest_pos) -> dict:
    ver, nodes, _tail = load_bin(path)
    models, names2, parent2, limbs = _objects(nodes)
    pos, _rot, _p70 = _fk(limbs, names2, parent2, _EULER_ORDER)
    err = [float(np.linalg.norm(pos[k] - tgt_id[k])) / 100.0
           for k in tgt_id if k in pos]
    return {
        "verify_max_err_cm": max(err) if err else 0.0,
        "verify_median_err_cm": float(np.median(err)) if err else 0.0,
    }
