"""SAM-3D 网格 + MHR-lod5 骨架（当前姿势）→ ASCII FBX（零第三方依赖）。

用途：把 SAM-3D 输出的 18439 顶点人形网格与 MHR 127 关节骨架的**当前姿势**
一起导出成可在 3ds Max / Maya / Blender 里旋转检视的单个 FBX。

约定：
- 网格顶点取 ``pose.json`` 的 ``world.mesh``（``ground_world_v1``，米、+Z 朝上）；
- 关节取 ``pose.json`` 的 ``world.generic_pose.pred_joint_coords``（127×3，米，
  与 ``world.mesh`` 同一世界系）；
- 骨架层级 / 骨名取自 MHR 资产 ``profile/mhr_rest_profile.v1.json``（由
  ``MHR-lod5.tpose.fbx`` 预处理生成，126 骨，root 为唯一根）；``joint_names[0]
  = body_world`` 只是关节槽位，没有对应骨；
- 骨骼局部旋转一律写 0：关节位置已经由 ``Lcl Translation`` 精确表达，导入端
  无论按哪种 Euler 顺序解释都不会让骨架错位（即使网格是静态的，骨架连线
  仍然完全正确）；
- 输出单位：厘米。FBX 原生单位就是厘米，所以 ``UnitScaleFactor=1``，同时把
  米制坐标乘 100 写成厘米值（不要再声明 100，否则导入端会再加一次单位换算）；
- 轴系：``UpAxis=2``（Z 上）、``FrontAxis=1 / FrontAxisSign=-1``（右手系，
  X×Z=-Y）。若写成 ``FrontAxisSign=+1`` 会构成左手系，导入端做镜像修正，
  表现为"姿势上下颠倒"。

网格不做蒙皮：SAM-3D 的顶点本身已经是目标姿势，静态挂到场景里即可；FBX 里
不带 Skin / Cluster，避免权重错误把网格拉坏。
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

UP_AXIS_Z = 2
UNIT_CM = 100.0          # 米 → 厘米（写入坐标时乘）
UNIT_SCALE_FACTOR = 1.0  # FBX 原生单位即厘米：1 单位 = 1 cm
BONE_SIZE_CM = 4.0

_ASSET_ROOT = (r"\\Tq0301016414\G\UGit\QingGeTools\ImagePose_Resources"
               r"\assets\mhr\profile")
SKELETON_FBX = (r"\\Tq0301016414\G\UGit\QingGeTools\ImagePose_Resources"
                r"\assets\mhr\skeleton\MHR-lod5.tpose.fbx")
# 仓库内副本：WSL 无法直连 UNC 共享，运行时优先用本地副本
_LOCAL_DIR = pathlib.Path(__file__).resolve().parent / "assets" / "mhr"
DEFAULT_REST_PROFILE = os.path.join(_ASSET_ROOT, "mhr_rest_profile.v1.json")
DEFAULT_SOURCE_PROFILE = os.path.join(_ASSET_ROOT, "mhr_source_profile.v1.json")

_ROOT_ENV = "MHR_REST_PROFILE"
_SRC_ENV = "MHR_SOURCE_PROFILE"
_RIG_CACHE: dict | None = None


def profile_paths() -> tuple[str, str]:
    """返回 (rest, source) profile 路径：env > 仓库副本 > UNC 共享。"""
    rest = os.environ.get(_ROOT_ENV)
    src = os.environ.get(_SRC_ENV)
    if not rest:
        rest = str(_LOCAL_DIR / "mhr_rest_profile.v1.json")
        if not os.path.exists(rest):
            rest = DEFAULT_REST_PROFILE
    if not src:
        src = str(_LOCAL_DIR / "mhr_source_profile.v1.json")
        if not os.path.exists(src):
            src = DEFAULT_SOURCE_PROFILE
    return rest, src


def get_rig() -> dict:
    """进程内缓存：骨架层级只需加载一次。"""
    global _RIG_CACHE
    if _RIG_CACHE is None:
        _RIG_CACHE = load_rig_profile()
    return _RIG_CACHE



def load_rig_profile(rest_path: str | None = None,
                     source_path: str | None = None) -> dict:
    """读取 MHR rest/source profile，返回骨名顺序、父子关系与关节索引。"""
    if rest_path is None or source_path is None:
        rest_default, src_default = profile_paths()
        rest_path = rest_path or rest_default
        source_path = source_path or src_default
    rest = json.loads(pathlib.Path(rest_path).read_text(encoding="utf-8"))
    src = json.loads(pathlib.Path(source_path).read_text(encoding="utf-8"))
    joint_names = list(src["joint_names"])
    bones = rest["bones"]
    missing = [n for n in bones if n not in joint_names]
    if missing:
        raise ValueError(f"rest profile 骨名不在 joint_names 中: {missing[:5]}")
    # 骨顺序沿用 joint_names（去掉 body_world 槽位），保证与关节坐标一一对应
    bone_order = [n for n in joint_names[1:] if n in bones]
    parents = {n: bones[n]["parent"] for n in bone_order}
    roots = [n for n in bone_order if parents[n] is None]
    if len(roots) != 1:
        raise ValueError(f"rest profile 期望单一根骨，实际 {roots}")
    return {
        "rest_path": rest_path,
        "source_path": source_path,
        "joint_names": joint_names,
        "bone_order": bone_order,
        "parents": parents,
        "index": {n: i for i, n in enumerate(joint_names)},
        "root": roots[0],
    }


def bone_positions(joints, rig: dict) -> list[tuple[str, str | None, np.ndarray]]:
    """按骨顺序取关节世界位置（米），返回 [(骨名, 父骨名, xyz)]。"""
    j = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    need = max(rig["index"][n] for n in rig["bone_order"]) + 1
    if j.shape[0] < need:
        raise ValueError(f"需要 {need} 个 MHR 关节，实际 {j.shape[0]}")
    return [(name, rig["parents"][name], j[rig["index"][name]])
            for name in rig["bone_order"]]


def _topo_order(rig: dict) -> list[str]:
    """父骨先于子骨的遍历顺序（profile 的 joint_names 顺序不保证层级）。"""
    order = rig["bone_order"]
    parents = rig["parents"]
    out: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        parent = parents[name]
        if parent is not None:
            visit(parent)
        seen.add(name)
        out.append(name)

    for name in order:
        visit(name)
    return out


def preserve_bone_lengths(joints, rig: dict, reference=None) -> np.ndarray:
    """把骨架每根骨的长度约束回 ``reference``（默认输入本身），仅改方向。

    骨骼吸附用的 LBS 位移场对相邻两个关节的混合权重不同，等价于各自套了不同的
    仿射，短骨（手指/腕）会被放大或压扁（相机近侧缩小、远侧放大）。这里自上而下
    做一次正向传播：子关节位置 = 父关节位置 + 单位方向 × 参考骨长，可以在保留
    修正后的姿态方向的同时逐骨恢复原始骨长。
    """
    target = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    j = target.copy()
    ref = target if reference is None else np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    index, parents = rig["index"], rig["parents"]
    for name in _topo_order(rig):
        parent = parents[name]
        if parent is None:
            continue
        a, b = index[name], index[parent]
        length = float(np.linalg.norm(ref[a] - ref[b]))
        delta = target[a] - target[b]
        dist = float(np.linalg.norm(delta))
        if length < 1e-9 or dist < 1e-9:
            continue
        j[a] = j[b] + delta * (length / dist)
    return j


def _n(value: float, nd: int = 4) -> str:
    text = f"{float(value):.{nd}f}"
    return "0." + "0" * nd if text.startswith("-0.") and float(text) == 0.0 else text


def _array(values, indent: str, per_line: int = 16, nd: int = 4) -> str:
    """FBX ASCII 数组：``*N { a: v,v,... }``（换行只是空白，多行更稳）。"""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    parts = [_n(v, nd) for v in flat.tolist()]
    head = f"{indent}*{len(parts)} {{\n{indent}\ta: "
    body = (",\n" + indent + "\t   ").join(
        ",".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line))
    return head + body + "\n" + indent + "}"


def _array_int(values, indent: str, per_line: int = 16) -> str:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    parts = [str(int(v)) for v in flat.tolist()]
    head = f"{indent}*{len(parts)} {{\n{indent}\ta: "
    body = (",\n" + indent + "\t   ").join(
        ",".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line))
    return head + body + "\n" + indent + "}"


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """面积加权顶点法线（面可能是三角/四边形，按扇形三角化）。"""
    v = np.asarray(verts, dtype=np.float64)
    normals = np.zeros_like(v)
    for face in faces:
        idx = np.asarray(face, dtype=np.int64)
        if len(idx) < 3:
            continue
        p0 = v[idx[0]]
        for k in range(1, len(idx) - 1):
            p1, p2 = v[idx[k]], v[idx[k + 1]]
            n = np.cross(p1 - p0, p2 - p0)
            normals[idx[0]] += n
            normals[idx[k]] += n
            normals[idx[k + 1]] += n
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return normals / length


def build_ascii_fbx(bones: list[tuple[str, str | None, np.ndarray]],
                    verts, faces, object_name: str = "SAM3D_MHR") -> str:
    """骨架 + 网格写成 FBX 7.4 ASCII 文本（输入为米，输出为厘米）。"""
    abs_pos = {name: pos for name, _p, pos in bones}
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3) * UNIT_CM
    normals = vertex_normals(v, faces)

    base_id = 1000000
    ids = {name: base_id + i for i, (name, _p, _x) in enumerate(bones)}
    attr_ids = {name: base_id + 100000 + i for i, (name, _p, _x) in enumerate(bones)}
    arm_id = base_id + 900000
    mesh_model_id = base_id + 950000
    geom_id = base_id + 960000
    stem = object_name.replace(" ", "_")

    model_count = len(bones) + 2          # Armature + 骨 + 网格
    attr_count = len(bones)               # 每根骨一个 LimbNode NodeAttribute
    total = 1 + model_count + attr_count + 1

    lines: list[str] = []
    add = lines.append
    add("; FBX 7.4.0 project file")
    add("; SAM-3D mesh + MHR-lod5 skeleton (current pose) by Phy-SIC ground_pipeline")
    add("FBXHeaderExtension:  {")
    add("\tFBXHeaderVersion: 1003")
    add("\tFBXVersion: 7400")
    add(f'\tCreator: "Phy-SIC mhr_mesh_fbx ({stem})"')
    add("\tCreationTimeStamp:  {")
    add("\t\tVersion: 1000")
    add("\t\tYear: 2026")
    add("\t\tMonth: 10")
    add("\t\tDay: 29")
    add("\t\tHour: 0")
    add("\t\tMinute: 0")
    add("\t\tSecond: 0")
    add("\t\tMillisecond: 0")
    add("\t}")
    add("}")
    add("GlobalSettings:  {")
    add("\tVersion: 1000")
    add("\tProperties70:  {")
    add(f'\t\tP: "UpAxis", "int", "Integer", "",{UP_AXIS_Z}')
    add('\t\tP: "UpAxisSign", "int", "Integer", "",1')
    add('\t\tP: "FrontAxis", "int", "Integer", "",1')
    add('\t\tP: "FrontAxisSign", "int", "Integer", "",-1')
    add('\t\tP: "CoordAxis", "int", "Integer", "",0')
    add('\t\tP: "CoordAxisSign", "int", "Integer", "",1')
    add('\t\tP: "OriginalUpAxis", "int", "Integer", "",2')
    add('\t\tP: "OriginalUpAxisSign", "int", "Integer", "",1')
    add(f'\t\tP: "UnitScaleFactor", "double", "Number", "",{_n(UNIT_SCALE_FACTOR)}')
    add(f'\t\tP: "OriginalUnitScaleFactor", "double", "Number", "",{_n(UNIT_SCALE_FACTOR)}')
    add('\t\tP: "TimeMode", "enum", "", "",11')
    add('\t\tP: "TimeSpanStart", "KTime", "Time", "",0')
    add('\t\tP: "TimeSpanStop", "KTime", "Time", "",46186158000')
    add("\t}")
    add("}")
    add("Documents:  {")
    add("\tCount: 1")
    add('\tDocument: 1000000000, "", "Scene" {')
    add("\t\tProperties70:  {")
    add('\t\t\tP: "SourceObject", "object", "", ""')
    add('\t\t\tP: "ActiveAnimStackName", "KString", "", "", ""')
    add("\t\t}")
    add("\t\tRootNode: 0")
    add("\t}")
    add("}")
    add("References:  {")
    add("}")
    add("Definitions:  {")
    add("\tVersion: 100")
    add(f"\tCount: {total}")
    add('\tObjectType: "GlobalSettings" {')
    add("\t\tCount: 1")
    add("\t}")
    add('\tObjectType: "Model" {')
    add(f"\t\tCount: {model_count}")
    add("\t}")
    add('\tObjectType: "NodeAttribute" {')
    add(f"\t\tCount: {attr_count}")
    add("\t}")
    add('\tObjectType: "Geometry" {')
    add("\t\tCount: 1")
    add("\t}")
    add("}")
    add("Objects:  {")

    # 骨架容器（Max: Dummy；Maya/Blender: 组节点）
    add(f'\tModel: {arm_id}, "Model::{stem}_Armature", "Null" {{')
    add("\t\tVersion: 232")
    add("\t\tProperties70:  {")
    add('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0.000000,0.000000,0.000000')
    add('\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0.000000,0.000000,0.000000')
    add('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1.000000,1.000000,1.000000')
    add('\t\t\tP: "RotationActive", "bool", "", "",1')
    add('\t\t\tP: "InheritType", "enum", "", "",1')
    add('\t\t\tP: "ScalingMax", "Vector3D", "Vector", "",0.000000,0.000000,0.000000')
    add("\t\t}")
    add("\t\tShading: T")
    add('\t\tCulling: "CullingOff"')
    add("\t}")

    # 骨：局部平移 = 世界位置差（局部旋转全 0，世界位置因此精确）
    for name, parent, pos in bones:
        local = (pos - (abs_pos[parent] if parent else np.zeros(3))) * UNIT_CM
        add(f'\tModel: {ids[name]}, "Model::{name}", "LimbNode" {{')
        add("\t\tVersion: 232")
        add("\t\tProperties70:  {")
        add('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",'
            f'{_n(local[0])},{_n(local[1])},{_n(local[2])}')
        add('\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0.000000,0.000000,0.000000')
        add('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1.000000,1.000000,1.000000')
        add('\t\t\tP: "RotationActive", "bool", "", "",1')
        add('\t\t\tP: "InheritType", "enum", "", "",1')
        add('\t\t\tP: "ScalingMax", "Vector3D", "Vector", "",0.000000,0.000000,0.000000')
        add('\t\t\tP: "DefaultAttributeIndex", "int", "Integer", "",0')
        add("\t\t}")
        add("\t\tShading: T")
        add('\t\tCulling: "CullingOff"')
        add("\t}")

    for name, _parent, _pos in bones:
        add(f'\tNodeAttribute: {attr_ids[name]}, "NodeAttribute::{name}",'
            ' "LimbNode" {')
        add('\t\tTypeFlags: "Skeleton"')
        add("\t\tProperties70:  {")
        add(f'\t\t\tP: "Size", "double", "Number", "",{_n(BONE_SIZE_CM)}')
        add("\t\t}")
        add("\t}")

    # 网格
    add(f'\tModel: {mesh_model_id}, "Model::{stem}_Mesh", "Mesh" {{')
    add("\t\tVersion: 232")
    add("\t\tProperties70:  {")
    add('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0.000000,0.000000,0.000000')
    add('\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0.000000,0.000000,0.000000')
    add('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1.000000,1.000000,1.000000')
    add('\t\t\tP: "RotationActive", "bool", "", "",1')
    add('\t\t\tP: "InheritType", "enum", "", "",1')
    add('\t\t\tP: "ScalingMax", "Vector3D", "Vector", "",0.000000,0.000000,0.000000')
    add('\t\t\tP: "DefaultAttributeIndex", "int", "Integer", "",0')
    add("\t\t}")
    add("\t\tShading: T")
    add('\t\tCulling: "CullingOff"')
    add("\t}")

    # PolygonVertexIndex：每个多边形最后一个索引取负（-(i+1)）
    pvi: list[int] = []
    for face in faces:
        idx = [int(i) for i in face]
        pvi.extend(idx[:-1])
        pvi.append(-idx[-1] - 1)

    add(f'\tGeometry: {geom_id}, "Geometry::{stem}_Mesh", "Mesh" {{')
    add("\t\tGeometryVersion: 124")
    add("\t\tVertices: " + _array(v, "\t\t", nd=4))
    add("\t\tPolygonVertexIndex: " + _array_int(pvi, "\t\t"))
    add('\t\tLayerElementNormal: 0 {')
    add("\t\t\tVersion: 101")
    add('\t\t\tName: ""')
    add('\t\t\tMappingInformationType: "ByVertice"')
    add('\t\t\tReferenceInformationType: "Direct"')
    add("\t\t\tNormals: " + _array(normals, "\t\t\t", nd=4))
    add("\t\t}")
    add("\t\tLayer: 0 {")
    add("\t\t\tVersion: 100")
    add("\t\t\tLayerElement:  {")
    add('\t\t\t\tType: "LayerElementNormal"')
    add("\t\t\t\tTypedIndex: 0")
    add("\t\t\t}")
    add("\t\t}")
    add("\t}")

    add("}")
    add("Connections:  {")
    add(f'\tC: "OO",{geom_id},{mesh_model_id}')
    add(f'\tC: "OO",{mesh_model_id},0')
    for name, _parent, _pos in bones:
        add(f'\tC: "OO",{attr_ids[name]},{ids[name]}')
    for name, parent, _pos in bones:
        parent_id = ids[parent] if parent else arm_id
        add(f'\tC: "OO",{ids[name]},{parent_id}')
    add(f'\tC: "OO",{arm_id},0')
    add("}")
    add("Takes:  {")
    add('\tCurrent: ""')
    add("}")
    return "\n".join(lines) + "\n"


def write_mesh_rig(path: str | pathlib.Path, verts, faces, joints,
                   object_name: str = "SAM3D_MHR", rig: dict | None = None,
                   note: str = "") -> dict:
    """写出「网格 + MHR 骨架」单个 FBX，返回摘要。"""
    rig = rig or get_rig()
    path = pathlib.Path(path)
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if not len(v):
        raise ValueError("网格顶点为空")
    bones = bone_positions(joints, rig)
    text = build_ascii_fbx(bones, v, faces, object_name=object_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    world_cm = v * UNIT_CM
    return {
        "path": str(path),
        "note": note,
        "bones": len(bones),
        "joints_used": int(len(np.asarray(joints).reshape(-1, 3))),
        "verts": int(len(v)),
        "faces": int(len(faces)),
        "z_range_cm": [float(world_cm[:, 2].min()), float(world_cm[:, 2].max())],
        "height_cm": float(world_cm[:, 2].max() - world_cm[:, 2].min()),
        "root": rig["root"],
        "skeleton_fbx": SKELETON_FBX,
        "rest_profile": rig["rest_path"],
        "bytes": path.stat().st_size,
    }
