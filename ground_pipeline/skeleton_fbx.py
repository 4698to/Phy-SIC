"""COCO-17 关节 → ASCII FBX 骨架（零第三方依赖）。

用途：把骨骼吸附（C2）后的 SAM-3D 人形骨架导出为可在 Blender / Maya / 3ds Max
里自由旋转检视的 3D 资产。只写骨架（LimbNode 链），不带网格与蒙皮。

坐标约定：
- 输入 ``joints`` 为世界系米制坐标（``ground_world_v1``：地面 z=0、+Z 朝上）；
  无地面帧传入 ``camera_v1`` 也可，此时看到的即是相机系摆位。
- 输出按 FBX 原生单位厘米写坐标（UP_AXIS_Z=2 声明 Z 轴向上，FBX 原生单位
  即厘米，故 UnitScaleFactor=1 而不是 100；声明 100 会让导入端再乘一次
  单位换算，表现为骨骼被缩放）。

骨架结构（骨节点位置即关节位置，父子关系决定连线）：
    root(pelvis)
    ├── l_hip → l_knee → l_ankle
    ├── r_hip → r_knee → r_ankle
    └── spine → chest
                ├── neck → head(nose)
                ├── l_shoulder → l_elbow → l_wrist
                └── r_shoulder → r_elbow → r_wrist
其中 spine = 0.5(root, chest)、chest = 0.5(l_shoulder, r_shoulder)、
neck = 0.5(chest, head)；COCO-17 本身没有这三个中间点，这里只为让骨架成链。
"""

from __future__ import annotations

import pathlib

import numpy as np

UP_AXIS_Z = 2
UNIT_CM = 100.0

# (骨名, 父骨名, 取点函数)；取点函数输入 (17,3) 关节
BONE_SPEC = (
    ("root", None, lambda j: 0.5 * (j[11] + j[12])),
    ("spine", "root", lambda j: 0.25 * (j[11] + j[12]) + 0.25 * (j[5] + j[6])),
    ("chest", "spine", lambda j: 0.5 * (j[5] + j[6])),
    ("neck", "chest", lambda j: 0.25 * (j[5] + j[6]) + 0.5 * j[0]),
    ("head", "neck", lambda j: j[0]),
    ("l_shoulder", "chest", lambda j: j[5]),
    ("l_elbow", "l_shoulder", lambda j: j[7]),
    ("l_wrist", "l_elbow", lambda j: j[9]),
    ("r_shoulder", "chest", lambda j: j[6]),
    ("r_elbow", "r_shoulder", lambda j: j[8]),
    ("r_wrist", "r_elbow", lambda j: j[10]),
    ("l_hip", "root", lambda j: j[11]),
    ("l_knee", "l_hip", lambda j: j[13]),
    ("l_ankle", "l_knee", lambda j: j[15]),
    ("r_hip", "root", lambda j: j[12]),
    ("r_knee", "r_hip", lambda j: j[14]),
    ("r_ankle", "r_knee", lambda j: j[16]),
)

COCO17_MIN = 17

# camera_v1（x 右 / y 下 / z 前）→ Z 向上检视坐标（X 右 / Y 前 / Z 上）。
CAM_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0]], dtype=np.float64)


def bone_positions(joints) -> list[tuple[str, str | None, np.ndarray]]:
    """按 BONE_SPEC 计算各骨节点的绝对位置（米）。"""
    j = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    if j.shape[0] < COCO17_MIN:
        raise ValueError(f"需要 COCO-17 关节（>=17），实际 {j.shape[0]}")
    return [(name, parent, np.asarray(fn(j), dtype=np.float64))
            for name, parent, fn in BONE_SPEC]


def _n(value: float) -> str:
    text = f"{float(value):.6f}"
    return "0.000000" if text == "-0.000000" else text


def build_ascii_fbx(bones: list[tuple[str, str | None, np.ndarray]],
                    object_name: str = "SAM3D_COCO17") -> str:
    """把骨节点列表写成 FBX 7.4 ASCII 文本。"""
    abs_pos = {name: pos for name, _parent, pos in bones}
    base_id = 1000000
    ids = {name: base_id + index for index, (name, _p, _v) in enumerate(bones)}
    attr_ids = {name: base_id + 100000 + index
                for index, (name, _p, _v) in enumerate(bones)}
    arm_id = base_id + 900000
    models_stem = object_name.replace(" ", "_")

    lines: list[str] = []
    add = lines.append
    add("; FBX 7.4.0 project file")
    add("; SAM-3D / MHR COCO-17 skeleton exported by Phy-SIC ground_pipeline")
    add("FBXHeaderExtension:  {")
    add("\tFBXHeaderVersion: 1003")
    add("\tFBXVersion: 7400")
    add(f'\tCreator: "Phy-SIC skeleton_fbx ({models_stem})"')
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
    add('\t\tP: "UnitScaleFactor", "double", "Number", "",1.0000')
    add('\t\tP: "OriginalUnitScaleFactor", "double", "Number", "",1.0000')
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
    add(f"\tCount: {2 + 2 * len(bones)}")
    add('\tObjectType: "GlobalSettings" {')
    add("\t\tCount: 1")
    add("\t}")
    add('\tObjectType: "Model" {')
    add(f"\t\tCount: {len(bones) + 1}")
    add("\t}")
    add('\tObjectType: "NodeAttribute" {')
    add(f"\t\tCount: {len(bones)}")
    add("\t}")
    add("}")
    add("Objects:  {")
    # 骨架根节点（Armature）：Max 里是 Dummy，Maya/Blender 里是骨架容器
    add(f'\tModel: {arm_id}, "Model::{models_stem}_Armature", "Null" {{')
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
    for name, parent, pos in bones:
        local = pos - (abs_pos[parent] if parent else np.zeros(3))
        local_cm = local * UNIT_CM
        add(f'\tModel: {ids[name]}, "Model::{models_stem}_{name}", "LimbNode" {{')
        add("\t\tVersion: 232")
        add("\t\tProperties70:  {")
        add('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",'
            f'{_n(local_cm[0])},{_n(local_cm[1])},{_n(local_cm[2])}')
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
    # 每根骨都必须有 NodeAttribute(LimbNode)，否则 Max/Maya 会导成空节点
    for name, _parent, _pos in bones:
        add(f'\tNodeAttribute: {attr_ids[name]}, "NodeAttribute::{models_stem}_{name}",'
            ' "LimbNode" {')
        add('\t\tTypeFlags: "Skeleton"')
        add("\t\tProperties70:  {")
        add('\t\t\tP: "Size", "double", "Number", "",10.000000')
        add("\t\t}")
        add("\t}")
    add("}")
    add("Connections:  {")
    for name, parent, _pos in bones:
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


def write_skeleton(path: str | pathlib.Path, joints, object_name: str,
                   note: str = "", frame: str = "ground_world_v1") -> dict:
    """写出单套骨架 FBX，返回本次导出的摘要（便于日志核对）。

    ``frame`` 为 ``ground_world_v1`` 时输入已是 Z 向上；为 ``camera_v1`` 时先旋转到
    Z 向上（X 右 / Y 前 / Z 上），否则导入后骨架会“躺倒”。
    """
    path = pathlib.Path(path)
    j = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    if frame == "camera_v1" and len(j):
        j = j @ CAM_TO_ZUP.T
    bones = bone_positions(j)
    text = build_ascii_fbx(bones, object_name=object_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "path": str(path),
        "note": note,
        "frame": frame,
        "bones": len(bones),
        "joints_used": int(min(len(j), COCO17_MIN)),
        "height_cm": float((j[:17, 2].max() - j[:17, 2].min()) * UNIT_CM),
        "feet_z_cm": float(min(j[15, 2], j[16, 2]) * UNIT_CM),
    }
