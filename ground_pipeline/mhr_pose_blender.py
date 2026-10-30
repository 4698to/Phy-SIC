"""用无头 Blender 把一份目标骨骼姿势套到 ``MHR-lod5.tpose.fbx`` 上再导出。

与 ``mhr_pose_fbx``（直接改二进制、逐字节保留原文件）互补：本脚本走 Blender
导入/导出，好处是不用手写 FBX 序列化，导出结果一定被 Blender / 3ds Max 正常读取；
代价是几何、节点 ID、连接顺序会被 Blender 重建（形体与骨骼位置不变）。

输入 ``--targets`` 是 ``mhr_pose_fbx.export_pose(..., targets_json=...)`` 写出的
JSON：坐标已在 FBX 自身空间（Y-up、厘米），等于 Blender 里 Armature 对象空间，
所以这里不做任何轴系 / 尺度换算——只是在对象空间里摆骨骼。

用法（WSL，无头）：
  blender -b --factory-startup --python ground_pipeline/mhr_pose_blender.py -- \
      --targets images/_wr_mhrpose/fbx/<stem>_pose_targets.json \
      --out     images/_wr_mhrpose/fbx_blender/<stem>_mhr_pose_fbx.fbx \
      [--template ground_pipeline/assets/mhr/MHR-lod5.tpose.fbx] [--no-verify]
"""

from __future__ import annotations

import argparse
import json
import sys

import bpy
from mathutils import Matrix, Vector

DEFAULT_TEMPLATE = ("/mnt/g/UGit/Phy-SIC/ground_pipeline/assets/mhr/"
                    "MHR-lod5.tpose.fbx")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--no-verify", action="store_true")
    return ap.parse_args(argv)


def import_fbx(path: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=False,
                             automatic_bone_orientation=False, global_scale=1.0)


def pick_armature():
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("导入后没有 Armature（LimbNode 未被识别为骨骼）")
    return max(arms, key=lambda o: len(o.data.bones))


def bone_depth(bone) -> int:
    d = 0
    while bone.parent is not None:
        bone, d = bone.parent, d + 1
    return d


def solve(arms, targets: dict) -> dict:
    """把每根骨的头挪到 targets[name]（Armature 对象空间，厘米）。"""
    bones = arms.data.bones
    pose = arms.pose.bones
    missing = [n for n in targets if n not in bones]

    # 注意：``bone.parent is b`` 在 Blender 里恒为 False（每次访问返回新的包装对象），
    # 必须按名字比较，否则 kids 全空、aim() 永远退化、所有骨只平移不旋转。
    kids = {b.name: [c for c in bones if c.parent and c.parent.name == b.name]
            for b in bones}

    def aim(name: str) -> Vector | None:
        """骨方向：优先用子骨头的平均方向，退化时用骨自身 head→tail。"""
        b = bones[name]
        vs = [Vector(targets[c.name]) - Vector(targets[name])
              for c in kids[name] if c.name in targets]
        vs = [v for v in vs if v.length > 1e-9]
        if vs:
            v = sum(vs, Vector()) / len(vs)
            if v.length > 1e-9:
                return v.normalized()
        d = (b.tail_local - b.head_local)
        return d.normalized() if d.length > 1e-9 else None

    err = {}
    for b in sorted(bones, key=bone_depth):
        if b.name not in targets:
            continue
        pb = pose[b.name]
        tgt_head = Vector(targets[b.name])
        rest_dir = (b.tail_local - b.head_local)
        rest_dir = rest_dir.normalized() if rest_dir.length > 1e-9 else None
        tgt_dir = aim(b.name)
        if rest_dir is not None and tgt_dir is not None:
            swing = rest_dir.rotation_difference(tgt_dir).to_matrix().to_4x4()
        else:
            swing = Matrix.Identity(4)
        rest_rot = b.matrix_local.to_3x3().to_4x4()
        pb.matrix = Matrix.Translation(tgt_head) @ (swing @ rest_rot)
        bpy.context.view_layer.update()
        err[b.name] = (pb.head - tgt_head).length

    return {"missing": missing, "err": err}


def export(out: str) -> None:
    bpy.ops.export_scene.fbx(
        filepath=out, check_existing=False, use_selection=False,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        global_scale=1.0, apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Z", axis_up="Y",
        use_mesh_modifiers=True, mesh_smooth_type="OFF",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        bake_anim=False, path_mode="AUTO", embed_textures=False,
        use_custom_props=True)


def verify(out: str, targets: dict) -> dict:
    import_fbx(out)
    arms = pick_armature()
    bones = arms.data.bones
    errs = {n: (Vector(bones[n].head_local) - Vector(t)).length
            for n, t in targets.items() if n in bones}
    return {"bones": len(bones),
            "missing": len(set(targets) - {b.name for b in bones}),
            "max_err_cm": max(errs.values()) if errs else float("nan"),
            "median_err_cm": (sorted(errs.values())[len(errs) // 2]
                              if errs else float("nan"))}


def eval_coords(ob) -> list:
    """depsgraph 求值后的顶点坐标（对象局部空间，已含修改器与形态键）。"""
    dg = bpy.context.evaluated_depsgraph_get()
    obe = ob.evaluated_get(dg)
    me = obe.to_mesh()
    co = [Vector(v.co) for v in me.vertices]
    obe.to_mesh_clear()
    return co


def bake_deform(meshes, arms) -> None:
    """把 ARMATURE 修改器的蒙皮形变烘进网格顶点（含全部形态键）。

    ``apply_pose_as_rest`` 只改骨骼 rest，不会烘网格；不先烘形变的话，导出物就是
    「T 姿势网格 + 已摆姿势骨骼」（网格与骨骼脱节）。

    带形态键的网格不能 ``modifier_apply``（Blender 报 "Modifier cannot be applied
    to a mesh with shape keys"）。蒙皮是逐顶点的仿射变换、只由权重决定、与顶点位置
    无关，所以把某个形态键单独置 1 求值，得到的正是该键坐标变形后的结果。

    ``mesh.vertices`` 与形态键数据是两份独立数组（实测互不同步），而 FBX 导出以
    ``mesh.vertices`` 作为基础几何、形态键另存为 BlendShape；只写形态键的话导出的
    基础网格仍是 T 姿势，所以两处都要写。
    """
    bpy.ops.object.mode_set(mode="OBJECT")
    for ob in meshes:
        if not any(m.type == "ARMATURE" and m.object is arms for m in ob.modifiers):
            continue
        me = ob.data
        sk = me.shape_keys
        if sk is None:
            co = eval_coords(ob)
            for v, c in zip(me.vertices, co):
                v.co = c
            me.update()
            continue
        blocks = list(sk.key_blocks)
        base = blocks[0]                      # Blender 约定 key_blocks[0] 为 Basis
        saved = {kb.name: kb.value for kb in blocks}
        baked = {}
        for kb in blocks:                     # 逐个置 1 求值，Basis 恒为 1
            for other in blocks:
                other.value = 1.0 if (other is kb or other is base) else 0.0
            bpy.context.view_layer.update()
            baked[kb.name] = eval_coords(ob)
            if kb is not base:
                kb.value = 0.0
        for kb in blocks:
            for pt, c in zip(kb.data, baked[kb.name]):
                pt.co = c
            kb.value = saved[kb.name]
        for v, c in zip(me.vertices, baked[base.name]):   # 基础几何也要同步
            v.co = c
        me.update()
        bpy.context.view_layer.update()


def rebind(meshes, arms) -> None:
    """烘完形变后补回 ARMATURE 修改器（新 rest 与新网格一致，形变为恒等）。"""
    for ob in meshes:
        if not any(m.type == "ARMATURE" for m in ob.modifiers):
            ob.modifiers.new(name="Armature", type="ARMATURE").object = arms


def apply_pose_as_rest(arms) -> None:
    """把当前 pose 固化成 rest pose。

    Blender 的 FBX 导出只写「rest pose + 动画」，不 bake 动画时静态 pose 会被
    忽略；因此把姿态固化成 rest，导出的就是静态姿势。
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = arms
    arms.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def main() -> None:
    a = parse_args()
    payload = json.load(open(a.targets, encoding="utf-8"))
    targets = payload["targets"]
    src = payload.get("source") or a.template

    import_fbx(src)
    arms = pick_armature()
    res = solve(arms, targets)
    mesh = [o for o in bpy.data.objects if o.type == "MESH"]
    print(f"[pose] armature={arms.name!r} bones={len(arms.data.bones)} "
          f"targets={len(targets)} missing={len(res['missing'])} "
          f"max_pose_err_cm={max(res['err'].values()):.4g} "
          f"meshes={[(o.name, len(o.data.vertices)) for o in mesh]}")

    bake_deform(mesh, arms)
    apply_pose_as_rest(arms)
    rebind(mesh, arms)
    export(a.out)
    if not a.no_verify:
        v = verify(a.out, targets)
        print(f"[verify] reimport bones={v['bones']} missing={v['missing']} "
              f"max_err_cm={v['max_err_cm']:.6g} "
              f"median_err_cm={v['median_err_cm']:.6g}")
    print(f"[out] {a.out}")


main()
