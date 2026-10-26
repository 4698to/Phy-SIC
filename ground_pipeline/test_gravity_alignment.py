"""重力对齐单元测试（合成数据，不依赖 GPU / SAM-3D 权重）。

运行：``python -m unittest ground_pipeline.test_gravity_alignment -v``
"""
from __future__ import annotations

import copy
import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from ground_pipeline.ground_retarget import (
    FIGURE_P,
    GroundTransform,
    alignment_rotation,
    apply_gravity_alignment,
    insert_gravity_xml_metadata,
    resolve_gravity_alignment,
)


def _axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rotation_from_to(source, target) -> np.ndarray:
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if cosine > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return _axis_angle(np.cross(a, helper), math.pi)
    return _axis_angle(cross, math.acos(cosine))


def _transform_for_figure_up(up) -> GroundTransform:
    """构造一个 GroundTransform，使 figure 系地面 up 等于给定向量。"""
    camera_up = FIGURE_P.T @ np.asarray(up, dtype=np.float64)
    rotation = _rotation_from_to(camera_up, [0.0, 0.0, 1.0])
    return GroundTransform(
        rotation,
        [0.0, 0.0, -1.0],
        source="synthetic",
        plane_normal_camera=np.asarray(up, dtype=np.float64),
    )


def _base_body() -> dict:
    pivot = np.array([0.10, -0.20, 0.90])
    joints = np.array(
        [
            pivot,
            pivot + [0.20, 0.00, 0.00],
            pivot + [0.00, 0.10, -0.50],
            pivot + [0.10, 0.20, -0.90],
        ]
    )
    keypoints_fig = np.array(
        [
            pivot,
            pivot + [0.00, 0.00, -0.90],
            pivot + [0.30, 0.00, 0.00],
        ]
    )
    keypoints_cam = keypoints_fig @ FIGURE_P  # kp_cam @ FIGURE_P.T == kp_fig
    generic_points = np.array(
        [
            pivot,
            pivot + [0.00, 0.00, -0.90],
            pivot + [0.40, 0.00, 0.00],
        ]
    )
    joint_cm = joints * 100.0
    # bip22_joints 与顶层同为米；pose_payload 用厘米
    bip = np.array(
        [
            pivot + [-0.05, -0.03, 0.00],
            pivot,
            pivot + [0.10, 0.00, -0.30],
        ]
    )
    bip_cm = bip * 100.0
    identity = np.eye(3).tolist()
    local = _axis_angle([0.0, 0.0, 1.0], 0.4).tolist()
    # pose_payload 的骨架行顺序与 bip22 一致：row0=root，row1=骨盆
    payload_cm = np.vstack([bip_cm[0], bip_cm[1], joint_cm[1:]])
    return {
        "joints_3d": joints.tolist(),
        "keypoints_3d_coco17": keypoints_cam.tolist(),
        "generic_pose": {
            "pred_joint_coords": generic_points.tolist(),
            "pred_keypoints_3d": generic_points.tolist(),
            "pred_global_rots": [identity, identity],
        },
        "retarget": {
            "bones": {
                "Bip001 Pelvis": {
                    "position": pivot.tolist(),
                    "global_rotation": identity,
                    "local_rotation": local,
                },
                "Bip001 Spine": {
                    "position": (pivot + [0.0, 0.0, 0.2]).tolist(),
                    "global_rotation": identity,
                    "local_rotation": identity,
                },
            },
            "root": {"bone": "Bip001 Pelvis", "position": pivot.tolist()},
        },
        "cs": {
            "bip22_joints": bip.tolist(),
            "pose_payload": {
                "data": [
                    payload_cm.tolist(),
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [1.0, 0.0, 0.0],
                    [[0, 1], [1, 2]],
                    [None, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
                    [pivot.tolist(), 0.5, 60.0, 1.0],
                    "cs_pose_validation.txt",
                ]
            },
            "points_53": [
                {"source_key": "root", "x": 1.0, "y": 2.0, "z": 50.0},
                {"source_key": "left_foot", "x": 20.0, "y": 5.0, "z": 10.0},
            ],
            "points_53_tsv": (
                "source_key\tx\ty\tz\n"
                "root\t1.000000\t2.000000\t50.000000\n"
                "left_foot\t20.000000\t5.000000\t10.000000\n"
            ),
        },
        "profile_rest": {"bones": ["Bip001 Pelvis"], "scope": "synthetic"},
    }


def _aligned(points, rotation, pivot, dz):
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = (arr - pivot) @ rotation.T + pivot
    out[:, 2] += dz
    return out


class GravityAlignmentTest(unittest.TestCase):
    def test_identity_alignment_keeps_values(self):
        transform = _transform_for_figure_up([0.0, 0.0, 1.0])
        rotation, angle = alignment_rotation(transform)
        self.assertAlmostEqual(angle, 0.0, places=9)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)

        body = _base_body()
        before = copy.deepcopy(body)
        report = apply_gravity_alignment(body, transform)
        self.assertEqual(report["status"], "applied")
        self.assertTrue(report["applied"])
        self.assertAlmostEqual(report["dz_m"], 0.0, places=6)
        self.assertEqual(report["source"], "joints")
        np.testing.assert_allclose(body["joints_3d"], before["joints_3d"], atol=1e-9)
        np.testing.assert_allclose(
            body["generic_pose"]["pred_joint_coords"],
            before["generic_pose"]["pred_joint_coords"],
            atol=1e-9,
        )

    def test_known_tilt_rotates_all_payloads(self):
        angle = math.radians(12.86)
        rotation = _axis_angle([1.0, 0.0, 0.0], angle)
        transform = _transform_for_figure_up(rotation.T @ np.array([0.0, 0.0, 1.0]))
        computed, computed_deg = alignment_rotation(transform)
        self.assertAlmostEqual(computed_deg, 12.86, places=6)
        np.testing.assert_allclose(computed, rotation, atol=1e-9)

        body = _base_body()
        before = copy.deepcopy(body)
        report = apply_gravity_alignment(body, transform)
        dz = report["dz_m"]
        pivot = np.asarray(report["pivot"], dtype=np.float64)
        self.assertEqual(report["source"], "joints")

        np.testing.assert_allclose(
            body["joints_3d"],
            _aligned(before["joints_3d"], rotation, pivot, dz),
            atol=1e-6,
        )
        keypoints_fig = np.asarray(before["keypoints_3d_coco17"]) @ FIGURE_P.T
        np.testing.assert_allclose(
            body["keypoints_3d_coco17"],
            _aligned(keypoints_fig, rotation, pivot, dz),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            body["generic_pose"]["pred_joint_coords"],
            _aligned(before["generic_pose"]["pred_joint_coords"], rotation, pivot, dz),
            atol=1e-6,
        )
        for name, row in before["retarget"]["bones"].items():
            got = body["retarget"]["bones"][name]
            np.testing.assert_allclose(
                got["global_rotation"],
                rotation @ np.asarray(row["global_rotation"]),
                atol=1e-9,
            )
            np.testing.assert_allclose(
                got["position"],
                _aligned([row["position"]], rotation, pivot, dz)[0],
                atol=1e-6,
            )
        self.assertEqual(body["retarget"]["frame"], "figure_v6")
        self.assertEqual(body["retarget"]["position_space"], "figure_v6_world")

    def test_generic_pivot_matches_skeleton_pivot(self):
        transform = _transform_for_figure_up(_axis_angle([0.0, 1.0, 0.0], 0.2).T @ np.array([0.0, 0.0, 1.0]))
        body = _base_body()
        before_joints = np.asarray(body["joints_3d"])
        before_generic = np.asarray(body["generic_pose"]["pred_joint_coords"])
        delta_before = before_generic[0] - before_joints[0]

        apply_gravity_alignment(body, transform)
        delta_after = (
            np.asarray(body["generic_pose"]["pred_joint_coords"])[0]
            - np.asarray(body["joints_3d"])[0]
        )
        np.testing.assert_allclose(delta_after, delta_before, atol=1e-9)
        self.assertFalse(body.get("warnings"))

    def test_generic_pivot_falls_back_when_deviating(self):
        transform = _transform_for_figure_up([0.0, 0.0, 1.0])
        body = _base_body()
        generic = np.asarray(body["generic_pose"]["pred_joint_coords"])
        generic[0] = generic[0] + [0.5, 0.0, 0.0]
        body["generic_pose"]["pred_joint_coords"] = generic.tolist()

        apply_gravity_alignment(body, transform)
        self.assertTrue(
            any("generic pelvis deviates" in item for item in body.get("warnings", []))
        )

    def test_local_rotation_unchanged(self):
        transform = _transform_for_figure_up(_axis_angle([1.0, 0.0, 0.0], 0.3).T @ np.array([0.0, 0.0, 1.0]))
        body = _base_body()
        before = copy.deepcopy(body["retarget"]["bones"])
        apply_gravity_alignment(body, transform)
        for name, row in before.items():
            np.testing.assert_allclose(
                body["retarget"]["bones"][name]["local_rotation"],
                row["local_rotation"],
                atol=0.0,
            )

    def test_second_application_rejected(self):
        transform = _transform_for_figure_up(_axis_angle([1.0, 0.0, 0.0], 0.3).T @ np.array([0.0, 0.0, 1.0]))
        body = _base_body()
        report = apply_gravity_alignment(body, transform)
        body["gravity_aligned"] = True
        body["gravity_alignment"] = report
        snapshot = copy.deepcopy(body)
        with self.assertRaises(ValueError):
            apply_gravity_alignment(body, transform)
        self.assertEqual(body, snapshot)

    def test_disabled_still_reports_angle_without_mutation(self):
        rotation = _axis_angle([0.0, 1.0, 0.0], math.radians(12.86))
        transform = _transform_for_figure_up(rotation.T @ np.array([0.0, 0.0, 1.0]))
        body = _base_body()
        snapshot = copy.deepcopy(body)
        _, angle = alignment_rotation(transform)
        self.assertAlmostEqual(angle, 12.86, places=6)
        self.assertEqual(body, snapshot)

        report = resolve_gravity_alignment(body, transform, align=False)
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "disabled")
        self.assertEqual(report["source"], "ground")
        self.assertFalse(report["applied"])
        self.assertAlmostEqual(report["angle_deg"], 12.86, places=6)
        self.assertFalse(body["gravity_aligned"])
        self.assertAlmostEqual(body["align_angle_deg"], 12.86, places=6)
        # 未开启时任何数值载荷都不变
        self.assertEqual(body["joints_3d"], snapshot["joints_3d"])
        self.assertEqual(body["retarget"]["bones"], snapshot["retarget"]["bones"])
        self.assertEqual(body["cs"]["bip22_joints"], snapshot["cs"]["bip22_joints"])
        self.assertEqual(
            body["cs"]["pose_payload"]["data"], snapshot["cs"]["pose_payload"]["data"]
        )

    def test_landing_uses_single_dz_for_mesh(self):
        transform = _transform_for_figure_up(_axis_angle([1.0, 0.0, 0.0], 0.25).T @ np.array([0.0, 0.0, 1.0]))
        rotation, _ = alignment_rotation(transform)
        body = _base_body()
        pivot = np.asarray(body["joints_3d"][0], dtype=np.float64)
        mesh_fig = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.2],
                [0.0, 0.5, 0.4],
                [-0.2, -0.2, 1.2],
            ]
        )
        mesh_vertices = mesh_fig @ FIGURE_P
        rotated = (mesh_fig - pivot) @ rotation.T + pivot
        expected_dz = -float(np.min(rotated[:, 2]))

        before_joints = np.asarray(body["joints_3d"], dtype=np.float64)
        report = apply_gravity_alignment(body, transform, mesh_vertices=mesh_vertices)
        self.assertEqual(report["source"], "sam3d_mesh")
        self.assertAlmostEqual(report["dz_m"], expected_dz, places=6)
        np.testing.assert_allclose(
            body["joints_3d"], _aligned(before_joints, rotation, pivot, expected_dz), atol=1e-6
        )

        bip = np.asarray(body["cs"]["bip22_joints"], dtype=np.float64)
        self.assertAlmostEqual(float(np.min(rotated[:, 2])) + expected_dz, 0.0, places=9)
        self.assertAlmostEqual(bip[0, 2], 0.0, places=9)
        self.assertEqual(body["profile_rest"], _base_body()["profile_rest"])

    def test_landing_joints_source_and_cs_units(self):
        transform = _transform_for_figure_up(_axis_angle([0.0, 1.0, 0.0], 0.3).T @ np.array([0.0, 0.0, 1.0]))
        rotation, _ = alignment_rotation(transform)
        body = _base_body()
        pivot = np.asarray(body["joints_3d"][0], dtype=np.float64)
        before = copy.deepcopy(body)

        report = apply_gravity_alignment(body, transform)
        dz = report["dz_m"]
        self.assertEqual(report["source"], "joints")
        rotated_joints = _aligned(before["joints_3d"], rotation, pivot, dz)
        self.assertAlmostEqual(float(np.min(rotated_joints[:, 2])), 0.0, delta=1e-6)

        pelvis_before = np.asarray(before["cs"]["bip22_joints"], dtype=np.float64)[1]
        pelvis_expected = rotation @ (pelvis_before - pivot) + pivot
        pelvis_expected[2] += dz
        pelvis_after = np.asarray(body["cs"]["bip22_joints"], dtype=np.float64)[1]
        np.testing.assert_allclose(pelvis_after, pelvis_expected, atol=1e-6)

        for point in body["cs"]["points_53"]:
            if str(point.get("source_key")) == "root":
                self.assertAlmostEqual(point["z"], 0.0, places=9)
                self.assertAlmostEqual(point["x"], pelvis_after[0] * 100.0, places=6)
                self.assertAlmostEqual(point["y"], pelvis_after[1] * 100.0, places=6)

        lines = body["cs"]["points_53_tsv"].strip().splitlines()
        headers = lines[0].split("\t")
        for line in lines[1:]:
            cols = line.split("\t")
            if cols[headers.index("source_key")] == "root":
                self.assertAlmostEqual(float(cols[headers.index("z")]), 0.0, places=6)
            else:
                self.assertNotAlmostEqual(float(cols[headers.index("z")]), 10.0, places=6)

        payload = np.asarray(body["cs"]["pose_payload"]["data"][0], dtype=np.float64)
        # payload 第 2 行为骨盆；第 1 行为 root，z 置 0 且 xy 取骨盆 xy
        np.testing.assert_allclose(payload[1], pelvis_after * 100.0, atol=1e-4)
        self.assertAlmostEqual(payload[0, 2], 0.0, places=6)
        self.assertAlmostEqual(payload[0, 0], payload[1, 0], places=6)
        self.assertAlmostEqual(payload[0, 1], payload[1, 1], places=6)

    def test_cs_camera_slot_stays_in_meters(self):
        transform = _transform_for_figure_up(
            _axis_angle([1.0, 0.0, 0.0], 0.2).T @ np.array([0.0, 0.0, 1.0])
        )
        rotation, _ = alignment_rotation(transform)
        body = _base_body()
        # center 是“骨盆→相机”的相对向量（figure 系、米），不是绝对相机位置
        center = np.array([0.30, -0.20, 0.40])
        body["cs"]["pose_payload"]["data"][6] = [center.tolist(), 0.5, 60.0, 1.0]

        report = apply_gravity_alignment(body, transform)
        dz = report["dz_m"]
        self.assertNotAlmostEqual(dz, 0.0, places=6)
        slot = body["cs"]["pose_payload"]["data"][6]
        # 相机与姿势刚性旋转：相对向量只旋转，不绕骨盆平移，也不叠加落地 dz
        np.testing.assert_allclose(slot[0], rotation @ center, atol=1e-6)
        # 相机中心是米制：与 pivot 同量级，不能被放大 100 倍
        self.assertLess(float(np.max(np.abs(slot[0]))), 10.0)
        self.assertEqual(slot[1:], [0.5, 60.0, 1.0])

        # 同一 payload 的米制/厘米制位置字段各自保持单位
        bip = np.asarray(body["cs"]["bip22_joints"], dtype=np.float64)
        payload = np.asarray(body["cs"]["pose_payload"]["data"][0], dtype=np.float64)
        self.assertLess(float(np.max(np.abs(bip))), 10.0)
        self.assertGreater(float(np.max(np.abs(payload))), 10.0)
        np.testing.assert_allclose(payload[1], bip[1] * 100.0, atol=1e-4)

    def test_no_ground_failure_semantics(self):
        body = _base_body()
        before = copy.deepcopy(body)
        report = resolve_gravity_alignment(
            body, None, align=True, ground_status="skipped", ground_error=None
        )
        self.assertEqual(report["status"], "ground_unavailable")
        self.assertFalse(report["applied"])
        self.assertFalse(body["gravity_aligned"])
        self.assertIsNone(body["align_angle_deg"])
        self.assertIsNone(report["angle_deg"])
        self.assertTrue(
            any(str(w).startswith("gravity alignment unavailable:") for w in body["warnings"])
        )
        # 未旋转：targetPose 等数值保持原样
        self.assertEqual(body["joints_3d"], before["joints_3d"])
        self.assertEqual(body["keypoints_3d_coco17"], before["keypoints_3d_coco17"])
        self.assertEqual(body["retarget"]["bones"], before["retarget"]["bones"])
        self.assertEqual(body["cs"]["bip22_joints"], before["cs"]["bip22_joints"])
        self.assertEqual(
            body["cs"]["pose_payload"]["data"], before["cs"]["pose_payload"]["data"]
        )
        # 三处镜像一致
        for block in (body["cs"], body["cs"]["pose_payload"], body["retarget"]):
            self.assertFalse(block["gravity_aligned"])
            self.assertIsNone(block["align_angle_deg"])

        errored = _base_body()
        report = resolve_gravity_alignment(
            errored, None, align=True, ground_status="error", ground_error="boom"
        )
        self.assertEqual(report["status"], "ground_error")
        self.assertEqual(report["error"], "boom")
        self.assertTrue(
            any(str(w) == "gravity alignment unavailable: boom" for w in errored["warnings"])
        )

    def test_xml_metadata_three_nodes_consistent(self):
        xml = (
            b'<?xml version="1.0" encoding="utf-8"?>'
            b"<poseResponse>"
            b'<poseResult gravity_aligned="false" align_angle_deg="9.900000"><targetPose/></poseResult>'
            b"<csPose/>"
            b"<targetPose/>"
            b"<restProfile/>"
            b"</poseResponse>"
        )
        aligned = insert_gravity_xml_metadata(
            xml, {"gravity_aligned": True, "align_angle_deg": 12.86}
        )
        root = ET.fromstring(aligned)
        for tag in ("poseResult", "csPose", "targetPose"):
            element = root.find(tag)
            self.assertEqual(element.get("gravity_aligned"), "true")
            self.assertEqual(element.get("align_angle_deg"), "12.860000")
        # 不再向 poseResult 追加子节点
        self.assertEqual([child.tag for child in root.find("poseResult")], ["targetPose"])

        skipped = insert_gravity_xml_metadata(
            xml, {"gravity_aligned": False, "align_angle_deg": None}
        )
        root = ET.fromstring(skipped)
        for tag in ("poseResult", "csPose", "targetPose"):
            element = root.find(tag)
            self.assertEqual(element.get("gravity_aligned"), "false")
            self.assertNotIn("align_angle_deg", element.attrib)

    def test_degenerate_ground_rejected(self):
        bad = GroundTransform(
            np.ones((3, 3)),
            [0.0, 0.0, 0.0],
            source="synthetic",
            plane_normal_camera=[0.0, 0.0, 1.0],
        )
        with self.assertRaises(ValueError):
            alignment_rotation(bad)

        opposite = _transform_for_figure_up([0.0, 0.0, -1.0])
        with self.assertRaises(ValueError):
            alignment_rotation(opposite)

        body = _base_body()
        snapshot = copy.deepcopy(body)
        with self.assertRaises(ValueError):
            apply_gravity_alignment(body, opposite)
        self.assertEqual(body, snapshot)


if __name__ == "__main__":
    unittest.main()
