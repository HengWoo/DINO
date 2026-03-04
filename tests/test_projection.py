from __future__ import annotations

import math

import numpy as np
import pytest

from dino.spatial.models import CameraIntrinsics, CameraPose
from dino.spatial.projection import (
    backproject_pixel_to_camera,
    backproject_to_world,
    camera_to_world,
    quaternion_to_rotation_matrix,
)


class TestQuaternionToRotationMatrix:
    def test_identity_quaternion(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90_degree_rotation_z(self):
        angle = math.pi / 2
        q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        R = quaternion_to_rotation_matrix(q)
        # Rotating [1, 0, 0] by 90 degrees around Z should give [0, 1, 0]
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_output_shape(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        assert R.shape == (3, 3)

    def test_orthogonal(self):
        angle = 1.23
        q = np.array([
            math.cos(angle / 2),
            math.sin(angle / 2) * 0.267,
            math.sin(angle / 2) * 0.535,
            math.sin(angle / 2) * 0.802,
        ])
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6


class TestBackprojectPixelToCamera:
    def test_principal_point_at_unit_depth(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        point = backproject_pixel_to_camera(320.0, 240.0, 1.0, intrinsics)
        np.testing.assert_array_almost_equal(point, [0.0, 0.0, 1.0])

    def test_offset_pixel(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        point = backproject_pixel_to_camera(820.0, 740.0, 1.0, intrinsics)
        np.testing.assert_array_almost_equal(point, [1.0, 1.0, 1.0])

    def test_depth_scaling(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        p1 = backproject_pixel_to_camera(420.0, 340.0, 1.0, intrinsics)
        p2 = backproject_pixel_to_camera(420.0, 340.0, 2.0, intrinsics)
        np.testing.assert_array_almost_equal(p2[:2], p1[:2] * 2.0)


class TestCameraToWorld:
    def test_identity_pose(self):
        pose = CameraPose(
            translation=np.zeros(3), rotation=np.eye(3), frame_idx=0
        )
        point = np.array([1.0, 2.0, 3.0])
        result = camera_to_world(point, pose)
        np.testing.assert_array_almost_equal(result, point)

    def test_translation_only(self):
        pose = CameraPose(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=np.eye(3),
            frame_idx=0,
        )
        point = np.array([0.0, 0.0, 0.0])
        result = camera_to_world(point, pose)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_rotation_and_translation(self):
        # 90 degree rotation around Z axis
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose = CameraPose(
            translation=np.array([10.0, 20.0, 0.0]),
            rotation=R,
            frame_idx=0,
        )
        point = np.array([1.0, 0.0, 0.0])
        result = camera_to_world(point, pose)
        # R @ [1,0,0] = [0,1,0], then + [10,20,0] = [10,21,0]
        np.testing.assert_array_almost_equal(result, [10.0, 21.0, 0.0])


class TestBackprojectToWorld:
    def test_principal_point_identity_pose(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        pose = CameraPose(
            translation=np.zeros(3), rotation=np.eye(3), frame_idx=0
        )
        result = backproject_to_world(320.0, 240.0, 5.0, intrinsics, pose)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 5.0])

    def test_round_trip(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        pose = CameraPose(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=np.eye(3),
            frame_idx=0,
        )
        result = backproject_to_world(420.0, 340.0, 5.0, intrinsics, pose)
        # Camera point: ((420-320)*5/500, (340-240)*5/500, 5) = (1.0, 1.0, 5.0)
        # World point: (1.0, 1.0, 5.0) + (1.0, 2.0, 3.0) = (2.0, 3.0, 8.0)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 8.0])
