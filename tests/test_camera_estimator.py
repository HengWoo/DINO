"""Tests for camera auto-estimation module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dino.spatial.camera_estimator import estimate_camera
from dino.spatial.models import CameraIntrinsics, CameraPose


class TestEstimateCamera:
    def test_returns_intrinsics_and_pose(self):
        intrinsics, pose = estimate_camera(1920, 1080)
        assert isinstance(intrinsics, CameraIntrinsics)
        assert isinstance(pose, CameraPose)

    def test_intrinsics_principal_point_centered(self):
        intrinsics, _ = estimate_camera(1920, 1080)
        assert intrinsics.cx == pytest.approx(960.0)
        assert intrinsics.cy == pytest.approx(540.0)
        assert intrinsics.width == 1920
        assert intrinsics.height == 1080

    def test_fov_affects_focal_length(self):
        """Wider FOV -> shorter focal length."""
        intrinsics_narrow, _ = estimate_camera(1920, 1080, fov_deg=50.0)
        intrinsics_wide, _ = estimate_camera(1920, 1080, fov_deg=90.0)
        assert intrinsics_narrow.fx > intrinsics_wide.fx

    def test_focal_length_formula(self):
        """Verify fx = width / (2 * tan(fov/2))."""
        fov_deg = 70.0
        width = 1920
        expected_fx = width / (2 * math.tan(math.radians(fov_deg / 2)))
        intrinsics, _ = estimate_camera(width, 1080, fov_deg=fov_deg)
        assert intrinsics.fx == pytest.approx(expected_fx)
        assert intrinsics.fy == pytest.approx(expected_fx)

    def test_pose_translation_shape(self):
        _, pose = estimate_camera(1920, 1080)
        assert pose.translation.shape == (3,)

    def test_pose_rotation_shape(self):
        _, pose = estimate_camera(1920, 1080)
        assert pose.rotation.shape == (3, 3)

    def test_camera_above_ground(self):
        """Camera Y translation should be positive (above ground)."""
        _, pose = estimate_camera(1920, 1080)
        # translation[1] is Y (height)
        assert pose.translation[1] > 0

    def test_rotation_is_valid(self):
        """Rotation matrix should be orthogonal (R^T R ≈ I)."""
        _, pose = estimate_camera(1920, 1080)
        identity = pose.rotation.T @ pose.rotation
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

    def test_different_resolutions(self):
        """Should work for various resolutions."""
        for w, h in [(640, 480), (1280, 720), (3840, 2160)]:
            intrinsics, pose = estimate_camera(w, h)
            assert intrinsics.width == w
            assert intrinsics.height == h
            assert intrinsics.cx == pytest.approx(w / 2)
            assert intrinsics.cy == pytest.approx(h / 2)
