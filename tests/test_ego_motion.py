"""Tests for ego-motion estimation."""
from __future__ import annotations

import numpy as np
import pytest

from dino.spatial.ego_motion import EgoMotionEstimator
from dino.spatial.models import CameraIntrinsics


class TestEgoMotionEstimator:
    @pytest.fixture()
    def intrinsics(self):
        return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    @pytest.fixture()
    def estimator(self, intrinsics):
        return EgoMotionEstimator(intrinsics)

    def test_first_frame_returns_identity(self, estimator):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pose = estimator.update(frame)
        assert pose.shape == (4, 4)
        np.testing.assert_array_almost_equal(pose, np.eye(4))

    def test_poses_accumulate(self, estimator):
        for _ in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            estimator.update(frame)
        assert len(estimator.poses) == 3

    def test_get_all_poses_format(self, estimator):
        for _ in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            estimator.update(frame)
        poses = estimator.get_all_poses()
        assert len(poses) == 3
        for p in poses:
            assert "position" in p
            assert len(p["position"]) == 3

    def test_identical_frames_no_motion(self, estimator):
        """Identical frames should produce near-zero displacement."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        estimator.update(frame.copy())
        pose = estimator.update(frame.copy())
        # Should be identity or near-identity
        np.testing.assert_array_almost_equal(pose[:3, 3], [0, 0, 0], decimal=1)

    def test_empty_frame_no_crash(self, estimator):
        """A uniform frame with no features should not crash."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = estimator.update(frame)
        assert pose.shape == (4, 4)
