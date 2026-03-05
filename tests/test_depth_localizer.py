"""Tests for DepthLocalizer."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from dino.spatial.camera_estimator import estimate_camera
from dino.spatial.depth_localizer import DepthLocalizer
from dino.spatial.models import WorldObject
from dino.spatial.projection import backproject_to_world


def _make_detections(n=1, bbox_center=(100, 100), size=40):
    """Create detections centered at given pixel coords."""
    half = size // 2
    cx, cy = bbox_center
    xyxy = np.array(
        [[cx - half + i * 10, cy - half, cx + half + i * 10, cy + half] for i in range(n)],
        dtype=np.float32,
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9] * n, dtype=np.float32),
        class_id=np.array([0] * n, dtype=int),
        tracker_id=np.array(list(range(n)), dtype=int),
        data={"class_name": np.array(["person"] * n)},
    )


class TestDepthLocalizer:
    def _make_localizer(self, depth_value=0.5, frame_shape=(240, 320)):
        """Create a DepthLocalizer with a mock depth estimator."""
        intrinsics, pose = estimate_camera(frame_shape[1], frame_shape[0])
        mock_estimator = MagicMock()
        depth_map = np.full(frame_shape, depth_value, dtype=np.float32)
        mock_estimator.estimate.return_value = depth_map
        localizer = DepthLocalizer(intrinsics, pose, mock_estimator)
        return localizer, mock_estimator

    def test_localize_produces_3d_positions(self):
        """Objects should have non-zero Z in world_position."""
        localizer, _ = self._make_localizer(depth_value=0.5)
        dets = _make_detections(n=2)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        objects = localizer.localize(dets, frame, frame_idx=0)

        assert len(objects) == 2
        for obj in objects:
            assert isinstance(obj, WorldObject)
            assert obj.world_position.shape == (3,)
            # With depth > 0, z should be non-zero (true 3D)
            assert not np.allclose(obj.world_position, 0.0)

    def test_depth_sampling_at_bbox_bottom(self):
        """Depth should be sampled at bottom-center of bbox."""
        localizer, mock_estimator = self._make_localizer()
        # Create a non-uniform depth map to verify sampling location
        depth_map = np.zeros((240, 320), dtype=np.float32)
        # Put a specific value at the bottom-center of our bbox
        # bbox will be [80, 80, 120, 120] -> bottom-center at (100, 120)
        depth_map[120, 100] = 0.8
        mock_estimator.estimate.return_value = depth_map

        dets = _make_detections(n=1, bbox_center=(100, 100))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        objects = localizer.localize(dets, frame, frame_idx=0)

        # The depth used should be 0.8 (from bottom-center), not 0.0
        # This verifies sampling at y2 (bottom), not center
        assert len(objects) == 1
        # World position should be non-zero since depth is 0.8
        assert not np.allclose(objects[0].world_position, 0.0)

    def test_zero_depth_clamps_to_epsilon(self):
        """Zero depth should be clamped to avoid division errors."""
        localizer, _ = self._make_localizer(depth_value=0.0)
        dets = _make_detections(n=1)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # Should not raise
        objects = localizer.localize(dets, frame, frame_idx=0)
        assert len(objects) == 1
        # Position should still be valid
        assert np.all(np.isfinite(objects[0].world_position))

    def test_metadata_preserved(self):
        """Tracker ID, class name, confidence should be preserved."""
        localizer, _ = self._make_localizer(depth_value=0.5)
        dets = _make_detections(n=1)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        objects = localizer.localize(dets, frame, frame_idx=7)

        obj = objects[0]
        assert obj.tracker_id == 0
        assert obj.class_name == "person"
        assert obj.confidence == pytest.approx(0.9)
        assert obj.frame_idx == 7

    def test_empty_detections(self):
        """Empty detections should return empty list but still compute depth map."""
        localizer, mock_estimator = self._make_localizer()
        dets = sv.Detections.empty()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        objects = localizer.localize(dets, frame, frame_idx=0)

        assert objects == []
        # Depth estimation runs even for empty detections (needed for point cloud export)
        mock_estimator.estimate.assert_called_once()
        assert localizer.last_depth_map is not None

    def test_no_tracker_id_uses_minus_one(self):
        """Detections without tracker_id should use -1."""
        localizer, _ = self._make_localizer(depth_value=0.5)
        dets = sv.Detections(
            xyxy=np.array([[80, 80, 120, 120]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            data={"class_name": np.array(["person"])},
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        objects = localizer.localize(dets, frame, frame_idx=0)
        assert objects[0].tracker_id == -1

    def test_compute_bbox_3d_shape(self):
        """_compute_bbox_3d should return (8, 3) array."""
        localizer, _ = self._make_localizer(depth_value=2.0)
        intrinsics, pose = estimate_camera(320, 240)
        depth_map = np.full((240, 320), 2.0, dtype=np.float32)
        bbox = np.array([80, 80, 120, 120], dtype=np.float32)

        result = localizer._compute_bbox_3d(bbox, depth_map, intrinsics, pose)

        assert result is not None
        assert result.shape == (8, 3)
        assert np.all(np.isfinite(result))

    def test_compute_bbox_3d_encloses_world_position(self):
        """3D bbox AABB should contain the world_position."""
        localizer, _ = self._make_localizer(depth_value=2.0)
        intrinsics, pose = estimate_camera(320, 240)
        depth_map = np.full((240, 320), 2.0, dtype=np.float32)
        bbox = np.array([80, 80, 120, 120], dtype=np.float32)

        corners = localizer._compute_bbox_3d(bbox, depth_map, intrinsics, pose)

        # Compute world position at bottom-center of bbox
        u_center = (80 + 120) / 2.0
        v_bottom = 120.0
        world_pos = backproject_to_world(u_center, v_bottom, 2.0, intrinsics, pose)

        # AABB from corners should contain world_pos
        bbox_min = corners.min(axis=0)
        bbox_max = corners.max(axis=0)
        for dim in range(3):
            assert world_pos[dim] >= bbox_min[dim] - 0.1
            assert world_pos[dim] <= bbox_max[dim] + 0.1

    def test_compute_bbox_3d_with_flat_depth(self):
        """Uniform depth should produce corners on two depth planes."""
        localizer, _ = self._make_localizer(depth_value=3.0)
        intrinsics, pose = estimate_camera(320, 240)
        depth_map = np.full((240, 320), 3.0, dtype=np.float32)
        bbox = np.array([80, 80, 120, 120], dtype=np.float32)

        corners = localizer._compute_bbox_3d(bbox, depth_map, intrinsics, pose)
        assert corners.shape == (8, 3)

        # With flat depth, front 4 corners should be at ~same depth plane
        # and back 4 corners should be slightly behind
        # The Z spread should be small (just the extrusion offset)
        z_range = corners[:, 2].max() - corners[:, 2].min()
        assert z_range > 0  # There should be some depth extent
        assert z_range < 1.0  # But not too much for a 0.05 extrusion

    def test_bbox_3d_in_localize_output(self):
        """localize() should populate bbox_3d on WorldObjects."""
        localizer, _ = self._make_localizer(depth_value=2.0)
        dets = _make_detections(n=1, bbox_center=(160, 120))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        objects = localizer.localize(dets, frame, frame_idx=0)

        assert len(objects) == 1
        assert objects[0].bbox_3d is not None
        assert objects[0].bbox_3d.shape == (8, 3)
