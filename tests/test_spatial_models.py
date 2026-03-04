from __future__ import annotations

import numpy as np
import pytest

from dino.spatial.models import CameraIntrinsics, CameraPose, WorldObject


class TestWorldObject:
    def test_construction_with_required_fields(self):
        obj = WorldObject(
            tracker_id=1,
            bbox_xyxy=np.array([10, 20, 100, 200], dtype=np.float32),
            world_position=np.array([55.0, 110.0, 0.0]),
        )
        assert obj.tracker_id == 1
        assert obj.bbox_xyxy is not None
        assert obj.world_position is not None

    def test_default_values(self):
        obj = WorldObject(
            tracker_id=0,
            bbox_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
            world_position=np.array([0.0, 0.0, 0.0]),
        )
        assert obj.persistent_id is None
        assert obj.class_name == ""
        assert obj.class_id == -1
        assert obj.confidence == 0.0
        assert obj.position_confidence == 1.0
        assert obj.frame_idx == 0
        assert obj.timestamp == 0.0
        assert obj.zone_id is None

    def test_numpy_array_shapes(self):
        obj = WorldObject(
            tracker_id=0,
            bbox_xyxy=np.array([10, 20, 100, 200], dtype=np.float32),
            world_position=np.array([1.0, 2.0, 3.0]),
        )
        assert obj.bbox_xyxy.shape == (4,)
        assert obj.world_position.shape == (3,)


class TestCameraIntrinsics:
    def test_construction(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 500.0
        assert intrinsics.cx == 320.0
        assert intrinsics.cy == 240.0
        assert intrinsics.width == 640
        assert intrinsics.height == 480

    def test_frozen(self):
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )
        with pytest.raises(AttributeError):
            intrinsics.fx = 600.0


class TestCameraPose:
    def test_construction(self):
        pose = CameraPose(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=np.eye(3),
            frame_idx=42,
        )
        assert pose.translation.shape == (3,)
        assert pose.rotation.shape == (3, 3)
        assert pose.frame_idx == 42

    def test_default_timestamp(self):
        pose = CameraPose(
            translation=np.zeros(3),
            rotation=np.eye(3),
            frame_idx=0,
        )
        assert pose.timestamp == 0.0
