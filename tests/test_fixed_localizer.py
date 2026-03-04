from __future__ import annotations

import logging

import numpy as np
import pytest
import supervision as sv

from dino.spatial.base_localizer import BaseLocalizer
from dino.spatial.fixed_camera_localizer import FixedCameraLocalizer


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestBaseLocalizer:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLocalizer()


class TestFixedCameraLocalizer:
    def test_localize_single_detection(self):
        localizer = FixedCameraLocalizer()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 110, 220]], dtype=np.float32),
            tracker_id=np.array([1]),
        )
        frame = _make_frame()
        objects = localizer.localize(detections, frame, frame_idx=0)

        assert len(objects) == 1
        np.testing.assert_array_almost_equal(
            objects[0].world_position, [60.0, 120.0, 0.0]
        )

    def test_localize_multiple_detections(self):
        localizer = FixedCameraLocalizer()
        detections = sv.Detections(
            xyxy=np.array(
                [[0, 0, 100, 100], [200, 200, 400, 400]], dtype=np.float32
            ),
            tracker_id=np.array([1, 2]),
        )
        frame = _make_frame()
        objects = localizer.localize(detections, frame, frame_idx=0)

        assert len(objects) == 2
        np.testing.assert_array_almost_equal(
            objects[0].world_position, [50.0, 50.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            objects[1].world_position, [300.0, 300.0, 0.0]
        )

    def test_localize_empty_detections(self):
        localizer = FixedCameraLocalizer()
        detections = sv.Detections.empty()
        frame = _make_frame()
        objects = localizer.localize(detections, frame, frame_idx=0)

        assert len(objects) == 0

    def test_preserves_class_info(self):
        localizer = FixedCameraLocalizer()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.85]),
            class_id=np.array([2]),
            tracker_id=np.array([7]),
            data={"class_name": np.array(["person"])},
        )
        frame = _make_frame()
        objects = localizer.localize(detections, frame, frame_idx=0)

        assert objects[0].class_name == "person"
        assert objects[0].class_id == 2
        assert objects[0].confidence == pytest.approx(0.85)

    def test_preserves_tracker_id(self):
        localizer = FixedCameraLocalizer()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            tracker_id=np.array([42]),
        )
        frame = _make_frame()
        objects = localizer.localize(detections, frame, frame_idx=0)

        assert objects[0].tracker_id == 42

    def test_frame_idx_passed_through(self):
        localizer = FixedCameraLocalizer()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            tracker_id=np.array([1]),
        )
        frame = _make_frame()
        objects = localizer.localize(detections, frame, frame_idx=99)

        assert objects[0].frame_idx == 99

    def test_no_tracker_id_warns_and_assigns_minus_one(self, caplog):
        """When tracker_id is None, all objects get -1 and a warning is logged."""
        localizer = FixedCameraLocalizer()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200], [0, 0, 50, 50]], dtype=np.float32),
        )
        frame = _make_frame()
        with caplog.at_level(logging.WARNING):
            objects = localizer.localize(detections, frame, frame_idx=0)
        assert len(objects) == 2
        assert all(obj.tracker_id == -1 for obj in objects)
        assert "tracker_id" in caplog.text
