"""Tests for viewer data contract — validates JSON schema expected by the 3D viewer."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import supervision as sv
from unittest.mock import MagicMock

from dino.config import PipelineConfig, SpatialConfig


def _make_test_video(path: str, num_frames: int = 10, fps: int = 30):
    h, w = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {path}")
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (i * 25, i * 10, 0)
        writer.write(frame)
    writer.release()


def _make_mock_detector(detections=None):
    detector = MagicMock()
    if detections is None:
        detector.detect.return_value = sv.Detections.empty()
    else:
        detector.detect.return_value = detections
    return detector


def _make_detections(n=1):
    xyxy = np.array([[50 + i * 10, 50, 90 + i * 10, 90] for i in range(n)], dtype=np.float32)
    confidence = np.array([0.9] * n, dtype=np.float32)
    class_id = np.array([0] * n, dtype=int)
    return sv.Detections(
        xyxy=xyxy, confidence=confidence, class_id=class_id,
        data={"class_name": np.array(["person"] * n)},
    )


class TestViewerDataContract:
    """Validate the full JSON schema the viewer JS app expects."""

    def _generate_json(self, tmp_path, with_zones=True, with_detections=True):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "spatial_results.json")
        _make_test_video(input_path, num_frames=6)

        zones_data = {
            "zones": [
                {"zone_id": "z1", "name": "Zone 1", "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]]},
                {"zone_id": "z2", "name": "Zone 2", "polygon": [[100, 100], [300, 100], [300, 230], [100, 230]]},
            ],
            "rules": [{"event_type": "occupied", "requires_any": ["person"]}],
        }

        spatial_kwargs = {}
        if with_zones:
            zones_path = str(tmp_path / "zones.json")
            with open(zones_path, "w") as f:
                json.dump(zones_data, f)
            spatial_kwargs["zones_path"] = zones_path

        dets = _make_detections(n=2) if with_detections else None
        detector = _make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"], stride=1)
        spatial_config = SpatialConfig(**spatial_kwargs)
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            return json.load(f)

    def test_full_schema_keys(self, tmp_path):
        data = self._generate_json(tmp_path)
        assert set(data.keys()) == {"metadata", "zones", "frames", "observations", "events"}

    def test_metadata_types(self, tmp_path):
        data = self._generate_json(tmp_path)
        m = data["metadata"]
        assert isinstance(m["fps"], (int, float))
        assert isinstance(m["width"], int)
        assert isinstance(m["height"], int)
        assert isinstance(m["total_frames"], int)
        assert isinstance(m["duration_sec"], (int, float))
        assert isinstance(m["stride"], int)

    def test_zones_serialization(self, tmp_path):
        data = self._generate_json(tmp_path)
        assert len(data["zones"]) == 2
        for z in data["zones"]:
            assert isinstance(z["zone_id"], str)
            assert isinstance(z["name"], str)
            assert isinstance(z["polygon"], list)
            assert all(len(pt) == 2 for pt in z["polygon"])

    def test_frame_objects_have_world_position(self, tmp_path):
        data = self._generate_json(tmp_path)
        assert len(data["frames"]) > 0
        frame = data["frames"][0]
        assert len(frame["objects"]) > 0
        for obj in frame["objects"]:
            wp = obj["world_position"]
            assert isinstance(wp, list)
            assert len(wp) == 3
            assert all(isinstance(v, (int, float)) for v in wp)

    def test_coordinate_space_consistency(self, tmp_path):
        data = self._generate_json(tmp_path)
        w, h = data["metadata"]["width"], data["metadata"]["height"]
        for frame in data["frames"]:
            for obj in frame["objects"]:
                wp = obj["world_position"]
                # World positions should be within reasonable bounds of video dimensions
                assert -w <= wp[0] <= w * 2, f"x={wp[0]} out of range for width={w}"
                assert -h <= wp[1] <= h * 2, f"y={wp[1]} out of range for height={h}"
