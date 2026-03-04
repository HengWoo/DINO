"""Tests for spatial pipeline components."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import supervision as sv

from dino.config import PipelineConfig, SpatialConfig


def _make_test_video(path: str, num_frames: int = 10, fps: int = 30):
    """Create a small test video (240x320, mp4v codec)."""
    h, w = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter at {path} with mp4v codec. "
            f"Check that OpenCV was built with mp4v support."
        )
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (i * 25, i * 10, 0)
        writer.write(frame)
    writer.release()


class TestSpatialConfig:
    def test_defaults(self):
        config = SpatialConfig()
        assert config.camera_mode == "fixed"
        assert config.match_distance == 50.0
        assert config.max_age_seconds == 30.0
        assert config.zones_path is None
        assert config.rules is None

    def test_invalid_camera_mode(self):
        with pytest.raises(ValueError, match="Unsupported camera_mode"):
            SpatialConfig(camera_mode="moving")

    def test_invalid_match_distance(self):
        with pytest.raises(ValueError, match="match_distance must be > 0"):
            SpatialConfig(match_distance=0)

    def test_invalid_max_age(self):
        with pytest.raises(ValueError, match="max_age_seconds must be > 0"):
            SpatialConfig(max_age_seconds=-1)

    def test_invalid_rules_missing_event_type(self):
        with pytest.raises(ValueError, match="missing required key 'event_type'"):
            SpatialConfig(rules=[{"requires_any": ["person"]}])


class TestLoadZonesFile:
    def test_load_valid_file(self, tmp_path):
        from dino.zones.loader import load_zones_file

        zones_data = {
            "zones": [
                {"zone_id": "z1", "name": "Zone 1", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            ],
            "rules": [
                {"event_type": "occupied", "requires_any": ["person"]},
            ],
        }
        path = tmp_path / "zones.json"
        path.write_text(json.dumps(zones_data))

        zones, rules = load_zones_file(path)
        assert len(zones) == 1
        assert zones[0].zone_id == "z1"
        assert zones[0].name == "Zone 1"
        assert zones[0].polygon.shape == (4, 2)
        assert len(rules) == 1
        assert rules[0].event_type == "occupied"

    def test_file_not_found(self):
        from dino.zones.loader import load_zones_file

        with pytest.raises(FileNotFoundError):
            load_zones_file("/nonexistent/zones.json")

    def test_missing_zones_key(self, tmp_path):
        from dino.zones.loader import load_zones_file

        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"rules": []}))
        with pytest.raises(ValueError, match="must contain a 'zones' key"):
            load_zones_file(path)

    def test_no_rules_section(self, tmp_path):
        from dino.zones.loader import load_zones_file

        zones_data = {
            "zones": [
                {"zone_id": "z1", "name": "Zone 1", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            ],
        }
        path = tmp_path / "zones.json"
        path.write_text(json.dumps(zones_data))

        zones, rules = load_zones_file(path)
        assert len(zones) == 1
        assert len(rules) == 0

    def test_missing_zone_keys(self, tmp_path):
        from dino.zones.loader import load_zones_file

        zones_data = {
            "zones": [
                {"zone_id": "z1", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            ],
        }
        path = tmp_path / "zones.json"
        path.write_text(json.dumps(zones_data))
        with pytest.raises(ValueError, match="missing required keys"):
            load_zones_file(path)

    def test_invalid_json(self, tmp_path):
        from dino.zones.loader import load_zones_file

        path = tmp_path / "bad.json"
        path.write_text("not valid json {")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_zones_file(path)


class TestZoneAnnotator:
    def _make_zone(self, zone_id="z1", name="Test Zone"):
        from dino.zones.models import ZoneDefinition

        return ZoneDefinition(
            zone_id=zone_id,
            name=name,
            polygon=np.array(
                [[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float64
            ),
        )

    def _make_frame(self, h=240, w=320):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_annotate_returns_copy(self):
        from dino.annotation.zone_annotator import ZoneAnnotator

        annotator = ZoneAnnotator()
        frame = self._make_frame()
        zone = self._make_zone()
        result = annotator.annotate(frame, [zone])
        # Input should not be modified
        assert np.array_equal(frame, np.zeros_like(frame))
        # Result should be a different array
        assert result is not frame

    def test_annotate_draws_on_frame(self):
        from dino.annotation.zone_annotator import ZoneAnnotator

        annotator = ZoneAnnotator()
        frame = self._make_frame()
        zone = self._make_zone()
        result = annotator.annotate(frame, [zone])
        # Output should differ from blank input (zone was drawn)
        assert not np.array_equal(result, frame)

    def test_active_zone_different_color(self):
        from dino.annotation.zone_annotator import ZoneAnnotator
        from dino.spatial.models import WorldObject
        from dino.zones.models import ZoneState

        annotator = ZoneAnnotator()
        zone = self._make_zone()

        # Inactive frame
        frame1 = self._make_frame()
        result_inactive = annotator.annotate(frame1, [zone])

        # Active frame (zone has present objects)
        frame2 = self._make_frame()
        obj = WorldObject(
            tracker_id=1,
            bbox_xyxy=np.array([10, 10, 50, 50]),
            world_position=np.array([30, 30, 0]),
            persistent_id=1,
            class_name="person",
        )
        zone_states = {
            "z1": ZoneState(zone_id="z1", present_objects={1: obj}),
        }
        result_active = annotator.annotate(frame2, [zone], zone_states=zone_states)

        # Active and inactive should produce different results
        assert not np.array_equal(result_inactive, result_active)

    def test_annotate_empty_zones(self):
        from dino.annotation.zone_annotator import ZoneAnnotator

        annotator = ZoneAnnotator()
        frame = self._make_frame()
        result = annotator.annotate(frame, [])
        # Should return a copy even with no zones
        assert result is not frame
        assert np.array_equal(result, frame)


class TestSpatialPipeline:
    def _make_mock_detector(self, detections=None):
        detector = MagicMock()
        if detections is None:
            detector.detect.return_value = sv.Detections.empty()
        else:
            detector.detect.return_value = detections
        return detector

    def _make_detections(self, n=1):
        """Create mock detections with n bounding boxes inside a 320x240 frame."""
        xyxy = np.array(
            [[50 + i * 10, 50, 90 + i * 10, 90] for i in range(n)],
            dtype=np.float32,
        )
        confidence = np.array([0.9] * n, dtype=np.float32)
        class_id = np.array([0] * n, dtype=int)
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={"class_name": np.array(["person"] * n)},
        )

    def test_run_produces_output_no_zones(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=5)

        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"], stride=1)
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        results = pipeline.run(input_path, output_path)

        assert (tmp_path / "output.mp4").exists()
        assert isinstance(results.frame_results, list)
        assert isinstance(results.observations, list)
        assert isinstance(results.events, list)

    def test_run_with_zones_and_detections(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=5)

        zones_data = {
            "zones": [
                {
                    "zone_id": "z1",
                    "name": "Zone 1",
                    "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
                },
            ],
            "rules": [
                {"event_type": "occupied", "requires_any": ["person"], "hysteresis": 1},
            ],
        }
        zones_path = str(tmp_path / "zones.json")
        with open(zones_path, "w") as f:
            json.dump(zones_data, f)

        dets = self._make_detections(n=1)
        detector = self._make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"], stride=1)
        spatial_config = SpatialConfig(zones_path=zones_path)
        pipeline = SpatialPipeline(detector, config, spatial_config)
        results = pipeline.run(input_path, output_path)

        assert len(results.frame_results) == 5
        # 1 ENTER on frame 0, 4 PRESENT on frames 1-4
        assert len(results.observations) == 5
        # 1 occupied event (hysteresis=1, latches after firing)
        assert len(results.events) == 1
        assert results.events[0]["event_type"] == "occupied"

    def test_run_input_not_found(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        with pytest.raises(FileNotFoundError):
            pipeline.run("/nonexistent/video.mp4", str(tmp_path / "out.mp4"))

    def test_json_export_structure(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        _make_test_video(input_path, num_frames=3)

        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        assert (tmp_path / "results.json").exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "frames" in data
        assert "observations" in data
        assert "events" in data

    def test_json_export_with_detections(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        _make_test_video(input_path, num_frames=3)

        zones_data = {
            "zones": [
                {
                    "zone_id": "z1",
                    "name": "Zone 1",
                    "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
                },
            ],
            "rules": [
                {"event_type": "occupied", "requires_any": ["person"]},
            ],
        }
        zones_path = str(tmp_path / "zones.json")
        with open(zones_path, "w") as f:
            json.dump(zones_data, f)

        dets = self._make_detections(n=1)
        detector = self._make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig(zones_path=zones_path)
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        # Verify frame result dict structure
        frame = data["frames"][0]
        assert "frame_idx" in frame
        assert "timestamp" in frame
        assert "objects" in frame
        assert len(frame["objects"]) > 0
        obj = frame["objects"][0]
        for key in ("persistent_id", "tracker_id", "class_name", "confidence",
                     "bbox", "world_position", "zone_id"):
            assert key in obj, f"Missing key {key} in object dict"

        # Verify observation dict structure
        assert len(data["observations"]) > 0
        obs = data["observations"][0]
        for key in ("zone_id", "persistent_id", "observation_type", "frame_idx", "timestamp"):
            assert key in obs, f"Missing key {key} in observation dict"

        # Verify event dict structure
        assert len(data["events"]) > 0
        event = data["events"][0]
        for key in ("zone_id", "event_type", "frame_idx", "timestamp", "details"):
            assert key in event, f"Missing key {key} in event dict"

    def test_progress_callback(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=6)

        calls = []
        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(
            input_path,
            output_path,
            progress_callback=lambda p, t: calls.append((p, t)),
        )

        assert len(calls) == 6
        # Callback receives (processed_count, total_frames)
        assert calls == [(i + 1, 6) for i in range(6)]

    def test_stride_skips_frames(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=9)

        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"], stride=3)
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        results = pipeline.run(input_path, output_path)

        # Frames 0, 3, 6 should be processed (3 frames)
        assert len(results.frame_results) == 3

    def test_inline_rules(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=3)

        zones_data = {
            "zones": [
                {
                    "zone_id": "z1",
                    "name": "Zone 1",
                    "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
                },
            ],
        }
        zones_path = str(tmp_path / "zones.json")
        with open(zones_path, "w") as f:
            json.dump(zones_data, f)

        dets = self._make_detections(n=1)
        detector = self._make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig(
            zones_path=zones_path,
            rules=[{"event_type": "occupied", "requires_any": ["person"]}],
        )
        pipeline = SpatialPipeline(detector, config, spatial_config)
        results = pipeline.run(input_path, output_path)

        assert len(results.events) == 1
        assert results.events[0]["event_type"] == "occupied"

    def test_inline_rules_override_file_rules(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=3)

        zones_data = {
            "zones": [
                {
                    "zone_id": "z1",
                    "name": "Zone 1",
                    "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
                },
            ],
            "rules": [
                {"event_type": "file_rule", "requires_any": ["person"]},
            ],
        }
        zones_path = str(tmp_path / "zones.json")
        with open(zones_path, "w") as f:
            json.dump(zones_data, f)

        dets = self._make_detections(n=1)
        detector = self._make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig(
            zones_path=zones_path,
            rules=[{"event_type": "inline_rule", "requires_any": ["person"]}],
        )
        pipeline = SpatialPipeline(detector, config, spatial_config)
        results = pipeline.run(input_path, output_path)

        event_types = [e["event_type"] for e in results.events]
        assert "inline_rule" in event_types
        assert "file_rule" not in event_types

    def test_progress_callback_exception_does_not_kill_pipeline(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=3)

        def bad_callback(p, t):
            raise RuntimeError("callback boom")

        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        # Should complete without raising
        results = pipeline.run(input_path, output_path, progress_callback=bad_callback)
        assert len(results.frame_results) == 3

    def test_json_export_atomic_no_partial_file(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        _make_test_video(input_path, num_frames=3)

        detector = self._make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        # JSON file should exist; temp file should not
        assert (tmp_path / "results.json").exists()
        assert not (tmp_path / "results.json.tmp").exists()
