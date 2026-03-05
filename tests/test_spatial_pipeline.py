"""Tests for spatial pipeline components."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.helpers import make_test_video, make_mock_detector, make_detections
from dino.config import PipelineConfig, SpatialConfig


class TestSpatialConfig:
    def test_defaults(self):
        config = SpatialConfig()
        assert config.camera_mode == "fixed"
        assert config.match_distance == 50.0
        assert config.max_age_seconds == 30.0
        assert config.zones_path is None
        assert config.rules is None
        assert config.depth_model is None
        assert config.camera_fov_deg == 70.0
        assert config.annotate_zones_on_video is True

    def test_depth_mode_valid(self):
        config = SpatialConfig(camera_mode="depth")
        assert config.camera_mode == "depth"

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

    def test_invalid_polygon_too_few_vertices(self, tmp_path):
        from dino.zones.loader import load_zones_file

        zones_data = {
            "zones": [
                {"zone_id": "z1", "name": "Bad Zone", "polygon": [[0, 0], [100, 0]]},
            ],
        }
        path = tmp_path / "zones.json"
        path.write_text(json.dumps(zones_data))

        with pytest.raises(ValueError, match="Invalid zone entry 0"):
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
    def test_run_produces_output_no_zones(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        make_test_video(input_path, num_frames=5)

        detector = make_mock_detector()
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
        make_test_video(input_path, num_frames=5)

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

        dets = make_detections(n=1)
        detector = make_mock_detector(dets)
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

        detector = make_mock_detector()
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
        make_test_video(input_path, num_frames=3)

        detector = make_mock_detector()
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
        make_test_video(input_path, num_frames=3)

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

        dets = make_detections(n=1)
        detector = make_mock_detector(dets)
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
        make_test_video(input_path, num_frames=6)

        calls = []
        detector = make_mock_detector()
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
        make_test_video(input_path, num_frames=9)

        detector = make_mock_detector()
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
        make_test_video(input_path, num_frames=3)

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

        dets = make_detections(n=1)
        detector = make_mock_detector(dets)
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
        make_test_video(input_path, num_frames=3)

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

        dets = make_detections(n=1)
        detector = make_mock_detector(dets)
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

    def test_progress_callback_exception_disables_callback(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        make_test_video(input_path, num_frames=3)

        call_count = 0

        def bad_callback(p, t):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("callback boom")

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        # Should complete without raising
        results = pipeline.run(input_path, output_path, progress_callback=bad_callback)
        assert len(results.frame_results) == 3
        # Callback should be called once, then disabled after failure
        assert call_count == 1

    def test_json_export_atomic_no_partial_file(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        make_test_video(input_path, num_frames=3)

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        # JSON file should exist; temp file should not
        assert (tmp_path / "results.json").exists()
        assert not (tmp_path / "results.json.tmp").exists()

    def test_detector_exception_wraps_with_frame_context(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        make_test_video(input_path, num_frames=3)

        detector = make_mock_detector()
        detector.detect.side_effect = ValueError("model OOM")
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)

        with pytest.raises(RuntimeError, match="Error processing frame 0") as exc_info:
            pipeline.run(input_path, output_path)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_invalid_inline_rule_no_conditions_raises(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig(
            rules=[{"event_type": "bad_rule"}],
        )
        with pytest.raises(ValueError, match="Invalid inline rule at index 0"):
            SpatialPipeline(detector, config, spatial_config)

    def test_json_export_has_metadata(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        num_frames = 5
        make_test_video(input_path, num_frames=num_frames, fps=30)

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert "metadata" in data
        m = data["metadata"]
        assert isinstance(m["fps"], float)
        assert isinstance(m["width"], int)
        assert isinstance(m["height"], int)
        assert isinstance(m["total_frames"], int)
        assert isinstance(m["duration_sec"], float)
        assert isinstance(m["stride"], int)
        assert m["width"] == 320
        assert m["height"] == 240
        assert m["fps"] == 30.0
        assert m["total_frames"] == num_frames
        assert m["stride"] == 1
        # Verify computed values
        assert data["metadata"]["duration_sec"] == pytest.approx(5 / 30, abs=0.01)

    def test_json_export_has_zones(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        make_test_video(input_path, num_frames=5)

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

        dets = make_detections(n=1)
        detector = make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig(zones_path=zones_path)
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert "zones" in data
        assert isinstance(data["zones"], list)
        assert len(data["zones"]) == 1
        z = data["zones"][0]
        assert z["zone_id"] == "z1"
        assert z["name"] == "Zone 1"
        assert isinstance(z["polygon"], list)
        assert all(len(pt) == 2 for pt in z["polygon"])

    def test_json_export_metadata_with_stride(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        make_test_video(input_path, num_frames=9)

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"], stride=3)
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        m = data["metadata"]
        assert m["stride"] == 3
        assert m["fps"] == 10.0  # 30 / 3
        assert m["total_frames"] == 3  # ceil(9/3)
        assert m["duration_sec"] == pytest.approx(3 / 10, abs=0.01)  # 0.3

    def test_json_export_metadata_without_zones(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        num_frames = 5
        make_test_video(input_path, num_frames=num_frames, fps=30)

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig()
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["width"] == 320
        assert data["metadata"]["height"] == 240
        assert data["metadata"]["fps"] == 30.0
        assert data["metadata"]["total_frames"] == num_frames
        assert data["metadata"]["stride"] == 1
        assert "zones" in data
        assert data["zones"] == []


class TestSpatialPipelineDepthMode:
    """Integration tests for camera_mode='depth'."""

    @staticmethod
    def _mock_transformers_pipeline():
        """Mock the transformers pipeline to return fake depth maps."""
        mock_pipe = MagicMock()
        # Return a fake 240x320 uint8 depth image (will be normalized to float32)
        fake_depth = np.full((240, 320), 128, dtype=np.uint8)
        mock_pipe.return_value = {"depth": fake_depth}
        return mock_pipe

    def test_run_with_depth_mode(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        make_test_video(input_path, num_frames=3)

        dets = make_detections(n=1)
        detector = make_mock_detector(dets)
        config = PipelineConfig(prompts=["person"])

        mock_pipe = self._mock_transformers_pipeline()
        with patch(
            "dino.spatial.depth_estimator.pipeline",
            return_value=mock_pipe,
        ):
            spatial_config = SpatialConfig(camera_mode="depth")
            pipeline = SpatialPipeline(detector, config, spatial_config)
            results = pipeline.run(input_path, output_path, json_output=json_path)

        assert len(results.frame_results) == 3
        # Objects should have 3D positions (z != 0)
        obj = results.frame_results[0]["objects"][0]
        wp = obj["world_position"]
        assert len(wp) == 3
        assert wp[2] != 0.0  # non-flat

    def test_json_export_has_camera_metadata(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        make_test_video(input_path, num_frames=2)

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])

        mock_pipe = self._mock_transformers_pipeline()
        with patch(
            "dino.spatial.depth_estimator.pipeline",
            return_value=mock_pipe,
        ):
            spatial_config = SpatialConfig(camera_mode="depth", camera_fov_deg=60.0)
            pipeline = SpatialPipeline(detector, config, spatial_config)
            pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        cam = data["metadata"]["camera"]
        assert cam is not None
        assert isinstance(cam["position"], list)
        assert len(cam["position"]) == 3
        assert isinstance(cam["rotation"], list)
        assert len(cam["rotation"]) == 3
        assert cam["fov_deg"] == 60.0
        assert "intrinsics" in cam
        assert cam["intrinsics"]["width"] == 320
        assert cam["intrinsics"]["height"] == 240
        # Camera trail should be present in depth mode with json_output
        trail = data["metadata"]["camera_trail"]
        assert isinstance(trail, list)
        assert len(trail) > 0
        assert "position" in trail[0]

    def test_fixed_mode_has_no_camera_metadata(self, tmp_path):
        from dino.spatial.spatial_pipeline import SpatialPipeline

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        make_test_video(input_path, num_frames=2)

        detector = make_mock_detector()
        config = PipelineConfig(prompts=["person"])
        spatial_config = SpatialConfig(camera_mode="fixed")
        pipeline = SpatialPipeline(detector, config, spatial_config)
        pipeline.run(input_path, output_path, json_output=json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert data["metadata"]["camera"] is None
        assert data["metadata"]["camera_trail"] is None
