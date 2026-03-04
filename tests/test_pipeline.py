import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv

from dino.config import PipelineConfig
from dino.pipeline.video_pipeline import VideoPipeline


def _make_test_video(path: str, num_frames: int = 10, fps: int = 30):
    """Create a small test video."""
    h, w = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (i * 25, i * 10, 0)  # varying colors
        writer.write(frame)
    writer.release()


class TestPipelineConfig:
    def test_default_prompts_required(self):
        with pytest.raises(ValueError, match="prompts must not be empty"):
            PipelineConfig()

    def test_custom_values(self):
        config = PipelineConfig(
            prompts=["person", "table"],
            stride=3,
        )
        assert config.prompts == ["person", "table"]
        assert config.stride == 3

    def test_invalid_stride_zero(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            PipelineConfig(prompts=["person"], stride=0)

    def test_invalid_stride_negative(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            PipelineConfig(prompts=["person"], stride=-1)

    def test_empty_prompts_rejected(self):
        with pytest.raises(ValueError, match="prompts must not be empty"):
            PipelineConfig(prompts=[])


class TestVideoPipeline:
    def test_process_video_creates_output(self, tmp_path):
        # Create test video
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=5)

        # Mock detector that returns empty detections
        mock_detector = MagicMock()
        mock_detector.detect.return_value = sv.Detections.empty()

        config = PipelineConfig(prompts=["person"], stride=1)
        pipeline = VideoPipeline(detector=mock_detector, config=config)
        pipeline.run(input_path, output_path)

        assert Path(output_path).exists()
        assert mock_detector.detect.call_count == 5

    def test_stride_skips_frames(self, tmp_path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=10)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = sv.Detections.empty()

        config = PipelineConfig(prompts=["person"], stride=3)
        pipeline = VideoPipeline(detector=mock_detector, config=config)
        pipeline.run(input_path, output_path)

        # With stride=3 on 10 frames: process frames 0, 3, 6, 9 = 4 frames
        assert mock_detector.detect.call_count == 4

    def test_json_results_exported(self, tmp_path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        json_path = str(tmp_path / "results.json")
        _make_test_video(input_path, num_frames=3)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        config = PipelineConfig(prompts=["person"], stride=1)
        pipeline = VideoPipeline(detector=mock_detector, config=config)
        pipeline.run(input_path, output_path, json_output=json_path)

        assert Path(json_path).exists()
        with open(json_path) as f:
            results = json.load(f)
        assert len(results) == 3
        assert "frame" in results[0]
        assert "detections" in results[0]

    def test_with_detections(self, tmp_path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=3)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={"class_name": np.array(["person"])},
        )

        config = PipelineConfig(prompts=["person"], stride=1)
        pipeline = VideoPipeline(detector=mock_detector, config=config)
        pipeline.run(input_path, output_path)

        assert Path(output_path).exists()

    def test_detections_to_dict_full_fields(self):
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.85]),
            class_id=np.array([2]),
            tracker_id=np.array([7]),
            data={"class_name": np.array(["person"])},
        )
        result = VideoPipeline._detections_to_dict(42, detections)
        assert result["frame"] == 42
        det = result["detections"][0]
        assert det["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert det["confidence"] == pytest.approx(0.85)
        assert det["class_id"] == 2
        assert det["tracker_id"] == 7
        assert det["class_name"] == "person"

    def test_detections_to_dict_empty(self):
        result = VideoPipeline._detections_to_dict(0, sv.Detections.empty())
        assert result["detections"] == []
        assert result["frame"] == 0

    def test_detections_to_dict_minimal_fields(self):
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
        )
        result = VideoPipeline._detections_to_dict(0, detections)
        det = result["detections"][0]
        assert "bbox" in det

    def test_input_not_found_raises(self, tmp_path):
        mock_detector = MagicMock()
        config = PipelineConfig(prompts=["person"])
        pipeline = VideoPipeline(detector=mock_detector, config=config)

        with pytest.raises(FileNotFoundError, match="Input video not found"):
            pipeline.run(str(tmp_path / "nonexistent.mp4"), str(tmp_path / "out.mp4"))

    def test_progress_callback_called(self, tmp_path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _make_test_video(input_path, num_frames=3)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = sv.Detections.empty()

        config = PipelineConfig(prompts=["person"])
        pipeline = VideoPipeline(detector=mock_detector, config=config)

        calls = []
        pipeline.run(input_path, output_path, progress_callback=lambda p, t: calls.append((p, t)))

        assert len(calls) == 3
        assert calls[-1][0] == 3  # processed count
