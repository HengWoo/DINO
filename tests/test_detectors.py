from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from dino.detectors.base import BaseDetector
from dino.detectors.grounding_dino import GroundingDINODetector


def _make_mock_inputs():
    """Create a MagicMock that behaves like processor output (supports .to() and dict access)."""
    inputs = MagicMock()
    inputs.__getitem__ = MagicMock(side_effect=lambda k: MagicMock())
    inputs.to.return_value = inputs
    return inputs


def _setup_mocks(mock_model_cls, mock_proc_cls, boxes, scores, labels):
    """Set up processor and model mocks with given detection results."""
    mock_processor = MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_processor
    mock_processor.return_value = _make_mock_inputs()

    mock_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.device = "cpu"

    mock_processor.post_process_grounded_object_detection.return_value = [
        {
            "boxes": MagicMock(
                cpu=MagicMock(
                    return_value=MagicMock(
                        numpy=MagicMock(return_value=boxes)
                    )
                )
            ),
            "scores": MagicMock(
                cpu=MagicMock(
                    return_value=MagicMock(
                        numpy=MagicMock(return_value=scores)
                    )
                )
            ),
            "labels": labels,
        }
    ]
    return mock_processor, mock_model


class TestBaseDetector:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseDetector()

    def test_subclass_must_implement_detect(self):
        class BadDetector(BaseDetector):
            pass

        with pytest.raises(TypeError):
            BadDetector()


class TestGroundingDINODetector:
    @patch("dino.detectors.grounding_dino.AutoProcessor")
    @patch("dino.detectors.grounding_dino.AutoModelForZeroShotObjectDetection")
    def test_detect_returns_sv_detections(self, mock_model_cls, mock_proc_cls):
        _setup_mocks(
            mock_model_cls,
            mock_proc_cls,
            boxes=np.array([[10.0, 20.0, 100.0, 200.0]], dtype=np.float32),
            scores=np.array([0.85], dtype=np.float32),
            labels=["person"],
        )

        detector = GroundingDINODetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, ["person"])

        assert isinstance(detections, sv.Detections)
        assert len(detections) == 1
        assert detections.xyxy.shape == (1, 4)

    @patch("dino.detectors.grounding_dino.AutoProcessor")
    @patch("dino.detectors.grounding_dino.AutoModelForZeroShotObjectDetection")
    def test_detect_empty_results(self, mock_model_cls, mock_proc_cls):
        _setup_mocks(
            mock_model_cls,
            mock_proc_cls,
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.array([], dtype=np.float32),
            labels=[],
        )

        detector = GroundingDINODetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, ["unicorn"])

        assert isinstance(detections, sv.Detections)
        assert len(detections) == 0

    @patch("dino.detectors.grounding_dino.AutoProcessor")
    @patch("dino.detectors.grounding_dino.AutoModelForZeroShotObjectDetection")
    def test_prompts_joined_with_period(self, mock_model_cls, mock_proc_cls):
        mock_processor, _ = _setup_mocks(
            mock_model_cls,
            mock_proc_cls,
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.array([], dtype=np.float32),
            labels=[],
        )

        detector = GroundingDINODetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame, ["person", "table", "plate"])

        # Verify processor was called with period-separated prompts
        call_args = mock_processor.call_args
        text_arg = call_args[1]["text"]
        assert text_arg == "person. table. plate."

    @patch("dino.detectors.grounding_dino.AutoProcessor")
    @patch("dino.detectors.grounding_dino.AutoModelForZeroShotObjectDetection")
    def test_is_base_detector_subclass(self, mock_model_cls, mock_proc_cls):
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        detector = GroundingDINODetector(device="cpu")
        assert isinstance(detector, BaseDetector)

    @patch("dino.detectors.grounding_dino.AutoProcessor")
    @patch("dino.detectors.grounding_dino.AutoModelForZeroShotObjectDetection")
    def test_multiple_detections(self, mock_model_cls, mock_proc_cls):
        _setup_mocks(
            mock_model_cls,
            mock_proc_cls,
            boxes=np.array(
                [[10, 20, 100, 200], [300, 400, 500, 600]], dtype=np.float32
            ),
            scores=np.array([0.9, 0.7], dtype=np.float32),
            labels=["person", "table"],
        )

        detector = GroundingDINODetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, ["person", "table"])

        assert len(detections) == 2
        assert detections.confidence[0] == pytest.approx(0.9)
        assert detections.confidence[1] == pytest.approx(0.7)
