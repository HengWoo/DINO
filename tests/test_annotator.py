import numpy as np
import supervision as sv

from dino.annotation.annotator import FrameAnnotator


class TestFrameAnnotator:
    def test_annotate_returns_numpy_array(self):
        annotator = FrameAnnotator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = sv.Detections.empty()
        result = annotator.annotate(frame, detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_annotate_does_not_modify_original_frame(self):
        annotator = FrameAnnotator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        annotator.annotate(frame, detections)
        np.testing.assert_array_equal(frame, original)

    def test_annotate_with_custom_labels(self):
        annotator = FrameAnnotator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        result = annotator.annotate(frame, detections, labels=["custom label"])
        assert isinstance(result, np.ndarray)

    def test_build_labels_with_all_fields(self):
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            tracker_id=np.array([5]),
            data={"class_name": np.array(["person"])},
        )
        labels = FrameAnnotator._build_labels(detections)
        assert len(labels) == 1
        assert "person" in labels[0]
        assert "#5" in labels[0]
        assert "0.90" in labels[0]

    def test_build_labels_without_tracker_id(self):
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={"class_name": np.array(["person"])},
        )
        labels = FrameAnnotator._build_labels(detections)
        assert len(labels) == 1
        assert "person" in labels[0]
        assert "#" not in labels[0]

    def test_build_labels_without_class_name(self):
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        labels = FrameAnnotator._build_labels(detections)
        assert len(labels) == 1
        assert "0.90" in labels[0]

    def test_build_labels_empty_detections(self):
        detections = sv.Detections.empty()
        labels = FrameAnnotator._build_labels(detections)
        assert labels == []
