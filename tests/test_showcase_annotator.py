from __future__ import annotations

import numpy as np
import supervision as sv

from dino.annotation.annotator import FrameAnnotator
from dino.annotation.showcase_annotator import ShowcaseAnnotator


class TestShowcaseAnnotator:
    def _make_frame(self, h: int = 480, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _make_detections(self, n: int = 3) -> sv.Detections:
        xyxy = np.array([[10 * i, 10 * i, 50 + 10 * i, 50 + 10 * i] for i in range(n)], dtype=np.float32)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.9] * n, dtype=np.float32),
            class_id=np.array(list(range(n)), dtype=int),
            data={"class_name": np.array(["person", "table", "plate"][:n])},
        )

    def test_is_subclass_of_frame_annotator(self):
        assert issubclass(ShowcaseAnnotator, FrameAnnotator)

    def test_annotate_returns_numpy_array(self):
        annotator = ShowcaseAnnotator(prompts=["person", "table", "plate"])
        frame = self._make_frame()
        detections = self._make_detections()
        result = annotator.annotate(frame, detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_hud_differs_from_base(self):
        """ShowcaseAnnotator output should differ from base due to HUD overlay."""
        frame = self._make_frame()
        detections = self._make_detections()

        base = FrameAnnotator()
        base_result = base.annotate(frame, detections)

        showcase = ShowcaseAnnotator(prompts=["person", "table", "plate"])
        showcase_result = showcase.annotate(frame, detections)

        assert not np.array_equal(base_result, showcase_result)

    def test_color_lookup_is_class(self):
        annotator = ShowcaseAnnotator(prompts=["person"])
        assert annotator._box_annotator.color_lookup == sv.ColorLookup.CLASS
        assert annotator._label_annotator.color_lookup == sv.ColorLookup.CLASS
        assert annotator._trace_annotator.color_lookup == sv.ColorLookup.CLASS

    def test_works_with_empty_detections(self):
        annotator = ShowcaseAnnotator(prompts=["person"])
        frame = self._make_frame()
        detections = sv.Detections.empty()
        result = annotator.annotate(frame, detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_does_not_modify_original_frame(self):
        annotator = ShowcaseAnnotator(prompts=["person", "table"])
        frame = self._make_frame()
        original = frame.copy()
        detections = self._make_detections(2)
        annotator.annotate(frame, detections)
        np.testing.assert_array_equal(frame, original)
