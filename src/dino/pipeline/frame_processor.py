from __future__ import annotations

import numpy as np
import supervision as sv

from dino.detectors.base import BaseDetector
from dino.tracking.tracker import ObjectTracker


class FrameProcessor:
    """Process a single frame: detect objects and update tracking."""

    def __init__(self, detector: BaseDetector, tracker: ObjectTracker | None = None):
        self.detector = detector
        self.tracker = tracker or ObjectTracker()

    def process(
        self, frame: np.ndarray, prompts: list[str]
    ) -> sv.Detections:
        """Detect objects and assign tracking IDs."""
        detections = self.detector.detect(frame, prompts)
        detections = self.tracker.update(detections)
        return detections
