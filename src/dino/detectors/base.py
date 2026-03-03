from abc import ABC, abstractmethod

import numpy as np
import supervision as sv


class BaseDetector(ABC):
    """Abstract base for all object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray, prompts: list[str]) -> sv.Detections:
        """Detect objects in a frame given text prompts.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            prompts: List of text descriptions to detect.

        Returns:
            sv.Detections with xyxy boxes, confidence scores, and class data.
        """
