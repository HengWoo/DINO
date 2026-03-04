"""Shared test helpers for the DINO test suite."""
from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import supervision as sv


def make_test_video(path: str, num_frames: int = 10, fps: int = 30):
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


def make_mock_detector(detections=None):
    """Create a mock detector that returns the given detections."""
    detector = MagicMock()
    if detections is None:
        detector.detect.return_value = sv.Detections.empty()
    else:
        detector.detect.return_value = detections
    return detector


def make_detections(n=1):
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
