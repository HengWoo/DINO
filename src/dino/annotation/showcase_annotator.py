from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import supervision as sv

from dino.annotation.annotator import FrameAnnotator


class ShowcaseAnnotator(FrameAnnotator):
    """Enhanced annotator with per-class colors and a HUD overlay."""

    def __init__(self, prompts: list[str], trace_length: int = 60):
        super().__init__(trace_length=trace_length)
        self.prompts = prompts
        self._box_annotator = sv.BoxAnnotator(
            color_lookup=sv.ColorLookup.CLASS,
        )
        self._label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.CLASS,
        )
        self._trace_annotator = sv.TraceAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            trace_length=trace_length,
            color_lookup=sv.ColorLookup.CLASS,
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        annotated = super().annotate(frame, detections, labels)
        annotated = self._draw_hud(annotated, detections)
        return annotated

    def _draw_hud(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw a semi-transparent HUD bar at the top of the frame."""
        _, w = frame.shape[:2]
        bar_height = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        total = len(detections)
        class_counts: Counter[str] = Counter()
        if "class_name" in detections.data:
            for name in detections.data["class_name"]:
                class_counts[str(name)] += 1

        parts = [f"Detections: {total}"]
        for name, count in class_counts.most_common():
            parts.append(f"{name}:{count}")
        text = "  ".join(parts)

        cv2.putText(
            frame,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame
