from __future__ import annotations

import numpy as np
import supervision as sv


class FrameAnnotator:
    """Annotate frames with bounding boxes, labels, and tracking traces."""

    def __init__(self, trace_length: int = 60):
        self._box_annotator = sv.BoxAnnotator()
        self._label_annotator = sv.LabelAnnotator()
        self._trace_annotator = sv.TraceAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            trace_length=trace_length,
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        """Draw boxes, labels, and traces on a frame."""
        annotated = frame.copy()

        if labels is None:
            labels = self._build_labels(detections)

        annotated = self._box_annotator.annotate(
            scene=annotated, detections=detections
        )
        annotated = self._label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        if detections.tracker_id is not None:
            annotated = self._trace_annotator.annotate(
                scene=annotated, detections=detections
            )

        return annotated

    @staticmethod
    def _build_labels(detections: sv.Detections) -> list[str]:
        """Build default labels from detection data."""
        labels = []
        for i in range(len(detections)):
            parts = []
            if "class_name" in detections.data and len(detections.data["class_name"]) > i:
                parts.append(str(detections.data["class_name"][i]))
            if detections.tracker_id is not None:
                parts.append(f"#{detections.tracker_id[i]}")
            if detections.confidence is not None:
                parts.append(f"{detections.confidence[i]:.2f}")
            labels.append(" ".join(parts))
        return labels
