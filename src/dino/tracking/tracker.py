from __future__ import annotations

import supervision as sv


class ObjectTracker:
    """ByteTrack object tracker wrapper."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new detections, returns detections with tracker_id."""
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.reset()
