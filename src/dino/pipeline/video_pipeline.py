from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from dino.annotation.annotator import FrameAnnotator
from dino.config import PipelineConfig
from dino.detectors.base import BaseDetector
from dino.tracking.tracker import ObjectTracker


class VideoPipeline:
    """End-to-end video processing: detect, track, annotate, export."""

    def __init__(self, detector: BaseDetector, config: PipelineConfig):
        self.detector = detector
        self.config = config
        self.tracker = ObjectTracker(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
        )
        self.annotator = FrameAnnotator(trace_length=config.trace_length)

    def run(
        self,
        input_path: str,
        output_path: str,
        json_output: str | None = None,
    ) -> list[dict]:
        """Process a video file end-to-end.

        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video.
            json_output: Optional path to export per-frame results as JSON.

        Returns:
            List of per-frame result dicts.
        """
        video_info = sv.VideoInfo.from_video_path(input_path)
        frame_generator = sv.get_video_frames_generator(input_path)

        results: list[dict] = []

        with sv.VideoSink(output_path, video_info) as sink:
            for frame_idx, frame in enumerate(frame_generator):
                if frame_idx % self.config.stride != 0:
                    continue

                detections = self.detector.detect(frame, self.config.prompts)
                detections = self.tracker.update(detections)

                annotated = self.annotator.annotate(frame, detections)
                sink.write_frame(annotated)

                frame_result = self._detections_to_dict(frame_idx, detections)
                results.append(frame_result)

        if json_output:
            Path(json_output).parent.mkdir(parents=True, exist_ok=True)
            with open(json_output, "w") as f:
                json.dump(results, f, indent=2)

        return results

    @staticmethod
    def _detections_to_dict(frame_idx: int, detections: sv.Detections) -> dict:
        """Convert frame detections to a serializable dict."""
        det_list = []
        for i in range(len(detections)):
            det = {
                "bbox": detections.xyxy[i].tolist(),
            }
            if detections.confidence is not None:
                det["confidence"] = float(detections.confidence[i])
            if detections.class_id is not None and len(detections.class_id) > i:
                det["class_id"] = int(detections.class_id[i])
            if detections.tracker_id is not None and len(detections.tracker_id) > i:
                det["tracker_id"] = int(detections.tracker_id[i])
            if "class_name" in detections.data and len(detections.data["class_name"]) > i:
                det["class_name"] = str(detections.data["class_name"][i])
            det_list.append(det)

        return {"frame": frame_idx, "detections": det_list}
