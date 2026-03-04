from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path

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
        progress_callback: Callable[[int, int], None] | None = None,
        sbs_output: str | None = None,
    ) -> list[dict]:
        """Process a video file end-to-end.

        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video.
            json_output: Optional path to export per-frame results as JSON.
            progress_callback: Optional callback(processed, total) called per frame.
            sbs_output: Optional path for side-by-side (original|annotated) video.

        Returns:
            List of per-frame result dicts.
        """
        video_info = sv.VideoInfo.from_video_path(input_path)
        if self.config.stride > 1:
            video_info.fps = video_info.fps / self.config.stride
        frame_generator = sv.get_video_frames_generator(input_path)

        total_frames = video_info.total_frames // self.config.stride

        sbs_info = None
        if sbs_output:
            sbs_info = sv.VideoInfo(
                width=video_info.width * 2,
                height=video_info.height,
                fps=video_info.fps,
                total_frames=video_info.total_frames,
            )

        results: list[dict] = []
        processed = 0

        with contextlib.ExitStack() as stack:
            sink = stack.enter_context(sv.VideoSink(output_path, video_info))
            sbs_sink = None
            if sbs_output and sbs_info:
                sbs_sink = stack.enter_context(sv.VideoSink(sbs_output, sbs_info))

            for frame_idx, frame in enumerate(frame_generator):
                if frame_idx % self.config.stride != 0:
                    continue

                detections = self.detector.detect(frame, self.config.prompts)
                detections = self.tracker.update(detections)

                annotated = self.annotator.annotate(frame, detections)
                sink.write_frame(annotated)

                if sbs_sink is not None:
                    sbs_frame = np.hstack([frame, annotated])
                    sbs_sink.write_frame(sbs_frame)

                frame_result = self._detections_to_dict(frame_idx, detections)
                results.append(frame_result)

                processed += 1
                if progress_callback is not None:
                    progress_callback(processed, total_frames)

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
