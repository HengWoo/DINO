from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from dino.annotation.annotator import FrameAnnotator
from dino.annotation.zone_annotator import ZoneAnnotator
from dino.config import PipelineConfig, SpatialConfig
from dino.detectors.base import BaseDetector
from dino.spatial.fixed_camera_localizer import FixedCameraLocalizer
from dino.spatial.registry import SpatialObjectRegistry
from dino.tracking.tracker import ObjectTracker
from dino.zones.event_rules import EventRule, RuleEngine
from dino.zones.loader import load_zones_file
from dino.zones.models import ZoneDefinition
from dino.zones.zone_manager import ZoneManager

logger = logging.getLogger(__name__)


@dataclass
class SpatialResults:
    """Results from a spatial pipeline run."""

    frame_results: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class SpatialPipeline:
    """End-to-end spatial video processing pipeline.

    detect -> track -> localize -> register -> zone update ->
    rule eval -> annotate -> export
    """

    def __init__(
        self,
        detector: BaseDetector,
        config: PipelineConfig,
        spatial_config: SpatialConfig,
    ):
        self.detector = detector
        self.config = config
        self.spatial_config = spatial_config

        self.tracker = ObjectTracker(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
        )
        self.annotator = FrameAnnotator(trace_length=config.trace_length)
        self.zone_annotator = ZoneAnnotator()

        # Localizer
        if spatial_config.camera_mode == "fixed":
            self.localizer = FixedCameraLocalizer()
        else:
            raise ValueError(f"Unsupported camera_mode: {spatial_config.camera_mode!r}")

        self.registry = SpatialObjectRegistry(
            match_distance=spatial_config.match_distance,
            max_age_seconds=spatial_config.max_age_seconds,
        )

        # Load zones and rules
        self.zones: list[ZoneDefinition] = []
        rules: list[EventRule] = []

        if spatial_config.zones_path:
            self.zones, file_rules = load_zones_file(spatial_config.zones_path)
            rules = file_rules

        # Inline rules override file rules
        if spatial_config.rules:
            rules = [
                EventRule(
                    event_type=r["event_type"],
                    requires_all=r.get("requires_all"),
                    requires_any=r.get("requires_any"),
                    requires_none=r.get("requires_none"),
                    min_count=r.get("min_count"),
                    hysteresis=r.get("hysteresis", 1),
                )
                for r in spatial_config.rules
            ]

        self.zone_manager = ZoneManager(zones=self.zones if self.zones else None)
        self.rule_engine = RuleEngine(rules=rules)

    def run(
        self,
        input_path: str,
        output_path: str,
        json_output: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SpatialResults:
        """Process a video file with full spatial pipeline.

        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video.
            json_output: Optional path to export spatial results as JSON.
            progress_callback: Optional callback(frame_idx, total_frames).

        Returns:
            SpatialResults with per-frame data, observations, and events.
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        video_info = sv.VideoInfo.from_video_path(input_path)
        adjusted_fps = (
            video_info.fps / self.config.stride
            if self.config.stride > 1
            else video_info.fps
        )
        output_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=adjusted_fps,
            total_frames=video_info.total_frames,
        )

        frame_generator = sv.get_video_frames_generator(input_path)
        total_frames = max(1, video_info.total_frames // self.config.stride)

        results = SpatialResults()

        with sv.VideoSink(output_path, output_info) as sink:
            for frame_idx, frame in enumerate(frame_generator):
                if frame_idx % self.config.stride != 0:
                    continue

                timestamp = frame_idx / video_info.fps if video_info.fps > 0 else 0.0
                frame_result = self._process_frame(
                    frame, frame_idx, timestamp, sink
                )
                results.frame_results.append(frame_result)

                # Accumulate observations and events
                results.observations.extend(frame_result.get("observations", []))
                results.events.extend(frame_result.get("events", []))

                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames)

        if json_output:
            Path(json_output).parent.mkdir(parents=True, exist_ok=True)
            with open(json_output, "w") as f:
                json.dump(
                    {
                        "frames": results.frame_results,
                        "observations": results.observations,
                        "events": results.events,
                    },
                    f,
                    indent=2,
                )

        return results

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        sink: sv.VideoSink,
    ) -> dict:
        """Process a single frame through the full spatial pipeline."""
        # Detect
        detections = self.detector.detect(frame, self.config.prompts)

        # Track
        detections = self.tracker.update(detections)

        # Localize
        world_objects = self.localizer.localize(detections, frame, frame_idx)

        # Set timestamp on all objects
        for obj in world_objects:
            obj.timestamp = timestamp

        # Register (persistent identity)
        for i, obj in enumerate(world_objects):
            world_objects[i] = self.registry.register(obj)

        # Zone update
        observations = self.zone_manager.update(world_objects, frame_idx, timestamp)

        # Rule evaluation
        events = []
        for zone in self.zones:
            state = self.zone_manager.get_zone_state(zone.zone_id)
            if state is not None:
                zone_events = self.rule_engine.evaluate(state, frame_idx, timestamp)
                events.extend(zone_events)

        # Annotate: base (boxes/labels/traces)
        annotated = self.annotator.annotate(frame, detections)

        # Annotate: zones
        if self.zones:
            zone_states = {}
            for zone in self.zones:
                state = self.zone_manager.get_zone_state(zone.zone_id)
                if state is not None:
                    zone_states[zone.zone_id] = state
            annotated = self.zone_annotator.annotate(annotated, self.zones, zone_states)

        # Write to sink
        sink.write_frame(annotated)

        return self._build_frame_result(
            frame_idx, timestamp, world_objects, observations, events
        )

    @staticmethod
    def _build_frame_result(
        frame_idx: int,
        timestamp: float,
        objects: list,
        observations: list,
        events: list,
    ) -> dict:
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "objects": [
                {
                    "persistent_id": obj.persistent_id,
                    "tracker_id": obj.tracker_id,
                    "class_name": obj.class_name,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox_xyxy.tolist(),
                    "world_position": obj.world_position.tolist(),
                    "zone_id": obj.zone_id,
                }
                for obj in objects
            ],
            "observations": [
                SpatialPipeline._observation_to_dict(obs) for obs in observations
            ],
            "events": [SpatialPipeline._event_to_dict(e) for e in events],
        }

    @staticmethod
    def _observation_to_dict(obs) -> dict:
        return {
            "zone_id": obs.zone_id,
            "persistent_id": obs.persistent_id,
            "observation_type": obs.observation_type.value,
            "frame_idx": obs.frame_idx,
            "timestamp": obs.timestamp,
        }

    @staticmethod
    def _event_to_dict(event) -> dict:
        return {
            "zone_id": event.zone_id,
            "event_type": event.event_type,
            "frame_idx": event.frame_idx,
            "timestamp": event.timestamp,
            "details": dict(event.details) if event.details else {},
        }
