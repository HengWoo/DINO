from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv

from dino.annotation.annotator import FrameAnnotator
from dino.annotation.zone_annotator import ZoneAnnotator
from dino.config import PipelineConfig, SpatialConfig
from dino.detectors.base import BaseDetector
from dino.spatial.depth_cloud_writer import DepthCloudWriter
from dino.spatial.ego_motion import EgoMotionEstimator
from dino.spatial.fixed_camera_localizer import FixedCameraLocalizer
from dino.spatial.models import WorldObject
from dino.spatial.registry import SpatialObjectRegistry
from dino.tracking.tracker import ObjectTracker
from dino.zones.event_rules import EventRule, RuleEngine
from dino.zones.loader import load_zones_file
from dino.zones.models import ZoneDefinition, ZoneEvent, ZoneObservation
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

        self._tracker = ObjectTracker(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
        )
        self._annotator = FrameAnnotator(trace_length=config.trace_length)
        self._zone_annotator = ZoneAnnotator()

        # Localizer
        self._camera_intrinsics = None
        self._camera_pose = None
        self._depth_estimator = None
        self._ego_motion: EgoMotionEstimator | None = None
        self._cloud_writer: DepthCloudWriter | None = None

        if spatial_config.camera_mode == "fixed":
            self._localizer = FixedCameraLocalizer()
        elif spatial_config.camera_mode == "depth":
            # Depth localizer needs video dimensions for camera estimation.
            # We load the model eagerly but defer localizer creation until run().
            from dino.spatial.depth_estimator import DepthEstimator

            self._depth_estimator = DepthEstimator(
                model_id=spatial_config.depth_model,
            )
            self._localizer = None  # finalized in run() once video dimensions are known
        else:
            raise ValueError(f"Unsupported camera_mode: {spatial_config.camera_mode!r}")

        self._registry = SpatialObjectRegistry(
            match_distance=spatial_config.match_distance,
            max_age_seconds=spatial_config.max_age_seconds,
        )

        # Load zones and rules
        self._zones: list[ZoneDefinition] = []
        rules: list[EventRule] = []

        if spatial_config.zones_path:
            self._zones, file_rules = load_zones_file(spatial_config.zones_path)
            rules = file_rules

        # Inline rules override file rules
        if spatial_config.rules:
            if rules:
                logger.warning(
                    "Inline rules override %d rules from zones file", len(rules)
                )
            rules = []
            for i, r in enumerate(spatial_config.rules):
                if "event_type" not in r:
                    raise ValueError(
                        f"Inline rule at index {i} is missing required key 'event_type'"
                    )
                try:
                    rules.append(EventRule(
                        event_type=r["event_type"],
                        requires_all=r.get("requires_all"),
                        requires_any=r.get("requires_any"),
                        requires_none=r.get("requires_none"),
                        min_count=r.get("min_count"),
                        hysteresis=r.get("hysteresis", 1),
                    ))
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid inline rule at index {i} "
                        f"(event_type={r.get('event_type', '?')}): {e}"
                    ) from e

        self._zone_manager = ZoneManager(zones=self._zones if self._zones else None)
        self._rule_engine = RuleEngine(rules=rules)

    def run(
        self,
        input_path: str,
        output_path: str,
        json_output: str | None = None,
        point_cloud_output: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SpatialResults:
        """Process a video file with full spatial pipeline.

        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video.
            json_output: Optional path to export spatial results as JSON.
            point_cloud_output: Optional path to export depth point clouds as binary .bin.
            progress_callback: Optional callback(processed_count, total_frames).

        Returns:
            SpatialResults with per-frame data, observations, and events.
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        video_info = sv.VideoInfo.from_video_path(input_path)

        # Finalize depth localizer now that we know video dimensions
        if self.spatial_config.camera_mode == "depth":
            from dino.spatial.camera_estimator import estimate_camera
            from dino.spatial.depth_localizer import DepthLocalizer

            intrinsics, pose = estimate_camera(
                video_info.width, video_info.height,
                self.spatial_config.camera_fov_deg,
            )
            self._localizer = DepthLocalizer(intrinsics, pose, self._depth_estimator)
            self._camera_intrinsics = intrinsics
            self._camera_pose = pose

            # Ego-motion estimator for camera trail (only when needed)
            if point_cloud_output or json_output:
                self._ego_motion = EgoMotionEstimator(intrinsics)

            # Point cloud writer
            if point_cloud_output:
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :3] = pose.rotation
                pose_4x4[:3, 3] = pose.translation
                self._cloud_writer = DepthCloudWriter(
                    point_cloud_output, intrinsics, pose_4x4
                )
                self._cloud_writer.open()

        if video_info.fps <= 0:
            raise ValueError(
                f"Input video reports fps={video_info.fps}. "
                f"A positive FPS is required for correct timestamps and output video."
            )
        adjusted_fps = (
            video_info.fps / self.config.stride
            if self.config.stride > 1
            else video_info.fps
        )
        total_frames = max(1, math.ceil(video_info.total_frames / self.config.stride))
        output_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=adjusted_fps,
            total_frames=total_frames,
        )

        frame_generator = sv.get_video_frames_generator(input_path)

        results = SpatialResults()
        processed = 0

        try:
            with sv.VideoSink(output_path, output_info) as sink:
                for frame_idx, frame in enumerate(frame_generator):
                    if frame_idx % self.config.stride != 0:
                        continue

                    timestamp = frame_idx / video_info.fps if video_info.fps > 0 else 0.0

                    try:
                        frame_result = self._process_frame(
                            frame, frame_idx, timestamp, sink
                        )
                    except (TypeError, ValueError, OSError) as e:
                        raise RuntimeError(
                            f"Error processing frame {frame_idx}: {e}"
                        ) from e

                    results.frame_results.append(frame_result)

                    # Accumulate observations and events
                    results.observations.extend(frame_result.get("observations", []))
                    results.events.extend(frame_result.get("events", []))

                    processed += 1
                    if progress_callback is not None:
                        try:
                            progress_callback(processed, total_frames)
                        except Exception:
                            logger.error(
                                "progress_callback raised an exception; "
                                "disabling further callbacks for this run",
                                exc_info=True,
                            )
                            progress_callback = None
        finally:
            if self._cloud_writer is not None:
                self._cloud_writer.close()

        if json_output:
            json_path = Path(json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = json_path.with_suffix(".json.tmp")
            try:
                camera_meta = None
                if self._camera_intrinsics is not None:
                    ci = self._camera_intrinsics
                    camera_meta = {
                        "position": self._camera_pose.translation.tolist(),
                        "rotation": self._camera_pose.rotation.tolist(),
                        "fov_deg": self.spatial_config.camera_fov_deg,
                        "intrinsics": {
                            "fx": ci.fx, "fy": ci.fy,
                            "cx": ci.cx, "cy": ci.cy,
                            "width": ci.width, "height": ci.height,
                        },
                    }

                camera_trail = None
                if self._ego_motion is not None:
                    camera_trail = self._ego_motion.get_all_poses()

                with open(tmp_path, "w") as f:
                    json.dump(
                        {
                            "metadata": {
                                "fps": float(adjusted_fps),
                                "width": video_info.width,
                                "height": video_info.height,
                                "total_frames": total_frames,
                                "duration_sec": round(total_frames / adjusted_fps, 3),
                                "stride": self.config.stride,
                                "camera": camera_meta,
                                "camera_trail": camera_trail,
                            },
                            "zones": [
                                {"zone_id": z.zone_id, "name": z.name, "polygon": z.polygon.tolist()}
                                for z in self._zones
                            ],
                            "frames": results.frame_results,
                            "observations": results.observations,
                            "events": results.events,
                        },
                        f,
                        indent=2,
                    )
                tmp_path.replace(json_path)
            except Exception as e:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(
                    f"Failed to write JSON results to {json_output}: {e}"
                ) from e

        return results

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        sink: sv.VideoSink,
    ) -> dict:
        """Process a single frame through the full spatial pipeline."""
        logger.debug("Processing frame %d (t=%.3f)", frame_idx, timestamp)

        # Detect
        detections = self.detector.detect(frame, self.config.prompts)

        # Track
        detections = self._tracker.update(detections)

        # Localize
        if self._localizer is None:
            raise RuntimeError(
                "Localizer not initialized. In depth mode, run() must be called "
                "to finalize the localizer with video dimensions."
            )
        world_objects = self._localizer.localize(detections, frame, frame_idx)

        # Ego-motion update + point cloud export (depth mode only)
        if self._ego_motion is not None:
            frame_pose = self._ego_motion.update(frame)
            if self._cloud_writer is not None:
                depth_map = self._localizer.last_depth_map
                if depth_map is not None:
                    self._cloud_writer.write_frame(depth_map, frame_pose, rgb_frame=frame)

        # Set timestamp on all objects
        for obj in world_objects:
            obj.timestamp = timestamp

        # Register (persistent identity)
        for i, obj in enumerate(world_objects):
            world_objects[i] = self._registry.register(obj)

        # Zone update
        observations = self._zone_manager.update(world_objects, frame_idx, timestamp)

        # Rule evaluation
        events: list[ZoneEvent] = []
        for zone in self._zones:
            state = self._zone_manager.get_zone_state(zone.zone_id)
            if state is not None:
                zone_events = self._rule_engine.evaluate(state, frame_idx, timestamp)
                events.extend(zone_events)

        # Annotate: base (boxes/labels/traces)
        annotated = self._annotator.annotate(frame, detections)

        # Annotate: zones (gated by config flag)
        if self._zones and self.spatial_config.annotate_zones_on_video:
            zone_states = {}
            for zone in self._zones:
                state = self._zone_manager.get_zone_state(zone.zone_id)
                if state is not None:
                    zone_states[zone.zone_id] = state
            annotated = self._zone_annotator.annotate(annotated, self._zones, zone_states)

        # Write to sink
        sink.write_frame(annotated)

        logger.debug(
            "Frame %d: %d objects, %d observations, %d events",
            frame_idx, len(world_objects), len(observations), len(events),
        )

        return self._build_frame_result(
            frame_idx, timestamp, world_objects, observations, events
        )

    @staticmethod
    def _build_frame_result(
        frame_idx: int,
        timestamp: float,
        objects: list[WorldObject],
        observations: list[ZoneObservation],
        events: list[ZoneEvent],
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
                    "bbox_3d": obj.bbox_3d.tolist() if obj.bbox_3d is not None else None,
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
    def _observation_to_dict(obs: ZoneObservation) -> dict:
        return {
            "zone_id": obs.zone_id,
            "persistent_id": obs.persistent_id,
            "observation_type": obs.observation_type.value,
            "frame_idx": obs.frame_idx,
            "timestamp": obs.timestamp,
        }

    @staticmethod
    def _event_to_dict(event: ZoneEvent) -> dict:
        return {
            "zone_id": event.zone_id,
            "event_type": event.event_type,
            "frame_idx": event.frame_idx,
            "timestamp": event.timestamp,
            "details": dict(event.details) if event.details else {},
        }
