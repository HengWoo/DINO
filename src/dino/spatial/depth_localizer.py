"""Depth-based localizer using monocular depth estimation + camera projection."""
from __future__ import annotations

import logging

import numpy as np
import supervision as sv

from dino.spatial.base_localizer import BaseLocalizer
from dino.spatial.depth_estimator import DepthEstimator
from dino.spatial.models import CameraIntrinsics, CameraPose, WorldObject
from dino.spatial.projection import backproject_to_world

logger = logging.getLogger(__name__)

# Minimum depth value to avoid division-by-zero in backprojection
_DEPTH_EPSILON = 1e-3


class DepthLocalizer(BaseLocalizer):
    """Localizer that uses monocular depth estimation + camera model.

    Samples depth at the bottom-center of each bounding box (the object's
    base/feet) and backprojects to 3D world coordinates.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        pose: CameraPose,
        depth_estimator: DepthEstimator,
    ):
        self._intrinsics = intrinsics
        self._pose = pose
        self._depth = depth_estimator

    def localize(
        self, detections: sv.Detections, frame: np.ndarray, frame_idx: int
    ) -> list[WorldObject]:
        depth_map = self._depth.estimate(frame)
        h, w = depth_map.shape

        if detections.tracker_id is None and len(detections) > 0:
            logger.warning(
                "Detections have no tracker_id — all %d objects will get "
                "tracker_id=-1, which causes identity collision in the registry.",
                len(detections),
            )

        objects: list[WorldObject] = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]

            # Sample depth at bottom-center of bbox (base of object)
            u = (x1 + x2) / 2.0
            v = float(y2)  # bottom of bbox

            depth_val = float(
                depth_map[min(int(v), h - 1), min(int(u), w - 1)]
            )

            # Clamp to epsilon to avoid backprojection errors
            depth_val = max(depth_val, _DEPTH_EPSILON)

            world_pos = backproject_to_world(
                u, v, depth_val, self._intrinsics, self._pose
            )

            obj = WorldObject(
                tracker_id=(
                    int(detections.tracker_id[i])
                    if detections.tracker_id is not None
                    else -1
                ),
                bbox_xyxy=detections.xyxy[i].copy(),
                world_position=world_pos,
                class_name=(
                    str(detections.data["class_name"][i])
                    if "class_name" in detections.data
                    else ""
                ),
                class_id=(
                    int(detections.class_id[i])
                    if detections.class_id is not None
                    else -1
                ),
                confidence=(
                    float(detections.confidence[i])
                    if detections.confidence is not None
                    else 0.0
                ),
                frame_idx=frame_idx,
            )
            objects.append(obj)

        return objects
