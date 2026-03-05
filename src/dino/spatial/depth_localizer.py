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
        self.last_depth_map: np.ndarray | None = None

    def localize(
        self, detections: sv.Detections, frame: np.ndarray, frame_idx: int
    ) -> list[WorldObject]:
        depth_map = self._depth.estimate(frame)
        self.last_depth_map = depth_map

        if len(detections) == 0:
            return []
        h, w = depth_map.shape

        if detections.tracker_id is None:
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
            # Compute 3D bounding box
            obj.bbox_3d = self._compute_bbox_3d(
                obj.bbox_xyxy, depth_map, self._intrinsics, self._pose
            )

            objects.append(obj)

        return objects

    @staticmethod
    def _compute_bbox_3d(
        bbox_xyxy: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        pose: CameraPose,
    ) -> np.ndarray | None:
        """Compute 8 world-space corners of a 3D bounding box.

        Samples depth at the 4 bbox corners + center, uses median as
        front-face depth, extrudes by a fixed offset for back face.
        Returns (8, 3) array of world-space corners.
        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = depth_map.shape
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Sample depth at 4 corners + center
        sample_points = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, cy)
        ]
        depths = []
        for u, v in sample_points:
            ui = max(0, min(int(u), w - 1))
            vi = max(0, min(int(v), h - 1))
            d = float(depth_map[vi, ui])
            if d > _DEPTH_EPSILON:
                depths.append(d)

        if not depths:
            logger.debug(
                "All depth samples below threshold for bbox [%.0f,%.0f,%.0f,%.0f]; "
                "returning None for bbox_3d",
                x1, y1, x2, y2,
            )
            return None

        front_depth = float(np.median(depths))
        back_depth = front_depth + 0.05

        # Backproject 4 corners at front and back depths → 8 points
        # Clamp to image bounds to avoid mirrored world points from negative coords
        corners_2d = [
            (max(0.0, min(float(x1), w - 1)), max(0.0, min(float(y1), h - 1))),
            (max(0.0, min(float(x2), w - 1)), max(0.0, min(float(y1), h - 1))),
            (max(0.0, min(float(x2), w - 1)), max(0.0, min(float(y2), h - 1))),
            (max(0.0, min(float(x1), w - 1)), max(0.0, min(float(y2), h - 1))),
        ]
        corners_3d = []
        for depth in (front_depth, back_depth):
            for u, v in corners_2d:
                pt = backproject_to_world(u, v, depth, intrinsics, pose)
                corners_3d.append(pt)

        return np.array(corners_3d)  # (8, 3)
