from __future__ import annotations

import logging

import numpy as np
import supervision as sv

from dino.spatial.base_localizer import BaseLocalizer
from dino.spatial.models import WorldObject

logger = logging.getLogger(__name__)


class FixedCameraLocalizer(BaseLocalizer):
    """Localizer for fixed/CCTV cameras.

    Identity mapping: world_position = bbox center in pixels [cx, cy, 0.0].
    No depth estimation or model loading required.
    """

    def localize(
        self, detections: sv.Detections, frame: np.ndarray, frame_idx: int
    ) -> list[WorldObject]:
        if detections.tracker_id is None and len(detections) > 0:
            logger.warning(
                "Detections have no tracker_id — all %d objects will get "
                "tracker_id=-1, which causes identity collision in the registry. "
                "Ensure a tracker (e.g., ByteTrack) is in the pipeline.",
                len(detections),
            )
        objects = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            obj = WorldObject(
                tracker_id=(
                    int(detections.tracker_id[i])
                    if detections.tracker_id is not None
                    else -1
                ),
                bbox_xyxy=detections.xyxy[i].copy(),
                world_position=np.array([cx, cy, 0.0]),
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
