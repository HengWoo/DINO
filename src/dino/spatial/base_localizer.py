from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import supervision as sv

from dino.spatial.models import WorldObject


class BaseLocalizer(ABC):
    """Abstract base for all localizers.

    Converts 2D pixel-space detections to world-space objects.
    """

    @abstractmethod
    def localize(
        self, detections: sv.Detections, frame: np.ndarray, frame_idx: int
    ) -> list[WorldObject]:
        """Convert 2D detections to world-space objects.

        Args:
            detections: sv.Detections with xyxy boxes and optional metadata.
            frame: BGR image as numpy array (H, W, 3).
            frame_idx: Current frame index.

        Returns:
            List of WorldObject instances with world_position set.
        """
