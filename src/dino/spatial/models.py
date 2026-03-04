from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class CameraPose:
    """Camera extrinsic parameters (world-from-camera transform)."""

    translation: np.ndarray  # (3,) world position
    rotation: np.ndarray  # (3,3) rotation matrix
    frame_idx: int
    timestamp: float = 0.0


@dataclass
class WorldObject:
    """A detected object with world-space position."""

    tracker_id: int
    bbox_xyxy: np.ndarray  # (4,) pixel-space bounding box
    world_position: np.ndarray  # (3,) world-space position [x,y,z]
    persistent_id: int | None = None
    class_name: str = ""
    class_id: int = -1
    confidence: float = 0.0
    position_confidence: float = 1.0
    frame_idx: int = 0
    timestamp: float = 0.0
    zone_id: str | None = None
