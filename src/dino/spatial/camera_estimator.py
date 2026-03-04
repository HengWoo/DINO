"""Heuristic camera parameter estimation for typical CCTV/overhead cameras."""
from __future__ import annotations

import math

import numpy as np

from dino.spatial.models import CameraIntrinsics, CameraPose


def estimate_camera(
    width: int,
    height: int,
    fov_deg: float = 70.0,
) -> tuple[CameraIntrinsics, CameraPose]:
    """Estimate camera intrinsics and pose from video dimensions.

    Assumptions for a typical overhead/CCTV camera:
    - Horizontal FOV as specified (default 70 degrees)
    - Camera mounted above the scene, looking down at ~45 degrees
    - World origin at center of floor plane

    Args:
        width: Video width in pixels.
        height: Video height in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        Tuple of (CameraIntrinsics, CameraPose).
    """
    fx = fy = width / (2 * math.tan(math.radians(fov_deg / 2)))
    cx, cy = width / 2.0, height / 2.0
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    # Camera at scale-relative height, tilted 45 degrees down
    cam_height = max(width, height) * 0.8
    translation = np.array([0.0, cam_height, 0.0])

    angle = math.radians(45)
    rotation = np.array([
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)],
    ])

    pose = CameraPose(
        translation=translation,
        rotation=rotation,
        frame_idx=0,
    )

    return intrinsics, pose
