from __future__ import annotations

import numpy as np

from dino.spatial.models import CameraIntrinsics, CameraPose


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    The quaternion is normalized before conversion to ensure a valid
    rotation matrix even if the input is not exactly unit length.

    Raises:
        ValueError: If the quaternion has near-zero norm.
    """
    norm = float(np.linalg.norm(q))
    if norm < 1e-10:
        raise ValueError(f"quaternion has near-zero norm ({norm})")
    q = q / norm
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def backproject_pixel_to_camera(
    u: float, v: float, depth: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Backproject a pixel (u, v) at given depth to camera-frame 3D point.

    Uses pinhole camera model: X = (u - cx) * depth / fx, etc.

    Returns:
        (3,) array [X, Y, Z] in camera frame.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    z = depth
    return np.array([x, y, z])


def camera_to_world(
    point_camera: np.ndarray, pose: CameraPose
) -> np.ndarray:
    """Transform a point from camera frame to world frame.

    Args:
        point_camera: (3,) point in camera coordinates.
        pose: CameraPose with rotation (world-from-camera) and translation.

    Returns:
        (3,) point in world coordinates: R @ p + t
    """
    return pose.rotation @ point_camera + pose.translation


def backproject_to_world(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> np.ndarray:
    """Full pipeline: pixel (u,v) + depth -> world-space 3D point."""
    point_camera = backproject_pixel_to_camera(u, v, depth, intrinsics)
    return camera_to_world(point_camera, pose)
