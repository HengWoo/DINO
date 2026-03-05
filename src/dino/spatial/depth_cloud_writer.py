"""Binary point cloud writer for depth map export."""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from dino.spatial.models import CameraIntrinsics

logger = logging.getLogger(__name__)


class DepthCloudWriter:
    """Sample depth maps to point clouds and write binary .bin files.

    Binary format:
        Header (8 bytes): uint32 num_frames, uint32 index_offset
        Frame data: [uint32 num_points, float32[num_points*3] xyz] per frame
        Index (at index_offset): [uint64 byte_offset, uint32 num_points] per frame
    """

    def __init__(
        self,
        output_path: str | Path,
        intrinsics: CameraIntrinsics,
        pose_matrix: np.ndarray,
        grid_step: int = 8,
    ):
        self.output_path = Path(output_path)
        self.intrinsics = intrinsics
        self.pose_matrix = pose_matrix  # 4x4 world_T_cam default pose
        self.grid_step = grid_step
        self.frame_offsets: list[tuple[int, int]] = []
        self._file: BinaryIO | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "wb")
        # Reserve header: num_frames (u32) + index_offset (u32)
        self._file.write(b"\x00" * 8)

    def write_frame(
        self, depth_map: np.ndarray, frame_pose: np.ndarray | None = None
    ) -> None:
        """Sample grid from HxW depth map, backproject to world, write float32 xyz."""
        if self._file is None:
            raise RuntimeError("Writer not opened. Call open() first.")

        h, w = depth_map.shape
        pose = frame_pose if frame_pose is not None else self.pose_matrix

        us = np.arange(0, w, self.grid_step)
        vs = np.arange(0, h, self.grid_step)
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.flatten(), vv.flatten()
        depths = depth_map[vv, uu]

        valid = depths > 0.1
        uu, vv, depths = uu[valid], vv[valid], depths[valid]

        # Backproject using intrinsics
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        x_cam = (uu - cx) * depths / fx
        y_cam = (vv - cy) * depths / fy
        z_cam = depths

        # Transform to world: pose is 4x4 world_T_cam
        pts_cam = np.stack(
            [x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=0
        )  # 4xN
        pts_world = (pose @ pts_cam)[:3].T  # Nx3
        pts_f32 = pts_world.astype(np.float32)

        offset = self._file.tell()
        n_points = len(pts_f32)
        self._file.write(struct.pack("<I", n_points))
        self._file.write(pts_f32.tobytes())
        self.frame_offsets.append((offset, n_points))

    def close(self) -> None:
        """Write frame index at end and update header."""
        if self._file is None:
            return
        index_offset = self._file.tell()
        if index_offset > 0xFFFFFFFF:
            logger.error(
                "Point cloud file exceeds 4GB (%d bytes). "
                "Increase grid_step or reduce frame count.",
                index_offset,
            )
        for off, n in self.frame_offsets:
            self._file.write(struct.pack("<QI", off, n))
        # Update header
        self._file.seek(0)
        self._file.write(struct.pack("<II", len(self.frame_offsets), index_offset))
        self._file.close()
        self._file = None
        logger.info(
            "Point cloud written: %d frames to %s",
            len(self.frame_offsets),
            self.output_path,
        )
