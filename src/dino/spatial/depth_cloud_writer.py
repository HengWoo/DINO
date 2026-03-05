"""Binary point cloud writer for depth map export."""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from dino.spatial.models import CameraIntrinsics

logger = logging.getLogger(__name__)

# V2 magic number: "DINO" in little-endian
_V2_MAGIC = 0x44494E4F
_V2_HEADER_SIZE = 12  # uint32 magic + uint16 version + uint16 flags + uint32 indexOffset
_LEGACY_HEADER_SIZE = 8  # uint32 num_frames + uint32 index_offset


class DepthCloudWriter:
    """Sample depth maps to point clouds and write binary .bin files.

    Binary format (V1 legacy, when rgb_frame is never provided):
        Header (8 bytes): uint32 num_frames, uint32 index_offset
        Frame data: [uint32 num_points, float32[num_points*3] xyz] per frame
        Index (at index_offset): [uint64 byte_offset, uint32 num_points] per frame

    Binary format (V2, when rgb_frame is provided on first call):
        Header (12 bytes): uint32 magic(0x44494E4F), uint16 version(2),
                           uint16 flags(bit0=has_rgb), uint32 indexOffset
        Frame data: [uint32 num_points, float32[N*3] xyz, uint8[N*3] rgb] per frame
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
        self._v2: bool | None = None  # None = not decided yet

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "wb")
        # Reserve max header space (12 bytes for V2, 8 for legacy)
        # We'll decide format on first write_frame call
        self._file.write(b"\x00" * _V2_HEADER_SIZE)

    def write_frame(
        self,
        depth_map: np.ndarray,
        frame_pose: np.ndarray | None = None,
        rgb_frame: np.ndarray | None = None,
    ) -> None:
        """Sample grid from HxW depth map, backproject to world, write xyz (+ optional rgb)."""
        if self._file is None:
            raise RuntimeError("Writer not opened. Call open() first.")

        # Decide format on first frame
        if self._v2 is None:
            self._v2 = rgb_frame is not None
            if not self._v2:
                # Legacy mode: rewind to 8-byte header
                self._file.seek(0)
                self._file.write(b"\x00" * _LEGACY_HEADER_SIZE)
                self._file.seek(_LEGACY_HEADER_SIZE)

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

        # Write RGB if V2
        if self._v2:
            if rgb_frame is not None:
                # Sample BGR frame at valid grid positions, convert to RGB
                rgb_samples = rgb_frame[vv, uu][:, ::-1]  # BGR → RGB
                self._file.write(rgb_samples.astype(np.uint8).tobytes())
            else:
                logger.warning(
                    "V2 format active but rgb_frame is None on frame %d; "
                    "writing zero-filled RGB to preserve binary layout.",
                    len(self.frame_offsets),
                )
                self._file.write(b"\x00" * (n_points * 3))

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
        if self._v2:
            flags = 0x0001  # has_rgb
            self._file.write(struct.pack("<IHHI", _V2_MAGIC, 2, flags, index_offset))
        else:
            self._file.write(struct.pack("<II", len(self.frame_offsets), index_offset))

        self._file.close()
        self._file = None
        logger.info(
            "Point cloud written: %d frames (%s) to %s",
            len(self.frame_offsets),
            "V2+RGB" if self._v2 else "legacy",
            self.output_path,
        )
