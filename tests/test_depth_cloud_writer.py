"""Tests for depth cloud binary writer."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from dino.spatial.depth_cloud_writer import DepthCloudWriter
from dino.spatial.models import CameraIntrinsics


class TestDepthCloudWriter:
    @pytest.fixture()
    def intrinsics(self):
        return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    @pytest.fixture()
    def pose_matrix(self):
        return np.eye(4)

    def test_write_single_frame(self, tmp_path, intrinsics, pose_matrix):
        output = tmp_path / "cloud.bin"
        writer = DepthCloudWriter(output, intrinsics, pose_matrix, grid_step=32)
        writer.open()

        depth_map = np.full((480, 640), 2.0, dtype=np.float32)
        writer.write_frame(depth_map)
        writer.close()

        assert output.exists()
        assert output.stat().st_size > 8

        # Parse header
        data = output.read_bytes()
        num_frames, index_offset = struct.unpack_from("<II", data, 0)
        assert num_frames == 1
        assert index_offset > 8

    def test_write_multiple_frames(self, tmp_path, intrinsics, pose_matrix):
        output = tmp_path / "cloud.bin"
        writer = DepthCloudWriter(output, intrinsics, pose_matrix, grid_step=64)
        writer.open()

        for _ in range(5):
            depth_map = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
            writer.write_frame(depth_map)
        writer.close()

        data = output.read_bytes()
        num_frames, index_offset = struct.unpack_from("<II", data, 0)
        assert num_frames == 5

        # Verify frame index entries
        for i in range(5):
            entry_off = index_offset + i * 12
            offset_low = struct.unpack_from("<Q", data, entry_off)[0]
            n_points = struct.unpack_from("<I", data, entry_off + 8)[0]
            assert n_points > 0
            assert offset_low >= 8

    def test_write_not_opened_raises(self, tmp_path, intrinsics, pose_matrix):
        output = tmp_path / "cloud.bin"
        writer = DepthCloudWriter(output, intrinsics, pose_matrix)
        with pytest.raises(RuntimeError, match="Writer not opened"):
            writer.write_frame(np.zeros((10, 10), dtype=np.float32))

    def test_close_without_open_no_crash(self, tmp_path, intrinsics, pose_matrix):
        writer = DepthCloudWriter(tmp_path / "cloud.bin", intrinsics, pose_matrix)
        writer.close()  # Should not raise

    def test_depth_below_threshold_filtered(self, tmp_path, intrinsics, pose_matrix):
        """Depth values <= 0.1 should be filtered out."""
        output = tmp_path / "cloud.bin"
        writer = DepthCloudWriter(output, intrinsics, pose_matrix, grid_step=1)
        writer.open()

        # All zeros — all should be filtered
        depth_map = np.zeros((4, 4), dtype=np.float32)
        writer.write_frame(depth_map)
        writer.close()

        data = output.read_bytes()
        # Read first frame: num_points should be 0
        n_points = struct.unpack_from("<I", data, 8)[0]
        assert n_points == 0

    def test_custom_frame_pose(self, tmp_path, intrinsics, pose_matrix):
        """Using a custom per-frame pose should not crash."""
        output = tmp_path / "cloud.bin"
        writer = DepthCloudWriter(output, intrinsics, pose_matrix, grid_step=64)
        writer.open()

        custom_pose = np.eye(4)
        custom_pose[0, 3] = 1.0  # translate x
        depth_map = np.full((480, 640), 2.0, dtype=np.float32)
        writer.write_frame(depth_map, frame_pose=custom_pose)
        writer.close()

        data = output.read_bytes()
        num_frames = struct.unpack_from("<I", data, 0)[0]
        assert num_frames == 1
