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

    def test_round_trip_point_data(self, tmp_path, intrinsics):
        """Read back point data and verify backprojection math."""
        pose = np.eye(4)
        output = tmp_path / "cloud.bin"
        # Use large grid_step so we get few, predictable points
        writer = DepthCloudWriter(output, intrinsics, pose, grid_step=640)
        writer.open()

        # Single depth value at pixel (0, 0), depth = 2.0
        depth_map = np.full((480, 640), 2.0, dtype=np.float32)
        writer.write_frame(depth_map)
        writer.close()

        data = output.read_bytes()
        n_points = struct.unpack_from("<I", data, 8)[0]
        assert n_points > 0

        # Parse float32 xyz triplets
        points = np.frombuffer(
            data, dtype=np.float32, count=n_points * 3, offset=12
        ).reshape(-1, 3)

        # With identity pose: x_cam = (u - cx) * depth / fx
        # First sample point is u=0, v=0, depth=2.0
        expected_x = (0 - intrinsics.cx) * 2.0 / intrinsics.fx
        expected_y = (0 - intrinsics.cy) * 2.0 / intrinsics.fy
        expected_z = 2.0
        np.testing.assert_almost_equal(points[0, 0], expected_x, decimal=3)
        np.testing.assert_almost_equal(points[0, 1], expected_y, decimal=3)
        np.testing.assert_almost_equal(points[0, 2], expected_z, decimal=3)

    def test_pose_translation_applied_to_points(self, tmp_path, intrinsics):
        """Points should be offset by the pose translation."""
        pose = np.eye(4)
        pose[0, 3] = 10.0  # translate +10 in X
        output = tmp_path / "cloud.bin"
        writer = DepthCloudWriter(output, intrinsics, pose, grid_step=640)
        writer.open()

        depth_map = np.full((480, 640), 2.0, dtype=np.float32)
        writer.write_frame(depth_map)
        writer.close()

        data = output.read_bytes()
        n_points = struct.unpack_from("<I", data, 8)[0]
        points = np.frombuffer(
            data, dtype=np.float32, count=n_points * 3, offset=12
        ).reshape(-1, 3)

        # All X coordinates should include the +10 offset
        # x_cam = (u - cx) * depth / fx, with cx=320 and first sample u=0:
        # x_cam = -1.28, x_world = -1.28 + 10.0 = 8.72
        # Compare with identity-pose result to verify offset
        writer2 = DepthCloudWriter(tmp_path / "ref.bin", intrinsics, np.eye(4), grid_step=640)
        writer2.open()
        writer2.write_frame(depth_map)
        writer2.close()
        ref_data = (tmp_path / "ref.bin").read_bytes()
        ref_n = struct.unpack_from("<I", ref_data, 8)[0]
        ref_pts = np.frombuffer(ref_data, dtype=np.float32, count=ref_n * 3, offset=12).reshape(-1, 3)
        # Offset should be ~10.0 in X
        np.testing.assert_almost_equal(points[:, 0] - ref_pts[:, 0], 10.0, decimal=2)

    def test_context_manager(self, tmp_path, intrinsics, pose_matrix):
        """Context manager should open/close correctly."""
        output = tmp_path / "cloud.bin"
        with DepthCloudWriter(output, intrinsics, pose_matrix, grid_step=64) as writer:
            depth_map = np.full((480, 640), 2.0, dtype=np.float32)
            writer.write_frame(depth_map)

        data = output.read_bytes()
        num_frames = struct.unpack_from("<I", data, 0)[0]
        assert num_frames == 1
