"""Tests for depth estimation module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDepthEstimator:
    def test_estimate_returns_depth_map(self):
        """Mock pipeline returns HxW float32 depth map."""
        from dino.spatial.depth_estimator import DepthEstimator

        # Mock the pipeline
        mock_pipe = MagicMock()
        fake_depth = np.random.randint(0, 256, (240, 320), dtype=np.uint8)
        mock_pipe.return_value = {"depth": fake_depth}

        with patch(
            "dino.spatial.depth_estimator.pipeline", return_value=mock_pipe
        ):
            estimator = DepthEstimator(device="cpu")

        frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        result = estimator.estimate(frame_bgr)

        assert result.shape == (240, 320)
        assert result.dtype == np.float32

    def test_estimate_normalized_range(self):
        """Depth values should be in [0, 1]."""
        from dino.spatial.depth_estimator import DepthEstimator

        mock_pipe = MagicMock()
        # Create depth with known range
        fake_depth = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        mock_pipe.return_value = {"depth": fake_depth}

        with patch(
            "dino.spatial.depth_estimator.pipeline", return_value=mock_pipe
        ):
            estimator = DepthEstimator(device="cpu")

        frame_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        result = estimator.estimate(frame_bgr)

        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(128 / 255.0)
        assert result[1, 0] == pytest.approx(1.0)

    def test_estimate_caches_nothing_by_default(self):
        """Each call invokes the pipeline (no built-in caching)."""
        from dino.spatial.depth_estimator import DepthEstimator

        mock_pipe = MagicMock()
        fake_depth = np.zeros((4, 4), dtype=np.uint8)
        mock_pipe.return_value = {"depth": fake_depth}

        with patch(
            "dino.spatial.depth_estimator.pipeline", return_value=mock_pipe
        ):
            estimator = DepthEstimator(device="cpu")

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        estimator.estimate(frame)
        estimator.estimate(frame)

        assert mock_pipe.call_count == 2
