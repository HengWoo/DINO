"""Monocular depth estimation using Depth Anything v2."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


class DepthEstimator:
    """Thin wrapper around transformers depth-estimation pipeline."""

    DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cpu",
    ):
        self._pipe = pipeline(
            "depth-estimation",
            model=model_id or self.DEFAULT_MODEL,
            device=device,
        )

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR frame.

        Returns:
            HxW float32 depth map normalized to [0, 1].
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pipe(Image.fromarray(rgb))
        depth = np.array(result["depth"], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-8:
            return np.zeros_like(depth)
        return (depth - d_min) / (d_max - d_min)
