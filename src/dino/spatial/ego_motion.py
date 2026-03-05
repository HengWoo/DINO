"""Per-frame camera pose estimation via ORB feature matching."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from dino.spatial.models import CameraIntrinsics

logger = logging.getLogger(__name__)


class EgoMotionEstimator:
    """Estimate camera ego-motion using ORB feature matching between consecutive frames."""

    def __init__(self, intrinsics: CameraIntrinsics, avg_depth: float = 2.0):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.intrinsics = intrinsics
        self.avg_depth = avg_depth
        self.prev_gray: np.ndarray | None = None
        self.prev_kp = None
        self.prev_des: np.ndarray | None = None
        self.cumulative_pose = np.eye(4)  # world_T_cam
        self.poses: list[np.ndarray] = []

    def _fallback_pose(self) -> np.ndarray:
        """Return the current cumulative pose (reuse last known position)."""
        pose = self.cumulative_pose.copy()
        self.poses.append(pose)
        return pose

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process one frame, return 4x4 world_T_cam pose."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
        except cv2.error as e:
            logger.warning("Ego-motion ORB failed: %s; reusing last pose", e)
            return self._fallback_pose()

        if self.prev_des is None or des is None or len(kp) < 10:
            logger.debug(
                "Ego-motion: insufficient keypoints (%d); reusing last pose",
                len(kp) if kp else 0,
            )
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return self._fallback_pose()

        matches = self.bf.match(self.prev_des, des)
        if len(matches) < 10:
            logger.debug(
                "Ego-motion: insufficient matches (%d); reusing last pose",
                len(matches),
            )
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return self._fallback_pose()

        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

        M, _inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC
        )
        if M is not None:
            tx, ty = M[0, 2], M[1, 2]
            fx = self.intrinsics.fx
            fy = self.intrinsics.fy
            dx = tx * self.avg_depth / fx
            dz = ty * self.avg_depth / fy
            delta = np.eye(4)
            delta[0, 3] = dx
            delta[2, 3] = dz
            self.cumulative_pose = self.cumulative_pose @ delta
        else:
            logger.debug("Ego-motion: RANSAC failed; reusing last pose")

        pose = self.cumulative_pose.copy()
        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        self.poses.append(pose)
        return pose

    def get_all_poses(self) -> list[dict]:
        """Return list of {position: [x, y, z]} dicts for JSON export."""
        return [{"position": p[:3, 3].tolist()} for p in self.poses]
