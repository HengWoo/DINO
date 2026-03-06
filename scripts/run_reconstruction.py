#!/usr/bin/env python3
"""Orchestrator: tries SLAM3R first, falls back to COLMAP dense MVS.

Usage:
    python scripts/run_reconstruction.py /path/to/video.mp4
    python scripts/run_reconstruction.py --video /path/to/video.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Dense room reconstruction via Modal")
    parser.add_argument("video", nargs="?", help="Path to input video")
    parser.add_argument("--video", dest="video_flag", help="Path to input video (alternative)")
    args = parser.parse_args()

    video = args.video or args.video_flag
    if not video:
        parser.error("Video path required")

    video_path = Path(video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent

    # Try SLAM3R first (better quality for monocular video)
    print(f"=== Trying SLAM3R reconstruction ===")
    result = subprocess.run(
        ["modal", "run", str(project_root / "modal_slam3r.py"), "--video", str(video_path)],
        cwd=str(project_root),
    )

    if result.returncode == 0:
        print("\nSLAM3R reconstruction succeeded!")
        return

    # Fall back to COLMAP dense MVS
    print(f"\n=== SLAM3R failed (exit {result.returncode}), trying COLMAP dense ===")
    result = subprocess.run(
        ["modal", "run", str(project_root / "modal_colmap_dense.py"), "--video", str(video_path)],
        cwd=str(project_root),
    )

    if result.returncode == 0:
        print("\nCOLMAP dense reconstruction succeeded!")
    else:
        print(f"\nCOLMAP dense also failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
