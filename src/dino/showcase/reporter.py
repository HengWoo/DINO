from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ShowcaseStats:
    """Aggregated statistics from pipeline results."""

    total_frames: int = 0
    video_duration_sec: float = 0.0
    total_detections: int = 0
    avg_detections_per_frame: float = 0.0
    class_counts: dict[str, int] = field(default_factory=dict)
    unique_tracker_ids: int = 0
    peak_detection_count: int = 0
    peak_detection_frame: int = 0
    per_frame_counts: list[int] = field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[dict], fps: float = 30.0) -> ShowcaseStats:
        total_frames = len(results)
        class_counter: Counter[str] = Counter()
        tracker_ids: set[int] = set()
        per_frame_counts: list[int] = []
        peak_count = 0
        peak_frame = 0

        for entry in results:
            dets = entry.get("detections", [])
            count = len(dets)
            per_frame_counts.append(count)
            if count > peak_count:
                peak_count = count
                peak_frame = entry.get("frame", 0)
            for det in dets:
                if "class_name" in det:
                    class_counter[det["class_name"]] += 1
                if "tracker_id" in det:
                    tracker_ids.add(det["tracker_id"])

        total_detections = sum(per_frame_counts)
        avg = total_detections / total_frames if total_frames > 0 else 0.0
        duration = total_frames / fps if fps > 0 else 0.0

        return cls(
            total_frames=total_frames,
            video_duration_sec=round(duration, 1),
            total_detections=total_detections,
            avg_detections_per_frame=round(avg, 1),
            class_counts=dict(class_counter.most_common()),
            unique_tracker_ids=len(tracker_ids),
            peak_detection_count=peak_count,
            peak_detection_frame=peak_frame,
            per_frame_counts=per_frame_counts,
        )


def extract_sample_frames(video_path: str, count: int = 8) -> list[np.ndarray]:
    """Read evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / count) for i in range(count)]
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _frame_to_base64(frame: np.ndarray, max_width: int = 640) -> str:
    """Encode a frame as a base64 PNG string, resized if needed."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".png", frame)
    return base64.b64encode(buf).decode("ascii")


class ShowcaseReporter:
    """Generate an HTML report from showcase stats and sample frames."""

    @staticmethod
    def generate(
        stats: ShowcaseStats,
        sample_frames: list[np.ndarray],
        output_dir: str,
    ) -> str:
        output_path = Path(output_dir) / "report.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame_images = [_frame_to_base64(f) for f in sample_frames]
        timeline_svg = _build_timeline_svg(stats.per_frame_counts)
        class_rows = "\n".join(
            f"<tr><td>{name}</td><td>{count}</td></tr>"
            for name, count in stats.class_counts.items()
        )
        frame_gallery = "\n".join(
            f'<img src="data:image/png;base64,{img}" style="max-width:320px;margin:4px;border-radius:4px;">'
            for img in frame_images
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DINO Showcase Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ color: #38bdf8; }}
  h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0; }}
  .stat-card {{ background: #1e293b; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat-card .value {{ font-size: 2em; font-weight: bold; color: #38bdf8; }}
  .stat-card .label {{ color: #94a3b8; font-size: 0.9em; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 500px; }}
  th, td {{ padding: 8px 16px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ color: #94a3b8; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }}
  .gallery img {{ border: 1px solid #334155; }}
  .timeline {{ margin: 20px 0; }}
</style>
</head>
<body>
<h1>DINO Showcase Report</h1>

<div class="stats-grid">
  <div class="stat-card"><div class="value">{stats.total_frames}</div><div class="label">Frames Processed</div></div>
  <div class="stat-card"><div class="value">{stats.video_duration_sec}s</div><div class="label">Video Duration</div></div>
  <div class="stat-card"><div class="value">{stats.total_detections}</div><div class="label">Total Detections</div></div>
  <div class="stat-card"><div class="value">{stats.avg_detections_per_frame}</div><div class="label">Avg per Frame</div></div>
  <div class="stat-card"><div class="value">{stats.unique_tracker_ids}</div><div class="label">Unique Tracks</div></div>
  <div class="stat-card"><div class="value">{stats.peak_detection_count}</div><div class="label">Peak Detections (frame {stats.peak_detection_frame})</div></div>
</div>

<h2>Detection Timeline</h2>
<div class="timeline">{timeline_svg}</div>

<h2>Sample Frames</h2>
<div class="gallery">
{frame_gallery}
</div>

<h2>Class Breakdown</h2>
<table>
<tr><th>Class</th><th>Count</th></tr>
{class_rows}
</table>

</body>
</html>"""

        output_path.write_text(html)
        return str(output_path)


def _build_timeline_svg(per_frame_counts: list[int]) -> str:
    """Build an inline SVG bar chart of per-frame detection counts."""
    if not per_frame_counts:
        return "<p>No data</p>"

    width = 800
    height = 120
    max_count = max(per_frame_counts) or 1
    n = len(per_frame_counts)
    bar_width = max(width / n, 1)

    bars = []
    for i, count in enumerate(per_frame_counts):
        bar_height = (count / max_count) * (height - 10)
        x = i * bar_width
        y = height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#38bdf8" opacity="0.8"/>'
        )

    rects = "\n".join(bars)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"'
        f' style="background:#1e293b;border-radius:8px;">\n{rects}\n</svg>'
    )
