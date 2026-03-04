from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from dino.showcase.reporter import (
    ShowcaseReporter,
    ShowcaseStats,
    _build_timeline_svg,
    extract_sample_frames,
)


def _make_sample_results(n_frames: int = 10) -> list[dict]:
    results = []
    for i in range(n_frames):
        dets = []
        for j in range(i % 4):
            dets.append({
                "bbox": [10, 10, 50, 50],
                "confidence": 0.9,
                "class_id": j % 2,
                "tracker_id": j + 1,
                "class_name": "person" if j % 2 == 0 else "table",
            })
        results.append({"frame": i, "detections": dets})
    return results


def _make_test_video(path: str, n_frames: int = 16, w: int = 320, h: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 15 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestShowcaseStats:
    def test_from_results_basic(self):
        results = _make_sample_results(10)
        stats = ShowcaseStats.from_results(results, fps=10.0)
        assert stats.total_frames == 10
        assert stats.video_duration_sec == 1.0
        assert stats.total_detections > 0
        assert "person" in stats.class_counts
        assert stats.unique_tracker_ids > 0
        assert stats.peak_detection_count == 3
        assert len(stats.per_frame_counts) == 10

    def test_from_empty_results(self):
        stats = ShowcaseStats.from_results([], fps=30.0)
        assert stats.total_frames == 0
        assert stats.total_detections == 0
        assert stats.avg_detections_per_frame == 0.0

    def test_avg_detections(self):
        results = _make_sample_results(10)
        stats = ShowcaseStats.from_results(results, fps=10.0)
        expected_avg = round(stats.total_detections / 10, 1)
        assert stats.avg_detections_per_frame == expected_avg


class TestExtractSampleFrames:
    def test_extracts_correct_count(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_test_video(path, n_frames=16)
            frames = extract_sample_frames(path, count=8)
            assert len(frames) == 8
            for frame in frames:
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (240, 320, 3)
        finally:
            os.unlink(path)

    def test_nonexistent_video_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_sample_frames("/nonexistent/video.mp4", count=8)


class TestBuildTimelineSvg:
    def test_produces_rect_elements(self):
        svg = _build_timeline_svg([1, 3, 2, 5, 4])
        assert "<rect" in svg
        assert "<svg" in svg

    def test_empty_data(self):
        result = _build_timeline_svg([])
        assert "No data" in result


class TestShowcaseReporter:
    def test_generates_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _make_sample_results(10)
            stats = ShowcaseStats.from_results(results, fps=10.0)

            # Create sample frames (plain colored frames)
            sample_frames = [
                np.full((240, 320, 3), i * 30 % 256, dtype=np.uint8)
                for i in range(8)
            ]

            report_path = ShowcaseReporter.generate(stats, sample_frames, tmpdir)
            assert os.path.exists(report_path)
            assert report_path.endswith("report.html")

    def test_html_contains_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _make_sample_results(10)
            stats = ShowcaseStats.from_results(results, fps=10.0)
            sample_frames = [
                np.full((240, 320, 3), 128, dtype=np.uint8) for _ in range(8)
            ]

            report_path = ShowcaseReporter.generate(stats, sample_frames, tmpdir)
            html = open(report_path).read()

            assert str(stats.total_frames) in html
            assert str(stats.total_detections) in html
            assert "person" in html

    def test_html_contains_base64_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _make_sample_results(10)
            stats = ShowcaseStats.from_results(results, fps=10.0)
            sample_frames = [
                np.full((240, 320, 3), 128, dtype=np.uint8) for _ in range(8)
            ]

            report_path = ShowcaseReporter.generate(stats, sample_frames, tmpdir)
            html = open(report_path).read()

            assert html.count("data:image/png;base64,") == 8

    def test_html_contains_svg_timeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _make_sample_results(10)
            stats = ShowcaseStats.from_results(results, fps=10.0)
            sample_frames = [
                np.full((240, 320, 3), 128, dtype=np.uint8) for _ in range(8)
            ]

            report_path = ShowcaseReporter.generate(stats, sample_frames, tmpdir)
            html = open(report_path).read()

            assert "<rect" in html
            assert "<svg" in html
