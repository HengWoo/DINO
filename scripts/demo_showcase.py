#!/usr/bin/env python3
"""CLI entry point for the full DINO showcase demo."""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from dino.annotation.showcase_annotator import ShowcaseAnnotator
from dino.config import PipelineConfig
from dino.detectors.grounding_dino import GroundingDINODetector
from dino.pipeline.video_pipeline import VideoPipeline
from dino.showcase.reporter import ShowcaseReporter, ShowcaseStats, extract_sample_frames

TABLE_PROMPTS = ["empty table", "person sitting at table", "plate of food on table", "dirty plate"]
SERVICE_PROMPTS = ["person raising hand", "server carrying food", "empty glass", "menu on table"]
STAFF_PROMPTS = ["person in uniform", "server", "chef"]

PROMPT_PRESETS = {
    "table": TABLE_PROMPTS,
    "service": SERVICE_PROMPTS,
    "staff": STAFF_PROMPTS,
    "all": TABLE_PROMPTS + SERVICE_PROMPTS + STAFF_PROMPTS,
}


def main():
    parser = argparse.ArgumentParser(description="DINO Full Showcase Demo")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--prompts", nargs="+", default=None, help="Custom text prompts"
    )
    parser.add_argument(
        "--preset",
        choices=list(PROMPT_PRESETS.keys()),
        default="all",
        help="Prompt preset (default: all)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--threshold", type=float, default=0.3, help="Box confidence threshold")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input video not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompts = args.prompts or PROMPT_PRESETS[args.preset]
        print(f"Prompts: {prompts}")

        config = PipelineConfig(
            prompts=prompts,
            box_threshold=args.threshold,
            stride=args.stride,
        )

        print("Loading Grounding DINO...")
        detector = GroundingDINODetector(
            box_threshold=args.threshold,
            device=args.device,
        )
        print(f"Device: {detector.device}")

        pipeline = VideoPipeline(detector=detector, config=config)
        pipeline.annotator = ShowcaseAnnotator(
            prompts=prompts,
            trace_length=config.trace_length,
        )

        annotated_path = str(output_dir / "annotated.mp4")
        sbs_path = str(output_dir / "side_by_side.mp4")
        json_path = str(output_dir / "results.json")

        print(f"Processing: {args.input}")
        print(f"Output dir: {output_dir}")

        progress_bar = tqdm(desc="Processing frames", unit="frame")

        def on_progress(processed: int, total: int) -> None:
            progress_bar.total = total
            progress_bar.n = processed
            progress_bar.refresh()

        results = pipeline.run(
            input_path=str(input_path),
            output_path=annotated_path,
            json_output=json_path,
            progress_callback=on_progress,
            sbs_output=sbs_path,
        )
        progress_bar.close()

        print("\nExtracting sample frames...")
        sample_frames = extract_sample_frames(annotated_path, count=8)

        import supervision as sv
        video_info = sv.VideoInfo.from_video_path(str(input_path))
        effective_fps = video_info.fps / config.stride if config.stride > 1 else video_info.fps

        print("Generating HTML report...")
        stats = ShowcaseStats.from_results(results, fps=effective_fps)
        report_path = ShowcaseReporter.generate(stats, sample_frames, str(output_dir))

        print(f"\nDone! Processed {stats.total_frames} frames, {stats.total_detections} detections.")
        print(f"  Annotated video: {annotated_path}")
        print(f"  Side-by-side:    {sbs_path}")
        print(f"  JSON results:    {json_path}")
        print(f"  HTML report:     {report_path}")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
