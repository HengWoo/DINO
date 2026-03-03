#!/usr/bin/env python3
"""CLI entry point for the video processing pipeline."""

import argparse
import sys
from pathlib import Path

from dino.config import PipelineConfig
from dino.detectors.grounding_dino import GroundingDINODetector
from dino.pipeline.video_pipeline import VideoPipeline

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
    parser = argparse.ArgumentParser(description="DINO Restaurant Visual Pipeline")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--json", default=None, help="Path to export JSON results")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Custom text prompts for detection",
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

    if not Path(args.input).exists():
        print(f"Error: input video not found: {args.input}", file=sys.stderr)
        sys.exit(1)

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

    json_path = args.json
    if json_path is None:
        json_path = str(Path(args.output).with_suffix(".json"))

    print(f"Processing: {args.input}")
    print(f"Output video: {args.output}")
    print(f"Output JSON: {json_path}")
    print(f"Stride: {args.stride}")

    results = pipeline.run(args.input, args.output, json_output=json_path)

    total_detections = sum(len(r["detections"]) for r in results)
    print(f"\nDone! Processed {len(results)} frames, {total_detections} total detections.")


if __name__ == "__main__":
    main()
