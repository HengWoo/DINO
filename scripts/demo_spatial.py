#!/usr/bin/env python3
"""CLI entry point for the spatial video processing pipeline."""

import argparse
import sys
import traceback
from pathlib import Path

from dino.config import PipelineConfig, SpatialConfig
from dino.detectors.grounding_dino import GroundingDINODetector
from dino.spatial.spatial_pipeline import SpatialPipeline


def main():
    parser = argparse.ArgumentParser(description="DINO Spatial Pipeline Demo")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--zones", default=None, help="Path to zones.json")
    parser.add_argument(
        "--prompts", nargs="+", required=True, help="Text prompts for detection"
    )
    parser.add_argument(
        "--camera-mode", default="fixed", choices=["fixed", "depth"],
        help="Camera mode ('fixed' for flat 2D, 'depth' for 3D with monocular depth)",
    )
    parser.add_argument(
        "--camera-fov", type=float, default=70.0,
        help="Camera horizontal FOV in degrees (depth mode only)",
    )
    parser.add_argument(
        "--depth-model", default=None,
        help="Depth estimation model ID (default: Depth-Anything-V2-Small)",
    )
    parser.add_argument(
        "--no-zone-overlay", action="store_true",
        help="Disable zone annotations on output video (show bboxes only)",
    )
    parser.add_argument(
        "--point-cloud", action="store_true",
        help="Export depth maps as binary point cloud (.bin)",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every Nth frame"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Box confidence threshold"
    )
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    try:
        if not Path(args.input).exists():
            print(f"Error: input video not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = PipelineConfig(
            prompts=args.prompts,
            stride=args.stride,
        )

        spatial_config = SpatialConfig(
            camera_mode=args.camera_mode,
            zones_path=args.zones,
            depth_model=args.depth_model,
            camera_fov_deg=args.camera_fov,
            annotate_zones_on_video=not args.no_zone_overlay,
        )

        print("Loading Grounding DINO...")
        detector = GroundingDINODetector(
            box_threshold=args.threshold,
            device=args.device,
        )
        print(f"Device: {detector.device}")

        pipeline = SpatialPipeline(
            detector=detector, config=config, spatial_config=spatial_config
        )

        output_video = str(output_dir / "spatial_annotated.mp4")
        json_output = str(output_dir / "spatial_results.json")
        point_cloud_output = (
            str(output_dir / "point_clouds.bin") if args.point_cloud else None
        )

        print(f"Processing: {args.input}")
        print(f"Output video: {output_video}")
        print(f"Output JSON: {json_output}")
        print(f"Zones: {args.zones or 'none'}")
        print(f"Stride: {args.stride}")

        def progress(processed, total):
            print(f"\rFrame {processed}/{total}", end="", flush=True)

        results = pipeline.run(
            args.input,
            output_video,
            json_output=json_output,
            point_cloud_output=point_cloud_output,
            progress_callback=progress,
        )

        print(f"\n\nDone! Processed {len(results.frame_results)} frames.")
        print(f"  Observations: {len(results.observations)}")
        print(f"  Events: {len(results.events)}")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
