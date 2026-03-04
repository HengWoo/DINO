#!/usr/bin/env python3
"""Minimal demo: run Grounding DINO on a single image."""

import argparse
import sys

import cv2
import supervision as sv

from dino.detectors.grounding_dino import GroundingDINODetector

DEFAULT_PROMPTS = ["person", "empty table", "plate of food", "chair"]


def main():
    parser = argparse.ArgumentParser(description="Grounding DINO single-image demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="Text prompts for detection",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Box confidence threshold"
    )
    parser.add_argument("--output", default=None, help="Path to save annotated image")
    args = parser.parse_args()

    try:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: cannot read image '{args.image}'", file=sys.stderr)
            sys.exit(1)

        print(f"Loading Grounding DINO...")
        detector = GroundingDINODetector(box_threshold=args.threshold)
        print(f"Device: {detector.device}")

        print(f"Detecting with prompts: {args.prompts}")
        detections = detector.detect(frame, args.prompts)

        print(f"\nFound {len(detections)} detection(s):")
        for i in range(len(detections)):
            box = detections.xyxy[i]
            conf = detections.confidence[i]
            label = detections.data["class_name"][i] if "class_name" in detections.data else "unknown"
            print(f"  [{i}] {label} ({conf:.2f}) at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

        if args.output:
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            labels = [
                f"{detections.data['class_name'][i] if 'class_name' in detections.data else 'unknown'} "
                f"{detections.confidence[i]:.2f}"
                for i in range(len(detections))
            ]
            annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
            if not cv2.imwrite(args.output, annotated):
                print(f"Error: failed to write image to '{args.output}'", file=sys.stderr)
                sys.exit(1)
            print(f"\nAnnotated image saved to: {args.output}")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
