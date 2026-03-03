# DINO — Restaurant Visual Pipeline

Spatial Intelligence for Restaurant Operations. Uses Grounding DINO for open-vocabulary object detection in restaurant footage, with ByteTrack tracking and table state detection.

## Quick Start

```bash
uv sync
uv run python scripts/demo_grounding_dino.py --image path/to/image.jpg
uv run python scripts/run_pipeline.py --input data/videos/sample.mp4 --output data/output/annotated.mp4
```

## Stack

- **Detection**: Grounding DINO via HuggingFace Transformers
- **Tracking**: ByteTrack via supervision
- **Video I/O**: supervision
- **Python tooling**: uv
