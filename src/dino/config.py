from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    prompts: list[str] = field(default_factory=list)
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    stride: int = 1
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    trace_length: int = 60
