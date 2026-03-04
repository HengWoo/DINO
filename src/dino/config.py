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

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError(f"box_threshold must be in [0, 1], got {self.box_threshold}")
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError(f"text_threshold must be in [0, 1], got {self.text_threshold}")
