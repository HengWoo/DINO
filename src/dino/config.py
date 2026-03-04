from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    prompts: list[str] = field(default_factory=list)
    stride: int = 1
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    trace_length: int = 60

    def __post_init__(self):
        if not self.prompts:
            raise ValueError("prompts must not be empty")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
