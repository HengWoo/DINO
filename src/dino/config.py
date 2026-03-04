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


@dataclass
class SpatialConfig:
    """Configuration for spatial processing."""

    camera_mode: str = "fixed"
    match_distance: float = 50.0
    max_age_seconds: float = 30.0
    zones_path: str | None = None
    rules: list[dict] | None = None

    def __post_init__(self):
        if self.camera_mode not in ("fixed",):
            raise ValueError(
                f"Unsupported camera_mode: {self.camera_mode!r}, "
                f"must be one of ('fixed',)"
            )
        if self.match_distance <= 0:
            raise ValueError(f"match_distance must be > 0, got {self.match_distance}")
        if self.max_age_seconds <= 0:
            raise ValueError(f"max_age_seconds must be > 0, got {self.max_age_seconds}")
