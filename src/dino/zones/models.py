from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from dino.spatial.models import WorldObject


class ObservationType(Enum):
    """Type of zone observation."""
    ENTER = "enter"
    EXIT = "exit"
    PRESENT = "present"


@dataclass
class ZoneDefinition:
    """Definition of a spatial zone."""
    zone_id: str
    name: str
    polygon: np.ndarray  # (M, 2) for 2D or (M, 3) for 3D
    metadata: dict = field(default_factory=dict)


@dataclass
class ZoneState:
    """Current state of a zone."""
    zone_id: str
    present_objects: dict[int, WorldObject] = field(default_factory=dict)
    last_observed_at: float = 0.0
    is_currently_observed: bool = False


@dataclass
class ZoneObservation:
    """Record of an object being observed in/around a zone."""
    zone_id: str
    persistent_id: int
    observation_type: ObservationType
    frame_idx: int
    timestamp: float = 0.0


@dataclass
class ZoneEvent:
    """A higher-level event triggered by rules."""
    zone_id: str
    event_type: str
    frame_idx: int
    timestamp: float = 0.0
    details: dict = field(default_factory=dict)
