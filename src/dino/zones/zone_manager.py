from __future__ import annotations

import numpy as np

from dino.spatial.models import WorldObject
from dino.zones.models import (
    ObservationType,
    ZoneDefinition,
    ZoneObservation,
    ZoneState,
)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting algorithm for point-in-polygon test.

    Args:
        point: (2,) array [x, y].
        polygon: (M, 2) array of polygon vertices.

    Returns:
        True if point is inside the polygon.
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class ZoneManager:
    """Manages spatial zones and tracks object containment."""

    def __init__(self, zones: list[ZoneDefinition] | None = None):
        self._zones: dict[str, ZoneDefinition] = {}
        self._states: dict[str, ZoneState] = {}
        self._observations: list[ZoneObservation] = []
        if zones:
            for zone in zones:
                self.add_zone(zone)

    def add_zone(self, zone: ZoneDefinition) -> None:
        """Add a zone to track."""
        self._zones[zone.zone_id] = zone
        self._states[zone.zone_id] = ZoneState(zone_id=zone.zone_id)

    def update(
        self, objects: list[WorldObject], frame_idx: int, timestamp: float = 0.0
    ) -> list[ZoneObservation]:
        """Update zone states with current objects.

        Checks containment for each object against each zone,
        generates enter/exit/present observations.

        Returns:
            List of observations generated this update.
        """
        new_observations: list[ZoneObservation] = []

        for zone_id, zone_def in self._zones.items():
            state = self._states[zone_id]
            current_ids: set[int] = set()

            for obj in objects:
                if obj.persistent_id is None:
                    continue
                pos_2d = obj.world_position[:2]
                if point_in_polygon(pos_2d, zone_def.polygon[:, :2]):
                    current_ids.add(obj.persistent_id)
                    if obj.persistent_id not in state.present_objects:
                        obs = ZoneObservation(
                            zone_id=zone_id,
                            persistent_id=obj.persistent_id,
                            observation_type=ObservationType.ENTER,
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                        )
                    else:
                        obs = ZoneObservation(
                            zone_id=zone_id,
                            persistent_id=obj.persistent_id,
                            observation_type=ObservationType.PRESENT,
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                        )
                    new_observations.append(obs)
                    state.present_objects[obj.persistent_id] = obj
                    obj.zone_id = zone_id

            # Check exits
            exited = set(state.present_objects.keys()) - current_ids
            for pid in exited:
                obs = ZoneObservation(
                    zone_id=zone_id,
                    persistent_id=pid,
                    observation_type=ObservationType.EXIT,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                )
                new_observations.append(obs)
                del state.present_objects[pid]

            state.last_observed_at = timestamp
            state.is_currently_observed = len(state.present_objects) > 0

        self._observations.extend(new_observations)
        return new_observations

    def get_zone_state(self, zone_id: str) -> ZoneState | None:
        """Get current state of a zone."""
        return self._states.get(zone_id)

    @property
    def observations(self) -> list[ZoneObservation]:
        """All recorded observations."""
        return list(self._observations)
