from __future__ import annotations

import numpy as np

from dino.spatial.models import WorldObject


class SpatialObjectRegistry:
    """Persistent object identity via 3D proximity matching.

    Maintains a registry of known objects and matches incoming detections
    to existing objects for persistent identity across tracker re-IDs.
    """

    def __init__(
        self,
        match_distance: float = 50.0,
        max_age_seconds: float = 30.0,
    ):
        self.match_distance = match_distance
        self.max_age_seconds = max_age_seconds
        self._next_id: int = 0
        self._tracker_to_persistent: dict[int, int] = {}
        self._objects: dict[int, WorldObject] = {}  # persistent_id -> last WorldObject

    def register(self, obj: WorldObject) -> WorldObject:
        """Register a world object and assign persistent identity.

        1. Fast path: tracker_id already mapped -> reuse persistent_id.
        2. Slow path: find nearest known object (same class, within match_distance).
        3. New object: allocate new persistent_id.

        Returns:
            WorldObject with persistent_id set.
        """
        self._evict(obj.timestamp)

        # Fast path: tracker_id already known
        if obj.tracker_id in self._tracker_to_persistent:
            pid = self._tracker_to_persistent[obj.tracker_id]
            if pid in self._objects:
                return self._update(obj, pid)
            # Stale mapping — fall through to slow path
            del self._tracker_to_persistent[obj.tracker_id]

        # Slow path: proximity match
        best_pid = self._find_nearest(obj)
        if best_pid is not None:
            self._tracker_to_persistent[obj.tracker_id] = best_pid
            return self._update(obj, best_pid)

        # New object
        pid = self._next_id
        self._next_id += 1
        self._tracker_to_persistent[obj.tracker_id] = pid
        return self._update(obj, pid)

    def _update(self, obj: WorldObject, persistent_id: int) -> WorldObject:
        obj.persistent_id = persistent_id
        self._objects[persistent_id] = obj
        return obj

    def _find_nearest(self, obj: WorldObject) -> int | None:
        best_dist = self.match_distance
        best_pid = None
        for pid, known in self._objects.items():
            if known.class_name != obj.class_name:
                continue
            dist = float(np.linalg.norm(obj.world_position - known.world_position))
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
        return best_pid

    def _evict(self, current_time: float) -> None:
        stale = [
            pid
            for pid, obj in self._objects.items()
            if current_time - obj.timestamp > self.max_age_seconds
        ]
        for pid in stale:
            del self._objects[pid]
            self._tracker_to_persistent = {
                tid: p
                for tid, p in self._tracker_to_persistent.items()
                if p != pid
            }
