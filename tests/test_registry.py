from __future__ import annotations

import numpy as np
import pytest

from dino.spatial.models import WorldObject
from dino.spatial.registry import SpatialObjectRegistry


def _make_object(
    tracker_id: int = 1,
    position: tuple[float, float, float] = (50.0, 50.0, 0.0),
    class_name: str = "person",
    timestamp: float = 0.0,
) -> WorldObject:
    return WorldObject(
        tracker_id=tracker_id,
        bbox_xyxy=np.array([0, 0, 100, 100], dtype=np.float32),
        world_position=np.array(position),
        class_name=class_name,
        timestamp=timestamp,
    )


class TestSpatialObjectRegistry:
    def test_register_new_object(self):
        registry = SpatialObjectRegistry()
        obj = _make_object(tracker_id=1)
        result = registry.register(obj)
        assert result.persistent_id == 0

    def test_tracker_id_cache_hit(self):
        registry = SpatialObjectRegistry()
        obj1 = _make_object(tracker_id=1, timestamp=0.0)
        obj2 = _make_object(tracker_id=1, timestamp=1.0)
        r1 = registry.register(obj1)
        r2 = registry.register(obj2)
        assert r1.persistent_id == r2.persistent_id

    def test_proximity_match(self):
        registry = SpatialObjectRegistry(match_distance=100.0)
        obj1 = _make_object(tracker_id=1, position=(50.0, 50.0, 0.0), timestamp=0.0)
        obj2 = _make_object(
            tracker_id=99, position=(55.0, 55.0, 0.0), timestamp=1.0
        )
        r1 = registry.register(obj1)
        r2 = registry.register(obj2)
        assert r1.persistent_id == r2.persistent_id

    def test_proximity_no_match_different_class(self):
        registry = SpatialObjectRegistry(match_distance=100.0)
        obj1 = _make_object(
            tracker_id=1, position=(50.0, 50.0, 0.0), class_name="person"
        )
        obj2 = _make_object(
            tracker_id=2, position=(55.0, 55.0, 0.0), class_name="cup"
        )
        r1 = registry.register(obj1)
        r2 = registry.register(obj2)
        assert r1.persistent_id != r2.persistent_id

    def test_proximity_no_match_too_far(self):
        registry = SpatialObjectRegistry(match_distance=10.0)
        obj1 = _make_object(tracker_id=1, position=(0.0, 0.0, 0.0))
        obj2 = _make_object(tracker_id=2, position=(100.0, 100.0, 0.0))
        r1 = registry.register(obj1)
        r2 = registry.register(obj2)
        assert r1.persistent_id != r2.persistent_id

    def test_eviction(self):
        registry = SpatialObjectRegistry(
            match_distance=100.0, max_age_seconds=5.0
        )
        obj1 = _make_object(tracker_id=1, position=(50.0, 50.0, 0.0), timestamp=0.0)
        r1 = registry.register(obj1)
        pid1 = r1.persistent_id

        # Same position but after eviction window, new tracker
        obj2 = _make_object(
            tracker_id=2, position=(50.0, 50.0, 0.0), timestamp=10.0
        )
        r2 = registry.register(obj2)
        assert r2.persistent_id != pid1

    def test_multiple_objects(self):
        registry = SpatialObjectRegistry()
        ids = set()
        for i in range(3):
            obj = _make_object(
                tracker_id=i, position=(i * 200.0, 0.0, 0.0)
            )
            r = registry.register(obj)
            ids.add(r.persistent_id)
        assert len(ids) == 3

    def test_register_returns_updated_object(self):
        registry = SpatialObjectRegistry()
        obj = _make_object(tracker_id=1)
        result = registry.register(obj)
        assert result.persistent_id is not None
        assert result is obj

    def test_stale_tracker_mapping_falls_through(self):
        """Tracker mapping to evicted object falls through to proximity/new."""
        registry = SpatialObjectRegistry(match_distance=100.0, max_age_seconds=5.0)
        # Register object with tracker_id=1
        obj1 = _make_object(tracker_id=1, position=(50.0, 50.0, 0.0), timestamp=0.0)
        r1 = registry.register(obj1)
        pid1 = r1.persistent_id

        # Object gets evicted, but tracker_id=1 appears again far away
        obj2 = _make_object(
            tracker_id=1, position=(500.0, 500.0, 0.0), timestamp=10.0
        )
        r2 = registry.register(obj2)
        # Should get new persistent_id since old was evicted
        assert r2.persistent_id != pid1

    def test_eviction_cleans_tracker_mappings(self):
        """Eviction should clean up corresponding tracker-to-persistent mappings."""
        registry = SpatialObjectRegistry(match_distance=100.0, max_age_seconds=5.0)
        obj1 = _make_object(tracker_id=1, position=(50.0, 50.0, 0.0), timestamp=0.0)
        registry.register(obj1)
        assert 1 in registry._tracker_to_persistent

        # Trigger eviction
        obj2 = _make_object(tracker_id=2, position=(500.0, 500.0, 0.0), timestamp=10.0)
        registry.register(obj2)
        # tracker_id=1 mapping should be cleaned up
        assert 1 not in registry._tracker_to_persistent

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="match_distance"):
            SpatialObjectRegistry(match_distance=0)
        with pytest.raises(ValueError, match="max_age_seconds"):
            SpatialObjectRegistry(max_age_seconds=-1)

    def test_nearest_among_multiple_candidates(self):
        """When multiple objects of same class exist, nearest one wins."""
        registry = SpatialObjectRegistry(match_distance=100.0)
        obj_a = _make_object(tracker_id=1, position=(0.0, 0.0, 0.0), timestamp=0.0)
        obj_b = _make_object(tracker_id=2, position=(50.0, 0.0, 0.0), timestamp=0.0)
        r_a = registry.register(obj_a)
        r_b = registry.register(obj_b)

        # New object closer to B
        obj_c = _make_object(tracker_id=3, position=(48.0, 0.0, 0.0), timestamp=1.0)
        r_c = registry.register(obj_c)
        assert r_c.persistent_id == r_b.persistent_id

    def test_empty_class_name_disables_proximity_match(self):
        """Objects with empty class_name skip proximity matching to avoid false merges."""
        registry = SpatialObjectRegistry(match_distance=100.0)
        obj1 = _make_object(tracker_id=1, position=(50.0, 50.0, 0.0), class_name="", timestamp=0.0)
        obj2 = _make_object(tracker_id=2, position=(50.0, 50.0, 0.0), class_name="", timestamp=1.0)
        r1 = registry.register(obj1)
        r2 = registry.register(obj2)
        # Should NOT merge — each gets its own persistent_id
        assert r1.persistent_id != r2.persistent_id

    def test_proximity_match_cleans_old_tracker_mapping(self):
        """When proximity re-IDs, old tracker mapping for that persistent_id is cleaned."""
        registry = SpatialObjectRegistry(match_distance=100.0)
        obj1 = _make_object(tracker_id=1, position=(50.0, 50.0, 0.0), timestamp=0.0)
        r1 = registry.register(obj1)
        pid = r1.persistent_id

        # New tracker_id at same position — proximity match should re-ID
        obj2 = _make_object(tracker_id=2, position=(52.0, 52.0, 0.0), timestamp=1.0)
        r2 = registry.register(obj2)
        assert r2.persistent_id == pid

        # Old tracker mapping (tracker_id=1 -> pid) should be cleaned up
        assert 1 not in registry._tracker_to_persistent
        assert registry._tracker_to_persistent[2] == pid
