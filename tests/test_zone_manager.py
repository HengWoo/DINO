"""Tests for zone manager — point-in-polygon and zone tracking."""
from __future__ import annotations

import numpy as np
import pytest

from dino.spatial.models import WorldObject
from dino.zones.models import ObservationType, ZoneDefinition
from dino.zones.zone_manager import ZoneManager, point_in_polygon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_square_zone(zone_id: str = "zone-a", size: float = 100.0) -> ZoneDefinition:
    """Create a square zone from (0,0) to (size, size)."""
    return ZoneDefinition(
        zone_id=zone_id,
        name=f"Square {zone_id}",
        polygon=np.array([
            [0.0, 0.0],
            [size, 0.0],
            [size, size],
            [0.0, size],
        ]),
    )


def _make_object(
    pid: int,
    x: float,
    y: float,
    class_name: str = "person",
    frame_idx: int = 0,
) -> WorldObject:
    """Create a WorldObject at (x, y, 0)."""
    return WorldObject(
        tracker_id=pid,
        bbox_xyxy=np.array([0, 0, 10, 10]),
        world_position=np.array([x, y, 0.0]),
        persistent_id=pid,
        class_name=class_name,
        frame_idx=frame_idx,
    )


# ===========================================================================
# TestPointInPolygon
# ===========================================================================

class TestPointInPolygon:
    """Tests for the ray-casting point_in_polygon function."""

    def test_point_inside_square(self) -> None:
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        assert point_in_polygon(np.array([50.0, 50.0]), square) is True

    def test_point_outside_square(self) -> None:
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        assert point_in_polygon(np.array([150.0, 50.0]), square) is False

    def test_point_on_edge(self) -> None:
        """Point on edge — ray-casting may return True or False; just verify no crash."""
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        result = point_in_polygon(np.array([0.0, 50.0]), square)
        assert isinstance(result, bool)

    def test_concave_polygon(self) -> None:
        """L-shaped concave polygon."""
        polygon = np.array([
            [0, 0],
            [100, 0],
            [100, 50],
            [50, 50],
            [50, 100],
            [0, 100],
        ], dtype=float)
        assert point_in_polygon(np.array([75.0, 25.0]), polygon) is True
        assert point_in_polygon(np.array([25.0, 75.0]), polygon) is True
        assert point_in_polygon(np.array([75.0, 75.0]), polygon) is False

    def test_degenerate_polygon_returns_false(self) -> None:
        """Polygon with fewer than 3 vertices returns False."""
        line = np.array([[0, 0], [100, 0]], dtype=float)
        assert point_in_polygon(np.array([50.0, 0.0]), line) is False

    def test_1d_polygon_returns_false(self) -> None:
        """1D array returns False."""
        bad = np.array([0, 0, 100])
        assert point_in_polygon(np.array([50.0, 0.0]), bad) is False


# ===========================================================================
# TestZoneManager
# ===========================================================================

class TestZoneManager:
    """Tests for ZoneManager containment tracking."""

    def test_add_zone(self) -> None:
        mgr = ZoneManager()
        zone = _make_square_zone()
        mgr.add_zone(zone)
        state = mgr.get_zone_state("zone-a")
        assert state is not None
        assert state.zone_id == "zone-a"

    def test_update_object_enters_zone(self) -> None:
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=50, y=50)
        observations = mgr.update([obj], frame_idx=0)
        assert len(observations) == 1
        assert observations[0].observation_type == ObservationType.ENTER
        assert observations[0].persistent_id == 1

    def test_update_object_exits_zone(self) -> None:
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj_in = _make_object(pid=1, x=50, y=50)
        mgr.update([obj_in], frame_idx=0)

        obj_out = _make_object(pid=1, x=200, y=200)
        observations = mgr.update([obj_out], frame_idx=1)
        exit_obs = [o for o in observations if o.observation_type == ObservationType.EXIT]
        assert len(exit_obs) == 1
        assert exit_obs[0].persistent_id == 1

    def test_update_object_remains(self) -> None:
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=50, y=50)
        mgr.update([obj], frame_idx=0)

        observations = mgr.update([obj], frame_idx=1)
        assert len(observations) == 1
        assert observations[0].observation_type == ObservationType.PRESENT

    def test_update_object_outside_all_zones(self) -> None:
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=200, y=200)
        observations = mgr.update([obj], frame_idx=0)
        assert len(observations) == 0
        assert obj.zone_id is None

    def test_multiple_zones(self) -> None:
        zone_a = _make_square_zone("zone-a", size=100)
        zone_b = ZoneDefinition(
            zone_id="zone-b",
            name="Zone B",
            polygon=np.array([
                [200, 200], [300, 200], [300, 300], [200, 300],
            ], dtype=float),
        )
        mgr = ZoneManager(zones=[zone_a, zone_b])
        obj = _make_object(pid=1, x=50, y=50)
        observations = mgr.update([obj], frame_idx=0)

        enter_zones = {o.zone_id for o in observations if o.observation_type == ObservationType.ENTER}
        assert "zone-a" in enter_zones
        assert "zone-b" not in enter_zones

    def test_get_zone_state(self) -> None:
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=50, y=50)
        mgr.update([obj], frame_idx=0, timestamp=1.0)

        state = mgr.get_zone_state("zone-a")
        assert state is not None
        assert 1 in state.present_objects
        assert state.is_currently_observed is True
        assert state.last_observed_at == 1.0

    def test_get_zone_state_returns_copy(self) -> None:
        """Mutating returned state should not affect internal state."""
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=50, y=50)
        mgr.update([obj], frame_idx=0)

        state = mgr.get_zone_state("zone-a")
        assert state is not None
        state.present_objects.clear()

        # Internal state should be unaffected
        internal = mgr.get_zone_state("zone-a")
        assert internal is not None
        assert 1 in internal.present_objects

    def test_observations_logged(self) -> None:
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=50, y=50)
        mgr.update([obj], frame_idx=0)
        mgr.update([obj], frame_idx=1)

        all_obs = mgr.observations
        assert len(all_obs) == 2
        assert all_obs[0].observation_type == ObservationType.ENTER
        assert all_obs[1].observation_type == ObservationType.PRESENT

    def test_load_zones_from_list(self) -> None:
        zones = [
            _make_square_zone("z1"),
            _make_square_zone("z2"),
        ]
        mgr = ZoneManager(zones=zones)
        assert mgr.get_zone_state("z1") is not None
        assert mgr.get_zone_state("z2") is not None

    def test_persistent_id_none_skipped(self) -> None:
        """Objects without persistent_id are skipped."""
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = WorldObject(
            tracker_id=1,
            bbox_xyxy=np.array([0, 0, 10, 10]),
            world_position=np.array([50.0, 50.0, 0.0]),
            persistent_id=None,
            class_name="person",
        )
        observations = mgr.update([obj], frame_idx=0)
        assert len(observations) == 0

    def test_object_disappears_generates_exit(self) -> None:
        """Object that was present but is absent from update generates EXIT."""
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj = _make_object(pid=1, x=50, y=50)
        mgr.update([obj], frame_idx=0)

        # Empty update — object disappeared entirely
        observations = mgr.update([], frame_idx=1)
        assert len(observations) == 1
        assert observations[0].observation_type == ObservationType.EXIT
        assert observations[0].persistent_id == 1

    def test_multiple_objects_same_zone(self) -> None:
        """Multiple objects can be present in the same zone."""
        mgr = ZoneManager(zones=[_make_square_zone()])
        obj1 = _make_object(pid=1, x=30, y=30)
        obj2 = _make_object(pid=2, x=70, y=70)
        observations = mgr.update([obj1, obj2], frame_idx=0)
        enters = [o for o in observations if o.observation_type == ObservationType.ENTER]
        assert len(enters) == 2
        assert {e.persistent_id for e in enters} == {1, 2}

    def test_overlapping_zones_first_zone_wins(self) -> None:
        """Object in overlapping zones gets zone_id of first matching zone."""
        zone_a = _make_square_zone("zone-a", size=200)
        zone_b = _make_square_zone("zone-b", size=200)
        mgr = ZoneManager(zones=[zone_a, zone_b])
        obj = _make_object(pid=1, x=50, y=50)
        observations = mgr.update([obj], frame_idx=0)
        # Should get ENTER for both zones
        assert len(observations) == 2
        # zone_id should be set to first zone
        assert obj.zone_id == "zone-a"


class TestZoneDefinitionValidation:
    """Tests for ZoneDefinition __post_init__ validation."""

    def test_empty_zone_id_raises(self) -> None:
        with pytest.raises(ValueError, match="zone_id"):
            ZoneDefinition(
                zone_id="",
                name="Test",
                polygon=np.array([[0, 0], [1, 0], [1, 1]]),
            )

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ZoneDefinition(
                zone_id="z1",
                name="",
                polygon=np.array([[0, 0], [1, 0], [1, 1]]),
            )

    def test_polygon_too_few_vertices_raises(self) -> None:
        with pytest.raises(ValueError, match="polygon"):
            ZoneDefinition(
                zone_id="z1",
                name="Test",
                polygon=np.array([[0, 0], [1, 0]]),
            )
