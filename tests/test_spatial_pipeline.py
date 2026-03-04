"""Tests for spatial pipeline components."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dino.config import SpatialConfig


class TestSpatialConfig:
    def test_defaults(self):
        config = SpatialConfig()
        assert config.camera_mode == "fixed"
        assert config.match_distance == 50.0
        assert config.max_age_seconds == 30.0
        assert config.zones_path is None
        assert config.rules is None

    def test_invalid_camera_mode(self):
        with pytest.raises(ValueError, match="Unsupported camera_mode"):
            SpatialConfig(camera_mode="moving")

    def test_invalid_match_distance(self):
        with pytest.raises(ValueError, match="match_distance must be > 0"):
            SpatialConfig(match_distance=0)

    def test_invalid_max_age(self):
        with pytest.raises(ValueError, match="max_age_seconds must be > 0"):
            SpatialConfig(max_age_seconds=-1)


class TestLoadZonesFile:
    def test_load_valid_file(self, tmp_path):
        from dino.zones.loader import load_zones_file
        zones_data = {
            "zones": [
                {"zone_id": "z1", "name": "Zone 1", "polygon": [[0,0],[100,0],[100,100],[0,100]]},
            ],
            "rules": [
                {"event_type": "occupied", "requires_any": ["person"]},
            ],
        }
        path = tmp_path / "zones.json"
        path.write_text(json.dumps(zones_data))

        zones, rules = load_zones_file(path)
        assert len(zones) == 1
        assert zones[0].zone_id == "z1"
        assert zones[0].name == "Zone 1"
        assert zones[0].polygon.shape == (4, 2)
        assert len(rules) == 1
        assert rules[0].event_type == "occupied"

    def test_file_not_found(self):
        from dino.zones.loader import load_zones_file
        with pytest.raises(FileNotFoundError):
            load_zones_file("/nonexistent/zones.json")

    def test_missing_zones_key(self, tmp_path):
        from dino.zones.loader import load_zones_file
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"rules": []}))
        with pytest.raises(ValueError, match="must contain a 'zones' key"):
            load_zones_file(path)

    def test_no_rules_section(self, tmp_path):
        from dino.zones.loader import load_zones_file
        zones_data = {
            "zones": [
                {"zone_id": "z1", "name": "Zone 1", "polygon": [[0,0],[100,0],[100,100],[0,100]]},
            ],
        }
        path = tmp_path / "zones.json"
        path.write_text(json.dumps(zones_data))

        zones, rules = load_zones_file(path)
        assert len(zones) == 1
        assert len(rules) == 0


class TestZoneAnnotator:
    def _make_zone(self, zone_id="z1", name="Test Zone"):
        from dino.zones.models import ZoneDefinition

        return ZoneDefinition(
            zone_id=zone_id,
            name=name,
            polygon=np.array(
                [[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float64
            ),
        )

    def _make_frame(self, h=240, w=320):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_annotate_returns_copy(self):
        from dino.annotation.zone_annotator import ZoneAnnotator

        annotator = ZoneAnnotator()
        frame = self._make_frame()
        zone = self._make_zone()
        result = annotator.annotate(frame, [zone])
        # Input should not be modified
        assert np.array_equal(frame, np.zeros_like(frame))
        # Result should be a different array
        assert result is not frame

    def test_annotate_draws_on_frame(self):
        from dino.annotation.zone_annotator import ZoneAnnotator

        annotator = ZoneAnnotator()
        frame = self._make_frame()
        zone = self._make_zone()
        result = annotator.annotate(frame, [zone])
        # Output should differ from blank input (zone was drawn)
        assert not np.array_equal(result, frame)

    def test_active_zone_different_color(self):
        from dino.annotation.zone_annotator import ZoneAnnotator
        from dino.spatial.models import WorldObject
        from dino.zones.models import ZoneState

        annotator = ZoneAnnotator()
        zone = self._make_zone()

        # Inactive frame
        frame1 = self._make_frame()
        result_inactive = annotator.annotate(frame1, [zone])

        # Active frame (zone has present objects)
        frame2 = self._make_frame()
        obj = WorldObject(
            tracker_id=1,
            bbox_xyxy=np.array([10, 10, 50, 50]),
            world_position=np.array([30, 30, 0]),
            persistent_id=1,
            class_name="person",
        )
        zone_states = {
            "z1": ZoneState(zone_id="z1", present_objects={1: obj}),
        }
        result_active = annotator.annotate(frame2, [zone], zone_states=zone_states)

        # Active and inactive should produce different results
        assert not np.array_equal(result_inactive, result_active)
