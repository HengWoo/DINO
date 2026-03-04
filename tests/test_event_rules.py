"""Tests for event rules and rule engine."""
from __future__ import annotations

import numpy as np

from dino.spatial.models import WorldObject
from dino.zones.event_rules import EventRule, RuleEngine
from dino.zones.models import ZoneState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_object(
    pid: int,
    class_name: str = "person",
) -> WorldObject:
    """Create a minimal WorldObject."""
    return WorldObject(
        tracker_id=pid,
        bbox_xyxy=np.array([0, 0, 10, 10]),
        world_position=np.array([0.0, 0.0, 0.0]),
        persistent_id=pid,
        class_name=class_name,
    )


def _make_state(zone_id: str, objects: list[WorldObject]) -> ZoneState:
    """Create a ZoneState with given objects."""
    return ZoneState(
        zone_id=zone_id,
        present_objects={obj.persistent_id: obj for obj in objects},  # type: ignore[misc]
        is_currently_observed=len(objects) > 0,
    )


# ===========================================================================
# TestEventRule
# ===========================================================================

class TestEventRule:
    """Tests for EventRule.evaluate()."""

    def test_requires_all_satisfied(self) -> None:
        rule = EventRule(event_type="dining", requires_all=["person", "plate of food"])
        state = _make_state("z1", [
            _make_object(1, "person"),
            _make_object(2, "plate of food"),
        ])
        assert rule.evaluate(state) is True

    def test_requires_all_not_satisfied(self) -> None:
        rule = EventRule(event_type="dining", requires_all=["person", "plate of food"])
        state = _make_state("z1", [_make_object(1, "person")])
        assert rule.evaluate(state) is False

    def test_requires_any_satisfied(self) -> None:
        rule = EventRule(event_type="drink", requires_any=["cup", "glass"])
        state = _make_state("z1", [_make_object(1, "cup")])
        assert rule.evaluate(state) is True

    def test_requires_any_not_satisfied(self) -> None:
        rule = EventRule(event_type="drink", requires_any=["cup", "glass"])
        state = _make_state("z1", [_make_object(1, "person")])
        assert rule.evaluate(state) is False

    def test_requires_none_satisfied(self) -> None:
        rule = EventRule(event_type="empty", requires_none=["person"])
        state = _make_state("z1", [_make_object(1, "chair")])
        assert rule.evaluate(state) is True

    def test_requires_none_not_satisfied(self) -> None:
        rule = EventRule(event_type="empty", requires_none=["person"])
        state = _make_state("z1", [_make_object(1, "person")])
        assert rule.evaluate(state) is False

    def test_min_count(self) -> None:
        rule = EventRule(event_type="crowd", min_count={"person": 2})
        state_ok = _make_state("z1", [
            _make_object(1, "person"),
            _make_object(2, "person"),
            _make_object(3, "person"),
        ])
        assert rule.evaluate(state_ok) is True

        state_low = _make_state("z1", [_make_object(1, "person")])
        assert rule.evaluate(state_low) is False

    def test_combined_rules(self) -> None:
        rule = EventRule(
            event_type="secure",
            requires_all=["camera"],
            requires_none=["person"],
        )
        state_ok = _make_state("z1", [_make_object(1, "camera")])
        assert rule.evaluate(state_ok) is True

        state_bad = _make_state("z1", [
            _make_object(1, "camera"),
            _make_object(2, "person"),
        ])
        assert rule.evaluate(state_bad) is False


# ===========================================================================
# TestRuleEngine
# ===========================================================================

class TestRuleEngine:
    """Tests for RuleEngine with hysteresis."""

    def test_evaluate_with_hysteresis(self) -> None:
        rule = EventRule(event_type="alert", requires_all=["person"], hysteresis=3)
        engine = RuleEngine(rules=[rule])
        state = _make_state("z1", [_make_object(1, "person")])

        # Frames 0, 1 — no event yet
        assert engine.evaluate(state, frame_idx=0) == []
        assert engine.evaluate(state, frame_idx=1) == []
        # Frame 2 — hysteresis met (3 consecutive)
        events = engine.evaluate(state, frame_idx=2)
        assert len(events) == 1
        assert events[0].event_type == "alert"

    def test_hysteresis_resets(self) -> None:
        rule = EventRule(event_type="alert", requires_all=["person"], hysteresis=3)
        engine = RuleEngine(rules=[rule])
        state_present = _make_state("z1", [_make_object(1, "person")])
        state_empty = _make_state("z1", [])

        # 2 consecutive, then break
        engine.evaluate(state_present, frame_idx=0)
        engine.evaluate(state_present, frame_idx=1)
        engine.evaluate(state_empty, frame_idx=2)  # resets counter

        # Restart — needs another 3 consecutive
        assert engine.evaluate(state_present, frame_idx=3) == []
        assert engine.evaluate(state_present, frame_idx=4) == []
        events = engine.evaluate(state_present, frame_idx=5)
        assert len(events) == 1

    def test_multiple_rules(self) -> None:
        rule_a = EventRule(event_type="rule-a", requires_all=["person"])
        rule_b = EventRule(event_type="rule-b", requires_all=["car"])
        engine = RuleEngine(rules=[rule_a, rule_b])
        state = _make_state("z1", [_make_object(1, "person")])

        events = engine.evaluate(state, frame_idx=0)
        types = {e.event_type for e in events}
        assert "rule-a" in types
        assert "rule-b" not in types

    def test_event_includes_zone_and_frame(self) -> None:
        rule = EventRule(event_type="test-event", requires_all=["person"])
        engine = RuleEngine(rules=[rule])
        state = _make_state("zone-x", [_make_object(1, "person")])

        events = engine.evaluate(state, frame_idx=42, timestamp=1.5)
        assert len(events) == 1
        assert events[0].zone_id == "zone-x"
        assert events[0].frame_idx == 42
        assert events[0].timestamp == 1.5
