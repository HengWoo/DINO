from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dino.zones.models import ZoneEvent, ZoneState


@dataclass
class EventRule:
    """Configurable rule for generating zone events."""
    event_type: str
    requires_all: list[str] | None = None
    requires_any: list[str] | None = None
    requires_none: list[str] | None = None
    min_count: dict[str, int] | None = None
    hysteresis: int = 1

    def evaluate(self, state: ZoneState) -> bool:
        """Check if rule conditions are met for the given zone state."""
        class_names = [obj.class_name for obj in state.present_objects.values()]
        class_set = set(class_names)

        if self.requires_all:
            if not all(c in class_set for c in self.requires_all):
                return False

        if self.requires_any:
            if not any(c in class_set for c in self.requires_any):
                return False

        if self.requires_none:
            if any(c in class_set for c in self.requires_none):
                return False

        if self.min_count:
            counts = Counter(class_names)
            for cls, min_n in self.min_count.items():
                if counts.get(cls, 0) < min_n:
                    return False

        return True


class RuleEngine:
    """Evaluates event rules against zone states with hysteresis."""

    def __init__(self, rules: list[EventRule]):
        self._rules = rules
        # (zone_id, event_type) -> consecutive frame count
        self._counters: dict[tuple[str, str], int] = {}

    def evaluate(
        self, state: ZoneState, frame_idx: int, timestamp: float = 0.0
    ) -> list[ZoneEvent]:
        """Evaluate all rules against a zone state.

        Returns events for rules that have met their hysteresis threshold.
        """
        events: list[ZoneEvent] = []
        for rule in self._rules:
            key = (state.zone_id, rule.event_type)
            if rule.evaluate(state):
                self._counters[key] = self._counters.get(key, 0) + 1
                if self._counters[key] >= rule.hysteresis:
                    events.append(
                        ZoneEvent(
                            zone_id=state.zone_id,
                            event_type=rule.event_type,
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                        )
                    )
                    self._counters[key] = 0
            else:
                self._counters[key] = 0
        return events
