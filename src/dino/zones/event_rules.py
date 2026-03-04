from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from dino.zones.models import ZoneEvent, ZoneState

logger = logging.getLogger(__name__)


@dataclass
class EventRule:
    """Configurable rule for generating zone events.

    At least one condition (requires_all, requires_any, requires_none,
    or min_count) must be specified.
    """
    event_type: str
    requires_all: list[str] | None = None
    requires_any: list[str] | None = None
    requires_none: list[str] | None = None
    min_count: dict[str, int] | None = None
    hysteresis: int = 1

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type must not be empty")
        if self.hysteresis < 1:
            raise ValueError(f"hysteresis must be >= 1, got {self.hysteresis}")
        if (
            self.requires_all is None
            and self.requires_any is None
            and self.requires_none is None
            and self.min_count is None
        ):
            raise ValueError(
                "At least one condition must be set "
                "(requires_all, requires_any, requires_none, or min_count)"
            )

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
    """Evaluates event rules against zone states with hysteresis.

    Hysteresis is latching: once an event fires, it will not fire again
    until the condition breaks and then re-meets the hysteresis threshold.
    """

    def __init__(self, rules: list[EventRule]):
        self._rules = rules
        # (zone_id, event_type) -> consecutive frame count
        self._counters: dict[tuple[str, str], int] = {}
        # (zone_id, event_type) -> whether currently latched (already fired)
        self._latched: dict[tuple[str, str], bool] = {}

    def evaluate(
        self, state: ZoneState, frame_idx: int, timestamp: float = 0.0
    ) -> list[ZoneEvent]:
        """Evaluate all rules against a zone state.

        Returns events for rules that have met their hysteresis threshold.
        Events fire once and latch until the condition breaks.
        """
        events: list[ZoneEvent] = []
        for rule in self._rules:
            key = (state.zone_id, rule.event_type)
            if rule.evaluate(state):
                if self._latched.get(key, False):
                    # Already fired, don't re-fire
                    continue
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
                    self._latched[key] = True
                    self._counters[key] = 0
                    logger.debug(
                        "Event %s fired for zone %s at frame %d",
                        rule.event_type, state.zone_id, frame_idx,
                    )
            else:
                self._counters[key] = 0
                self._latched[key] = False
        return events
