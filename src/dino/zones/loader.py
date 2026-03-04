from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dino.zones.models import ZoneDefinition
from dino.zones.event_rules import EventRule


def load_zones_file(path: str | Path) -> tuple[list[ZoneDefinition], list[EventRule]]:
    """Load zone definitions and event rules from a JSON file.

    JSON format:
    {
        "zones": [
            {"zone_id": "z1", "name": "Zone 1", "polygon": [[0,0],[100,0],[100,100],[0,100]]},
            ...
        ],
        "rules": [
            {"event_type": "occupied", "requires_any": ["person"]},
            ...
        ]
    }

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If JSON is invalid or missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zones file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Zones file {path} contains invalid JSON: {e}") from e

    if "zones" not in data:
        raise ValueError("Zones file must contain a 'zones' key")

    zones = []
    for i, z in enumerate(data["zones"]):
        required = ("zone_id", "name", "polygon")
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(
                f"Zone entry {i} in {path} is missing required keys: {missing}"
            )
        try:
            zones.append(ZoneDefinition(
                zone_id=z["zone_id"],
                name=z["name"],
                polygon=np.array(z["polygon"], dtype=np.float64),
                metadata=z.get("metadata", {}),
            ))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid zone entry {i} (zone_id={z.get('zone_id', '?')}) "
                f"in {path}: {e}"
            ) from e

    rules = []
    for i, r in enumerate(data.get("rules", [])):
        if "event_type" not in r:
            raise ValueError(
                f"Rule entry {i} in {path} is missing required key 'event_type'"
            )
        try:
            rules.append(EventRule(
                event_type=r["event_type"],
                requires_all=r.get("requires_all"),
                requires_any=r.get("requires_any"),
                requires_none=r.get("requires_none"),
                min_count=r.get("min_count"),
                hysteresis=r.get("hysteresis", 1),
            ))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid rule entry {i} (event_type={r.get('event_type', '?')}) "
                f"in {path}: {e}"
            ) from e

    return zones, rules
