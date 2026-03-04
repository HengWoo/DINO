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
        ValueError: If JSON is missing required "zones" key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zones file not found: {path}")

    data = json.loads(path.read_text())

    if "zones" not in data:
        raise ValueError("Zones file must contain a 'zones' key")

    zones = []
    for z in data["zones"]:
        zones.append(ZoneDefinition(
            zone_id=z["zone_id"],
            name=z["name"],
            polygon=np.array(z["polygon"], dtype=np.float64),
            metadata=z.get("metadata", {}),
        ))

    rules = []
    for r in data.get("rules", []):
        rules.append(EventRule(
            event_type=r["event_type"],
            requires_all=r.get("requires_all"),
            requires_any=r.get("requires_any"),
            requires_none=r.get("requires_none"),
            min_count=r.get("min_count"),
            hysteresis=r.get("hysteresis", 1),
        ))

    return zones, rules
