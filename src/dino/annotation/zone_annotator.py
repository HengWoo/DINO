from __future__ import annotations

import cv2
import numpy as np

from dino.zones.models import ZoneDefinition, ZoneState


class ZoneAnnotator:
    """Annotator that draws zone polygons on video frames.

    Active zones (containing objects) are drawn in active_color,
    inactive zones in zone_color. Zones are drawn as semi-transparent
    filled polygons with solid outlines and text labels.
    """

    def __init__(
        self,
        zone_color: tuple[int, int, int] = (0, 255, 0),
        active_color: tuple[int, int, int] = (0, 165, 255),
        alpha: float = 0.2,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.zone_color = zone_color
        self.active_color = active_color
        self.alpha = alpha

    def annotate(
        self,
        frame: np.ndarray,
        zones: list[ZoneDefinition],
        zone_states: dict[str, ZoneState] | None = None,
    ) -> np.ndarray:
        """Draw zone overlays on a frame.

        Args:
            frame: BGR image (H, W, 3)
            zones: Zone definitions to draw
            zone_states: Optional mapping of zone_id -> ZoneState.
                Zones with present_objects are drawn as active.

        Returns:
            Annotated copy of the frame (input is not modified).
        """
        annotated = frame.copy()
        if not zones:
            return annotated

        overlay = annotated.copy()

        for zone in zones:
            polygon_pts = zone.polygon[:, :2].astype(np.int32)

            # Determine if zone is active
            is_active = False
            object_count = 0
            if zone_states and zone.zone_id in zone_states:
                state = zone_states[zone.zone_id]
                object_count = len(state.present_objects)
                is_active = object_count > 0

            color = self.active_color if is_active else self.zone_color

            # Fill polygon on overlay
            cv2.fillPoly(overlay, [polygon_pts], color)

            # Blend overlay with annotated frame
            cv2.addWeighted(overlay, self.alpha, annotated, 1 - self.alpha, 0, annotated)
            # Reset overlay for next zone
            overlay = annotated.copy()

            # Draw outline
            cv2.polylines(annotated, [polygon_pts], isClosed=True, color=color, thickness=2)

            # Draw label
            centroid = polygon_pts.mean(axis=0).astype(int)
            label = zone.name
            if is_active:
                label += f" ({object_count})"
            cv2.putText(
                annotated,
                label,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return annotated
