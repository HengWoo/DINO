import numpy as np
import supervision as sv

from dino.tracking.tracker import ObjectTracker


class TestObjectTracker:
    def test_returns_detections_with_tracker_id(self):
        tracker = ObjectTracker()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        tracked = tracker.update(detections)
        assert isinstance(tracked, sv.Detections)
        assert tracked.tracker_id is not None
        assert len(tracked.tracker_id) == 1

    def test_consistent_ids_across_frames(self):
        tracker = ObjectTracker()

        det1 = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        tracked1 = tracker.update(det1)

        # Slightly moved box — same object
        det2 = sv.Detections(
            xyxy=np.array([[12, 22, 102, 202]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        tracked2 = tracker.update(det2)

        assert tracked1.tracker_id[0] == tracked2.tracker_id[0]

    def test_empty_detections(self):
        tracker = ObjectTracker()
        detections = sv.Detections.empty()
        tracked = tracker.update(detections)
        assert isinstance(tracked, sv.Detections)
        assert len(tracked) == 0

    def test_multiple_objects_get_different_ids(self):
        tracker = ObjectTracker()
        detections = sv.Detections(
            xyxy=np.array(
                [[10, 20, 100, 200], [300, 400, 500, 600]], dtype=np.float32
            ),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 1]),
        )
        tracked = tracker.update(detections)
        assert len(tracked.tracker_id) == 2
        assert tracked.tracker_id[0] != tracked.tracker_id[1]

    def test_reset_clears_state(self):
        tracker = ObjectTracker()
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        tracker.update(detections)
        tracker.reset()
        # After reset, tracker state should be clean
        tracked = tracker.update(detections)
        assert tracked.tracker_id is not None
