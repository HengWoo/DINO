from __future__ import annotations

import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from dino.detectors.base import BaseDetector

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroundingDINODetector(BaseDetector):
    """Grounding DINO detector via HuggingFace Transformers."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        self.device = device or _auto_device()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )

    def detect(self, frame: np.ndarray, prompts: list[str]) -> sv.Detections:
        """Detect objects matching text prompts in a frame."""
        image = Image.fromarray(frame[..., ::-1])  # BGR -> RGB
        text = ". ".join(prompts) + "."

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        raw_class_ids = [self._label_to_class_id(label, prompts) for label in labels]

        # Filter out unmatched detections (class_id == -1)
        mask = np.array([cid >= 0 for cid in raw_class_ids], dtype=bool)
        if len(mask) > 0 and not mask.all():
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [l for l, m in zip(labels, mask) if m]
            raw_class_ids = [c for c, m in zip(raw_class_ids, mask) if m]

        class_ids = np.array(raw_class_ids, dtype=int) if raw_class_ids else np.array([], dtype=int)

        return sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids,
            data={"class_name": np.array(labels) if labels else np.array([])},
        )

    @staticmethod
    def _label_to_class_id(label: str, prompts: list[str]) -> int:
        """Map a detection label back to a prompt index. Returns -1 if no match."""
        label_lower = label.lower().strip()
        # Prefer exact match
        for i, prompt in enumerate(prompts):
            if label_lower == prompt.lower():
                return i
        # Fall back to containment (label in prompt only)
        for i, prompt in enumerate(prompts):
            if label_lower in prompt.lower():
                return i
        return -1
