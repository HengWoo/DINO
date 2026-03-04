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


def _available_devices() -> set[str]:
    devices = {"cpu"}
    if torch.cuda.is_available():
        devices.add("cuda")
    if torch.backends.mps.is_available():
        devices.add("mps")
    return devices


class GroundingDINODetector(BaseDetector):
    """Grounding DINO detector via HuggingFace Transformers."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        resolved_device = device or _auto_device()
        if device is not None:
            available = _available_devices()
            if device not in available:
                raise ValueError(
                    f"Device '{device}' is not available. "
                    f"Available devices: {sorted(available)}"
                )
        self.device = resolved_device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load processor for model '{model_id}'. "
                f"Check your internet connection and model ID."
            ) from e

        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
                self.device
            )
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"Out of memory loading model '{model_id}' on device '{self.device}'. "
                f"Try using --device cpu or a smaller model."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_id}' on device '{self.device}': {e}"
            ) from e

    def detect(self, frame: np.ndarray, prompts: list[str]) -> sv.Detections:
        """Detect objects matching text prompts in a frame."""
        if frame is None:
            raise ValueError("frame must not be None")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be a 3-channel image (H, W, 3), got shape {frame.shape}"
            )
        if not prompts:
            raise ValueError("prompts must not be empty")

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

        raw_class_ids = np.array(
            [self._label_to_class_id(label, prompts) for label in labels],
            dtype=int,
        ) if labels else np.array([], dtype=int)

        mask = raw_class_ids >= 0
        if len(mask) > 0 and not mask.all():
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [l for l, m in zip(labels, mask) if m]
            raw_class_ids = raw_class_ids[mask]

        return sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=raw_class_ids,
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
        # Fall back to bidirectional containment
        for i, prompt in enumerate(prompts):
            prompt_lower = prompt.lower()
            if label_lower in prompt_lower or prompt_lower in label_lower:
                return i
        return -1
