"""RT-DETR-v2 comic detector (ogkalu/comic-text-and-bubble-detector, Apache-2.0)."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from mangatl.detection.base import Detector, RawDetection
from mangatl.gpu import release_cuda_memory

log = logging.getLogger(__name__)


class RtDetrDetector(Detector):
    def __init__(self, model_dir: Path, device: str = "cuda", threshold: float = 0.35) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self._torch = torch
        self.device = device
        self.threshold = threshold
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir).to(device).eval()
        self.id2label: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}

    def _detect_tile(self, image: Image.Image, x_offset: int = 0) -> list[RawDetection]:
        torch = self._torch
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(rgb.height, rgb.width)]
        )[0]
        detections = []
        for score, label, box in zip(
            results["scores"].tolist(),
            results["labels"].tolist(),
            results["boxes"].tolist(),
            strict=True,
        ):
            x0, y0, x1, y1 = (round(v) for v in box)
            x0, x1 = max(0, x0), min(rgb.width, x1)
            y0, y1 = max(0, y0), min(rgb.height, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            detections.append(
                RawDetection(
                    label=self.id2label[label],  # type: ignore[arg-type]
                    box=(x0 + x_offset, y0, x1 + x_offset, y1),
                    score=float(score),
                )
            )
        return detections

    def detect(self, image: Image.Image) -> list[RawDetection]:
        # The model sees a 640x640 resize: a two-page spread is split so each half keeps
        # enough resolution, then boxes are mapped back to spread coordinates.
        if image.width / image.height > 1.1:
            mid = image.width // 2
            left = self._detect_tile(image.crop((0, 0, mid, image.height)))
            right = self._detect_tile(image.crop((mid, 0, image.width, image.height)), mid)
            return left + right
        return self._detect_tile(image)

    def close(self) -> None:
        del self.model
        release_cuda_memory()
