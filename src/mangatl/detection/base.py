"""Detector interface: finds bubbles and text blocks on a page."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from PIL import Image

DetectionLabel = Literal["bubble", "text_bubble", "text_free"]


@dataclass(frozen=True)
class RawDetection:
    label: DetectionLabel
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels
    score: float


class Detector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> list[RawDetection]:
        """Detect bubbles and text blocks on one page (RGB or L image)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model memory."""
