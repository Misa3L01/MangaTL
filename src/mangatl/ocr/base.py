"""OCR interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float  # 0..1, geometric mean of the generated tokens' probabilities


class OcrEngine(ABC):
    @abstractmethod
    def read(self, crops: list[Image.Image]) -> list[OcrResult]:
        """Recognize the text in each crop (one text block per crop)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model memory."""
