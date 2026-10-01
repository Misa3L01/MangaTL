"""Reading-order interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from mangatl.models import BBox

Direction = Literal["rtl", "ltr"]


class ReadingOrderer(ABC):
    @abstractmethod
    def order(
        self, boxes: list[BBox], page_width: int, page_height: int, is_spread: bool
    ) -> list[int]:
        """Return the indices of `boxes` in reading order."""


def coarse_position(box: BBox, page_width: int, page_height: int) -> str:
    """3x3 grid position, e.g. 'top-right' (given to the translator as context)."""
    col = ("left", "center", "right")[min(2, int(3 * box.cx / max(1, page_width)))]
    row = ("top", "middle", "bottom")[min(2, int(3 * box.cy / max(1, page_height)))]
    return f"{row}-{col}"
