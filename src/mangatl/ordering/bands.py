"""Band heuristic: rows of vertically overlapping boxes, each row read right-to-left (JA).

Good for most pages; complex panel layouts (a tall panel beside stacked ones) need panel
detection, planned for Phase 2. The translator may also suggest corrections.
"""

from __future__ import annotations

from mangatl.models import BBox
from mangatl.ordering.base import Direction, ReadingOrderer


class BandOrderer(ReadingOrderer):
    def __init__(self, direction: Direction = "rtl", min_overlap: float = 0.4) -> None:
        self.direction = direction
        self.min_overlap = min_overlap

    def _order_group(self, boxes: list[BBox], indices: list[int]) -> list[int]:
        remaining = sorted(indices, key=lambda i: (boxes[i].y0, boxes[i].cy))
        ordered: list[int] = []
        while remaining:
            anchor = boxes[remaining[0]]
            band = [remaining[0]]
            for i in remaining[1:]:
                b = boxes[i]
                overlap = min(anchor.y1, b.y1) - max(anchor.y0, b.y0)
                if overlap >= self.min_overlap * min(anchor.h, b.h):
                    band.append(i)
            band.sort(key=lambda i: boxes[i].cx, reverse=self.direction == "rtl")
            ordered.extend(band)
            remaining = [i for i in remaining if i not in band]
        return ordered

    def order(
        self, boxes: list[BBox], page_width: int, page_height: int, is_spread: bool
    ) -> list[int]:
        indices = list(range(len(boxes)))
        if not is_spread:
            return self._order_group(boxes, indices)
        mid = page_width / 2
        right = [i for i in indices if boxes[i].cx >= mid]
        left = [i for i in indices if boxes[i].cx < mid]
        first, second = (right, left) if self.direction == "rtl" else (left, right)
        return self._order_group(boxes, first) + self._order_group(boxes, second)
