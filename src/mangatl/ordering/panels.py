"""Panel-aware reading order.

Panels are found as the areas NOT connected to the white page margin (gutters): the ink of
panel borders is thickened slightly so small gaps do not let the flood fill leak in.
Panels are ordered with the band heuristic (rows, right-to-left for Japanese), then the
regions inside each panel. Regions outside every panel count as their own tiny panel.
"""

from __future__ import annotations

import cv2
import numpy as np

from mangatl.models import BBox
from mangatl.ordering.bands import BandOrderer
from mangatl.ordering.base import Direction, ReadingOrderer

MIN_PANEL_AREA = 0.015  # fraction of the page
MAX_PANEL_AREA = 0.97  # a "panel" covering the whole page means no gutters were found


def detect_panels(gray: np.ndarray) -> list[BBox]:
    h, w = gray.shape
    ink = (gray < 200).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    free = (1 - ink).astype(np.uint8)
    padded = np.pad(free, 1, constant_values=1)
    flood = padded.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    gutter = flood[1:-1, 1:-1] == 2
    panels = (~gutter).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(panels, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_PANEL_AREA * h * w or bw * bh > MAX_PANEL_AREA * h * w:
            continue
        if min(bw, bh) < 0.05 * min(h, w):
            continue
        boxes.append(BBox(x0=int(x), y0=int(y), x1=int(x + bw), y1=int(y + bh)))
    # Drop panels nested inside another one (insets are read with their parent).
    return [
        b
        for b in boxes
        if not any(
            o is not b and o.intersection_area(b) >= 0.9 * b.area and o.area > b.area for o in boxes
        )
    ]


class PanelOrderer(ReadingOrderer):
    def __init__(self, direction: Direction = "rtl") -> None:
        self.direction = direction
        self.bands = BandOrderer(direction)
        self.last_panels: list[int | None] = []

    def order(
        self,
        boxes: list[BBox],
        page_width: int,
        page_height: int,
        is_spread: bool,
        gray: np.ndarray | None = None,
    ) -> list[int]:
        panels = detect_panels(gray) if gray is not None else []
        if len(panels) < 2:
            self.last_panels = [None] * len(boxes)
            return self.bands.order(boxes, page_width, page_height, is_spread)

        owner: list[int | None] = []
        for b in boxes:
            inside = [
                i for i, p in enumerate(panels) if p.x0 <= b.cx <= p.x1 and p.y0 <= b.cy <= p.y1
            ]
            owner.append(min(inside, key=lambda i: panels[i].area) if inside else None)

        # Units to order: every panel with regions, plus each orphan region on its own.
        units: list[tuple[BBox, list[int]]] = []
        for i, panel in enumerate(panels):
            members = [k for k, o in enumerate(owner) if o == i]
            if members:
                units.append((panel, members))
        units += [(boxes[k], [k]) for k, o in enumerate(owner) if o is None]

        unit_order = self.bands.order([u[0] for u in units], page_width, page_height, is_spread)
        order: list[int] = []
        panel_number: dict[int, int] = {}
        for rank, u in enumerate(unit_order, start=1):
            members = units[u][1]
            inner = self.bands.order([boxes[k] for k in members], page_width, page_height, False)
            for k in (members[j] for j in inner):
                order.append(k)
                panel_number[k] = rank
        self.last_panels = [
            panel_number.get(k) if owner[k] is not None else None for k in range(len(boxes))
        ]
        return order
