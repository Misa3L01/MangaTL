from __future__ import annotations

import numpy as np

from mangatl.models import BBox
from mangatl.ordering import BandOrderer, coarse_position


def box(x0: int, y0: int, w: int = 100, h: int = 200) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + h)


def test_japanese_rows_right_to_left_then_down() -> None:
    boxes = [box(100, 50), box(700, 60), box(400, 40), box(650, 700), box(80, 720)]
    order = BandOrderer("rtl").order(boxes, 1000, 1500, is_spread=False)
    assert order == [1, 2, 0, 3, 4]


def test_english_rows_left_to_right() -> None:
    boxes = [box(700, 50), box(100, 60)]
    assert BandOrderer("ltr").order(boxes, 1000, 1500, is_spread=False) == [1, 0]


def test_boxes_without_vertical_overlap_are_separate_rows() -> None:
    # A low box on the left must not be read before a higher box on the right.
    boxes = [box(100, 400), box(700, 50)]
    assert BandOrderer("rtl").order(boxes, 1000, 1500, is_spread=False) == [1, 0]


def test_spread_reads_right_page_first() -> None:
    boxes = [box(100, 50), box(1200, 900), box(1700, 50)]
    order = BandOrderer("rtl").order(boxes, 2000, 1500, is_spread=True)
    assert order == [2, 1, 0]


def _panel_page() -> np.ndarray:
    """Top wide panel; below, a tall panel on the right and two stacked ones on the left."""
    import cv2

    page = np.full((1500, 1000), 255, np.uint8)
    for x0, y0, x1, y1 in [
        (20, 20, 980, 380),
        (700, 400, 980, 1480),
        (20, 400, 680, 850),
        (20, 870, 680, 1480),
    ]:
        cv2.rectangle(page, (x0, y0), (x1, y1), 0, 4)
        page[y0 + 30 : y1 - 30 : 7, x0 + 30 : x1 - 30 : 7] = 90  # some "art" inside
    return page


def test_panels_are_detected() -> None:
    from mangatl.ordering import detect_panels

    panels = sorted(detect_panels(_panel_page()), key=lambda b: (b.y0, b.x0))
    assert len(panels) == 4
    assert abs(panels[0].x0 - 20) <= 6 and abs(panels[0].y1 - 380) <= 6


def test_panel_order_fixes_tall_panel_layout() -> None:
    from mangatl.ordering import PanelOrderer

    boxes = [
        box(750, 1100, 100, 150),  # 0: tall right panel, low
        box(300, 450, 100, 150),  # 1: left upper panel
        box(300, 1000, 100, 150),  # 2: left lower panel
        box(450, 100, 100, 150),  # 3: top panel
    ]
    rows = BandOrderer("rtl").order(boxes, 1000, 1500, is_spread=False)
    assert rows != [3, 0, 1, 2], "rows alone read the tall panel too late"
    orderer = PanelOrderer("rtl")
    order = orderer.order(boxes, 1000, 1500, is_spread=False, gray=_panel_page())
    assert order == [3, 0, 1, 2]
    assert orderer.last_panels == [2, 3, 4, 1]


def test_panel_order_falls_back_to_rows_without_panels() -> None:
    from mangatl.ordering import PanelOrderer

    boxes = [box(100, 50), box(700, 60)]
    blank = np.full((1500, 1000), 255, np.uint8)
    orderer = PanelOrderer("rtl")
    assert orderer.order(boxes, 1000, 1500, False, gray=blank) == [1, 0]
    assert orderer.last_panels == [None, None]


def test_coarse_position() -> None:
    assert coarse_position(box(850, 50), 1000, 1500) == "top-right"
    assert coarse_position(box(450, 650), 1000, 1500) == "middle-center"
    assert coarse_position(box(0, 1300), 1000, 1500) == "bottom-left"
