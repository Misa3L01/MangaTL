"""Region analysis on synthetic pages: interior masks, glyph masks, border protection."""

from __future__ import annotations

import cv2
import numpy as np

from mangatl.config import DetectionConfig, InpaintConfig
from mangatl.detection.base import RawDetection
from mangatl.detection.regions import build_regions
from mangatl.inpainting.fill import FillInpainter
from mangatl.models import BBox, Region

CENTER, AXES = (300, 300), (150, 220)


def screentone(h: int = 600, w: int = 600) -> np.ndarray:
    """Dotted gray background like manga tone."""
    page = np.full((h, w), 235, np.uint8)
    page[::4, ::4] = 60
    page[2::4, 2::4] = 60
    return page


def bubble_page(brush: bool = False) -> tuple[np.ndarray, list[RawDetection]]:
    page = screentone()
    cv2.ellipse(page, CENTER, AXES, 0, 0, 360, 255, -1)
    cv2.ellipse(page, CENTER, AXES, 0, 0, 360, 0, 3)  # outline
    if brush:
        # Huge stroke that crosses the outline, like hand-drawn screaming.
        cv2.line(page, (300, 120), (300, 540), 0, 40)
    else:
        for x in (250, 300, 350):  # three vertical "text columns"
            for y in range(200, 400, 30):
                cv2.rectangle(page, (x - 10, y), (x + 10, y + 18), 0, -1)
    dets = [
        RawDetection("bubble", (150, 80, 450, 520), 0.97),
        RawDetection("text_bubble", (230, 110 if brush else 195, 370, 540 if brush else 423), 0.95),
    ]
    return page, dets


def interior_on_page(region_draft) -> np.ndarray:
    x0, y0, x1, y1 = region_draft.crop
    full = np.zeros((600, 600), bool)
    full[y0:y1, x0:x1] = region_draft.interior
    return full


def split_box_page() -> tuple[np.ndarray, list[RawDetection]]:
    """Caption box whose first line of text touches both sides of the outline."""
    page = screentone()
    cv2.rectangle(page, (150, 100), (450, 500), 255, -1)
    cv2.rectangle(page, (150, 100), (450, 500), 0, 3)
    cv2.rectangle(page, (200, 110), (215, 120), 0, -1)  # glyph above the dividing line
    for x in range(152, 448, 24):  # the dividing line: glyphs joined by a stroke
        cv2.rectangle(page, (x, 125), (min(x + 20, 448), 145), 0, -1)
    cv2.rectangle(page, (152, 134), (448, 136), 0, -1)
    for x in (220, 300, 380):  # more text below
        for y in range(170, 300, 30):
            cv2.rectangle(page, (x - 10, y), (x + 10, y + 18), 0, -1)
    dets = [
        RawDetection("bubble", (148, 98, 453, 503), 0.97),
        RawDetection("text_bubble", (155, 105, 445, 320), 0.95),
    ]
    return page, dets


def text_mask_on_page(draft) -> np.ndarray:
    x0, y0, x1, y1 = draft.crop
    full = np.zeros((600, 600), bool)
    full[y0:y1, x0:x1] = draft.text_mask
    return full


def test_join_text_areas_cleans_text_split_off_by_a_line_touching_the_outline() -> None:
    page, dets = split_box_page()
    [plain] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    [joined] = build_regions(page, dets, DetectionConfig(), InpaintConfig(join_text_areas=True))
    before, after = text_mask_on_page(plain), text_mask_on_page(joined)
    top_glyph = (slice(110, 121), slice(200, 216))
    line = (slice(126, 145), slice(170, 430))
    assert before[top_glyph].mean() < 0.5 and before[line].mean() < 0.5
    assert after[top_glyph].all() and after[line].mean() > 0.9
    # The outline itself is never part of the mask.
    assert not after[100:503, 148:153].any() and not after[100:503, 448:453].any()
    assert not after[98:103, 148:453].any()


def test_join_text_areas_never_touches_open_outlines() -> None:
    page = np.full((600, 600), 255, np.uint8)  # white page around the box
    cv2.ellipse(page, CENTER, AXES, 0, 0, 360, 0, 3)
    page[250:350, 145:160] = 255  # gap in the outline: the inside reaches the page
    for x in (250, 300, 350):
        for y in range(200, 400, 30):
            cv2.rectangle(page, (x - 10, y), (x + 10, y + 18), 0, -1)
    dets = [
        RawDetection("bubble", (150, 80, 450, 520), 0.97),
        RawDetection("text_bubble", (230, 195, 370, 423), 0.95),
    ]
    [plain] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    [joined] = build_regions(page, dets, DetectionConfig(), InpaintConfig(join_text_areas=True))
    assert "contorno abierto" in joined.notes
    # Open containers reach the page around them: the option leaves them exactly as before.
    assert np.array_equal(text_mask_on_page(plain), text_mask_on_page(joined))


def test_interior_follows_the_ellipse_and_text_mask_spares_the_outline() -> None:
    page, dets = bubble_page()
    [draft] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    assert draft.type == "speech_bubble"
    assert draft.background_uniform
    assert draft.clean_coverage > 0.95

    ellipse = np.zeros((600, 600), np.uint8)
    cv2.ellipse(ellipse, CENTER, (AXES[0] - 2, AXES[1] - 2), 0, 0, 360, 1, -1)
    interior = interior_on_page(draft)
    iou = (interior & ellipse.astype(bool)).sum() / (interior | ellipse.astype(bool)).sum()
    assert iou > 0.9

    x0, y0, x1, y1 = draft.crop
    glyphs = np.zeros((600, 600), bool)
    glyphs[y0:y1, x0:x1] = draft.text_mask
    outline = np.zeros((600, 600), np.uint8)
    cv2.ellipse(outline, CENTER, AXES, 0, 0, 360, 1, 5)
    assert not (glyphs & outline.astype(bool)).any(), "the glyph mask must never touch the outline"
    text_pixels = page < 100
    text_pixels &= interior
    assert glyphs[text_pixels].mean() > 0.99, "all text ink inside the bubble must be cleaned"


def test_fill_cleans_text_and_keeps_outline_and_tone() -> None:
    page, dets = bubble_page()
    [draft] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    labels = np.zeros_like(page)
    x0, y0, x1, y1 = draft.crop
    labels[y0:y1, x0:x1][draft.text_mask] = 1
    region = Region(
        id="P001-B01",
        type="speech_bubble",
        bbox=BBox(x0=230, y0=195, x1=370, y1=423),
        mask_label=1,
        background_color=draft.background_color,
    )
    cleaned = FillInpainter().clean(page, labels, [region])
    inside = interior_on_page(draft)
    ring = np.zeros((600, 600), np.uint8)
    cv2.ellipse(ring, CENTER, (AXES[0] - 12, AXES[1] - 12), 0, 0, 360, 1, -1)
    assert cleaned[ring.astype(bool)].min() >= 250, "no ink left inside the bubble"
    assert np.array_equal(cleaned[~inside], page[~inside]), "nothing outside the bubble changes"


def test_brush_lettering_crossing_the_outline_is_not_cleanable() -> None:
    page, dets = bubble_page(brush=True)
    [draft] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    assert draft.clean_coverage < InpaintConfig().min_coverage


def test_rectangular_box_is_a_narration_box() -> None:
    page = screentone()
    cv2.rectangle(page, (100, 100), (400, 300), 255, -1)
    cv2.rectangle(page, (100, 100), (400, 300), 0, 3)
    cv2.rectangle(page, (200, 180), (300, 200), 0, -1)
    dets = [
        RawDetection("bubble", (95, 95, 405, 305), 0.97),
        RawDetection("text_bubble", (190, 170, 310, 210), 0.95),
    ]
    [draft] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    assert draft.type == "narration_box"


def test_low_score_detections_are_ignored() -> None:
    page, _ = bubble_page()
    dets = [RawDetection("text_free", (10, 10, 60, 60), 0.2)]
    assert build_regions(page, dets, DetectionConfig(), InpaintConfig()) == []


def test_text_outside_bubbles_on_art_is_text_on_art() -> None:
    page = screentone()
    cv2.putText(page, "SFX", (220, 320), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 8)
    dets = [RawDetection("text_free", (210, 240, 400, 340), 0.9)]
    [draft] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    assert draft.type == "text_on_art"
