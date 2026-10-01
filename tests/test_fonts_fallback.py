from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mangatl.config import DetectionConfig, InpaintConfig
from mangatl.detection.base import RawDetection
from mangatl.detection.regions import build_regions
from mangatl.fonts import fit_to_font

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "ComicNeue-Bold.ttf"


def test_missing_macrons_fall_back_to_base_letters() -> None:
    text, replaced = fit_to_font("Saitō-kun y Gōda, ¿qué pasó?", FONT)
    assert text == "Saito-kun y Goda, ¿qué pasó?"
    assert set(replaced) == {"ō"}


def test_characters_without_fallback_are_dropped() -> None:
    text, replaced = fit_to_font("Hola 漢", FONT)
    assert text == "Hola " and replaced == ["漢"]


def test_ink_outside_the_detected_box_is_cleaned_in_closed_bubbles() -> None:
    page = np.full((600, 600), 235, np.uint8)
    page[::4, ::4] = 60
    cv2.rectangle(page, (100, 100), (500, 500), 255, -1)
    cv2.rectangle(page, (100, 100), (500, 500), 0, 3)
    cv2.rectangle(page, (280, 250), (320, 350), 0, -1)  # detected text
    cv2.rectangle(page, (130, 130), (160, 160), 0, -1)  # big emphasis character, outside the box
    dets = [
        RawDetection("bubble", (95, 95, 505, 505), 0.97),
        RawDetection("text_bubble", (270, 240, 330, 360), 0.95),
    ]
    [draft] = build_regions(page, dets, DetectionConfig(), InpaintConfig())
    x0, y0, _, _ = draft.crop
    mask = np.zeros((600, 600), bool)
    mask[y0 : y0 + draft.text_mask.shape[0], x0 : x0 + draft.text_mask.shape[1]] = draft.text_mask
    assert mask[135:155, 135:155].all(), "stray character inside the bubble is cleaned too"
    assert not mask[98:103, 98:502].any(), "outline untouched"
