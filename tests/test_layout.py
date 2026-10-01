"""Typesetting: line widths follow the bubble shape, font size search, hyphenation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mangatl.typesetting.hyphenation import Hyphenator
from mangatl.typesetting.layout import TextFitter, UsableArea, estimate_capacity, wrap

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "ComicNeue-Bold.ttf"
TEXT = "¡Solo he dormido dos horas en toda la noche y mañana tengo guardia otra vez!"


def ellipse(w: int = 260, h: int = 360) -> np.ndarray:
    mask = np.zeros((h + 20, w + 20), np.uint8)
    cv2.ellipse(mask, ((w + 20) // 2, (h + 20) // 2), (w // 2, h // 2), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def rectangle(w: int, h: int) -> np.ndarray:
    return np.ones((h, w), bool)


def test_oval_gives_shorter_lines_at_top_and_bottom() -> None:
    area = UsableArea(ellipse(), margin_ratio=0.08)
    slots = area.slots(5, 40)
    assert slots is not None
    widths = [s.width for s in slots]
    assert widths[2] > widths[0] and widths[2] > widths[-1]
    centers = {round(s.center) for s in slots}
    assert len(centers) == 1, "all lines share one vertical axis"


def test_fit_places_every_word_inside_the_bubble() -> None:
    area = UsableArea(ellipse(), margin_ratio=0.08)
    fitter = TextFitter(FONT, hyphenator=Hyphenator())
    layout = fitter.fit(TEXT, area, 12, 60)
    assert layout.fits
    assert (
        " ".join(layout.lines).replace("- ", "").replace("-", "").split()
        == TEXT.replace("-", "").split()
    )
    measure = fitter.measure_for(layout.font_size)
    for line, slot in zip(layout.lines, layout.slots, strict=True):
        assert measure(line) <= slot.width + 1


def test_bigger_bubble_gets_bigger_font() -> None:
    fitter = TextFitter(FONT)
    small = fitter.fit(TEXT, UsableArea(ellipse(200, 280), 0.08), 8, 80)
    big = fitter.fit(TEXT, UsableArea(ellipse(400, 560), 0.08), 8, 80)
    assert small.fits and big.fits
    assert big.font_size > small.font_size


def test_does_not_fit_below_minimum_size() -> None:
    fitter = TextFitter(FONT)
    layout = fitter.fit(TEXT * 3, UsableArea(ellipse(80, 100), 0.08), 20, 30)
    assert not layout.fits


def test_no_hyphenation_when_not_needed() -> None:
    fitter = TextFitter(FONT, hyphenator=Hyphenator())
    layout = fitter.fit(
        "Así es la realidad del residente.", UsableArea(rectangle(500, 300), 0.05), 10, 40
    )
    assert layout.fits
    assert not any(line.endswith("-") for line in layout.lines)


def test_hyphenates_long_words_in_narrow_bubbles() -> None:
    fitter = TextFitter(FONT, hyphenator=Hyphenator(), hyphen_gain=1.0)
    area = UsableArea(rectangle(120, 400), 0.02)
    layout = fitter.fit("la otorrinolaringología", area, 18, 18)
    assert layout.fits
    assert any(line.endswith("-") for line in layout.lines)


@pytest.mark.parametrize("word", ["Saitō", "Universidad", "hola", "1234567"])
def test_hyphenator_protects_names_short_words_and_numbers(word: str) -> None:
    assert Hyphenator().splits(word) == []


def test_hyphenator_respects_protected_glossary_words() -> None:
    assert Hyphenator().splits("residente")
    assert Hyphenator(protected={"residente"}).splits("residente") == []


def test_hyphenator_keeps_punctuation_on_the_right_side() -> None:
    splits = Hyphenator().splits("¡transfusión!")
    assert splits
    head, tail = splits[0]
    assert head.startswith("¡") and head.endswith("-") and tail.endswith("!")


def test_wrap_returns_none_when_a_word_cannot_fit() -> None:
    assert wrap(["larguísima"], [10], lambda s: len(s) * 10.0, None) is None


def test_capacity_grows_with_area() -> None:
    small = estimate_capacity(UsableArea(ellipse(200, 280), 0.08), 22, 1.12)
    big = estimate_capacity(UsableArea(ellipse(400, 560), 0.08), 22, 1.12)
    assert 0 < small < big
