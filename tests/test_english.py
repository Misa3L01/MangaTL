"""English source: line joining, language detection and English OCR on a rendered sample."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont

from mangatl.ocr.english import join_lines


def test_join_lines_orders_and_glues_hyphenated_words() -> None:
    lines = [
        (40.0, 10.0, "TO THE CONVIC-", 0.9),
        (10.0, 12.0, "SHE CLUNG", 1.0),
        (70.0, 11.0, "TION THAT...", 0.8),
    ]
    text, conf = join_lines(lines)
    assert text == "SHE CLUNG TO THE CONVICTION THAT..."
    assert conf == pytest.approx(0.9)


def test_join_lines_keeps_real_hyphens_and_empty_input() -> None:
    assert join_lines([(0.0, 0.0, "WELL-", 1.0), (20.0, 0.0, "...OK", 1.0)])[0] == "WELL- ...OK"
    assert join_lines([]) == ("", 0.0)


def _lettering(text: str) -> Image.Image:
    font = ImageFont.truetype("arial.ttf", 34) if _has_arial() else ImageFont.load_default()
    img = Image.new("RGB", (round(font.getlength(text)) + 24, 70), "white")
    ImageDraw.Draw(img).text((12, 14), text, font=font, fill="black")
    return img


def _has_arial() -> bool:
    try:
        ImageFont.truetype("arial.ttf", 10)
    except OSError:
        return False
    return True


def test_ppocr_reads_english_lettering() -> None:
    pytest.importorskip("rapidocr")
    from mangatl.ocr.english import PaddleRapidOcrEngine

    [result] = PaddleRapidOcrEngine().read(
        [_lettering("WHY DOES EVERYONE THINK US DOCTORS ARE RICH?")]
    )
    assert result.text.replace(" ", "") == "WHYDOESEVERYONETHINKUSDOCTORSARERICH?"
    assert result.confidence > 0.8
