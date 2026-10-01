from __future__ import annotations

from pathlib import Path

import pytest

from mangatl.config import Settings
from mangatl.fonts import OPTIONAL_CHARS, missing_glyphs

STYLES = ["normal", "shout", "whisper", "thought", "narration", "sfx"]


@pytest.mark.parametrize("style", STYLES)
def test_bundled_fonts_cover_spanish(style: str, project_root: Path) -> None:
    settings = Settings(root=project_root)
    font = settings.resolve(getattr(settings.typesetting.fonts, style))
    assert font.is_file(), f"Falta la fuente {font}"
    assert missing_glyphs(font) == []
    assert missing_glyphs(font, OPTIONAL_CHARS) == []


def test_missing_glyphs_reports_uncovered_chars(project_root: Path) -> None:
    font = project_root / "assets" / "fonts" / "ComicNeue-Regular.ttf"
    assert missing_glyphs(font, "a漢ñ") == ["漢"]


def test_bundled_fonts_ship_their_licenses(project_root: Path) -> None:
    fonts = project_root / "assets" / "fonts"
    assert (fonts / "OFL-ComicNeue.txt").is_file()
    assert (fonts / "OFL-Bangers.txt").is_file()
