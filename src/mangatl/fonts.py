"""Font validation: every configured font must render Spanish text."""

from __future__ import annotations

import string
import unicodedata
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont

# Characters a Spanish translation cannot do without.
REQUIRED_CHARS = string.ascii_letters + string.digits + ".,;:!?¡¿-'\"()" + "áéíóúüñÁÉÍÓÚÜÑ"
# Nice to have; missing ones are replaced at render time (… -> ..., — -> -, « » -> " ").
OPTIONAL_CHARS = "…—«»“”"


@lru_cache(maxsize=32)
def font_charset(font_path: str) -> frozenset[int]:
    with TTFont(font_path, lazy=True) as font:
        return frozenset((font.getBestCmap() or {}).keys())


def fit_to_font(text: str, font_path: Path) -> tuple[str, list[str]]:
    """Replace characters the font cannot draw: 'Saitō' -> 'Saito' (base letter), else drop.

    Returns the new text and the characters that were replaced.
    """
    charset = font_charset(str(font_path))
    out: list[str] = []
    replaced: list[str] = []
    for ch in text:
        if ch.isspace() or ord(ch) in charset:
            out.append(ch)
            continue
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        if base and all(ord(c) in charset for c in base):
            out.append(base)
        replaced.append(ch)
    return "".join(out), replaced


def missing_glyphs(font_path: Path, chars: str = REQUIRED_CHARS) -> list[str]:
    with TTFont(font_path, lazy=True) as font:
        cmap = font.getBestCmap() or {}
    return [c for c in dict.fromkeys(chars) if ord(c) not in cmap]


def font_family_name(font_path: Path) -> str:
    with TTFont(font_path, lazy=True) as font:
        name = font["name"]
        full = name.getDebugName(4) or name.getDebugName(1)
    return full or font_path.stem
