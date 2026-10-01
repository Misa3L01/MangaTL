"""Page sources: every input kind is normalized into an ordered list of page images."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"})


@dataclass(frozen=True)
class PageRef:
    number: int  # 1-based position in the source
    name: str  # original name without extension, used when exporting
    key: str  # source-specific locator (file path, archive member, PDF page index)


def natural_key(text: str) -> list[object]:
    """Sort key that orders 'p2' before 'p10' (case-insensitive)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_page_range(spec: str | None) -> set[int] | None:
    """'3-7,10' -> {3, 4, 5, 6, 7, 10}. None or '' means all pages."""
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start < 1 or end < start:
                raise ValueError(f"Rango de páginas inválido: {part!r}")
            pages.update(range(start, end + 1))
        else:
            number = int(part)
            if number < 1:
                raise ValueError(f"Número de página inválido: {part!r}")
            pages.add(number)
    return pages


class PageSource(ABC):
    kind: str = "folder"

    def __init__(self, path: Path) -> None:
        self.path = path

    @abstractmethod
    def pages(self) -> list[PageRef]:
        """All pages in reading order."""

    @abstractmethod
    def load(self, ref: PageRef) -> Image.Image:
        """Decode one page at its native resolution."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def __enter__(self) -> PageSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
