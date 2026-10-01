"""Stage 0: normalize any input into numbered page PNGs inside the project."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from mangatl.ingest.base import PageRef, PageSource, natural_key, parse_page_range
from mangatl.ingest.sources import open_source
from mangatl.models import Page
from mangatl.project_io import ProjectPaths

log = logging.getLogger(__name__)

# Width/height above this ratio is treated as a two-page spread.
SPREAD_ASPECT = 1.1

__all__ = [
    "PageRef",
    "PageSource",
    "ingest",
    "is_effectively_gray",
    "natural_key",
    "open_source",
    "parse_page_range",
]


def is_effectively_gray(img: Image.Image, tolerance: int = 6) -> bool:
    """True for grayscale images, including RGB scans whose channels are (almost) equal."""
    if img.mode in ("L", "1", "I;16"):
        return True
    rgb = np.asarray(img.convert("RGB").reduce(4), dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return float(np.percentile(spread, 99.5)) <= tolerance


def ingest(
    input_path: Path,
    paths: ProjectPaths,
    selection: set[int] | None = None,
    on_page: Callable[[], None] | None = None,
) -> tuple[str, list[Page]]:
    """Copy the selected pages into work/pages as PNG. Returns (source kind, pages)."""
    paths.pages.mkdir(parents=True, exist_ok=True)
    with open_source(input_path) as source:
        refs = source.pages()
        if not refs:
            raise ValueError(f"No se encontraron páginas en {input_path}")
        if selection is not None:
            missing = sorted(n for n in selection if n > len(refs))
            if missing:
                log.warning(
                    "Se ignoran páginas fuera de rango (%s); la entrada tiene %d páginas",
                    ", ".join(map(str, missing[:10])),
                    len(refs),
                )
            refs = [r for r in refs if r.number in selection]
        pages: list[Page] = []
        for ref in refs:
            img = source.load(ref)
            gray = is_effectively_gray(img)
            img = img.convert("L" if gray else "RGB")
            out = paths.pages / f"{ref.number:04d}.png"
            img.save(out, optimize=False, compress_level=3)
            pages.append(
                Page(
                    number=ref.number,
                    source_name=ref.name,
                    image_path=paths.rel(out),
                    width=img.width,
                    height=img.height,
                    grayscale=gray,
                    is_spread=img.width / img.height > SPREAD_ASPECT,
                )
            )
            if on_page:
                on_page()
        return source.kind, pages
