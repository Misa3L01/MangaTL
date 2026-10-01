from __future__ import annotations

from pathlib import Path

import pytest

from mangatl.config import Settings
from mangatl.models import BBox, ChapterMeta, Page, Project, Region

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default settings rooted in a temporary directory (no config.toml involved)."""
    return Settings(root=tmp_path)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


def make_region(page: int, n: int, text: str, **kw: object) -> Region:
    return Region(
        id=f"P{page:03d}-B{n:02d}",
        type=kw.pop("type", "speech_bubble"),  # type: ignore[arg-type]
        bbox=BBox(x0=10, y0=10, x1=100, y1=200),
        reading_order=n,
        ocr_text=text,
        ocr_confidence=0.99,
        capacity_chars=60,
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def small_project() -> Project:
    """Three pages of Japanese text, no images (translation-level tests)."""
    texts = {
        1: ["研修医というのは要するに見習いだ", "もっと術野を広げろ斉藤", "はい！"],
        2: ["今夜は初日という事で特別にもう一人当直がいる", "ではよろしく"],
        3: ["血圧が下がっています！！"],
    }
    pages = [
        Page(
            number=n,
            source_name=f"{n:03d}",
            image_path=f"work/pages/{n:04d}.png",
            width=1000,
            height=1500,
            regions=[make_region(n, i, t) for i, t in enumerate(lines, start=1)],
        )
        for n, lines in texts.items()
    ]
    return Project(meta=ChapterMeta(series="Black Jack ni Yoroshiku", chapter="1"), pages=pages)
