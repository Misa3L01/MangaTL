"""Project data model, persisted as `<chapter>.mangatl.json`.

Every pipeline stage reads and writes this structure, so any stage can be re-run, edited or
resumed without repeating the expensive ones.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

RegionType = Literal["speech_bubble", "narration_box", "text_on_art", "sfx"]
RegionStatus = Literal["auto", "edited", "needs_review", "skipped"]
TextStyle = Literal["normal", "shout", "whisper", "thought", "narration", "sfx"]
SourceLang = Literal["ja", "en", "auto"]
ReadingDirection = Literal["rtl", "ltr"]
StageName = Literal["ingest", "detect", "ocr", "translate", "inpaint", "typeset", "export"]

STAGES: tuple[StageName, ...] = (
    "ingest",
    "detect",
    "ocr",
    "translate",
    "inpaint",
    "typeset",
    "export",
)
# Region types the Phase 1 renderer can clean and letter (uniform background).
RENDERABLE_TYPES: frozenset[str] = frozenset({"speech_bubble", "narration_box"})


def utcnow() -> datetime:
    return datetime.now(UTC)


class BBox(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def expand(self, px: int, width: int, height: int) -> BBox:
        return BBox(
            x0=max(0, self.x0 - px),
            y0=max(0, self.y0 - px),
            x1=min(width, self.x1 + px),
            y1=min(height, self.y1 + px),
        )

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersection_area(self, other: BBox) -> int:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(0, w) * max(0, h)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


class Region(BaseModel):
    id: str  # e.g. "P007-B03": page number + reading order
    type: RegionType
    bbox: BBox  # union of the text blocks
    text_boxes: list[BBox] = Field(default_factory=list)  # individual text blocks (OCR crops)
    bubble_bbox: BBox | None = None  # window holding the container interior (layout/cleaning)
    polygon: list[tuple[int, int]] = Field(default_factory=list)  # simplified interior contour
    mask_label: int = 0  # value of this region in the page's bubble label mask
    detection_score: float = 0.0
    reading_order: int = 0
    panel: int | None = None
    position: str = ""  # coarse position on the page, e.g. "top-right"
    vertical: bool = False
    capacity_chars: int | None = None
    background_uniform: bool = True
    background_color: tuple[int, int, int] = (255, 255, 255)  # RGB used to clean the region
    # False when the fill cannot reach most of the ink (lettering crossing the outline).
    cleanable: bool = True
    # How the original text is removed: flat fill, LaMa inpainting, or not at all.
    clean_method: Literal["fill", "lama", "none"] = "fill"

    ocr_text: str = ""
    ocr_confidence: float | None = None

    source_text_corrected: str | None = None
    speaker: str | None = None
    translation: str | None = None
    shorter_alternative: str | None = None
    translator_note: str | None = None
    translation_confidence: float | None = None
    suggested_reading_order: int | None = None

    style: TextStyle = "normal"
    font: str | None = None  # overrides the style font
    font_size: int | None = None  # last size used by the typesetter
    font_size_fixed: bool = False  # True when the user chose `font_size` in the editor
    # Text box moved/resized by the user in the editor: lettering goes in this rectangle
    # instead of following the bubble shape (cleaning is not affected).
    text_box_override: BBox | None = None
    fits: bool | None = None
    used_shorter: bool = False
    status: RegionStatus = "auto"
    notes: list[str] = Field(default_factory=list)

    @property
    def text_for_translation(self) -> str:
        return self.source_text_corrected or self.ocr_text


class Page(BaseModel):
    number: int  # 1-based position in the source (PDF page, archive order...)
    source_name: str  # original file name (without extension) used for exports
    image_path: str  # relative to the project directory
    width: int
    height: int
    grayscale: bool = False
    is_spread: bool = False
    bubble_mask_path: str | None = None
    text_mask_path: str | None = None
    clean_path: str | None = None
    rendered_path: str | None = None
    regions: list[Region] = Field(default_factory=list)


class GlossaryEntry(BaseModel):
    source: str  # term as written in the original
    target: str  # fixed translation / romanization
    category: Literal["character", "place", "technique", "term", "other"] = "other"
    notes: str | None = None  # speech style, how they address others, etc.
    status: Literal["approved", "pending"] = "pending"
    chapter: str | None = None


class StageRecord(BaseModel):
    status: Literal["done", "partial", "failed"] = "done"
    seconds: float = 0.0
    finished_at: datetime = Field(default_factory=utcnow)
    details: dict[str, object] = Field(default_factory=dict)


class ChapterMeta(BaseModel):
    series: str
    chapter: str
    source_lang: SourceLang = "ja"
    # Manga keeps right-to-left pages even in official English editions (Yen Press, etc.).
    reading_direction: ReadingDirection = "rtl"
    target_variant: str = "es-419"
    honorifics: str = "keep"
    sfx_mode: str = "annotate"
    translator_backend: str = "ollama"
    translator_model: str | None = None
    pipeline_version: str = ""
    input_path: str = ""
    input_kind: Literal["folder", "archive", "pdf", "image"] = "folder"
    created_at: datetime = Field(default_factory=utcnow)


class Project(BaseModel):
    format_version: int = 1
    meta: ChapterMeta
    pages: list[Page] = Field(default_factory=list)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    chapter_summary: str | None = None
    block_summaries: list[str] = Field(default_factory=list)
    pending_glossary: list[GlossaryEntry] = Field(default_factory=list)

    def regions(self) -> list[tuple[Page, Region]]:
        return [(page, region) for page in self.pages for region in page.regions]

    def find_region(self, region_id: str) -> Region | None:
        for _, region in self.regions():
            if region.id == region_id:
                return region
        return None
