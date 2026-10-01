"""Structured translator output (validated with Pydantic, enforced in Ollama via `format`)."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from mangatl.models import TextStyle

GlossaryCategory = Literal["character", "place", "technique", "term", "other"]


class RegionTranslation(BaseModel):
    id: str
    # Echo of the original text (fixed if the OCR was wrong). Local models must always fill it:
    # it anchors each translation to its region and lets us detect shifted answers.
    source_text_corrected: str | None = Field(
        None, description="Original text of this region, copied; fix evident OCR mistakes"
    )
    speaker: str = Field("desconocido", description="Character speaking, or 'desconocido'")
    translation: str = Field(description="Final Spanish text for the balloon")
    style: TextStyle = "normal"
    reading_order: int | None = Field(
        None, description="Corrected position on the page, only if the given order is wrong"
    )
    fits_capacity: bool = True
    shorter_alternative: str | None = Field(
        None, description="Shorter version, only when the translation exceeds max_chars"
    )
    translator_note: str | None = Field(
        None, description="Brief note on wordplay or cultural references, if important"
    )
    confidence: float = Field(0.8, ge=0.0, le=1.0)

    @field_validator("translation")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> Any:
        if isinstance(value, int | float):
            return min(1.0, max(0.0, float(value)))
        return value


class GlossaryProposal(BaseModel):
    source: str = Field(description="Term as written in the original")
    target: str = Field(description="Spanish rendering / Hepburn romanization to keep fixed")
    category: GlossaryCategory = "other"
    notes: str | None = Field(None, description="Speech style, how they address others, etc.")


class BlockTranslation(BaseModel):
    regions: list[RegionTranslation]
    block_summary: str = Field(description="2-3 sentences in Spanish: what happens in these pages")
    new_glossary_entries: list[GlossaryProposal] = Field(default_factory=list)


class ChapterTranslation(BaseModel):
    """Shape requested from the manual backend (whole chapter or one part of it)."""

    regions: list[RegionTranslation]
    chapter_summary: str | None = None
    new_glossary_entries: list[GlossaryProposal] = Field(default_factory=list)


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema with every $ref inlined (grammar converters handle it more reliably)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items() if k not in ("title", "description")}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
