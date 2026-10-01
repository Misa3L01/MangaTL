"""Translator interface and the chapter-level driver shared by automatic backends.

Never translate balloon by balloon: the chapter is split into blocks of a few pages and
every request carries the glossary, the story so far and the last translated lines.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from mangatl.models import GlossaryEntry, Page, Project, Region
from mangatl.translation.alignment import normalize
from mangatl.translation.postprocess import clean_translation, vosotros_forms
from mangatl.translation.prompts import BlockContext, pivot_english_prompt, system_prompt
from mangatl.translation.schema import BlockTranslation, GlossaryProposal, RegionTranslation

log = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.5
STORY_BLOCKS = 4  # block summaries kept in the "story so far"


class TranslationError(RuntimeError):
    pass


@dataclass
class TranslationStats:
    blocks: int = 0
    regions: int = 0
    missing: list[str] = field(default_factory=list)
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    generation_seconds: float = 0.0
    context_peak: int = 0  # largest prompt + output of a single request, in tokens

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.generation_seconds if self.generation_seconds else 0.0


class Translator(ABC):
    name: str = "base"
    model: str | None = None

    def __init__(self) -> None:
        self.stats = TranslationStats()

    @abstractmethod
    def translate_block(self, system: str, ctx: BlockContext) -> BlockTranslation:
        """Translate the regions of `ctx`; must return one entry per region id when possible."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (e.g. unload the model from VRAM)."""


def translatable(region: Region) -> bool:
    return bool(region.ocr_text.strip()) and region.status != "edited"


def plan_blocks(pages: list[Page], pages_per_block: int) -> list[list[Page]]:
    with_text = [p for p in pages if any(translatable(r) for r in p.regions)]
    return [with_text[i : i + pages_per_block] for i in range(0, len(with_text), pages_per_block)]


def merge_glossary(
    existing: list[GlossaryEntry], proposals: list[GlossaryProposal], chapter: str
) -> int:
    known = {e.source.strip() for e in existing}
    added = 0
    for p in proposals:
        if not p.source.strip() or not p.target.strip() or p.source.strip() in known:
            continue
        existing.append(
            GlossaryEntry(
                source=p.source.strip(),
                target=p.target.strip(),
                category=p.category,
                notes=p.notes,
                status="pending",
                chapter=chapter,
            )
        )
        known.add(p.source.strip())
        added += 1
    return added


def grounded_proposals(
    proposals: list[GlossaryProposal], regions: list[Region]
) -> list[GlossaryProposal]:
    """Keep glossary proposals whose source term really appears in the translated text."""
    corpus = normalize("".join(r.text_for_translation for r in regions))
    kept = []
    for p in proposals:
        key = normalize(p.source)
        # Drop honorific suffixes so '牛田さん' also matches '牛田'.
        stem = re.sub(r"(さん|くん|君|ちゃん|様|さま|先生|先輩)$", "", key) or key
        if stem and stem in corpus:
            kept.append(p)
        else:
            log.debug("Se descarta la entrada de glosario sin respaldo en el texto: %s", p.source)
    return kept


def apply_translation(region: Region, rt: RegionTranslation, variant: str = "es-419") -> None:
    region.translation = clean_translation(rt.translation)
    if rt.shorter_alternative:
        rt = rt.model_copy(
            update={"shorter_alternative": clean_translation(rt.shorter_alternative)}
        )
    corrected = (rt.source_text_corrected or "").strip()
    region.source_text_corrected = corrected if corrected and corrected != region.ocr_text else None
    region.speaker = rt.speaker or "desconocido"
    style = rt.style
    if region.type == "narration_box" and style == "normal":
        style = "narration"
    region.style = style
    region.shorter_alternative = rt.shorter_alternative or None
    region.translator_note = rt.translator_note or None
    region.translation_confidence = rt.confidence
    region.suggested_reading_order = (
        rt.reading_order if rt.reading_order and rt.reading_order != region.reading_order else None
    )
    region.notes = [
        n for n in region.notes if not n.startswith(("traducción dudosa", "sin traducción"))
    ]
    region.notes = [n for n in region.notes if not n.startswith("usa «vosotros»")]
    if rt.confidence < LOW_CONFIDENCE and region.status == "auto":
        region.status = "needs_review"
        region.notes.append(f"traducción dudosa (confianza {rt.confidence:.2f})")
    wrong = vosotros_forms(region.translation) if variant != "es-ES" else []
    if wrong:
        region.notes.append(f"usa «vosotros» ({', '.join(wrong)}), incorrecto en {variant}")
        if region.status == "auto":
            region.status = "needs_review"


def translate_chapter(
    project: Project,
    translator: Translator,
    glossary: list[GlossaryEntry],
    pages_per_block: int,
    previous_lines: int,
    on_block: Callable[[int, int], None] | None = None,
    pivot: bool = False,
    previous_chapters: list[str] | None = None,
) -> TranslationStats:
    """Translate every page of `project` in context-aware blocks (mutates the project).

    `pivot`: two passes per block, Japanese -> English draft -> Spanish (the second pass also
    sees the Japanese original). Roughly doubles the translation time.
    """
    meta = project.meta
    system = system_prompt(meta, pivot=pivot)
    system_en = pivot_english_prompt() if pivot else ""
    blocks = plan_blocks(project.pages, pages_per_block)
    stats = translator.stats
    story: list[str] = []
    history: list[Region] = []
    project.block_summaries = []

    for index, block in enumerate(blocks, start=1):
        regions = [r for p in block for r in p.regions if translatable(r)]
        ctx = BlockContext(
            series=meta.series,
            chapter=meta.chapter,
            pages=[p.number for p in block],
            regions=regions,
            glossary=glossary + project.pending_glossary,
            story_so_far=" ".join(story[-STORY_BLOCKS:]),
            previous_lines=history[-previous_lines:] if previous_lines else [],
            previous_chapters=list(previous_chapters or []) if index == 1 else [],
        )
        proposals: list[GlossaryProposal] = []
        if pivot:
            draft = translator.translate_block(system_en, ctx)
            proposals.extend(draft.new_glossary_entries)
            english = {rt.id: rt.translation for rt in draft.regions}
            proxies = [
                r.model_copy(
                    update={
                        "ocr_text": english.get(r.id, r.text_for_translation),
                        "source_text_corrected": None,
                    }
                )
                for r in regions
            ]
            ctx = replace(
                ctx, regions=proxies, extra={r.id: {"ja": r.text_for_translation} for r in regions}
            )
        result = translator.translate_block(system, ctx)
        proposals.extend(result.new_glossary_entries)
        by_id: dict[str, RegionTranslation] = {}
        for rt in result.regions:
            by_id.setdefault(rt.id, rt)
        for region in regions:
            rt = by_id.get(region.id)
            if rt is None:
                stats.missing.append(region.id)
                region.notes = [n for n in region.notes if not n.startswith("sin traducción")]
                region.notes.append("sin traducción: el modelo no devolvió esta región")
                if region.status == "auto":
                    region.status = "needs_review"
                continue
            if pivot:
                # The echo refers to the English draft: never store it as corrected Japanese.
                rt = rt.model_copy(update={"source_text_corrected": None})
            apply_translation(region, rt, meta.target_variant)
            stats.regions += 1
        merge_glossary(
            project.pending_glossary,
            grounded_proposals(proposals, regions),
            meta.chapter,
        )
        if result.block_summary:
            story.append(result.block_summary.strip())
        history.extend(r for r in regions if r.translation)
        stats.blocks += 1
        if on_block:
            on_block(index, len(blocks))

    project.block_summaries = story
    project.chapter_summary = " ".join(story) or None
    return stats
