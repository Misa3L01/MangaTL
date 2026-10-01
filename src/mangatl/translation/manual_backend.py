"""Manual backend: translate with the claude.ai chat (free, highest quality).

`export-prompt` writes the full instructions plus the numbered chapter text, split into
parts that fit comfortably in one message. `import-translation` extracts the JSON from the
pasted answer (even surrounded by text or inside code blocks), validates it and reports
missing or unknown ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from mangatl.models import GlossaryEntry, Project, Region
from mangatl.project_io import ProjectPaths
from mangatl.translation.base import (
    apply_translation,
    grounded_proposals,
    merge_glossary,
    translatable,
)
from mangatl.translation.json_extract import extract_json_objects
from mangatl.translation.prompts import glossary_lines, region_payload, system_prompt
from mangatl.translation.schema import GlossaryProposal, RegionTranslation

HEADER = """\
=====================================================================
MangaTL · prompt para el chat de claude.ai
Serie: {series} · Capítulo: {chapter} · Parte {part} de {total}
Cómo usarlo: copia TODO este texto en el chat (adjunta las imágenes de
prompts/paginas/ si las generaste), copia la respuesta completa a un
archivo .txt y ejecuta:
  uv run mangatl import-translation <proyecto.mangatl.json> <respuesta.txt>
=====================================================================
"""

ANSWER_FORMAT = """\
# Answer format
Reply with ONE JSON object inside a ```json code block:
{
  "regions": [
    {"id": "P007-B01", "source_text_corrected": "…original text…", "speaker": "…",
     "translation": "…", "style": "normal", "confidence": 0.9}
  ],
  "chapter_summary": "3-5 sentences in Spanish summarizing the chapter (or this part)",
  "new_glossary_entries": [
    {"source": "斉藤", "target": "Saitō", "category": "character", "notes": "…"}
  ]
}
Optional per-region keys, only when they apply: reading_order, fits_capacity,
shorter_alternative, translator_note. Include every id listed below exactly once.
"""


def _region_lines(regions: list[Region]) -> list[str]:
    return [json.dumps(region_payload(r), ensure_ascii=False) for r in regions]


def _split_pages(project: Project, max_chars: int) -> list[list[Region]]:
    parts: list[list[Region]] = [[]]
    size = 0
    for page in project.pages:
        regions = [r for r in page.regions if translatable(r)]
        if not regions:
            continue
        page_size = sum(len(line) + 1 for line in _region_lines(regions))
        if parts[-1] and size + page_size > max_chars:
            parts.append([])
            size = 0
        parts[-1].extend(regions)
        size += page_size
    return [p for p in parts if p]


def export_prompt(
    project: Project,
    paths: ProjectPaths,
    glossary: list[GlossaryEntry],
    max_chars: int,
    previous_summaries: list[str] | None = None,
    images: bool = False,
) -> list[Path]:
    """Write prompts/parte-N.txt (and optional numbered page images). Returns the files."""
    paths.prompts.mkdir(parents=True, exist_ok=True)
    for old in paths.prompts.glob("parte-*.txt"):
        old.unlink()
    meta = project.meta
    parts = _split_pages(project, max_chars)
    files: list[Path] = []
    for n, regions in enumerate(parts, start=1):
        pages = sorted({int(r.id[1:4]) for r in regions})
        body = [
            HEADER.format(series=meta.series, chapter=meta.chapter, part=n, total=len(parts)),
            system_prompt(meta),
            ANSWER_FORMAT,
            f"SERIES: {meta.series} — CHAPTER {meta.chapter}",
            "GLOSSARY (mandatory renderings):",
            *(glossary_lines(glossary + project.pending_glossary) or ["(empty)"]),
            "PREVIOUS CHAPTERS:",
            *(previous_summaries or ["(none)"]),
        ]
        if len(parts) > 1:
            body.append(
                f"This is part {n} of {len(parts)} of the chapter. If earlier parts were sent in "
                "this conversation, keep names, voices and terms consistent with them."
            )
        body += [
            f"PAGES IN THIS PART: {', '.join(map(str, pages))}",
            "REGIONS (one JSON object per line, in reading order):",
            *_region_lines(regions),
        ]
        out = paths.prompts / f"parte-{n}.txt"
        out.write_text("\n".join(body) + "\n", encoding="utf-8")
        files.append(out)
    if images:
        files.extend(export_numbered_pages(project, paths))
    return files


def export_numbered_pages(project: Project, paths: ProjectPaths) -> list[Path]:
    """Page images with each region's id drawn next to it (to attach to the chat)."""
    out_dir = paths.prompts / "paginas"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for page in project.pages:
        if not any(translatable(r) for r in page.regions):
            continue
        img = Image.open(paths.abs(page.image_path)).convert("RGB")
        scale = min(1.0, 1600 / img.height)
        if scale < 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", max(14, img.height // 60))
        except OSError:
            font = ImageFont.load_default()
        for r in page.regions:
            if not translatable(r):
                continue
            box = [round(v * scale) for v in r.bbox.as_tuple()]
            draw.rectangle(box, outline=(255, 0, 0), width=3)
            label = r.id.split("-")[1]
            tb = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(tb, fill=(255, 0, 0))
            draw.text((box[0], box[1]), label, fill=(255, 255, 255), font=font)
        out = out_dir / f"{page.number:04d}.jpg"
        img.save(out, quality=85)
        files.append(out)
    return files


@dataclass
class ImportReport:
    applied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    glossary_added: int = 0
    objects_found: int = 0


def import_translation(project: Project, text: str) -> ImportReport:
    """Apply a pasted answer (one or several JSON objects) to the project."""
    report = ImportReport()
    objects = [o for o in extract_json_objects(text) if isinstance(o.get("regions"), list)]
    report.objects_found = len(objects)
    expected = {r.id: r for _, r in project.regions() if translatable(r)}
    seen: set[str] = set()
    proposals: list[GlossaryProposal] = []
    summaries: list[str] = []

    for obj in objects:
        for item in obj["regions"]:
            rid = str(item.get("id", "?")) if isinstance(item, dict) else "?"
            try:
                rt = RegionTranslation.model_validate(item)
            except ValidationError as exc:
                report.invalid.append(f"{rid}: {exc.errors()[0]['msg']}")
                continue
            region = expected.get(rt.id)
            if region is None:
                report.unknown.append(rt.id)
                continue
            if rt.id in seen:
                continue
            apply_translation(region, rt, project.meta.target_variant)
            if region.status == "needs_review" and any(
                n.startswith("sin traducción") for n in region.notes
            ):
                region.status = "auto"
            region.notes = [n for n in region.notes if not n.startswith("sin traducción")]
            seen.add(rt.id)
            report.applied.append(rt.id)
        for raw in obj.get("new_glossary_entries") or []:
            try:
                proposals.append(GlossaryProposal.model_validate(raw))
            except ValidationError:
                continue
        summary = obj.get("chapter_summary") or obj.get("block_summary")
        if isinstance(summary, str) and summary.strip():
            summaries.append(summary.strip())

    report.missing = [rid for rid in expected if rid not in seen]
    all_regions = list(expected.values())
    report.glossary_added = merge_glossary(
        project.pending_glossary, grounded_proposals(proposals, all_regions), project.meta.chapter
    )
    if summaries:
        project.block_summaries = summaries
        project.chapter_summary = " ".join(summaries)
    return report
