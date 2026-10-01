"""Per-series memory: series/<serie>/glossary.json and summaries.json.

- The translator always receives the APPROVED glossary entries (mandatory renderings) and the
  summaries of the last chapters.
- Entries proposed by the model are stored as "pending" until the user approves them
  (`mangatl glossary approve`), so a bad guess never propagates to later chapters on its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from mangatl.config import Settings
from mangatl.models import GlossaryEntry, Project
from mangatl.project_io import slugify

log = logging.getLogger(__name__)
_ENTRIES = TypeAdapter(list[GlossaryEntry])


def series_dir(settings: Settings, series: str) -> Path:
    return settings.resolve(settings.paths.series_dir) / slugify(series)


def _glossary_file(settings: Settings, series: str) -> Path:
    return series_dir(settings, series) / "glossary.json"


def _summaries_file(settings: Settings, series: str) -> Path:
    return series_dir(settings, series) / "summaries.json"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_glossary(settings: Settings, series: str) -> list[GlossaryEntry]:
    """All entries of the series (approved and pending)."""
    path = _glossary_file(settings, series)
    if not path.is_file():
        return []
    try:
        return _ENTRIES.validate_json(path.read_bytes())
    except ValidationError as exc:
        log.warning("Glosario inválido en %s: %s", path, exc.errors()[0]["msg"])
        return []


def write_glossary(settings: Settings, series: str, entries: list[GlossaryEntry]) -> Path:
    path = _glossary_file(settings, series)
    _write_json(path, [e.model_dump(mode="json") for e in entries])
    return path


def load_glossary(settings: Settings, series: str) -> list[GlossaryEntry]:
    """Approved entries: the ones sent to the translator as mandatory."""
    return [e for e in read_glossary(settings, series) if e.status == "approved"]


def add_pending(settings: Settings, series: str, proposals: list[GlossaryEntry]) -> int:
    """Store the model's proposals as pending (terms already in the glossary are skipped)."""
    entries = read_glossary(settings, series)
    known = {e.source for e in entries}
    added = 0
    for p in proposals:
        if p.source in known:
            continue
        entries.append(p.model_copy(update={"status": "pending"}))
        known.add(p.source)
        added += 1
    if added:
        write_glossary(settings, series, entries)
    return added


def approve(settings: Settings, series: str, sources: list[str] | None) -> list[str]:
    """Approve the given terms (None = every pending entry). Returns the approved terms."""
    entries = read_glossary(settings, series)
    approved = []
    for e in entries:
        if e.status == "pending" and (sources is None or e.source in sources):
            e.status = "approved"
            approved.append(e.source)
    if approved:
        write_glossary(settings, series, entries)
    return approved


def upsert(
    settings: Settings,
    series: str,
    source: str,
    target: str | None = None,
    category: str | None = None,
    notes: str | None = None,
) -> GlossaryEntry:
    """Create or edit an entry. Anything the user writes is approved."""
    entries = read_glossary(settings, series)
    entry = next((e for e in entries if e.source == source), None)
    if entry is None:
        if target is None:
            raise ValueError(f"El término «{source}» no existe: indica también --target")
        entry = GlossaryEntry(source=source, target=target)
        entries.append(entry)
    if target is not None:
        entry.target = target
    if category is not None:
        entry.category = category  # type: ignore[assignment]
    if notes is not None:
        entry.notes = notes or None
    entry.status = "approved"
    write_glossary(settings, series, entries)
    return entry


def remove(settings: Settings, series: str, source: str) -> bool:
    entries = read_glossary(settings, series)
    kept = [e for e in entries if e.source != source]
    if len(kept) == len(entries):
        return False
    write_glossary(settings, series, kept)
    return True


def save_summary(settings: Settings, series: str, chapter: str, summary: str) -> None:
    path = _summaries_file(settings, series)
    data: dict[str, str] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("summaries.json inválido en %s: se reemplaza", path)
    data[chapter] = summary
    _write_json(path, data)


def remember_chapter(settings: Settings, project: Project) -> int:
    """After translating: store the chapter summary and the model's glossary proposals
    (as pending) in the series memory. Returns the number of new pending entries."""
    series, chapter = project.meta.series, project.meta.chapter
    if project.chapter_summary:
        save_summary(settings, series, chapter, project.chapter_summary)
    return add_pending(settings, series, project.pending_glossary)


def _chapter_key(chapter: str) -> tuple[float, str]:
    try:
        return float(chapter), chapter
    except ValueError:
        return float("inf"), chapter


def load_previous_summaries(
    settings: Settings, series: str, last: int = 3, before: str | None = None
) -> list[str]:
    """Summaries of the last `last` chapters (optionally only those before `before`)."""
    path = _summaries_file(settings, series)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    items = sorted(data.items(), key=lambda kv: _chapter_key(kv[0]))
    if before is not None:
        items = [kv for kv in items if _chapter_key(kv[0]) < _chapter_key(before)]
    return [f"Capítulo {k}: {v}" for k, v in items[-last:]]
