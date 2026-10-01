"""Review report: a self-contained HTML page listing every region marked needs_review."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

from mangatl.models import STAGES, Page, Project, Region
from mangatl.project_io import ProjectPaths

THUMB = 360
STAGE_LABELS = {
    "ingest": "Ingesta",
    "detect": "Detección",
    "ocr": "OCR",
    "translate": "Traducción",
    "inpaint": "Limpieza",
    "typeset": "Rotulado",
    "export": "Exportación",
}

CSS = """
:root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e0da; --warn:#b26a00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151517; --card:#1f1f22; --ink:#ececef; --muted:#9a9aa2;
          --line:#333338; --warn:#f0a53a; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink);
       font:15px/1.5 system-ui, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 20px; }
table.times { border-collapse: collapse; margin: 0 0 28px; }
table.times td, table.times th {
  padding: 4px 12px; border-bottom: 1px solid var(--line); text-align: left;
}
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px; margin: 0 0 16px; display: grid; gap: 16px;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr); }
@media (max-width: 800px) { .card { grid-template-columns: 1fr; } }
.card img { max-width: 100%; border-radius: 6px; border: 1px solid var(--line); background: #fff; }
.label { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.id { font-weight: 600; font-size: 1.05rem; }
.reason { color: var(--warn); }
.ja { font-size: 1.1rem; }
dl { margin: 0; } dt { margin-top: 8px; } dd { margin: 0; }
"""


def _thumb(path: Path, box: tuple[int, int, int, int]) -> str:
    with Image.open(path) as img:
        crop = img.convert("RGB").crop(box)
    crop.thumbnail((THUMB, THUMB * 2))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _box(page: Page, region: Region, pad: int = 24) -> tuple[int, int, int, int]:
    b = (region.bubble_bbox or region.bbox).expand(pad, page.width, page.height)
    return b.as_tuple()


def _card(paths: ProjectPaths, page: Page, region: Region) -> str:
    box = _box(page, region)
    original = _thumb(paths.abs(page.image_path), box)
    result = (
        f'<img alt="resultado" src="{_thumb(paths.abs(page.rendered_path), box)}">'
        if page.rendered_path
        else "<p class='label'>sin rotular todavía</p>"
    )
    conf = f"{region.ocr_confidence:.2f}" if region.ocr_confidence is not None else "?"
    rows = [
        (
            "Texto original (OCR, confianza " + conf + ")",
            f"<span class='ja'>{html.escape(region.text_for_translation or '—')}</span>",
        ),
        ("Traducción", html.escape(region.translation or "—")),
    ]
    if region.shorter_alternative:
        rows.append(("Versión corta", html.escape(region.shorter_alternative)))
    if region.translator_note:
        rows.append(("Nota del traductor", html.escape(region.translator_note)))
    fields = "".join(f"<dt class='label'>{k}</dt><dd>{v}</dd>" for k, v in rows)
    reason = html.escape("; ".join(region.notes) or "—")
    return (
        "<section class='card'>"
        f"<div><div class='label'>Original</div><img alt='original' src='{original}'></div>"
        f"<div><div class='label'>Resultado</div>{result}</div>"
        f"<div><div class='id'>{region.id}</div>"
        f"<div class='label'>{region.type} · página {page.number}</div>"
        f"<p class='reason'>{reason}</p><dl>{fields}</dl></div>"
        "</section>"
    )


def review_regions(project: Project) -> list[tuple[Page, Region]]:
    return [(p, r) for p, r in project.regions() if r.status == "needs_review"]


def write_review_report(project: Project, paths: ProjectPaths) -> Path:
    """Write <out>/revision.html and return its path."""
    items = review_regions(project)
    meta = project.meta
    times = "".join(
        f"<tr><td>{STAGE_LABELS[s]}</td><td>{project.stages[s].seconds:.1f} s</td></tr>"
        for s in STAGES
        if s in project.stages
    )
    total = sum(r.seconds for r in project.stages.values())
    cards = "".join(_card(paths, p, r) for p, r in items) or "<p>No hay regiones para revisar.</p>"
    doc = (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Revisión · {html.escape(meta.series)} {html.escape(meta.chapter)}</title>"
        f"<style>{CSS}</style></head><body><main>"
        f"<h1>Revisión · {html.escape(meta.series)} · capítulo {html.escape(meta.chapter)}</h1>"
        f"<p class='sub'>{len(items)} región(es) a revisar de "
        f"{sum(1 for _ in project.regions())} · "
        f"{len(project.pages)} páginas · traductor {html.escape(meta.translator_backend)}"
        f"{' (' + html.escape(meta.translator_model) + ')' if meta.translator_model else ''}</p>"
        f"<table class='times'><tr><th>Etapa</th><th>Tiempo</th></tr>{times}"
        f"<tr><th>Total</th><th>{total:.1f} s</th></tr></table>"
        f"{cards}</main></body></html>"
    )
    out = paths.root / "revision.html"
    out.write_text(doc, encoding="utf-8")
    return out
