"""Stages 5-7: clean the original text, letter the translation, export PNG/CBZ."""

from __future__ import annotations

import logging
import shutil
import statistics
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from mangatl.config import Settings
from mangatl.fonts import fit_to_font
from mangatl.inpainting.base import Inpainter
from mangatl.inpainting.fill import FillInpainter
from mangatl.models import RENDERABLE_TYPES, Page, Project, Region
from mangatl.project_io import ProjectPaths
from mangatl.translation.postprocess import clean_translation
from mangatl.typesetting.hyphenation import Hyphenator
from mangatl.typesetting.layout import (
    Layout,
    Slot,
    TextFitter,
    UsableArea,
    estimate_capacity,
    load_font,
)

log = logging.getLogger(__name__)

Progress = Callable[[], None]


def normalize_text(text: str) -> str:
    """Same clean-up as at translation time (idempotent): also fixes hand-edited text."""
    return clean_translation(text)


def _has_text(region: Region) -> bool:
    return bool(region.translation and region.translation.strip())


def is_annotation(region: Region, sfx_mode: str = "annotate") -> bool:
    """Sound effect drawn on the artwork, annotated next to the original (not replaced)."""
    return (
        sfx_mode == "annotate"
        and region.style == "sfx"
        and region.type in ("text_on_art", "sfx")
        and region.status != "skipped"
        and _has_text(region)
    )


ART_MIN_OCR = 0.5


def unreliable_art_text(region: Region) -> bool:
    """Text outside speech bubbles read with low OCR confidence: usually signage or stylized
    lettering (signs on a light panel are classified as caption boxes). Translating OCR
    garbage and erasing the drawing does more harm than keeping the original."""
    return (
        region.type in ("text_on_art", "narration_box")
        and (region.ocr_confidence or 0.0) < ART_MIN_OCR
    )


def empty_translation(region: Region) -> bool:
    """Only punctuation ('¡', '…') for an original that has words: the model gave up."""
    has_words = any(ch.isalnum() for ch in region.ocr_text or "")
    return has_words and not any(ch.isalnum() for ch in region.translation or "")


def should_render(region: Region, sfx_mode: str = "annotate") -> bool:
    """True when the original text is removed and the translation is lettered in its place."""
    if region.status == "skipped" or not _has_text(region) or empty_translation(region):
        return False
    if region.clean_method == "none" or (region.clean_method == "fill" and not region.cleanable):
        return False  # Phase 1 projects: uncleanable regions keep the original
    if unreliable_art_text(region):
        return False
    if region.style == "sfx" and region.type in ("text_on_art", "sfx"):
        return sfx_mode == "replace"
    return region.type in RENDERABLE_TYPES or region.clean_method == "lama"


def _mode(page: Page) -> str:
    return "L" if page.grayscale else "RGB"


# ---------------------------------------------------------------------------- inpaint
def run_inpaint(
    project: Project,
    paths: ProjectPaths,
    settings: Settings,
    pages: list[Page],
    progress: Progress | None = None,
) -> dict[str, object]:
    sfx_mode = project.meta.sfx_mode
    fill = FillInpainter()
    lama: Inpainter | None = None
    paths.clean.mkdir(parents=True, exist_ok=True)
    counts = {"fill": 0, "lama": 0}
    try:
        for page in pages:
            image = np.asarray(Image.open(paths.abs(page.image_path)).convert(_mode(page)))
            for r in page.regions:
                art = "texto sobre dibujo con OCR dudoso: se conserva el original"
                empty = "traducción vacía (solo signos): se conserva el original"
                r.notes = [n for n in r.notes if n not in (art, empty)]
                if unreliable_art_text(r) and _has_text(r):
                    r.notes.append(art)
                if empty_translation(r):
                    r.notes.append(empty)
                    if r.status == "auto":
                        r.status = "needs_review"
            regions = [r for r in page.regions if should_render(r, sfx_mode)]
            if page.text_mask_path and regions:
                labels = np.asarray(Image.open(paths.abs(page.text_mask_path)))
                flat = [r for r in regions if r.clean_method == "fill"]
                textured = [r for r in regions if r.clean_method == "lama"]
                if flat:
                    image = fill.clean(image, labels, flat)
                if textured:
                    lama = lama or _load_lama(settings)
                    image = lama.clean(image, labels, textured)
                counts["fill"] += len(flat)
                counts["lama"] += len(textured)
            out = paths.clean / f"{page.number:04d}.png"
            Image.fromarray(image).save(out, compress_level=3)
            page.clean_path = paths.rel(out)
            if progress:
                progress()
    finally:
        if lama is not None:
            lama.close()
    return {"cleaned_fill": counts["fill"], "cleaned_lama": counts["lama"]}


def _load_lama(settings: Settings) -> Inpainter:
    import torch

    from mangatl.inpainting.lama import LamaInpainter
    from mangatl.model_store import lama_path

    device = "cuda" if settings.gpu.device == "cuda" and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("LaMa en CPU: la limpieza sobre dibujo será lenta")
    return LamaInpainter(
        lama_path(settings), device=device, max_side=settings.inpaint.lama_max_side
    )


# ---------------------------------------------------------------------------- typeset
ART_MARGIN = 0.03  # text on artwork: the "container" is already the text box plus a margin


class PageTypesetter:
    def __init__(
        self, settings: Settings, protected_words: set[str], sfx_mode: str = "annotate"
    ) -> None:
        self.settings = settings
        self.cfg = settings.typesetting
        self.sfx_mode = sfx_mode
        self.hyphenator = Hyphenator("es", protected_words) if self.cfg.hyphenate else None
        self._fitters: dict[str, TextFitter] = {}

    def margin(self, region: Region) -> float:
        return ART_MARGIN if region.type == "text_on_art" else self.cfg.margin_ratio

    def stroke(self, region: Region, size: int) -> int:
        """White outline width: needed wherever the text sits on artwork."""
        if region.type == "text_on_art" or (
            region.clean_method == "lama" and not region.background_uniform
        ):
            return max(2, round(size * self.cfg.stroke_ratio))
        return 0

    def font_path(self, region: Region) -> Path:
        if region.font:
            return self.settings.resolve(Path(region.font))
        return self.settings.resolve(getattr(self.cfg.fonts, region.style))

    def fitter(self, path: Path) -> TextFitter:
        key = str(path)
        if key not in self._fitters:
            self._fitters[key] = TextFitter(path, self.cfg.line_spacing, self.hyphenator)
        return self._fitters[key]

    def size_bounds(self, page: Page, region: Region) -> tuple[int, int]:
        scale = page.height / self.cfg.reference_page_height
        lo = max(6, round(self.cfg.min_font_px * scale))
        hi = max(lo, round(self.cfg.max_font_px * scale))
        if region.style == "whisper":
            hi = max(lo, round(hi * 0.85))
        if region.font_size and region.font_size_fixed:
            return region.font_size, region.font_size  # size fixed by the user
        return lo, hi

    def area(self, labels: np.ndarray, region: Region, margin_ratio: float) -> UsableArea:
        b = region.bubble_bbox or region.bbox
        if region.text_box_override is not None:
            o = region.text_box_override
            return UsableArea(np.ones((max(1, o.h), max(1, o.w)), bool), 0.0)
        if region.type == "text_on_art":
            # No bubble shape to follow: the whole (widened) box is usable.
            x0, y0, x1, y1 = self.art_box(region, labels.shape)
            return UsableArea(np.ones((y1 - y0, x1 - x0), bool), margin_ratio)
        interior = labels[b.y0 : b.y1, b.x0 : b.x1] == region.mask_label
        return UsableArea(interior, margin_ratio)

    def art_box(self, region: Region, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Text on artwork: vertical Japanese columns become horizontal Spanish lines, so the
        box is widened to at least ~0.9x its height, centered and kept inside the page."""
        h, w = shape[:2]
        b = region.bubble_bbox or region.bbox
        width = min(w, max(b.w, round(0.9 * b.h)))
        x0 = min(max(0, round(b.cx - width / 2)), w - width)
        return x0, b.y0, x0 + width, min(h, b.y1)

    def origin(self, region: Region, shape: tuple[int, ...]) -> tuple[int, int]:
        if region.text_box_override is not None:
            return region.text_box_override.x0, region.text_box_override.y0
        if region.type == "text_on_art":
            x0, y0, _, _ = self.art_box(region, shape)
            return x0, y0
        b = region.bubble_bbox or region.bbox
        return b.x0, b.y0

    def forced_layout(self, region: Region, text: str, shape: tuple[int, ...], size: int) -> Layout:
        """Last resort when nothing fits: a plain rectangle as wide as the bubble (at least six
        characters), growing downwards until every word is placed. Never leaves a cleaned
        bubble empty; the region is flagged for review anyway."""
        b = region.bubble_bbox or region.bbox
        fitter = self.fitter(self.font_path(region))
        width = max(b.w, size * 6)
        for grow in (1, 2, 3, 5):
            height = max(b.h, size * 2) * grow
            area = UsableArea(np.ones((height, width), bool), 0.0)
            lay = fitter.fit(text, area, max(6, size - 3), size)
            if lay.lines:
                # Center the block on the bubble: shift slots into bubble coordinates.
                dx, dy = (width - b.w) / 2, (height - b.h) / 2
                slots = [
                    Slot(y=s.y - dy, left=round(s.left - dx), right=round(s.right - dx))
                    for s in lay.slots
                ]
                return Layout(lay.font_size, lay.lines, slots, False)
        return Layout(size, [], [], False)

    def text_for(self, region: Region, shorter: bool = False) -> str:
        """Normalized text, with characters the font lacks replaced (ō -> o)."""
        raw = region.shorter_alternative if shorter else region.translation
        text, replaced = fit_to_font(normalize_text(raw or ""), self.font_path(region))
        if replaced:
            note = f"sin glifo en la fuente: {''.join(sorted(set(replaced)))}"
            region.notes = [n for n in region.notes if not n.startswith("sin glifo")] + [note]
        return text

    def layout_page(self, page: Page, labels: np.ndarray) -> dict[str, Layout]:
        layouts: dict[str, Layout] = {}
        areas: dict[str, UsableArea] = {}
        for region in page.regions:
            if not should_render(region, self.sfx_mode):
                continue
            region.used_shorter = False
            region.notes = [n for n in region.notes if not n.startswith("sin glifo")]
            area = self.area(labels, region, self.margin(region))
            areas[region.id] = area
            fitter = self.fitter(self.font_path(region))
            lo, hi = self.size_bounds(page, region)
            layout = fitter.fit(self.text_for(region), area, lo, hi)
            if not layout.fits and region.shorter_alternative:
                short = fitter.fit(self.text_for(region, shorter=True), area, lo, hi)
                if short.fits:
                    layout, region.used_shorter = short, True
            layouts[region.id] = layout

        # Consistent sizes on a page: short bubbles must not get much bigger text than the rest.
        sizes = [lay.font_size for lay in layouts.values() if lay.fits]
        if len(sizes) >= 2:
            cap = max(1, round(statistics.median(sizes) * self.cfg.page_size_spread))
            for region in page.regions:
                lay = layouts.get(region.id)
                if lay is None or not lay.fits or lay.font_size <= cap:
                    continue
                lo, _ = self.size_bounds(page, region)
                refit = self.fitter(self.font_path(region)).fit(
                    self.text_for(region, shorter=region.used_shorter),
                    areas[region.id],
                    min(lo, cap),
                    cap,
                )
                if refit.fits:
                    layouts[region.id] = refit

        for region in page.regions:
            lay = layouts.get(region.id)
            if lay is None:
                continue
            was_overflow = any(n.startswith("no cabe") for n in region.notes)
            region.notes = [n for n in region.notes if not n.startswith("no cabe")]
            if lay.fits:
                if was_overflow and region.status == "needs_review" and not region.notes:
                    region.status = "auto"  # it fits now (e.g. shorter version or edit)
                region.font_size, region.fits = lay.font_size, True
                continue
            # Best effort (no margin, slightly smaller), always flagged for review.
            lo, _ = self.size_bounds(page, region)
            loose = self.area(labels, region, 0.0)
            text = self.text_for(region, shorter=bool(region.shorter_alternative))
            lay = self.fitter(self.font_path(region)).fit(text, loose, max(6, lo - 3), lo)
            if not lay.lines and region.type != "text_on_art":
                lay = self.forced_layout(region, text, labels.shape, lo)
            layouts[region.id] = lay
            region.notes.append("no cabe al tamaño mínimo: revisar")
            if region.status == "auto":
                region.status = "needs_review"
            region.font_size, region.fits = lay.font_size, False
        return layouts

    def draw(self, image: Image.Image, page: Page, layouts: dict[str, Layout]) -> Image.Image:
        draw = ImageDraw.Draw(image)
        color = ImageColor.getcolor(self.cfg.text_color, image.mode)
        white = ImageColor.getcolor("#ffffff", image.mode)
        for region in page.regions:
            lay = layouts.get(region.id)
            if lay is None or not lay.lines:
                continue
            ox, oy = self.origin(region, (page.height, page.width))
            font = load_font(str(self.font_path(region)), lay.font_size)
            stroke = self.stroke(region, lay.font_size)
            for line, slot in zip(lay.lines, lay.slots, strict=False):
                draw.text(
                    (ox + slot.center, oy + slot.y),
                    line,
                    font=font,
                    fill=color,
                    anchor="mm",
                    stroke_width=stroke,
                    stroke_fill=white,
                )
        return image

    def draw_annotations(self, image: Image.Image, page: Page) -> int:
        """sfx_mode = annotate: small translation next to each sound effect, original kept."""
        draw = ImageDraw.Draw(image)
        color = ImageColor.getcolor(self.cfg.text_color, image.mode)
        white = ImageColor.getcolor("#ffffff", image.mode)
        font_path = self.settings.resolve(self.cfg.fonts.sfx)
        scale = page.height / self.cfg.reference_page_height
        size = max(8, round(self.cfg.min_font_px * scale))
        done = 0
        for region in page.regions:
            if not is_annotation(region, self.sfx_mode):
                continue
            text, _ = fit_to_font(normalize_text(region.translation or ""), font_path)
            box = region.bbox
            width = min(page.width, max(round(box.w * 1.5), size * 6))
            area = UsableArea(
                np.ones((round(size * self.cfg.line_spacing * 3) + 4, width), bool), 0.0
            )
            lay = self.fitter(font_path).fit(text, area, max(6, size - 4), size)
            if not lay.lines:
                continue
            line_h = lay.font_size * self.cfg.line_spacing
            top = min(s.y for s in lay.slots[: len(lay.lines)]) - line_h / 2
            block_h = len(lay.lines) * line_h
            x0 = min(max(0, box.cx - width / 2), page.width - width)
            y0 = box.y1 + 4 if box.y1 + 4 + block_h <= page.height else max(0, box.y0 - 4 - block_h)
            font = load_font(str(font_path), lay.font_size)
            for line, slot in zip(lay.lines, lay.slots, strict=False):
                draw.text(
                    (x0 + slot.center, y0 + slot.y - top),
                    line,
                    font=font,
                    fill=color,
                    anchor="mm",
                    stroke_width=max(2, round(lay.font_size * self.cfg.stroke_ratio)),
                    stroke_fill=white,
                )
            region.font_size, region.fits = lay.font_size, True
            done += 1
        return done


def find_overflows(
    project: Project, paths: ProjectPaths, settings: Settings, pages: list[Page]
) -> list[tuple[Region, int]]:
    """Regions whose translation does not fit even at the minimum size (dry run, no drawing).

    Returns each region with a character budget for a shorter version.
    """
    sfx_mode = project.meta.sfx_mode
    typesetter = PageTypesetter(settings, protected_words(project), sfx_mode)
    found: list[tuple[Region, int]] = []
    for page in pages:
        if not page.bubble_mask_path or not any(should_render(r, sfx_mode) for r in page.regions):
            continue
        labels = np.asarray(Image.open(paths.abs(page.bubble_mask_path)))
        probe = page.model_copy(deep=True)  # layout_page updates regions; keep the real ones
        layouts = typesetter.layout_page(probe, labels)
        real = {r.id: r for r in page.regions}
        for r in probe.regions:
            if r.id not in layouts or r.fits is not False:
                continue
            region = real[r.id]
            lo, _ = typesetter.size_bounds(page, region)
            area = typesetter.area(labels, region, typesetter.margin(region))
            budget = estimate_capacity(area, lo, typesetter.cfg.line_spacing)
            found.append((region, max(4, min(budget, int(len(region.translation or "") * 0.75)))))
    return found


def protected_words(project: Project) -> set[str]:
    words: set[str] = set()
    for entry in project.pending_glossary:
        words.update(entry.target.split())
    for _, region in project.regions():
        if region.speaker and region.speaker != "desconocido":
            words.update(region.speaker.split())
    return words


def run_typeset(
    project: Project,
    paths: ProjectPaths,
    settings: Settings,
    pages: list[Page],
    progress: Progress | None = None,
) -> dict[str, object]:
    typesetter = PageTypesetter(settings, protected_words(project), project.meta.sfx_mode)
    rendered_dir = paths.work / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    lettered = review = annotated = 0
    for page in pages:
        if not page.clean_path:
            raise RuntimeError(
                f"La página {page.number} no tiene versión limpia; ejecuta la etapa inpaint"
            )
        image = Image.open(paths.abs(page.clean_path)).convert(_mode(page))
        labels = (
            np.asarray(Image.open(paths.abs(page.bubble_mask_path)))
            if page.bubble_mask_path
            else np.zeros((page.height, page.width), np.uint8)
        )
        layouts = typesetter.layout_page(page, labels)
        typesetter.draw(image, page, layouts)
        annotated += typesetter.draw_annotations(image, page)
        out = rendered_dir / f"{page.number:04d}.png"
        image.save(out, compress_level=3)
        page.rendered_path = paths.rel(out)
        lettered += len(layouts)
        review += sum(1 for r in page.regions if r.id in layouts and r.fits is False)
        if progress:
            progress()
    return {"lettered": lettered, "overflow": review, "sfx_annotated": annotated}


# ---------------------------------------------------------------------------- export
def render_single_page(
    project: Project, paths: ProjectPaths, settings: Settings, page: Page
) -> Path:
    """Clean + letter one page (editor 're-render') and refresh its exported PNG."""
    run_inpaint(project, paths, settings, [page])
    run_typeset(project, paths, settings, [page])
    out_dir = paths.output_pages
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{page.source_name}.png"
    shutil.copyfile(paths.abs(page.rendered_path or page.image_path), dst)
    return dst


def run_export(project: Project, paths: ProjectPaths, settings: Settings) -> dict[str, object]:
    out_dir = paths.output_pages
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    files: list[Path] = []
    for page in sorted(project.pages, key=lambda p: p.number):
        src = paths.abs(page.rendered_path or page.image_path)
        dst = out_dir / f"{page.source_name}.png"
        shutil.copyfile(src, dst)
        files.append(dst)
    formats = settings.export.formats
    details: dict[str, object] = {"pages": len(files)}
    if "cbz" in formats:
        with zipfile.ZipFile(paths.cbz, "w", compression=zipfile.ZIP_STORED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        details["cbz"] = str(paths.cbz)
    if "pdf" in formats:
        details["pdf"] = str(write_pdf(files, paths.pdf))
    return details


def write_pdf(images: list[Path], out: Path) -> Path:
    """One page per image, lossless (the PNG data is embedded as is), 1 px = 1 pt."""
    import pymupdf

    doc = pymupdf.open()
    try:
        for img in images:
            with Image.open(img) as im:
                width, height = im.size
            page = doc.new_page(width=width, height=height)
            page.insert_image(page.rect, filename=str(img))
        doc.save(out, garbage=3, deflate=True)
    finally:
        doc.close()
    return out
