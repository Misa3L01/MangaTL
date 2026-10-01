"""Stages 1-3: detection + region analysis + reading order, then OCR.

These are the only GPU stages before translation. They normally run in a child process
(see `mangatl.stages.vision_worker`) so the CUDA context is released before the LLM loads.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np
from PIL import Image

from mangatl.config import Settings
from mangatl.detection.regions import RegionDraft, build_regions
from mangatl.models import RENDERABLE_TYPES, BBox, Page, Project, Region
from mangatl.ordering import BandOrderer, PanelOrderer, coarse_position
from mangatl.project_io import ProjectPaths
from mangatl.typesetting.layout import UsableArea, estimate_capacity

log = logging.getLogger(__name__)

Progress = Callable[[], None]


def _quiet_transformers() -> None:
    """Hide transformers' per-model 'Loading weights' progress bars (we show our own)."""
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()


def _device(settings: Settings) -> str:
    import torch

    _quiet_transformers()

    if settings.gpu.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if settings.gpu.device == "cuda":
        log.warning("CUDA no disponible: la detección y el OCR se ejecutarán en CPU (más lento)")
    return "cpu"


def font_scale(settings: Settings, page: Page) -> float:
    return page.height / settings.typesetting.reference_page_height


def _to_bbox(box: tuple[int, int, int, int]) -> BBox:
    return BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])


def _make_region(
    page: Page, draft: RegionDraft, rank: int, settings: Settings, label: int
) -> Region:
    assert draft.interior is not None and draft.crop is not None
    ts = settings.typesetting
    scale = font_scale(settings, page)
    area = UsableArea(draft.interior, ts.margin_ratio)
    text_bbox = draft.text_bbox
    status = "auto"
    notes = list(draft.notes)
    cleanable = draft.clean_coverage >= settings.inpaint.min_coverage
    method = "fill"
    if draft.needs_lama:
        method = "lama" if settings.inpaint.use_lama else "none"
    if method == "lama":
        if draft.type != "text_on_art" and not cleanable:
            notes.append(f"el texto cruza el borde del globo ({draft.clean_coverage:.0%}): LaMa")
    elif method == "none":
        # LaMa disabled: keep the original text of regions a fill cannot clean.
        status = "skipped" if draft.type not in RENDERABLE_TYPES else "needs_review"
        notes.append("no se puede limpiar sin LaMa (inpaint.use_lama = false): se conserva")
    return Region(
        id=f"P{page.number:03d}-B{rank:02d}",
        type=draft.type,
        bbox=text_bbox,
        text_boxes=[_to_bbox(b) for b in draft.text_boxes],
        bubble_bbox=_to_bbox(draft.crop),
        polygon=draft.polygon,
        mask_label=label,
        detection_score=round(draft.score, 3),
        reading_order=rank,
        position=coarse_position(draft.order_bbox, page.width, page.height),
        vertical=text_bbox.h > 1.3 * text_bbox.w,
        capacity_chars=estimate_capacity(area, ts.base_font_px * scale, ts.line_spacing),
        background_uniform=draft.background_uniform,
        background_color=draft.background_color,
        cleanable=cleanable,
        clean_method=method,
        style="narration" if draft.type == "narration_box" else "normal",
        status=status,
        notes=notes,
    )


def run_detection(
    project: Project,
    paths: ProjectPaths,
    settings: Settings,
    pages: list[Page],
    progress: Progress | None = None,
) -> dict[str, object]:
    from mangatl.detection.rtdetr import RtDetrDetector
    from mangatl.model_store import detector_dir

    detector = RtDetrDetector(
        detector_dir(settings), device=_device(settings), threshold=settings.detection.threshold
    )
    direction = project.meta.reading_direction
    use_panels = settings.detection.panel_order
    orderer = PanelOrderer(direction) if use_panels else BandOrderer(direction)
    paths.masks.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        for page in pages:
            img = Image.open(paths.abs(page.image_path))
            gray = np.asarray(img.convert("L"))
            rgb = None if page.grayscale else np.asarray(img.convert("RGB"))
            drafts = build_regions(
                gray, detector.detect(img), settings.detection, settings.inpaint, rgb
            )
            boxes = [d.order_bbox for d in drafts]
            if isinstance(orderer, PanelOrderer):
                order = orderer.order(boxes, page.width, page.height, page.is_spread, gray)
                panel_of = orderer.last_panels
            else:
                order = orderer.order(boxes, page.width, page.height, page.is_spread)
                panel_of = [None] * len(drafts)
            if len(order) > 255:
                log.warning("Página %d: más de 255 regiones; se ignoran las sobrantes", page.number)
                order = order[:255]

            interiors = np.zeros(gray.shape, np.uint8)
            glyphs = np.zeros(gray.shape, np.uint8)
            regions = []
            for rank, idx in enumerate(order, start=1):
                d = drafts[idx]
                assert d.interior is not None and d.crop is not None and d.text_mask is not None
                x0, y0, x1, y1 = d.crop
                interiors[y0:y1, x0:x1][d.interior] = rank
                glyphs[y0:y1, x0:x1][d.text_mask] = rank
                region = _make_region(page, d, rank, settings, rank)
                region.panel = panel_of[idx]
                regions.append(region)

            stem = f"{page.number:04d}"
            Image.fromarray(interiors).save(paths.masks / f"{stem}_bubbles.png")
            Image.fromarray(glyphs).save(paths.masks / f"{stem}_text.png")
            page.bubble_mask_path = paths.rel(paths.masks / f"{stem}_bubbles.png")
            page.text_mask_path = paths.rel(paths.masks / f"{stem}_text.png")
            page.regions = regions
            total += len(regions)
            if progress:
                progress()
    finally:
        detector.close()
    return {"regions": total}


def _ocr_crop(img: Image.Image, box: BBox, pad: int = 4) -> Image.Image:
    b = box.expand(pad, img.width, img.height)
    return img.crop(b.as_tuple())


_CJK = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_ENGLISH_ACCENTS = str.maketrans(
    "ÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛáéíóúàèìòùâêîôû", "AEIOUAEIOUAEIOUAEIOUaeiouaeiouaeiou"
)


def detect_language(paths: ProjectPaths, pages: list[Page], sample: int = 30) -> str:
    """'ja' or 'en' from a multilingual OCR pass over the first text blocks of the chapter."""
    from mangatl.ocr.english import PaddleRapidOcrEngine

    crops = []
    for page in pages:
        img = Image.open(paths.abs(page.image_path))
        crops += [_ocr_crop(img, b) for r in page.regions for b in (r.text_boxes or [r.bbox])]
        if len(crops) >= sample:
            break
    if not crops:
        return "ja"
    texts = [r.text for r in PaddleRapidOcrEngine().read(crops[:sample])]
    letters = sum(1 for t in texts for ch in t if ch.isalnum())
    cjk = sum(len(_CJK.findall(t)) for t in texts)
    return "ja" if letters and cjk / letters > 0.3 else "en"


def run_ocr(
    project: Project,
    paths: ProjectPaths,
    settings: Settings,
    pages: list[Page],
    progress: Progress | None = None,
) -> dict[str, object]:
    details: dict[str, object] = {}
    if project.meta.source_lang == "auto":
        project.meta.source_lang = detect_language(paths, pages)  # type: ignore[assignment]
        details["detected_language"] = project.meta.source_lang
        log.info(
            "Idioma detectado: %s", "japonés" if project.meta.source_lang == "ja" else "inglés"
        )
    japanese = project.meta.source_lang == "ja"
    if japanese:
        from mangatl.model_store import ja_ocr_dir
        from mangatl.ocr.manga_ocr_engine import MangaOcrEngine

        engine = MangaOcrEngine(ja_ocr_dir(settings), _device(settings), settings.ocr.batch_size)
    else:
        from mangatl.ocr.english import PaddleRapidOcrEngine

        engine = PaddleRapidOcrEngine()
    low = 0
    try:
        for page in pages:
            img = Image.open(paths.abs(page.image_path))
            crops: list[Image.Image] = []
            owners: list[int] = []
            for k, region in enumerate(page.regions):
                boxes = region.text_boxes or [region.bbox]
                # Vertical Japanese blocks: columns right-to-left; English: top-to-bottom.
                key = (lambda b: (-b.cx, b.y0)) if japanese else (lambda b: (b.y0, b.x0))
                for box in sorted(boxes, key=key):
                    crops.append(_ocr_crop(img, box))
                    owners.append(k)
            results = engine.read(crops) if crops else []
            for k, region in enumerate(page.regions):
                parts = [r for r, o in zip(results, owners, strict=True) if o == k]
                if japanese:
                    region.ocr_text = "".join(p.text for p in parts)
                else:
                    region.ocr_text = " ".join(p.text for p in parts if p.text).translate(
                        _ENGLISH_ACCENTS
                    )
                region.ocr_confidence = min((p.confidence for p in parts), default=0.0)
                if region.status == "skipped":
                    continue
                # Re-running OCR replaces its previous verdict.
                region.notes = [n for n in region.notes if not n.startswith("OCR dudoso")]
                if region.status == "needs_review" and region.clean_method != "none":
                    region.status = "auto"
                if not region.ocr_text or region.ocr_confidence < settings.ocr.min_confidence:
                    region.status = "needs_review"
                    region.notes.append(f"OCR dudoso (confianza {region.ocr_confidence:.2f})")
                    low += 1
            if progress:
                progress()
    finally:
        engine.close()
    return {**details, "language": project.meta.source_lang, "low_confidence": low}
