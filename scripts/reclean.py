"""Re-clean an already processed project with other inpaint settings, in a separate folder.

    uv run python scripts/reclean.py --project <p.mangatl.json> --out tmp/exp-e/<tag> \
        [--pages 61-84] [--set inpaint.text_dilate_px=4 ...] [--crops P061-B01,P067-B10]

Rebuilds the detections from the saved regions (bubble and text boxes), recomputes the
cleaning masks with the given settings and cleans the regions the project letters (the same
choice `mangatl render` makes). Cleaning changes can thus be compared without running the
detector, the OCR or the LLM again. Writes <out>/clean/NNNN.png, <out>/report.json and, for
--crops, images "original | saved clean | new clean". The project is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mangatl import config
from mangatl.config import load_settings
from mangatl.runtime_env import apply_runtime_env

DARK = 110  # gray level counted as ink when measuring leftovers


def detections_from_regions(regions: list) -> list:
    """RawDetections equivalent to the detector output that produced `regions`."""
    from mangatl.detection.base import RawDetection

    dets, bubbles = [], set()
    for r in regions:
        if r.bubble_bbox is not None:
            box = r.bubble_bbox.as_tuple()
            if box not in bubbles:
                bubbles.add(box)
                dets.append(RawDetection("bubble", box, 1.0))
        loose_art = r.bubble_bbox is None and r.type in ("text_on_art", "narration_box", "sfx")
        label = "text_free" if loose_art else "text_bubble"
        for tb in r.text_boxes or [r.bbox]:
            dets.append(RawDetection(label, tb.as_tuple(), 1.0))
    return dets


def _match(drafts: list, region) -> object | None:
    boxes = {tb.as_tuple() for tb in region.text_boxes or [region.bbox]}
    for d in drafts:
        if set(d.text_boxes) == boxes:
            return d
    return None


def leftover_ink(original: np.ndarray, clean: np.ndarray, region) -> tuple[int, int]:
    """(ink pixels still there, ink pixels before) inside the region's text boxes, shrunk
    3 px so a bubble outline crossing the edge of a box is not counted."""
    left = before = 0
    for tb in region.text_boxes or [region.bbox]:
        x0, y0, x1, y1 = tb.x0 + 3, tb.y0 + 3, tb.x1 - 3, tb.y1 - 3
        if x1 <= x0 or y1 <= y0:
            continue
        o = original[y0:y1, x0:x1] < DARK
        left += int((o & (clean[y0:y1, x0:x1] < DARK)).sum())
        before += int(o.sum())
    return left, before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", default=None)
    parser.add_argument("--set", action="append", metavar="CLAVE=VALOR")
    parser.add_argument("--crops", default="", help="IDs separados por comas")
    args = parser.parse_args()

    settings = load_settings()
    apply_runtime_env(settings)
    try:
        settings = config.apply_overrides(settings, list(args.set or []))
    except ValueError as exc:
        raise SystemExit(f"--set: {exc}") from exc

    from mangatl.detection.regions import build_regions
    from mangatl.ingest import parse_page_range
    from mangatl.inpainting.fill import FillInpainter
    from mangatl.project_io import ProjectPaths, load_project
    from mangatl.stages.render import _load_lama, should_render

    project = load_project(args.project)
    paths = ProjectPaths(args.project)
    selection = parse_page_range(args.pages)
    pages = [p for p in project.pages if selection is None or p.number in selection]
    out_clean = args.out / "clean"
    out_clean.mkdir(parents=True, exist_ok=True)
    crops = {c.strip() for c in args.crops.split(",") if c.strip()}
    fill, lama = FillInpainter(), None
    report: dict[str, object] = {"overrides": args.set or [], "regions": {}}
    totals = {"fill": 0, "lama": 0, "ink_left": 0, "ink_before": 0, "lama_mask_px": 0}
    try:
        for page in pages:
            img = Image.open(paths.abs(page.image_path))
            gray = np.asarray(img.convert("L"))
            rgb = None if page.grayscale else np.asarray(img.convert("RGB"))
            drafts = build_regions(
                gray,
                detections_from_regions(page.regions),
                settings.detection,
                settings.inpaint,
                rgb,
            )
            labels = np.zeros(gray.shape, np.uint8)
            for region in sorted(page.regions, key=lambda r: r.mask_label):
                d = _match(drafts, region)
                if d is None:
                    print(f"{region.id}: sin borrador equivalente, se omite")
                    continue
                x0, y0, x1, y1 = d.crop
                labels[y0:y1, x0:x1][d.text_mask] = region.mask_label
                method = "fill"
                if d.needs_lama:
                    method = "lama" if settings.inpaint.use_lama else "none"
                region.clean_method = method
                region.cleanable = d.clean_coverage >= settings.inpaint.min_coverage
            regions = [r for r in page.regions if should_render(r, project.meta.sfx_mode)]
            image = np.asarray(img.convert("L" if page.grayscale else "RGB"))
            flat = [r for r in regions if r.clean_method == "fill"]
            textured = [r for r in regions if r.clean_method == "lama"]
            if flat:
                image = fill.clean(image, labels, flat)
            if textured:
                lama = lama or _load_lama(settings)
                image = lama.clean(image, labels, textured)
            Image.fromarray(image).save(out_clean / f"{page.number:04d}.png", compress_level=3)

            new_gray = np.asarray(Image.fromarray(image).convert("L"))
            for r in regions:
                mask_px = int((labels == r.mask_label).sum())
                left, before = leftover_ink(gray, new_gray, r)
                report["regions"][r.id] = {
                    "method": r.clean_method,
                    "mask_px": mask_px,
                    "box_px": int(sum(tb.area for tb in r.text_boxes or [r.bbox])),
                    "ink_left": left,
                    "ink_before": before,
                }
                totals[r.clean_method] += 1
                if r.clean_method == "fill":
                    totals["ink_left"] += left
                    totals["ink_before"] += before
                else:
                    totals["lama_mask_px"] += mask_px
                if r.id in crops:
                    _crop(paths, page, r, gray, new_gray, args.out)
    finally:
        if lama is not None:
            lama.close()
    report["totals"] = totals
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{len(pages)} páginas: {totals['fill']} con relleno (tinta restante "
        f"{totals['ink_left']} de {totals['ink_before']} px), {totals['lama']} con LaMa "
        f"(máscara total {totals['lama_mask_px']} px) -> {args.out}"
    )


def _crop(paths, page, region, gray: np.ndarray, new_gray: np.ndarray, out: Path) -> None:
    b, pad = region.bbox, 40
    y0, y1 = max(0, b.y0 - pad), min(gray.shape[0], b.y1 + pad)
    x0, x1 = max(0, b.x0 - pad), min(gray.shape[1], b.x1 + pad)
    saved = (
        np.asarray(Image.open(paths.abs(page.clean_path)).convert("L")) if page.clean_path else gray
    )
    panels = [gray[y0:y1, x0:x1], saved[y0:y1, x0:x1], new_gray[y0:y1, x0:x1]]
    h, w = panels[0].shape
    canvas = np.full((h, 3 * w + 20), 255, np.uint8)
    for i, panel in enumerate(panels):
        canvas[:, i * (w + 10) : i * (w + 10) + w] = panel
    crop = Image.fromarray(canvas)
    if max(crop.size) > 1200:
        s = 1200 / max(crop.size)
        crop = crop.resize((round(crop.width * s), round(crop.height * s)))
    (out / "crops").mkdir(exist_ok=True)
    crop.save(out / "crops" / f"{region.id}.png")


if __name__ == "__main__":
    main()
