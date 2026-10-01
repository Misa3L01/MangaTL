"""--debug images: detections with reading order, and masks (interior + text to clean)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mangatl.models import Page
from mangatl.project_io import ProjectPaths

TYPE_COLORS = {
    "speech_bubble": (0, 150, 255),
    "narration_box": (160, 0, 255),
    "text_on_art": (255, 60, 60),
    "sfx": (255, 140, 0),
}
STATUS_COLORS = {
    "auto": (0, 170, 0),
    "edited": (0, 120, 255),
    "needs_review": (255, 170, 0),
    "skipped": (150, 150, 150),
}


def _label_font(page: Page) -> ImageFont.ImageFont:
    size = max(14, page.height // 70)
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_regions(paths: ProjectPaths, page: Page) -> None:
    """Boxes colored by type, reading order number and status."""
    img = Image.open(paths.abs(page.image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _label_font(page)
    for r in page.regions:
        color = TYPE_COLORS.get(r.type, (0, 0, 0))
        if r.bubble_bbox:
            draw.rectangle(r.bubble_bbox.as_tuple(), outline=color, width=2)
        for tb in r.text_boxes or [r.bbox]:
            draw.rectangle(tb.as_tuple(), outline=STATUS_COLORS[r.status], width=3)
        if len(r.polygon) >= 3:
            draw.line([*r.polygon, r.polygon[0]], fill=color, width=2)
        label = f"{r.reading_order}"
        x, y = r.bbox.x1 + 4, r.bbox.y0
        tb = draw.textbbox((x, y), label, font=font)
        draw.rectangle(
            (tb[0] - 3, tb[1] - 3, tb[2] + 3, tb[3] + 3), fill=(255, 255, 0), outline=(0, 0, 0)
        )
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    paths.debug.mkdir(parents=True, exist_ok=True)
    img.save(paths.debug / f"{page.number:04d}_1_regiones.png")


def draw_masks(paths: ProjectPaths, page: Page) -> None:
    """Interior masks in translucent blue, pixels to clean in red."""
    if not (page.bubble_mask_path and page.text_mask_path):
        return
    base = np.asarray(Image.open(paths.abs(page.image_path)).convert("RGB")).astype(np.float32)
    interiors = np.asarray(Image.open(paths.abs(page.bubble_mask_path))) > 0
    glyphs = np.asarray(Image.open(paths.abs(page.text_mask_path))) > 0
    out = base.copy()
    out[interiors] = out[interiors] * 0.6 + np.array([60, 140, 255]) * 0.4
    out[glyphs] = np.array([255, 0, 0])
    paths.debug.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.astype(np.uint8)).save(paths.debug / f"{page.number:04d}_2_mascaras.png")
