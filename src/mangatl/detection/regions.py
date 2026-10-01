"""Turn raw detections into regions: group text into bubbles, compute masks, classify.

For every region we compute:
- the *interior mask* of its container (bubble or box), used to fit the translation;
- the *text mask*: the original glyph pixels (dilated) that must be cleaned, always kept
  inside the eroded interior so bubble outlines are never touched;
- whether the background is uniform (Phase 1 only cleans uniform backgrounds).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mangatl.config import DetectionConfig, InpaintConfig
from mangatl.detection.base import RawDetection
from mangatl.models import BBox, RegionType

Box = tuple[int, int, int, int]
MIN_AREA_PX = 20  # smaller bright specks are glyph counters or screentone, not bubble areas


@dataclass
class RegionDraft:
    type: RegionType
    text_boxes: list[Box]
    container: Box | None
    score: float
    interior: np.ndarray | None = None  # bool mask, crop coordinates
    crop: Box | None = None  # crop window of `interior` in page coordinates
    text_mask: np.ndarray | None = None  # bool mask, crop coordinates
    background_color: tuple[int, int, int] = (255, 255, 255)
    background_uniform: bool = True
    clean_coverage: float = 1.0
    needs_lama: bool = False
    polygon: list[tuple[int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def text_bbox(self) -> BBox:
        xs0, ys0, xs1, ys1 = zip(*self.text_boxes, strict=True)
        return BBox(x0=min(xs0), y0=min(ys0), x1=max(xs1), y1=max(ys1))

    @property
    def order_bbox(self) -> BBox:
        """Box used for reading order: the container when there is one."""
        if self.container:
            return BBox(
                x0=self.container[0],
                y0=self.container[1],
                x1=self.container[2],
                y1=self.container[3],
            )
        return self.text_bbox


def _area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a: Box, b: Box) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _iou(a: Box, b: Box) -> float:
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def dedupe(dets: list[RawDetection], iou: float = 0.6) -> list[RawDetection]:
    """Greedy NMS per label (the detector is NMS-free but may repeat a box)."""
    kept: list[RawDetection] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(k.label != det.label or _iou(k.box, det.box) < iou for k in kept):
            kept.append(det)
    return kept


def _expand(b: Box, px: int, w: int, h: int) -> Box:
    return (max(0, b[0] - px), max(0, b[1] - px), min(w, b[2] + px), min(h, b[3] + px))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill regions not reachable from the crop border (text strokes inside a bubble)."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def _convex_hull(mask: np.ndarray, every_part: bool = False) -> np.ndarray:
    """Convex hull of the largest blob of `mask` (or of all its blobs together)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = np.zeros(mask.shape, np.uint8)
    if contours:
        points = np.vstack(contours) if every_part else max(contours, key=cv2.contourArea)
        cv2.fillPoly(hull, [cv2.convexHull(points)], 1)
    return hull.astype(bool)


def _box_mask(
    shape: tuple[int, ...], crop_box: Box, text_boxes: list[Box], grow: int
) -> np.ndarray:
    """Text boxes (grown by `grow` px) as a mask in crop coordinates."""
    x0, y0, x1, y1 = crop_box
    mask = np.zeros(shape[:2], bool)
    for tb in text_boxes:
        bx0, by0 = max(0, tb[0] - x0 - grow), max(0, tb[1] - y0 - grow)
        bx1, by1 = min(x1 - x0, tb[2] - x0 + grow), min(y1 - y0, tb[3] - y0 + grow)
        mask[by0:by1, bx0:bx1] = True
    return mask


def text_areas(
    gray_crop: np.ndarray, crop_box: Box, text_boxes: list[Box], grow: int
) -> np.ndarray:
    """Bright enclosed areas of the container that touch its text (crop coordinates).

    A line of text that touches the outline on both sides splits a bubble into two bright
    areas; `container_interior` keeps only the one holding most of the text, so the glyphs
    of the other one and of the dividing line were never cleaned. Areas touching the crop
    border are excluded: they belong to the artwork or leak in from a neighboring box.
    """
    seed = _box_mask(gray_crop.shape, crop_box, text_boxes, 0)
    near = _box_mask(gray_crop.shape, crop_box, text_boxes, grow)
    bg_level = float(np.percentile(gray_crop[seed], 90)) if seed.any() else 255.0
    bright = gray_crop > max(100.0, bg_level - 30)
    kernel = np.ones((3, 3), np.uint8)
    core = cv2.erode(bright.astype(np.uint8), kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    if n <= 1:
        return np.zeros_like(bright)
    edge = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    touching = np.bincount(labels[near], minlength=n)
    keep = [
        i
        for i in range(1, n)
        if i not in edge and touching[i] > 0 and stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_PX
    ]
    if not keep:
        return np.zeros_like(bright)
    areas = cv2.dilate(np.isin(labels, keep).astype(np.uint8), kernel).astype(bool) & bright
    return _fill_holes(areas)


def _bodies(mask: np.ndarray) -> np.ndarray:
    """`_main_body` of every separate area of `mask` (areas split by a line of text)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    out = np.zeros(mask.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_PX:
            out |= _main_body(labels == i)
    return out


def _main_body(mask: np.ndarray) -> np.ndarray:
    """Largest blob after a morphological opening: drops tails and thin strips that leak in
    from a neighboring box, which would otherwise stretch the convex hull over outlines."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    side = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    k = max(3, int(0.1 * side)) | 1
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=4)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _border_contact(mask: np.ndarray) -> float:
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(edges.mean()) if edges.size else 0.0


def container_interior(
    gray: np.ndarray, container: Box, text_boxes: list[Box]
) -> tuple[np.ndarray, Box, list[str]]:
    """Bright connected area inside `container` that holds the text, with holes filled."""
    h, w = gray.shape
    crop_box = _expand(container, 4, w, h)
    x0, y0, x1, y1 = crop_box
    crop = gray[y0:y1, x0:x1]
    notes: list[str] = []

    seed = np.zeros(crop.shape, bool)
    for tb in text_boxes:
        sx0, sy0 = max(0, tb[0] - x0), max(0, tb[1] - y0)
        sx1, sy1 = min(x1 - x0, tb[2] - x0), min(y1 - y0, tb[3] - y0)
        seed[sy0:sy1, sx0:sx1] = True

    # Background level measured where the text is (glyphs are sparse): anti-aliased gray
    # outline pixels then count as "dark" and keep the bubble closed.
    bg_level = float(np.percentile(crop[seed], 90)) if seed.any() else 255.0
    bright = crop > max(100.0, bg_level - 30)
    # A 1 px erosion breaks leaks through hairline gaps in the outline.
    kernel = np.ones((3, 3), np.uint8)
    core = cv2.erode(bright.astype(np.uint8), kernel)

    n, labels = cv2.connectedComponents(core, connectivity=4)
    votes = np.bincount(labels[seed & (core > 0)], minlength=n) if n > 1 else np.zeros(1)
    label = int(np.argmax(votes[1:]) + 1) if n > 1 and votes[1:].max(initial=0) > 0 else 0
    # A real bubble interior at least holds its text; a few bright specks (screentone) do not.
    if label and _fill_holes(labels == label).sum() < 0.6 * seed.sum():
        label = 0
    if label == 0:
        notes.append("sin fondo claro alrededor del texto")
        interior = np.zeros_like(bright)
        interior[
            max(0, container[1] - y0) : container[3] - y0,
            max(0, container[0] - x0) : container[2] - x0,
        ] = True
        return interior, crop_box, notes
    component = cv2.dilate((labels == label).astype(np.uint8), kernel).astype(bool) & bright
    interior = _fill_holes(component)

    if _border_contact(interior) > 0.15:
        # Open or borderless container: clip to the detected box so we never spill onto art.
        notes.append("contorno abierto")
        clip = np.zeros_like(interior)
        clip[
            max(0, container[1] - y0) : container[3] - y0,
            max(0, container[0] - x0) : container[2] - x0,
        ] = True
        interior &= clip
    return interior, crop_box, notes


def text_pixels(
    gray_crop: np.ndarray,
    interior: np.ndarray,
    text_boxes: list[Box],
    crop_box: Box,
    cfg: InpaintConfig,
    whole_interior: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Glyph pixels to clean, background pixels, background uniformity, ink coverage.

    With `whole_interior`, every dark pixel inside the (eroded) interior is text, not only
    the ones near the detected text boxes: catches large characters, furigana or emphasis
    marks the detector box missed. Only safe for closed containers with a flat background.
    """
    x0, y0, x1, y1 = crop_box
    protect = max(1, cfg.border_protect_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * protect + 1, 2 * protect + 1))
    safe = cv2.erode(interior.astype(np.uint8), kernel).astype(bool)

    boxes = np.zeros_like(interior)
    grow = cfg.text_box_grow_px
    for tb in text_boxes:
        bx0, by0 = max(0, tb[0] - x0 - grow), max(0, tb[1] - y0 - grow)
        bx1, by1 = min(x1 - x0, tb[2] - x0 + grow), min(y1 - y0, tb[3] - y0 + grow)
        boxes[by0:by1, bx0:bx1] = True
    if whole_interior:
        boxes = safe.copy()

    light = safe & (gray_crop > 200)
    if not light.any():
        # Non-white bubble: take the brightest half of the interior as background.
        values = gray_crop[safe] if safe.any() else gray_crop.ravel()
        light = safe & (gray_crop >= np.median(values))
    bg = int(np.median(gray_crop[light])) if light.any() else 255
    dark = gray_crop < (bg - cfg.text_contrast)
    glyphs = dark & boxes & safe
    # Share of the ink inside the text boxes that the fill can reach. Big brush lettering
    # that crosses the bubble outline is left mostly outside the safe interior. Only ink
    # within the bubble's convex hull counts: the corners of a rectangular text box stick
    # out of oval bubbles onto the artwork.
    ink = dark & boxes & _convex_hull(interior)
    coverage = float(glyphs.sum()) / max(1, int(ink.sum()))

    dil = max(0, cfg.text_dilate_px)
    if dil:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
        glyphs = cv2.dilate(glyphs.astype(np.uint8), k).astype(bool) & safe

    # Uniformity: outside the text area, the interior should be flat.
    outside = safe & ~boxes
    if outside.sum() < 50:
        outside = safe & ~glyphs
    ring = gray_crop[outside]
    # Robust flatness test: few pixels far from the background (screentone dots are >=12 %)
    # and little spread among the rest (gradients). A stray mark does not break it.
    # Without enough background samples uniformity cannot be verified: assume it is not.
    deviation = np.abs(ring.astype(int) - bg)
    inliers = ring[deviation <= 40]
    uniform = bool(
        ring.size >= 50
        and np.mean(deviation > 40) < 0.04
        and inliers.size > 0
        and np.std(inliers) < 12
    )
    return glyphs, light, uniform, coverage


def _polygon(interior: np.ndarray, crop_box: Box) -> tuple[list[tuple[int, int]], bool]:
    """Simplified outline in page coordinates, and whether it is (nearly) rectangular."""
    contours, _ = cv2.findContours(
        interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [], False
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    _, _, w, h = cv2.boundingRect(cnt)
    rect_fill = cv2.contourArea(cnt) / float(w * h) if w * h else 0.0
    is_rect = len(approx) <= 6 and rect_fill > 0.93
    pts = [(int(p[0][0]) + crop_box[0], int(p[0][1]) + crop_box[1]) for p in approx]
    return pts, is_rect


def build_regions(
    gray: np.ndarray,
    detections: list[RawDetection],
    det_cfg: DetectionConfig,
    inpaint_cfg: InpaintConfig,
    rgb: np.ndarray | None = None,
) -> list[RegionDraft]:
    """`gray` drives the analysis; `rgb` (optional, color pages) only sets the fill color."""
    h, w = gray.shape
    dets = dedupe(detections)
    bubbles = [d for d in dets if d.label == "bubble" and d.score >= det_cfg.bubble_threshold]
    texts = [d for d in dets if d.label != "bubble" and d.score >= det_cfg.text_threshold]

    groups: dict[int, list[RawDetection]] = {}
    loose: list[RawDetection] = []
    for t in texts:
        best, best_key = None, None
        for i, b in enumerate(bubbles):
            ioa = _inter(t.box, b.box) / max(1, _area(t.box))
            if ioa >= 0.6:
                key = (-ioa, _area(b.box))
                if best_key is None or key < best_key:
                    best, best_key = i, key
        if best is None:
            loose.append(t)
        else:
            groups.setdefault(best, []).append(t)

    drafts: list[RegionDraft] = []
    for i, members in groups.items():
        bubble = bubbles[i]
        boxes = [m.box for m in members]
        interior, crop_box, notes = container_interior(gray, bubble.box, boxes)
        drafts.append(
            RegionDraft(
                type="speech_bubble",
                text_boxes=boxes,
                container=bubble.box,
                score=max(m.score for m in members),
                interior=interior,
                crop=crop_box,
                notes=notes,
            )
        )

    for t in loose:
        region_type: RegionType = "speech_bubble" if t.label == "text_bubble" else "text_on_art"
        # Treat the text box (plus a margin) as its own container and check the background.
        margin = max(6, int(0.15 * min(t.box[2] - t.box[0], t.box[3] - t.box[1])))
        container = _expand(t.box, margin, w, h)
        interior, crop_box, notes = container_interior(gray, container, [t.box])
        drafts.append(
            RegionDraft(
                type=region_type,
                text_boxes=[t.box],
                container=None,
                score=t.score,
                interior=interior,
                crop=crop_box,
                notes=notes,
            )
        )

    for d in drafts:
        assert d.interior is not None and d.crop is not None
        x0, y0, x1, y1 = d.crop
        glyphs, light, uniform, coverage = text_pixels(
            gray[y0:y1, x0:x1], d.interior, d.text_boxes, d.crop, inpaint_cfg
        )
        d.text_mask, d.background_uniform, d.clean_coverage = glyphs, uniform, coverage
        # Every dark mark enclosed by a flat bubble background is text. Artwork entering
        # through an open outline touches the crop border, so it is never part of the
        # interior (see container_interior) and cannot be erased here.
        light_container = d.container is not None and not any(
            n.startswith("sin fondo claro") for n in d.notes
        )
        if uniform and light_container:
            # Large characters that touch the outline are not enclosed by the background, so
            # they fall outside the interior. For (nearly) convex containers - boxes, ovals -
            # the convex hull recovers them while its eroded edge still spares the outline.
            # Spiky shout bubbles keep the plain interior: their hull covers artwork.
            area = d.interior
            join = inpaint_cfg.join_text_areas
            if join:
                gray_crop = gray[y0:y1, x0:x1]
                area = area | text_areas(
                    gray_crop, d.crop, d.text_boxes, inpaint_cfg.text_box_grow_px
                )
            core = _bodies(area) if join else _main_body(d.interior)
            hull = _convex_hull(core, every_part=join)
            body = core
            if join:
                # The line of text that splits the areas counts as part of the bubble.
                body = core | (_box_mask(core.shape, d.crop, d.text_boxes, 0) & hull)
            solidity = body.sum() / max(1, hull.sum())
            clean_area = hull if solidity > 0.92 else area
            d.text_mask, _, _, _ = text_pixels(
                gray[y0:y1, x0:x1], clean_area, d.text_boxes, d.crop, inpaint_cfg, True
            )
        source = rgb[y0:y1, x0:x1] if rgb is not None else gray[y0:y1, x0:x1][..., None]
        if light.any():
            med = [int(v) for v in np.median(source[light], axis=0).ravel()]
            r, g, b = med if len(med) == 3 else med * 3
            d.background_color = (r, g, b)
        bg_gray = int(np.median(gray[y0:y1, x0:x1][light])) if light.any() else 255
        d.polygon, is_rect = _polygon(d.interior, d.crop)
        if d.type == "text_on_art" and uniform and bg_gray > 200:
            # Text on a plain light area (caption without border): cleanable like a box.
            d.type = "narration_box"
        elif d.type == "speech_bubble" and is_rect and d.container is not None:
            d.type = "narration_box"
        if not uniform:
            d.notes.append("fondo no uniforme")

        # Anything a flat fill cannot clean goes to LaMa, with a mask suited to the case.
        d.needs_lama = d.type == "text_on_art" or not uniform or coverage < inpaint_cfg.min_coverage
        if d.needs_lama:
            d.text_mask = lama_mask(gray[y0:y1, x0:x1], d, inpaint_cfg, uniform)
    return drafts


def lama_mask(
    gray_crop: np.ndarray, d: RegionDraft, cfg: InpaintConfig, uniform: bool
) -> np.ndarray:
    """Mask for LaMa (crop coordinates).

    - Text over artwork or screentone: the whole text box (glyphs cannot be told apart from
      tone dots, and LaMa redraws the texture).
    - Lettering that crosses a bubble outline: its dark strokes inside the text boxes,
      outline pieces included (LaMa redraws the outline).
    """
    assert d.crop is not None
    x0, y0, x1, y1 = d.crop
    grow = cfg.text_box_grow_px
    boxes = np.zeros(gray_crop.shape, bool)
    for tb in d.text_boxes:
        bx0, by0 = max(0, tb[0] - x0 - grow), max(0, tb[1] - y0 - grow)
        bx1, by1 = min(x1 - x0, tb[2] - x0 + grow), min(y1 - y0, tb[3] - y0 + grow)
        boxes[by0:by1, bx0:bx1] = True
    if d.type == "text_on_art" or not uniform:
        mask = boxes
    else:
        light = gray_crop[d.interior] if d.interior is not None and d.interior.any() else gray_crop
        bg = float(np.percentile(light, 90))
        mask = boxes & (gray_crop < bg - cfg.text_contrast)
    dil = max(1, cfg.text_dilate_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)
