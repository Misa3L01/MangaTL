"""Fit a text into the real shape of a bubble.

The usable area is the bubble interior shrunk by a margin (distance transform). Each line
gets the width available at its own height, so oval bubbles get shorter lines at the top
and bottom. The font size is the largest one (binary search) for which the words fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from mangatl.typesetting.hyphenation import Hyphenator

Measure = Callable[[str], float]


@lru_cache(maxsize=256)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@dataclass(frozen=True)
class Slot:
    y: float  # vertical center of the line (usable-area coordinates)
    left: int
    right: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def center(self) -> float:
        return (self.left + self.right) / 2


@dataclass(frozen=True)
class Layout:
    font_size: int
    lines: list[str]
    slots: list[Slot]
    fits: bool


class UsableArea:
    """Row-wise horizontal extent of the usable part of an interior mask."""

    def __init__(self, interior: np.ndarray, margin_ratio: float, min_margin: int = 2) -> None:
        mask = interior.astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            self.mask = mask.astype(bool)
            self.empty = True
            return
        short_side = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        margin = max(min_margin, margin_ratio * short_side)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        usable = dist > margin
        if not usable.any():
            usable = dist >= dist.max() * 0.5
        self.mask = usable
        uy, ux = np.nonzero(usable)
        self.empty = False
        self.top, self.bottom = int(uy.min()), int(uy.max())
        self.cy = float(uy.mean())
        self.cx = float(ux.mean())
        self.area = int(usable.sum())
        self._runs: dict[int, tuple[int, int] | None] = {}

    @property
    def height(self) -> int:
        return 0 if self.empty else self.bottom - self.top + 1

    def run_at(self, y: int) -> tuple[int, int] | None:
        """Contiguous usable run on row `y` that contains (or is closest to) the center."""
        if y in self._runs:
            return self._runs[y]
        run = None
        if 0 <= y < self.mask.shape[0]:
            row = self.mask[y]
            xs = np.flatnonzero(row)
            if xs.size:
                cx = round(self.cx)
                breaks = np.flatnonzero(np.diff(xs) > 1)
                starts = np.concatenate(([xs[0]], xs[breaks + 1]))
                ends = np.concatenate((xs[breaks], [xs[-1]]))
                inside = np.flatnonzero((starts <= cx) & (ends >= cx))
                k = (
                    inside[0]
                    if inside.size
                    else int(np.argmin(np.minimum(abs(starts - cx), abs(ends - cx))))
                )
                run = (int(starts[k]), int(ends[k]) + 1)
        self._runs[y] = run
        return run

    def slots(self, n_lines: int, line_height: float) -> list[Slot] | None:
        block = n_lines * line_height
        if self.empty or block > self.height + line_height * 0.35:
            return None
        top = self.cy - block / 2
        top = min(
            max(top, self.top - line_height * 0.15), self.bottom + 1 - block + line_height * 0.15
        )
        slots = []
        for i in range(n_lines):
            y0 = top + i * line_height
            # Only the core of the line (x-height band) must be inside the usable area.
            rows = range(math.floor(y0 + 0.2 * line_height), math.ceil(y0 + 0.8 * line_height))
            left, right = -1, 1 << 30
            for y in rows:
                run = self.run_at(y)
                if run is None:
                    return None
                left, right = max(left, run[0]), min(right, run[1])
            # All lines share one vertical axis (like hand lettering): the slot is the widest
            # span centered on the axis that stays inside the usable area.
            half = min(self.cx - left, right - self.cx)
            if 2 * half < line_height * 0.8:
                return None
            slots.append(
                Slot(
                    y=y0 + line_height / 2, left=round(self.cx - half), right=round(self.cx + half)
                )
            )
        return slots


def wrap(
    words: list[str], widths: list[float], measure: Measure, hyphenator: Hyphenator | None
) -> list[str] | None:
    """Greedy fill of `words` into lines of the given widths; None if they do not fit."""
    queue = list(words)
    lines: list[str] = []
    for width in widths:
        if not queue:
            break
        line = ""
        while queue:
            word = queue[0]
            candidate = f"{line} {word}" if line else word
            if measure(candidate) <= width:
                line = candidate
                queue.pop(0)
                continue
            if hyphenator is not None:
                for head, tail in hyphenator.splits(word):
                    trial = f"{line} {head}" if line else head
                    if measure(trial) <= width:
                        line = trial
                        queue[0] = tail
                        break
            break
        if not line:
            return None
        lines.append(line)
    return lines if not queue else None


class TextFitter:
    def __init__(
        self,
        font_path: Path,
        line_spacing: float = 1.12,
        hyphenator: Hyphenator | None = None,
        hyphen_gain: float = 1.12,
    ) -> None:
        self.font_path = str(font_path)
        self.line_spacing = line_spacing
        self.hyphenator = hyphenator
        # Hyphenate only when it allows a font at least this much larger than without it.
        self.hyphen_gain = hyphen_gain

    def measure_for(self, size: int) -> Measure:
        font = load_font(self.font_path, size)
        return font.getlength

    def try_size(
        self, text: str, area: UsableArea, size: int, hyphenator: Hyphenator | None = None
    ) -> Layout | None:
        words = text.split()
        if not words:
            return Layout(size, [], [], True)
        measure = self.measure_for(size)
        line_h = size * self.line_spacing
        max_lines = max(1, int((area.height + line_h * 0.35) // line_h))
        for n in range(1, max_lines + 1):
            slots = area.slots(n, line_h)
            if slots is None:
                continue
            lines = wrap(words, [s.width for s in slots], measure, hyphenator)
            if lines is None or len(lines) != n:
                continue
            balanced = self._balance(words, slots, measure, size, hyphenator, lines)
            return balanced or Layout(size, lines, slots, True)
        return None

    def _balance(
        self,
        words: list[str],
        slots: list[Slot],
        measure: Measure,
        size: int,
        hyphenator: Hyphenator | None,
        greedy: list[str],
    ) -> Layout | None:
        """Narrow all lines as much as possible without adding lines or hyphens."""
        hyphens = sum(line.endswith("-") for line in greedy)
        lo, hi, best = 0.45, 1.0, None
        for _ in range(8):
            k = (lo + hi) / 2
            lines = wrap(words, [s.width * k for s in slots], measure, hyphenator)
            if (
                lines is not None
                and len(lines) == len(slots)
                and sum(line.endswith("-") for line in lines) <= hyphens
            ):
                best, hi = lines, k
            else:
                lo = k
        if best is None:
            return None
        return Layout(size, best, slots, True)

    def _search(
        self, text: str, area: UsableArea, lo: int, hi: int, hyphenator: Hyphenator | None
    ) -> Layout | None:
        best = self.try_size(text, area, lo, hyphenator)
        if best is None:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            layout = self.try_size(text, area, mid, hyphenator)
            if layout is not None:
                best, lo = layout, mid
            else:
                hi = mid - 1
        return best

    def fit(self, text: str, area: UsableArea, min_size: int, max_size: int) -> Layout:
        """Largest size in [min_size, max_size] that fits; fits=False if even min_size fails."""
        hi = max(min_size, max_size)
        plain = self._search(text, area, min_size, hi, None)
        hyph = self._search(text, area, min_size, hi, self.hyphenator) if self.hyphenator else None
        if plain is not None and (
            hyph is None or hyph.font_size < plain.font_size * self.hyphen_gain
        ):
            return plain
        if hyph is not None:
            return hyph
        return Layout(min_size, [], [], False)


def estimate_capacity(area: UsableArea, base_size: float, line_spacing: float) -> int:
    """Rough number of characters that fit at the base font size (given to the translator)."""
    if area.empty:
        return 0
    char_area = (base_size * 0.52) * (base_size * line_spacing)
    return max(4, int(area.area / char_area * 0.8))
