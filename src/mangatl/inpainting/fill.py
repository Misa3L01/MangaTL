"""Uniform-background cleaning: paint glyph pixels with the bubble's background color.

The glyph mask was built inside the eroded bubble interior, so outlines and artwork outside
the mask are never modified. Textured backgrounds need LaMa (Phase 2).
"""

from __future__ import annotations

import numpy as np

from mangatl.inpainting.base import Inpainter
from mangatl.models import Region


class FillInpainter(Inpainter):
    def clean(
        self, image: np.ndarray, text_labels: np.ndarray, regions: list[Region]
    ) -> np.ndarray:
        out = image.copy()
        for region in regions:
            if region.mask_label <= 0:
                continue
            mask = text_labels == region.mask_label
            if not mask.any():
                continue
            if out.ndim == 2:
                out[mask] = region.background_color[0]
            else:
                out[mask] = np.array(region.background_color, dtype=out.dtype)
        return out
