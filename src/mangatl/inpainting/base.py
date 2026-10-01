"""Inpainting interface: remove the original text from a page."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mangatl.models import Region


class Inpainter(ABC):
    @abstractmethod
    def clean(
        self, image: np.ndarray, text_labels: np.ndarray, regions: list[Region]
    ) -> np.ndarray:
        """Return a copy of `image` (HxW or HxWx3 uint8) with the text of `regions` removed.

        `text_labels` holds, for every pixel to clean, the `mask_label` of its region.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model memory."""
