"""LaMa on a synthetic page: text over screentone is removed, the rest is untouched."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mangatl.config import load_settings
from mangatl.model_store import ModelNotDownloadedError, lama_path
from mangatl.models import BBox, Region

torch = pytest.importorskip("torch")


def _lama_file():
    try:
        return lama_path(load_settings())
    except ModelNotDownloadedError:
        return None


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_lama_file() is None, reason="LaMa no descargado (mangatl setup)"),
]


def test_lama_removes_text_on_screentone() -> None:
    from mangatl.inpainting.lama import LamaInpainter

    page = np.full((400, 400), 200, np.uint8)
    page[::4, ::4] = 90  # screentone
    clean_reference = page.copy()
    cv2.putText(page, "BAM", (110, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.4, 0, 12)
    labels = np.zeros_like(page)
    labels[160:250, 100:300] = 1
    region = Region(
        id="P001-B01", type="text_on_art", bbox=BBox(x0=100, y0=160, x1=300, y1=250), mask_label=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lama = LamaInpainter(_lama_file(), device=device)
    out = lama.clean(page, labels, [region])
    lama.close()

    inside = labels == 1
    assert (out[inside] < 40).mean() < 0.02, "the black lettering is gone"
    assert abs(float(out[inside].mean()) - float(clean_reference[inside].mean())) < 25
    assert np.array_equal(out[~inside], page[~inside]), "pixels outside the mask are untouched"
