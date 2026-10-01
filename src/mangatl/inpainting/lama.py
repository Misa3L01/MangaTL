"""LaMa inpainting (anime-manga-big-lama: big-lama finetuned on manga; TorchScript by IOPaint).

Used where a flat fill cannot work: text on artwork or screentone, and lettering that crosses
a bubble outline. Each region is processed on a crop with surrounding context (resized to at
most `max_side`), and only the masked pixels are pasted back.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from mangatl.gpu import release_cuda_memory
from mangatl.inpainting.base import Inpainter
from mangatl.models import Region

log = logging.getLogger(__name__)


class LamaInpainter(Inpainter):
    def __init__(
        self, model_path: Path, device: str = "cuda", max_side: int = 1024, context: float = 0.6
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.max_side = max_side
        self.context = context
        self.model = torch.jit.load(str(model_path), map_location="cpu").to(device).eval()

    def _inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = mask.shape
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            size = (max(8, round(w * scale)), max(8, round(h * scale)))
            rgb_in = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            mask_in = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        else:
            rgb_in, mask_in = rgb, mask.astype(np.uint8)
        ih, iw = mask_in.shape
        ph, pw = (-ih) % 8, (-iw) % 8  # LaMa needs sides divisible by 8
        rgb_in = np.pad(rgb_in, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask_in = np.pad(mask_in, ((0, ph), (0, pw)), mode="constant")
        image_t = torch.from_numpy(rgb_in).permute(2, 0, 1)[None].float().div(255).to(self.device)
        mask_t = torch.from_numpy((mask_in > 0).astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(image_t, mask_t)
        result = (out[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy())[:ih, :iw]
        if scale < 1.0:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
        return result

    def clean(
        self, image: np.ndarray, text_labels: np.ndarray, regions: list[Region]
    ) -> np.ndarray:
        gray = image.ndim == 2
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if gray else image.copy()
        page_h, page_w = text_labels.shape
        for region in regions:
            mask = text_labels == region.mask_label
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            pad = int(self.context * max(y1 - y0, x1 - x0)) + 16
            cy0, cy1 = max(0, y0 - pad), min(page_h, y1 + pad)
            cx0, cx1 = max(0, x0 - pad), min(page_w, x1 + pad)
            crop_mask = mask[cy0:cy1, cx0:cx1]
            try:
                filled = self._inpaint_crop(rgb[cy0:cy1, cx0:cx1], crop_mask)
            except self._torch.cuda.OutOfMemoryError:
                log.warning("%s: sin VRAM para LaMa; se reintenta a menor resolución", region.id)
                release_cuda_memory()
                self.max_side = max(256, self.max_side // 2)
                filled = self._inpaint_crop(rgb[cy0:cy1, cx0:cx1], crop_mask)
            target = rgb[cy0:cy1, cx0:cx1]
            target[crop_mask] = filled[crop_mask]
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray else rgb
        return out

    def close(self) -> None:
        del self.model
        release_cuda_memory()
