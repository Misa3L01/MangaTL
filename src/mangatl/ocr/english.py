"""English OCR for comic lettering: PaddleOCR's PP-OCR models through RapidOCR.

RapidOCR (Apache-2.0) runs the PP-OCR detection/recognition models with onnxruntime on the
CPU, without installing PaddlePaddle. It was chosen over EasyOCR after a comparison on the
English edition of Black Jack ni Yoroshiku: near-perfect on all-caps lettering (EasyOCR
mixed up case and read "A" as "4"). The multilingual model also reads Japanese signage,
which `--src auto` uses to detect the source language.

Text blocks are already located by the detector; the lines found inside each block are
joined in reading order, gluing words split with a hyphen ("CONVIC-" + "TION").
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from mangatl.ocr.base import OcrEngine, OcrResult


def join_lines(lines: list[tuple[float, float, str, float]]) -> tuple[str, float]:
    """Lines as (y_center, x_left, text, confidence) -> (paragraph, mean confidence)."""
    if not lines:
        return "", 0.0
    lines = sorted(lines, key=lambda t: (round(t[0] / 12), t[1]))
    text = ""
    for _, _, piece, _ in lines:
        piece = piece.strip()
        if not piece:
            continue
        if text.endswith("-") and len(text) > 1 and text[-2].isalpha() and piece[:1].isalpha():
            text = text[:-1] + piece
        else:
            text = f"{text} {piece}" if text else piece
    text = re.sub(r"\s+", " ", text).strip()
    conf = float(np.mean([c for *_, c in lines]))
    return text, round(conf, 4)


def _prepare(crop: Image.Image, min_height: int = 64) -> np.ndarray:
    """Grayscale RGB array, upscaled when the crop is tiny (small lettering)."""
    img = crop.convert("L").convert("RGB")
    if img.height < min_height:
        scale = min_height / img.height
        img = img.resize((round(img.width * scale), min_height), Image.Resampling.LANCZOS)
    return np.asarray(img)


class PaddleRapidOcrEngine(OcrEngine):
    def __init__(self) -> None:
        from rapidocr import LangRec, RapidOCR

        self.engine = RapidOCR(params={"Rec.lang_type": LangRec.EN, "Global.log_level": "error"})

    def read(self, crops: list[Image.Image]) -> list[OcrResult]:
        results = []
        for crop in crops:
            out = self.engine(_prepare(crop))
            lines = []
            if out is not None and out.txts:
                for box, text, conf in zip(out.boxes, out.txts, out.scores, strict=False):
                    ys = [p[1] for p in box]
                    lines.append(
                        (float(np.mean(ys)), float(min(p[0] for p in box)), text, float(conf))
                    )
            text, conf = join_lines(lines)
            results.append(OcrResult(text=text, confidence=conf))
        return results
