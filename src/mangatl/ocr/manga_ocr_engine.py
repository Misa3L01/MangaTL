"""Japanese OCR with manga-ocr (kha-white/manga-ocr-base, Apache-2.0).

Uses the model directly instead of the `MangaOcr` wrapper to batch crops and to obtain a
confidence score from the generation probabilities.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from mangatl.gpu import release_cuda_memory
from mangatl.ocr.base import OcrEngine, OcrResult


class MangaOcrEngine(OcrEngine):
    def __init__(self, model_dir: Path, device: str = "cuda", batch_size: int = 16) -> None:
        import torch
        from manga_ocr.ocr import MangaOcrModel, post_process
        from transformers import AutoTokenizer, ViTImageProcessor

        self._torch = torch
        self._post_process = post_process
        self.device = device
        self.batch_size = batch_size
        self.processor = ViTImageProcessor.from_pretrained(model_dir)
        # Explicit tokenizer type: transformers>=5.13 misdetects it (same workaround as manga-ocr).
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, tokenizer_type="bert-japanese")
        self.model = MangaOcrModel.from_pretrained(model_dir).to(device).eval()
        self._skip_ids = {
            i
            for i in (
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.sep_token_id,
            )
            if i is not None
        }

    def read(self, crops: list[Image.Image]) -> list[OcrResult]:
        results: list[OcrResult] = []
        for start in range(0, len(crops), self.batch_size):
            results.extend(self._read_batch(crops[start : start + self.batch_size]))
        return results

    def _read_batch(self, crops: list[Image.Image]) -> list[OcrResult]:
        torch = self._torch
        images = [c.convert("L").convert("RGB") for c in crops]
        pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                pixel_values,
                max_length=300,
                output_scores=True,
                return_dict_in_generate=True,
            )
            scores = self.model.compute_transition_scores(
                out.sequences,
                out.scores,
                beam_indices=getattr(out, "beam_indices", None),
                normalize_logits=True,
            )
        sequences = out.sequences[:, -scores.shape[1] :].cpu().tolist()
        scores_list = scores.float().cpu().tolist()
        results = []
        for full, tokens, logps in zip(
            out.sequences.cpu().tolist(), sequences, scores_list, strict=True
        ):
            kept = [lp for tok, lp in zip(tokens, logps, strict=True) if tok not in self._skip_ids]
            finite = [lp for lp in kept if math.isfinite(lp)]
            confidence = math.exp(sum(finite) / len(finite)) if finite else 0.0
            text = self._post_process(self.tokenizer.decode(full, skip_special_tokens=True))
            results.append(OcrResult(text=text, confidence=round(confidence, 4)))
        return results

    def close(self) -> None:
        del self.model
        release_cuda_memory()
