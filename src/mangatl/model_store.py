"""Locate downloaded Hugging Face model snapshots (offline, inside models/hf)."""

from __future__ import annotations

from pathlib import Path

from mangatl.config import Settings

OCR_PATTERNS = ("*.json", "*.bin", "*.safetensors", "*.txt")


class ModelNotDownloadedError(RuntimeError):
    pass


def local_snapshot(repo: str, revision: str, patterns: tuple[str, ...] | list[str]) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo, revision=revision, allow_patterns=list(patterns), local_files_only=True
            )
        )
    except Exception as exc:
        raise ModelNotDownloadedError(
            f"El modelo {repo} no está descargado. Ejecuta: uv run mangatl setup"
        ) from exc


def detector_dir(settings: Settings) -> Path:
    d = settings.detection
    return local_snapshot(d.model_repo, d.model_revision, d.model_files)


def lama_path(settings: Settings) -> Path:
    path = settings.resolve(settings.inpaint.lama_file)
    if not path.is_file():
        raise ModelNotDownloadedError(
            "El modelo LaMa no está descargado. Ejecuta: uv run mangatl setup "
            "(o desactívalo con inpaint.use_lama = false)"
        )
    return path


def ja_ocr_dir(settings: Settings) -> Path:
    o = settings.ocr
    return local_snapshot(o.ja_model_repo, o.ja_model_revision, OCR_PATTERNS)
