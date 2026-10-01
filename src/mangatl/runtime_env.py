"""Process-level environment: keep every cache and temp file inside the project drive.

`env.ps1` sets the same variables for the whole shell session (uv needs them before Python
starts). This module re-applies them from Python so that `mangatl` never writes model
caches or temp files to C: even if `env.ps1` was not loaded. It must run before importing
torch, transformers or huggingface_hub.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from mangatl.config import Settings

log = logging.getLogger(__name__)


def runtime_env_vars(settings: Settings) -> dict[str, str]:
    root = settings.root
    models = settings.resolve(settings.paths.models_dir)
    tmp = settings.resolve(settings.paths.tmp_dir)
    local = root / ".local"
    return {
        "MANGATL_ROOT": str(root),
        "HF_HOME": str(models / "hf"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TORCH_HOME": str(models / "torch"),
        "XDG_CACHE_HOME": str(local / "xdg-cache"),
        "CUDA_CACHE_PATH": str(local / "nv-compute-cache"),
        "TEMP": str(tmp),
        "TMP": str(tmp),
    }


def apply_runtime_env(settings: Settings) -> None:
    for key, value in runtime_env_vars(settings).items():
        current = os.environ.get(key)
        if current and Path(current) != Path(value) and key not in ("TEMP", "TMP"):
            log.debug("Se reemplaza %s=%s por %s", key, current, value)
        os.environ[key] = value
    tmp = Path(os.environ["TEMP"])
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)


def paths_to_audit(settings: Settings) -> dict[str, Path]:
    """Locations that must live on the project drive (used by `mangatl setup`)."""
    import sys

    env = runtime_env_vars(settings)
    return {
        "Proyecto": settings.root,
        "Entorno virtual (.venv)": Path(sys.prefix),
        "Python base": Path(sys.base_prefix),
        "Hugging Face (HF_HOME)": Path(env["HF_HOME"]),
        "PyTorch (TORCH_HOME)": Path(env["TORCH_HOME"]),
        "Caché CUDA (CUDA_CACHE_PATH)": Path(env["CUDA_CACHE_PATH"]),
        "Temporales (TEMP)": Path(env["TEMP"]),
        "Ollama (binarios)": settings.resolve(settings.translator.ollama.install_dir),
        "Ollama (modelos)": settings.resolve(settings.translator.ollama.models_dir),
        "Caché de uv (UV_CACHE_DIR)": Path(os.environ.get("UV_CACHE_DIR", "(sin definir)")),
    }


def same_drive(path: Path, reference: Path) -> bool:
    return path.drive.upper() == reference.drive.upper()
