from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mangatl.config import Settings
from mangatl.runtime_env import apply_runtime_env, runtime_env_vars, same_drive


def test_apply_runtime_env_redirects_caches_into_project(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Record originals so monkeypatch restores them after apply_runtime_env mutates os.environ.
    for key in runtime_env_vars(settings):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)

    apply_runtime_env(settings)

    root = tmp_path.resolve()
    assert Path(os.environ["HF_HOME"]) == root / "models" / "hf"
    assert Path(os.environ["TORCH_HOME"]) == root / "models" / "torch"
    assert Path(os.environ["TEMP"]) == root / "tmp"
    assert Path(os.environ["CUDA_CACHE_PATH"]) == root / ".local" / "nv-compute-cache"
    assert (root / "tmp").is_dir()
    assert Path(tempfile.gettempdir()) == root / "tmp"


def test_every_runtime_path_is_under_root(settings: Settings, tmp_path: Path) -> None:
    for key, value in runtime_env_vars(settings).items():
        if key.startswith(("HF_HUB_",)):
            continue
        assert Path(value).is_relative_to(tmp_path.resolve()), key


def test_same_drive() -> None:
    assert same_drive(Path("R:/a/b"), Path("r:/c"))
    assert not same_drive(Path("C:/Users"), Path("R:/Manga-Translate"))
