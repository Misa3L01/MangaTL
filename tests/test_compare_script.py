"""scripts/compare_models.py helpers: per-run setting overrides and GPU layer parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mangatl.config import Settings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py"


def _script():
    spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_overrides_parse_toml_values_and_bare_strings(tmp_path: Path) -> None:
    cm = _script()
    base = Settings(root=tmp_path)
    s = cm.apply_overrides(
        base,
        [
            "translator.pages_per_block=2",
            "translator.pivot_english=true",
            "translator.ollama.model=qwen3.5:4b",
            'translator.ollama.kv_cache_type="f16"',
        ],
    )
    assert s.translator.pages_per_block == 2 and s.translator.pivot_english is True
    assert s.translator.ollama.model == "qwen3.5:4b"
    assert s.translator.ollama.kv_cache_type == "f16"
    assert base.translator.pages_per_block == 4  # the original is untouched


def test_overrides_reject_unknown_keys(tmp_path: Path) -> None:
    cm = _script()
    with pytest.raises(SystemExit):
        cm.apply_overrides(Settings(root=tmp_path), ["translator.nope=1"])
    with pytest.raises(SystemExit):
        cm.apply_overrides(Settings(root=tmp_path), ["translator.pages_per_block"])


def test_gpu_layers_reads_the_last_load() -> None:
    cm = _script()
    log = (
        "load_tensors: offloaded 34/34 layers to GPU\n...\n"
        "load_tensors: offloaded 18/34 layers to GPU\n"
    )
    assert cm.gpu_layers(log) == "18/34"
    assert cm.gpu_layers("nada") is None
