from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mangatl.config import Settings, load_settings


def test_defaults_match_user_choices(settings: Settings) -> None:
    tr = settings.translator
    assert tr.backend == "ollama"
    assert tr.target_variant == "es-419"
    assert tr.honorifics == "keep"
    assert tr.sfx_mode == "annotate"
    assert tr.ollama.model == "qwen3.5:9b"
    assert tr.ollama.think is False
    assert settings.typesetting.fonts.normal.name == "ComicNeue-Bold.ttf"
    assert settings.typesetting.fonts.shout.name == "Bangers-Regular.ttf"
    assert settings.setup.min_free_disk_gb == 3.0


def test_toml_file_overrides_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[translator]\ntarget_variant = "es-MX"\n'
        '[translator.ollama]\nmodel = "translategemma:4b"\n',
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert s.translator.target_variant == "es-MX"
    assert s.translator.ollama.model == "translategemma:4b"
    assert s.translator.honorifics == "keep"  # untouched keys keep their default


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[translator.ollama]\nmodel = "from-toml"\n', encoding="utf-8")
    monkeypatch.setenv("MANGATL_TRANSLATOR__OLLAMA__MODEL", "from-env")
    assert load_settings(cfg).translator.ollama.model == "from-env"


def test_invalid_variant_is_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[translator]\ntarget_variant = "pt-BR"\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_missing_explicit_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml")


def test_resolve_relative_to_root(settings: Settings, tmp_path: Path) -> None:
    assert settings.resolve(Path("models")) == (tmp_path / "models").resolve()
    absolute = tmp_path / "elsewhere"
    assert settings.resolve(absolute) == absolute


def test_stale_config_is_detected(tmp_path: Path, project_root: Path) -> None:
    from mangatl.bootstrap import missing_config_keys

    old = tmp_path / "config.toml"
    old.write_text("[translator.ollama]\nnum_ctx = 8192\n", encoding="utf-8")
    missing = missing_config_keys(old, project_root / "config.example.toml")
    assert "translator.ollama.model" in missing and "inpaint.min_coverage" in missing
    assert "translator.ollama.num_ctx" not in missing


def test_example_config_matches_code_defaults(project_root: Path) -> None:
    """config.example.toml must document exactly the defaults the code uses."""
    from_example = load_settings(project_root / "config.example.toml")
    defaults = Settings()
    assert from_example.model_dump(exclude={"root"}) == defaults.model_dump(exclude={"root"})
