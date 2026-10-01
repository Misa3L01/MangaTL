"""Series glossary and summaries (series/<serie>/), and the `mangatl glossary` commands."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mangatl.config import Settings
from mangatl.models import GlossaryEntry, Project
from mangatl.runtime_env import runtime_env_vars
from mangatl.translation import glossary as g

SERIES = "Black Jack ni Yoroshiku"


def entry(source: str, target: str, status: str = "pending") -> GlossaryEntry:
    return GlossaryEntry(source=source, target=target, category="character", status=status)


def test_pending_entries_are_not_sent_until_approved(settings: Settings) -> None:
    assert g.add_pending(settings, SERIES, [entry("斉藤", "Saitō"), entry("牛田", "Ushida")]) == 2
    assert g.load_glossary(settings, SERIES) == []
    assert g.approve(settings, SERIES, ["斉藤"]) == ["斉藤"]
    assert [e.source for e in g.load_glossary(settings, SERIES)] == ["斉藤"]
    assert g.approve(settings, SERIES, None) == ["牛田"]


def test_existing_terms_are_not_duplicated_or_downgraded(settings: Settings) -> None:
    g.upsert(settings, SERIES, "斉藤", "Saitō", "character", "Habla con cortesía")
    assert g.add_pending(settings, SERIES, [entry("斉藤", "Saito")]) == 0
    [e] = g.read_glossary(settings, SERIES)
    assert (e.target, e.status, e.notes) == ("Saitō", "approved", "Habla con cortesía")


def test_upsert_edits_and_requires_target_for_new_terms(settings: Settings) -> None:
    g.add_pending(settings, SERIES, [entry("永大", "Universidad de Yōda")])
    edited = g.upsert(settings, SERIES, "永大", target="Eidai", category="place")
    assert (edited.target, edited.category, edited.status) == ("Eidai", "place", "approved")
    with pytest.raises(ValueError):
        g.upsert(settings, SERIES, "新しい")


def test_remove(settings: Settings) -> None:
    g.upsert(settings, SERIES, "斉藤", "Saitō")
    assert g.remove(settings, SERIES, "斉藤")
    assert not g.remove(settings, SERIES, "斉藤")


def test_summaries_are_saved_and_the_last_ones_loaded(settings: Settings) -> None:
    for ch in ("1", "2", "10", "3"):
        g.save_summary(settings, SERIES, ch, f"Resumen {ch}")
    assert g.load_previous_summaries(settings, SERIES, last=2) == [
        "Capítulo 3: Resumen 3",
        "Capítulo 10: Resumen 10",
    ]
    assert g.load_previous_summaries(settings, SERIES, last=5, before="3") == [
        "Capítulo 1: Resumen 1",
        "Capítulo 2: Resumen 2",
    ]


def test_remember_chapter(settings: Settings, small_project: Project) -> None:
    small_project.chapter_summary = "Saitō empieza su residencia."
    small_project.pending_glossary = [entry("斉藤", "Saitō")]
    assert g.remember_chapter(settings, small_project) == 1
    assert g.load_previous_summaries(settings, small_project.meta.series) == [
        "Capítulo 1: Saitō empieza su residencia."
    ]


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in runtime_env_vars(Settings(root=tmp_path)):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("MANGATL_ROOT", str(tmp_path))
    monkeypatch.delenv("MANGATL_CONFIG", raising=False)
    return tmp_path


def test_glossary_cli_flow(cli_env: Path) -> None:
    from mangatl.cli import app

    settings = Settings(root=cli_env)
    g.add_pending(settings, SERIES, [entry("斉藤", "Saitō"), entry("牛田", "Uchida")])
    runner = CliRunner()

    listed = runner.invoke(app, ["glossary", "list", "--series", SERIES, "--pending"])
    assert listed.exit_code == 0 and "斉藤" in listed.output and "pendiente" in listed.output

    fixed = runner.invoke(
        app, ["glossary", "edit", "--series", SERIES, "牛田", "--target", "Ushida"]
    )
    assert fixed.exit_code == 0, fixed.output
    approved = runner.invoke(app, ["glossary", "approve", "--series", SERIES, "--all"])
    assert approved.exit_code == 0 and "斉藤" in approved.output
    assert {e.source: e.target for e in g.load_glossary(settings, SERIES)} == {
        "斉藤": "Saitō",
        "牛田": "Ushida",
    }

    missing = runner.invoke(app, ["glossary", "remove", "--series", SERIES, "存在しない"])
    assert missing.exit_code == 1
