"""End-to-end on real fixture pages without Ollama.

translate --backend manual (ingest, detection, OCR in a child process, prompt export)
-> import a fake pasted answer -> render (clean, letter, export PNG + CBZ).
Needs the downloaded models (uv run mangatl setup); skipped otherwise.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest
from rich.console import Console

from mangatl.config import load_settings
from mangatl.model_store import ModelNotDownloadedError, detector_dir, ja_ocr_dir
from mangatl.pipeline import Pipeline, TranslateOptions
from mangatl.project_io import load_project
from mangatl.runtime_env import apply_runtime_env
from mangatl.translation.manual_backend import import_translation

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bj"


def _models_ready() -> bool:
    settings = load_settings()
    apply_runtime_env(settings)
    try:
        detector_dir(settings)
        ja_ocr_dir(settings)
    except ModelNotDownloadedError:
        return False
    return True


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _models_ready(), reason="modelos no descargados (mangatl setup)"),
]


def test_full_pipeline_with_manual_backend(tmp_path: Path) -> None:
    chapter = tmp_path / "capitulo"
    chapter.mkdir()
    for f in sorted(FIXTURES.glob("*.png")):
        shutil.copy(f, chapter / f.name)
    out = tmp_path / "salida"
    settings = load_settings()
    with (tmp_path / "log.txt").open("w", encoding="utf-8") as log_file:
        _run(settings, Console(file=log_file), chapter, out)


def _run(settings, console: Console, chapter: Path, out: Path) -> None:
    pipeline = Pipeline(settings, console)
    project_file = pipeline.translate(
        TranslateOptions(
            chapter, "Black Jack ni Yoroshiku", "1", backend="manual", out_dir=out, debug=True
        )
    )
    project = load_project(project_file)
    assert [p.number for p in project.pages] == [1, 2, 3]
    assert project.stages["translate"].status == "partial"
    assert (out / "prompts" / "parte-1.txt").is_file()
    assert list((out / "work" / "debug").glob("*_1_regiones.png"))

    regions = [r for _, r in project.regions()]
    assert len(regions) >= 15
    speech = [r for r in regions if r.type in ("speech_bubble", "narration_box") and r.cleanable]
    assert len(speech) >= 10
    assert sum(1 for r in speech if r.ocr_text and (r.ocr_confidence or 0) > 0.9) >= 10
    assert all(r.id.startswith(f"P{p.number:03d}-") for p in project.pages for r in p.regions)
    # Page 2 (fixture bj_009) holds the rectangular narration boxes of the chapter.
    assert sum(1 for r in project.pages[1].regions if r.type == "narration_box") >= 4

    answer = {
        "regions": [
            {"id": r.id, "translation": f"Traducción de prueba número {i} para este globo."}
            for i, r in enumerate(regions)
        ]
    }
    report = import_translation(project, f"```json\n{json.dumps(answer)}\n```")
    assert not report.missing
    from mangatl.project_io import save_project

    save_project(project, project_file)

    pipeline.render(project_file, None, debug_images=False)
    project = load_project(project_file)
    rendered = [r for _, r in project.regions() if r.font_size]
    assert len(rendered) >= 10
    assert sum(1 for r in rendered if r.fits) >= 0.8 * len(rendered)
    assert sorted(p.name for p in (out / "pages").glob("*.png")) == [
        "bj_007.png",
        "bj_009.png",
        "bj_020.png",
    ]
    with zipfile.ZipFile(
        project_file.with_name(project_file.name.replace(".mangatl.json", ".cbz"))
    ) as zf:
        assert len(zf.namelist()) == 3
