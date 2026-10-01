"""Project file round-trip and the clean -> letter -> export stages on a synthetic page."""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mangatl.config import Settings
from mangatl.models import BBox, ChapterMeta, Page, Project, Region
from mangatl.project_io import ProjectPaths, load_project, project_file_for, save_project, slugify
from mangatl.stages.render import (
    normalize_text,
    run_export,
    run_inpaint,
    run_typeset,
    should_render,
)


def test_slugify_and_project_file_name(tmp_path: Path) -> None:
    assert slugify("Black Jack ni Yoroshiku") == "black-jack-ni-yoroshiku"
    assert slugify("  ¿?  ") == "capitulo"
    assert project_file_for(tmp_path, "Mi Serie", "12").name == "mi-serie-12.mangatl.json"


def test_project_round_trip(tmp_path: Path, small_project: Project) -> None:
    path = tmp_path / "p.mangatl.json"
    save_project(small_project, path)
    assert load_project(path) == small_project
    assert not list(tmp_path.glob(".tmp-*")), "atomic write leaves no temp files"


def test_normalize_text() -> None:
    assert normalize_text("¡Hola！  ¿Qué...?") == "¡Hola! ¿Qué…?"


def _synthetic_project(tmp_path: Path) -> tuple[Project, ProjectPaths]:
    paths = ProjectPaths(tmp_path / "t.mangatl.json")
    for d in (paths.pages, paths.masks):
        d.mkdir(parents=True)
    page_img = np.full((800, 600), 230, np.uint8)
    page_img[::4, ::4] = 40  # screentone
    cv2.ellipse(page_img, (300, 400), (180, 250), 0, 0, 360, 255, -1)
    cv2.ellipse(page_img, (300, 400), (180, 250), 0, 0, 360, 0, 3)
    cv2.rectangle(page_img, (280, 300), (320, 500), 0, -1)  # "text"
    Image.fromarray(page_img).save(paths.pages / "0001.png")

    interior = np.zeros_like(page_img)
    cv2.ellipse(interior, (300, 400), (177, 247), 0, 0, 360, 1, -1)
    glyphs = np.zeros_like(page_img)
    glyphs[297:503, 277:323] = 1
    Image.fromarray(interior).save(paths.masks / "0001_bubbles.png")
    Image.fromarray(glyphs).save(paths.masks / "0001_text.png")

    region = Region(
        id="P001-B01",
        type="speech_bubble",
        bbox=BBox(x0=280, y0=300, x1=320, y1=500),
        bubble_bbox=BBox(x0=116, y0=146, x1=484, y1=654),
        mask_label=1,
        translation="¡Solo he dormido dos horas en toda la noche!",
    )
    page = Page(
        number=1,
        source_name="pagina_01",
        image_path="work/pages/0001.png",
        width=600,
        height=800,
        grayscale=True,
        bubble_mask_path="work/masks/0001_bubbles.png",
        text_mask_path="work/masks/0001_text.png",
        regions=[region],
    )
    return Project(meta=ChapterMeta(series="t", chapter="1"), pages=[page]), paths


def test_clean_letter_and_export(tmp_path: Path, project_root: Path) -> None:
    settings = Settings(root=project_root)
    project, paths = _synthetic_project(tmp_path)
    run_inpaint(project, paths, settings, project.pages)
    run_typeset(project, paths, settings, project.pages)
    details = run_export(project, paths, settings)

    original = np.asarray(Image.open(paths.pages / "0001.png"))
    clean = np.asarray(Image.open(paths.abs(project.pages[0].clean_path)))
    rendered = np.asarray(Image.open(paths.abs(project.pages[0].rendered_path)))
    inside = np.asarray(Image.open(paths.masks / "0001_bubbles.png")) > 0

    assert clean[300:500, 280:320].min() >= 250, "original text removed"
    assert np.array_equal(clean[~inside], original[~inside]), "art outside the bubble untouched"
    assert np.array_equal(rendered[~inside], original[~inside]), "lettering stays inside the bubble"
    assert (rendered[inside] < 100).sum() > 500, "translation was drawn"
    region = project.pages[0].regions[0]
    assert region.fits and region.font_size

    assert details["pages"] == 1
    assert (paths.output_pages / "pagina_01.png").is_file()
    with zipfile.ZipFile(paths.cbz) as zf:
        assert zf.namelist() == ["pagina_01.png"]
    import pymupdf

    with pymupdf.open(paths.pdf) as pdf:
        assert pdf.page_count == 1
        assert (pdf[0].rect.width, pdf[0].rect.height) == (600, 800)


def test_find_overflows_gives_a_budget_for_a_shorter_version(
    tmp_path: Path, project_root: Path
) -> None:
    from mangatl.stages.render import find_overflows

    settings = _big_font_settings(project_root)
    project, paths = _synthetic_project(tmp_path)
    region = project.pages[0].regions[0]
    assert find_overflows(project, paths, settings, project.pages) == []

    region.translation = " ".join(["Esta traducción es muchísimo más larga de lo que cabe"] * 12)
    [(found, budget)] = find_overflows(project, paths, settings, project.pages)
    assert found is region
    assert 4 <= budget < len(region.translation) * 0.8
    assert region.fits is None, "the dry run must not modify the real region"


def _big_font_settings(project_root: Path) -> Settings:
    """The synthetic page is small (800 px): raise the minimum size so long text overflows."""
    return Settings(root=project_root, typesetting={"min_font_px": 40, "max_font_px": 64})


def test_shorter_alternative_is_used_when_the_translation_does_not_fit(
    tmp_path: Path, project_root: Path
) -> None:
    settings = _big_font_settings(project_root)
    project, paths = _synthetic_project(tmp_path)
    region = project.pages[0].regions[0]
    region.translation = " ".join(["Esta traducción es muchísimo más larga de lo que cabe"] * 12)
    region.shorter_alternative = "¡Dormí dos horas!"
    run_inpaint(project, paths, settings, project.pages)
    run_typeset(project, paths, settings, project.pages)
    assert region.used_shorter and region.fits and region.status == "auto"


def test_a_cleaned_bubble_never_stays_empty(tmp_path: Path, project_root: Path) -> None:
    """Even when nothing fits, the translation is drawn (flagged for review), never blank."""
    settings = Settings(root=project_root, typesetting={"min_font_px": 60, "max_font_px": 80})
    project, paths = _synthetic_project(tmp_path)
    region = project.pages[0].regions[0]
    region.translation = " ".join(["Una traducción larguísima que no cabe de ninguna manera"] * 10)
    run_inpaint(project, paths, settings, project.pages)
    run_typeset(project, paths, settings, project.pages)
    rendered = np.asarray(Image.open(paths.abs(project.pages[0].rendered_path)))
    inside = np.asarray(Image.open(paths.masks / "0001_bubbles.png")) > 0
    assert region.fits is False and region.status == "needs_review"
    assert (rendered[inside] < 100).sum() > 500, "text was drawn despite the overflow"


def test_punctuation_only_translation_keeps_the_original() -> None:
    from mangatl.stages.render import should_render

    base = {"id": "P001-B01", "type": "speech_bubble", "bbox": BBox(x0=0, y0=0, x1=1, y1=1)}
    assert not should_render(Region(**base, ocr_text="でした！！", translation="¡"))
    assert should_render(Region(**base, ocr_text="……", translation="…"))


def test_unreliable_text_on_art_keeps_the_original() -> None:
    from mangatl.stages.render import should_render

    base = {
        "id": "P001-B01",
        "type": "text_on_art",
        "bbox": BBox(x0=0, y0=0, x1=1, y1=1),
        "clean_method": "lama",
        "translation": "Clínica Wafu",
    }
    assert not should_render(Region(**base, ocr_confidence=0.3))
    assert should_render(Region(**base, ocr_confidence=0.95))


def test_regions_without_translation_or_uncleanable_are_not_touched() -> None:
    base = {"id": "P001-B01", "type": "speech_bubble", "bbox": BBox(x0=0, y0=0, x1=1, y1=1)}
    assert not should_render(Region(**base))
    assert not should_render(Region(**base, translation="Hola", cleanable=False))
    assert not should_render(Region(**base, translation="Hola", status="skipped"))
    assert not should_render(Region(**{**base, "type": "text_on_art"}, translation="Hola"))
    assert should_render(Region(**base, translation="Hola", status="needs_review"))
