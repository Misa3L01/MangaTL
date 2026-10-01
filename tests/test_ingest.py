from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mangatl.ingest import ingest, is_effectively_gray, natural_key, open_source, parse_page_range
from mangatl.project_io import ProjectPaths


def _page(path: Path, size=(60, 90), color=255, mode="L") -> None:
    Image.new(mode, size, color).save(path)


def test_natural_key_orders_numbers_numerically() -> None:
    names = ["p10.png", "p2.png", "p1.png", "P3.png", "p1a.png"]
    assert sorted(names, key=natural_key) == ["p1.png", "p1a.png", "p2.png", "P3.png", "p10.png"]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("3-5", {3, 4, 5}), ("1,4-5, 9", {1, 4, 5, 9}), ("", None), (None, None)],
)
def test_parse_page_range(spec: str | None, expected: set[int] | None) -> None:
    assert parse_page_range(spec) == expected


@pytest.mark.parametrize("spec", ["0", "5-3", "x"])
def test_parse_page_range_rejects_invalid(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_page_range(spec)


def test_folder_source_natural_order_and_ignores_junk(tmp_path: Path) -> None:
    for name in ("10.png", "2.png", "1.jpg", "notes.txt"):
        if name.endswith("txt"):
            (tmp_path / name).write_text("x")
        else:
            _page(tmp_path / name)
    (tmp_path / ".hidden").mkdir()
    _page(tmp_path / ".hidden" / "0.png")
    with open_source(tmp_path) as src:
        assert [p.name for p in src.pages()] == ["1", "2", "10"]
        assert src.kind == "folder"


def test_archive_source_zip_and_cbz(tmp_path: Path) -> None:
    for suffix in (".zip", ".cbz"):
        archive = tmp_path / f"chapter{suffix}"
        with zipfile.ZipFile(archive, "w") as zf:
            for name in ("ch/10.png", "ch/2.png", "ch/1.png", "__MACOSX/ch/._1.png", "info.txt"):
                if name.endswith(".png"):
                    buf = tmp_path / "tmp.png"
                    _page(buf)
                    zf.write(buf, name)
                else:
                    zf.writestr(name, "x")
        with open_source(archive) as src:
            assert [p.name for p in src.pages()] == ["1", "2", "10"]
            assert src.load(src.pages()[0]).size == (60, 90)


def test_pdf_source_keeps_native_resolution(tmp_path: Path) -> None:
    import pymupdf

    pdf = tmp_path / "vol.pdf"
    doc = pymupdf.open()
    for shade in (40, 200):
        img_path = tmp_path / f"{shade}.png"
        _page(img_path, size=(300, 450), color=shade)
        page = doc.new_page(width=150, height=225)  # 2 px per point
        page.insert_image(page.rect, filename=str(img_path))
    doc.save(pdf)
    doc.close()

    with open_source(pdf) as src:
        refs = src.pages()
        assert [r.name for r in refs] == ["vol_001", "vol_002"]
        first = src.load(refs[0])
        assert first.size == (300, 450)
        assert abs(int(np.asarray(first.convert("L")).mean()) - 40) <= 2


def test_ingest_selection_spreads_and_grayscale(tmp_path: Path) -> None:
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    _page(src_dir / "1.png")
    _page(src_dir / "2.png", size=(180, 90))  # landscape: spread
    Image.new("RGB", (60, 90), (200, 30, 30)).save(src_dir / "3.png")
    paths = ProjectPaths(tmp_path / "out" / "x.mangatl.json")

    kind, pages = ingest(src_dir, paths, selection={2, 3})
    assert kind == "folder"
    assert [p.number for p in pages] == [2, 3]
    assert pages[0].is_spread and not pages[1].is_spread
    assert pages[0].grayscale and not pages[1].grayscale
    assert (paths.root / pages[0].image_path).is_file()


def test_rgb_scan_of_gray_page_counts_as_gray() -> None:
    gray = Image.new("RGB", (40, 40), (128, 129, 127))
    assert is_effectively_gray(gray)
    assert not is_effectively_gray(Image.new("RGB", (40, 40), (200, 40, 40)))
