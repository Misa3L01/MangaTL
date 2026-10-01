"""Editor API on a synthetic project (no GPU: only fill cleaning and typesetting)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mangatl.config import Settings
from mangatl.project_io import load_project, save_project
from mangatl.server.app import create_app
from mangatl.server.store import ProjectStore

from .test_render import _synthetic_project


@pytest.fixture
def client(tmp_path: Path, project_root: Path) -> tuple[TestClient, Path]:
    settings = Settings(root=project_root)
    project, paths = _synthetic_project(tmp_path / "out")
    save_project(project, paths.project_file)
    store = ProjectStore(settings, tmp_path / "out")
    return TestClient(create_app(settings, store)), paths.project_file


def test_lists_and_loads_projects(client) -> None:
    api, _ = client
    [summary] = api.get("/api/projects").json()
    assert summary["id"] == "t" and summary["pages"] == 1
    project = api.get("/api/projects/t").json()
    assert project["pages"][0]["regions"][0]["id"] == "P001-B01"
    assert api.get("/api/projects/nope").status_code == 404


def test_edit_render_and_export(client) -> None:
    api, project_file = client
    patch = {"translation": "¡Solo dormí dos horas!", "style": "shout", "font_size": 30}
    region = api.patch("/api/projects/t/regions/P001-B01", json=patch).json()
    assert region["status"] == "edited" and region["style"] == "shout"
    assert region["font_size"] == 30 and region["font_size_fixed"] is True

    page = api.post("/api/projects/t/pages/1/render").json()
    assert page["has_rendered"] and page["regions"][0]["font_size"] == 30
    image = api.get("/api/projects/t/pages/1/image/rendered")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"

    moved = {"text_box_override": {"x0": 150, "y0": 300, "x1": 450, "y1": 500}, "auto_size": True}
    region = api.patch("/api/projects/t/regions/P001-B01", json=moved).json()
    assert region["text_box_override"]["x0"] == 150 and region["font_size_fixed"] is False
    api.post("/api/projects/t/pages/1/render")
    assert load_project(project_file).pages[0].regions[0].text_box_override is not None

    reset = api.patch("/api/projects/t/regions/P001-B01", json={"reset_text_box": True}).json()
    assert reset["text_box_override"] is None

    exported = api.post("/api/projects/t/export").json()
    assert exported["pages"] == 1 and exported["review"].endswith("revision.html")


def test_unknown_region_and_page(client) -> None:
    api, _ = client
    assert (
        api.patch("/api/projects/t/regions/P009-B09", json={"translation": "x"}).status_code == 404
    )
    assert api.post("/api/projects/t/pages/7/render").status_code == 404


def test_glossary_endpoints(client, project_root: Path, tmp_path: Path, monkeypatch) -> None:
    api, _ = client
    series = "Serie de prueba"
    # Keep the series folder out of the real project.
    from mangatl.translation import glossary as g

    monkeypatch.setattr(g, "series_dir", lambda settings, s: tmp_path / "series" / s)
    assert (
        api.put(f"/api/series/{series}/glossary/斉藤", json={"target": "Saitō"}).status_code == 200
    )
    g.add_pending(
        Settings(root=project_root), series, [g.GlossaryEntry(source="牛田", target="Uchida")]
    )
    entries = api.get(f"/api/series/{series}/glossary").json()
    assert {e["source"]: e["status"] for e in entries} == {"斉藤": "approved", "牛田": "pending"}
    assert api.post(f"/api/series/{series}/glossary/approve", json={"sources": None}).json() == {
        "approved": ["牛田"]
    }
    assert api.delete(f"/api/series/{series}/glossary/牛田").status_code == 200
    assert api.delete(f"/api/series/{series}/glossary/牛田").status_code == 404


def test_fonts_endpoint(client) -> None:
    api, _ = client
    fonts = api.get("/api/fonts").json()
    assert fonts["styles"]["shout"].endswith("Bangers-Regular.ttf")
    assert "assets/fonts/ComicNeue-Bold.ttf" in fonts["available"]
