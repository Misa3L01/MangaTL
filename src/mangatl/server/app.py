"""HTTP API of the local editor. Bound to 127.0.0.1 only; error messages are in Spanish."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mangatl.config import Settings
from mangatl.models import BBox, GlossaryEntry, Page, Project, RegionStatus, TextStyle
from mangatl.server.store import ProjectNotFoundError, ProjectStore

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
FONT_SUFFIXES = (".ttf", ".otf", ".ttc")
NO_CACHE = {"Cache-Control": "no-store"}

MISSING_UI = """<!doctype html><html lang="es"><meta charset="utf-8"><title>MangaTL</title>
<body style="font-family:system-ui;max-width:640px;
margin:48px auto;padding:0 16px">
<h1>MangaTL</h1><p>La API está funcionando, pero falta compilar la interfaz web.</p>
<p>En PowerShell, desde la raíz del proyecto:</p>
<pre>. .\\env.ps1
.\\scripts\\build-web.ps1</pre>
<p>Luego vuelve a ejecutar <code>uv run mangatl ui</code>.</p></body></html>"""


class RegionPatch(BaseModel):
    translation: str | None = None
    shorter_alternative: str | None = None
    style: TextStyle | None = None
    font: str | None = None  # "" resets to the style font
    font_size: int | None = Field(None, ge=6, le=400)
    auto_size: bool | None = None  # True drops a fixed font size
    text_box_override: BBox | None = None
    reset_text_box: bool = False
    status: RegionStatus | None = None


class ApproveRequest(BaseModel):
    sources: list[str] | None = None  # None = all pending


class GlossaryUpsert(BaseModel):
    target: str
    category: Literal["character", "place", "technique", "term", "other"] | None = None
    notes: str | None = None


def _page_payload(page: Page) -> dict[str, object]:
    data = page.model_dump(mode="json")
    data["has_rendered"] = bool(page.rendered_path)
    data["has_clean"] = bool(page.clean_path)
    data["review"] = sum(1 for r in page.regions if r.status == "needs_review")
    return data


def _project_payload(project: Project, project_id: str) -> dict[str, object]:
    return {
        "id": project_id,
        "meta": project.meta.model_dump(mode="json"),
        "stages": {k: v.model_dump(mode="json") for k, v in project.stages.items()},
        "chapter_summary": project.chapter_summary,
        "pending_glossary": [g.model_dump(mode="json") for g in project.pending_glossary],
        "pages": [_page_payload(p) for p in project.pages],
    }


def create_app(settings: Settings, store: ProjectStore) -> FastAPI:
    app = FastAPI(title="MangaTL", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.exception_handler(ProjectNotFoundError)
    def _not_found(_request, exc: ProjectNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": f"No existe el proyecto «{exc.args[0]}»"}, status_code=404)

    def _page(project: Project, number: int) -> Page:
        page = next((p for p in project.pages if p.number == number), None)
        if page is None:
            raise HTTPException(404, f"La página {number} no está en el proyecto")
        return page

    # ------------------------------------------------------------ projects
    @app.get("/api/projects")
    def list_projects() -> list[dict[str, object]]:
        return store.summaries()

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, object]:
        project, _ = store.load(project_id)
        return _project_payload(project, project_id)

    @app.get("/api/projects/{project_id}/pages/{number}/image/{kind}")
    def page_image(
        project_id: str, number: int, kind: Literal["original", "clean", "rendered"]
    ) -> FileResponse:
        project, paths = store.load(project_id)
        page = _page(project, number)
        rel = {
            "original": page.image_path,
            "clean": page.clean_path,
            "rendered": page.rendered_path,
        }[kind]
        if not rel:
            rel = page.image_path  # not processed yet: show the original
        return FileResponse(paths.abs(rel), media_type="image/png", headers=NO_CACHE)

    @app.patch("/api/projects/{project_id}/regions/{region_id}")
    def patch_region(project_id: str, region_id: str, patch: RegionPatch) -> dict[str, object]:
        with store.editing(project_id) as (project, _):
            region = project.find_region(region_id)
            if region is None:
                raise HTTPException(404, f"No existe la región {region_id}")
            changed = False
            if patch.translation is not None and patch.translation != region.translation:
                region.translation, changed = patch.translation.strip(), True
            if patch.shorter_alternative is not None:
                region.shorter_alternative = patch.shorter_alternative.strip() or None
                changed = True
            if patch.style is not None and patch.style != region.style:
                region.style, changed = patch.style, True
            if patch.font is not None:
                region.font, changed = (patch.font or None), True
            if patch.auto_size:
                region.font_size_fixed, changed = False, True
            elif patch.font_size is not None:
                region.font_size, region.font_size_fixed, changed = patch.font_size, True, True
            if patch.reset_text_box:
                region.text_box_override, changed = None, True
            elif patch.text_box_override is not None:
                region.text_box_override, changed = patch.text_box_override, True
            if patch.status is not None:
                region.status = patch.status
            elif changed:
                region.status = "edited"  # protects the edit from re-translation
            return region.model_dump(mode="json")

    @app.post("/api/projects/{project_id}/pages/{number}/render")
    def render_page(project_id: str, number: int) -> dict[str, object]:
        from mangatl.stages.render import render_single_page

        with store.editing(project_id) as (project, paths):
            page = _page(project, number)
            if not page.text_mask_path:
                raise HTTPException(409, "La página aún no pasó por la detección")
            render_single_page(project, paths, settings, page)
            return _page_payload(page)

    @app.post("/api/projects/{project_id}/export")
    def export(project_id: str) -> dict[str, object]:
        from mangatl.report import write_review_report
        from mangatl.stages.render import run_export

        with store.editing(project_id) as (project, paths):
            if not all(p.rendered_path for p in project.pages):
                raise HTTPException(409, "Hay páginas sin rotular: re-renderízalas antes")
            details = run_export(project, paths, settings)
            details["review"] = str(write_review_report(project, paths))
            return details

    # ------------------------------------------------------------ glossary
    @app.get("/api/series/{series}/glossary")
    def glossary(series: str) -> list[dict[str, object]]:
        from mangatl.translation.glossary import read_glossary

        return [e.model_dump(mode="json") for e in read_glossary(settings, series)]

    @app.post("/api/series/{series}/glossary/approve")
    def glossary_approve(series: str, body: ApproveRequest) -> dict[str, object]:
        from mangatl.translation.glossary import approve

        return {"approved": approve(settings, series, body.sources)}

    @app.put("/api/series/{series}/glossary/{source}")
    def glossary_upsert(series: str, source: str, body: GlossaryUpsert) -> dict[str, object]:
        from mangatl.translation.glossary import upsert

        entry: GlossaryEntry = upsert(
            settings, series, source, body.target, body.category, body.notes
        )
        return entry.model_dump(mode="json")

    @app.delete("/api/series/{series}/glossary/{source}")
    def glossary_delete(series: str, source: str) -> dict[str, object]:
        from mangatl.translation.glossary import remove

        if not remove(settings, series, source):
            raise HTTPException(404, f"«{source}» no está en el glosario")
        return {"removed": source}

    # ------------------------------------------------------------ fonts
    @app.get("/api/fonts")
    def fonts() -> dict[str, object]:
        fonts_dir = settings.resolve(settings.paths.fonts_dir)
        available = sorted(
            p.relative_to(settings.root).as_posix()
            for p in fonts_dir.rglob("*")
            if p.suffix.lower() in FONT_SUFFIXES
        )
        styles = {k: Path(v).as_posix() for k, v in settings.typesetting.fonts.model_dump().items()}
        return {"styles": styles, "available": available}

    # ------------------------------------------------------------ frontend
    if (WEB_DIST / "index.html").is_file():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def missing_ui() -> str:
            return MISSING_UI

    return app
