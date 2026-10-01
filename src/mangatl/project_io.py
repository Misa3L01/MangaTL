"""Load/save the project file and resolve its working paths."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from mangatl.models import Project

PROJECT_SUFFIX = ".mangatl.json"


class ProjectPaths:
    """Layout of an output folder:

    <out>/<name>.mangatl.json   project file
    <out>/work/pages/           normalized source pages
    <out>/work/masks/           bubble label masks and text masks
    <out>/work/clean/           pages with the original text removed
    <out>/work/debug/           --debug images
    <out>/pages/                translated pages (final PNG)
    <out>/<name>.cbz            translated chapter
    <out>/prompts/              manual backend prompts
    """

    def __init__(self, project_file: Path) -> None:
        self.project_file = project_file.resolve()
        self.root = self.project_file.parent
        self.name = self.project_file.name.removesuffix(PROJECT_SUFFIX)
        self.work = self.root / "work"
        self.pages = self.work / "pages"
        self.masks = self.work / "masks"
        self.clean = self.work / "clean"
        self.debug = self.work / "debug"
        self.output_pages = self.root / "pages"
        self.prompts = self.root / "prompts"
        self.cbz = self.root / f"{self.name}.cbz"
        self.pdf = self.root / f"{self.name}.pdf"

    def rel(self, path: Path) -> str:
        return Path(os.path.relpath(path, self.root)).as_posix()

    def abs(self, rel_path: str) -> Path:
        return self.root / rel_path


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "-", text.strip().lower(), flags=re.UNICODE)
    return re.sub(r"-{2,}", "-", text).strip("-") or "capitulo"


def project_file_for(out_dir: Path, series: str, chapter: str) -> Path:
    return out_dir / f"{slugify(series)}-{slugify(chapter)}{PROJECT_SUFFIX}"


def load_project(path: Path) -> Project:
    return Project.model_validate_json(path.read_text(encoding="utf-8"))


def save_project(project: Project, path: Path) -> None:
    """Atomic write: a crash never leaves a half-written project file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = project.model_dump_json(indent=2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
