"""Project access for the editor: discovery, locking, edits, per-page re-render and export."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from mangatl.config import Settings
from mangatl.models import Project
from mangatl.project_io import PROJECT_SUFFIX, ProjectPaths, load_project, save_project


class ProjectNotFoundError(KeyError):
    pass


class ProjectStore:
    """Projects are the *.mangatl.json files under `root` (the output folder by default).

    Every request reads the file from disk, so edits made with the CLI in parallel are seen;
    writes go through a per-project lock and the atomic `save_project`.
    """

    def __init__(self, settings: Settings, root: Path) -> None:
        self.settings = settings
        self.root = root.resolve()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ---------------------------------------------------------------- discovery
    def files(self) -> dict[str, Path]:
        if self.root.is_file():
            candidates = [self.root]
        else:
            candidates = sorted(self.root.glob(f"*{PROJECT_SUFFIX}")) + sorted(
                self.root.glob(f"*/*{PROJECT_SUFFIX}")
            )
        return {p.name.removesuffix(PROJECT_SUFFIX): p for p in candidates}

    def path(self, project_id: str) -> Path:
        try:
            return self.files()[project_id]
        except KeyError as exc:
            raise ProjectNotFoundError(project_id) from exc

    def summaries(self) -> list[dict[str, object]]:
        out = []
        for pid, file in self.files().items():
            try:
                project = load_project(file)
            except Exception:  # a corrupt file must not break the list
                continue
            regions = [r for _, r in project.regions()]
            out.append(
                {
                    "id": pid,
                    "series": project.meta.series,
                    "chapter": project.meta.chapter,
                    "source_lang": project.meta.source_lang,
                    "model": project.meta.translator_model,
                    "pages": len(project.pages),
                    "regions": len(regions),
                    "review": sum(1 for r in regions if r.status == "needs_review"),
                    "translated": all(project.stages.get(s) for s in ("translate", "typeset")),
                    "updated": datetime.fromtimestamp(file.stat().st_mtime, UTC).isoformat(),
                }
            )
        return sorted(out, key=lambda s: str(s["updated"]), reverse=True)

    # ---------------------------------------------------------------- access
    def _lock(self, project_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(project_id, threading.Lock())

    def load(self, project_id: str) -> tuple[Project, ProjectPaths]:
        file = self.path(project_id)
        return load_project(file), ProjectPaths(file)

    @contextmanager
    def editing(self, project_id: str) -> Iterator[tuple[Project, ProjectPaths]]:
        """Load, let the caller modify, and save atomically under the project lock."""
        with self._lock(project_id):
            project, paths = self.load(project_id)
            yield project, paths
            save_project(project, paths.project_file)
