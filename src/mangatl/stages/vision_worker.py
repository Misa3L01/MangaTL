"""Child process for the GPU vision stages (detection + OCR).

When this process exits, all of its VRAM (CUDA context included) returns to the system, so
the LLM that runs next has the whole GPU. Usage (internal):

    python -m mangatl.stages.vision_worker <project.mangatl.json> --stages detect,ocr [--pages 3-7]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from mangatl.config import Settings, load_settings
from mangatl.runtime_env import apply_runtime_env

log = logging.getLogger("mangatl.vision")


def run_stages(
    project_file: Path, stages: list[str], selection: set[int] | None, settings: Settings
) -> None:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    from mangatl.gpu import release_cuda_memory
    from mangatl.logging_setup import console
    from mangatl.models import StageRecord
    from mangatl.project_io import ProjectPaths, load_project, save_project
    from mangatl.stages.vision import run_detection, run_ocr

    paths = ProjectPaths(project_file)
    project = load_project(project_file)
    pages = [p for p in project.pages if selection is None or p.number in selection]
    runners = {"detect": (run_detection, "Detección"), "ocr": (run_ocr, "OCR")}

    for stage in stages:
        runner, label = runners[stage]
        start = time.perf_counter()
        with Progress(
            TextColumn(f"[bold]{label}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=len(pages))

            def step(task_id: int = task) -> None:
                progress.advance(task_id)

            try:
                details = runner(project, paths, settings, pages, step)
            except Exception as exc:
                if not _is_cuda_oom(exc) or settings.gpu.device == "cpu":
                    raise
                log.warning("%s: sin memoria en la GPU; se reintenta en CPU (más lento)", label)
                release_cuda_memory()
                settings = settings.model_copy(
                    update={"gpu": settings.gpu.model_copy(update={"device": "cpu"})}
                )
                progress.reset(task)
                details = runner(project, paths, settings, pages, step)
                details["fallback"] = "cpu"
        seconds = time.perf_counter() - start
        project.stages[stage] = StageRecord(seconds=round(seconds, 2), details=dict(details))
        save_project(project, project_file)
        log.info("%s: %.1f s", label, seconds)


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


def main(argv: list[str] | None = None) -> int:
    from mangatl.ingest import parse_page_range
    from mangatl.logging_setup import setup_logging

    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--stages", default="detect,ocr")
    parser.add_argument("--pages", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    apply_runtime_env(settings)
    setup_logging(settings, args.verbose)
    try:
        run_stages(args.project, args.stages.split(","), parse_page_range(args.pages), settings)
    except Exception:
        log.exception("Falló la etapa de visión")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
