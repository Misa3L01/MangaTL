"""Pipeline orchestration: stage order, resume, page selection, VRAM hand-off, timings."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from mangatl import PIPELINE_VERSION, debug
from mangatl.config import Settings
from mangatl.ingest import ingest, parse_page_range
from mangatl.models import STAGES, ChapterMeta, Page, Project, StageRecord
from mangatl.project_io import ProjectPaths, load_project, project_file_for, save_project, slugify
from mangatl.report import STAGE_LABELS, write_review_report

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


@dataclass
class TranslateOptions:
    input_path: Path
    series: str
    chapter: str
    src: str = "ja"
    direction: str = "rtl"
    model: str | None = None
    backend: str | None = None
    out_dir: Path | None = None
    pages: str | None = None
    from_stage: str | None = None
    debug: bool = False


class Pipeline:
    def __init__(self, settings: Settings, console: Console) -> None:
        self.settings = settings
        self.console = console

    # ---------------------------------------------------------------- helpers
    @contextmanager
    def _progress(self, label: str, total: int) -> Iterator[Callable[[], None]]:
        with Progress(
            TextColumn(f"[bold]{label}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=total)
            yield lambda: progress.advance(task)

    def _record(
        self, project: Project, file: Path, stage: str, start: float, **details: object
    ) -> None:
        seconds = time.perf_counter() - start
        project.stages[stage] = StageRecord(seconds=round(seconds, 2), details=details)
        save_project(project, file)
        log.info("%s: %.1f s", STAGE_LABELS[stage], seconds)

    @staticmethod
    def _select(project: Project, selection: set[int] | None) -> list[Page]:
        return [p for p in project.pages if selection is None or p.number in selection]

    # ---------------------------------------------------------------- stages
    def run_vision(self, project_file: Path, stages: list[str], pages_spec: str | None) -> Project:
        if not self.settings.pipeline.isolate_vision:
            from mangatl.stages.vision_worker import run_stages

            run_stages(project_file, stages, parse_page_range(pages_spec), self.settings)
            return load_project(project_file)
        cmd = [
            sys.executable,
            "-m",
            "mangatl.stages.vision_worker",
            str(project_file),
            "--stages",
            ",".join(stages),
        ]
        if pages_spec:
            cmd += ["--pages", pages_spec]
        log.info("Etapas de visión en un proceso aparte (libera toda la VRAM al terminar)")
        result = subprocess.run(cmd, env=dict(os.environ), check=False)
        if result.returncode != 0:
            raise PipelineError("Falló la detección/OCR; revisa logs/mangatl.log")
        return load_project(project_file)

    def run_translate_stage(self, project: Project, project_file: Path, pages: list[Page]) -> bool:
        """Returns False when the chapter is waiting for a manual translation."""
        from mangatl.translation.glossary import (
            load_glossary,
            load_previous_summaries,
            remember_chapter,
        )

        tcfg = self.settings.translator
        backend = project.meta.translator_backend
        glossary = load_glossary(self.settings, project.meta.series)
        previous = load_previous_summaries(
            self.settings, project.meta.series, before=project.meta.chapter
        )
        start = time.perf_counter()
        sub = (
            project.model_copy(update={"pages": pages})
            if len(pages) != len(project.pages)
            else project
        )

        if backend == "manual":
            from mangatl.translation.manual_backend import export_prompt

            files = export_prompt(
                sub,
                ProjectPaths(project_file),
                glossary,
                tcfg.manual_part_chars,
                previous,
            )
            project.stages["translate"] = StageRecord(
                status="partial", seconds=0.0, details={"backend": "manual", "prompts": len(files)}
            )
            save_project(project, project_file)
            self.console.print("\n[bold]Traducción manual:[/] se generaron los prompts:")
            for f in files:
                self.console.print(f"  • {f}")
            self.console.print(
                "Pégalos en el chat de claude.ai (una parte por mensaje), guarda la respuesta "
                "en un archivo y ejecuta:\n"
                f'  uv run mangatl import-translation "{project_file}" <respuesta.txt>\n'
                f'  uv run mangatl render "{project_file}"'
            )
            return False

        from mangatl.gpu import query_nvidia_smi
        from mangatl.ollama_runtime import OllamaRuntime
        from mangatl.stages.render import find_overflows
        from mangatl.translation.base import translate_chapter
        from mangatl.translation.ollama_backend import OllamaTranslator
        from mangatl.translation.postprocess import clean_translation
        from mangatl.translation.prompts import system_prompt

        smi = query_nvidia_smi()
        if smi and smi.memory_free_mb < self.settings.gpu.warn_free_vram_mb:
            log.warning(
                "Solo hay %d MiB de VRAM libre: parte del LLM irá a la RAM y será más lento. "
                "Cierra navegador/juegos si puedes.",
                smi.memory_free_mb,
            )
        runtime = OllamaRuntime(self.settings)
        if not runtime.is_running() and not tcfg.ollama.auto_start:
            raise PipelineError(
                "Ollama no está en ejecución (translator.ollama.auto_start = false)"
            )
        with runtime.running():
            translator = OllamaTranslator(self.settings)
            blocks = max(1, -(-len([p for p in pages if p.regions]) // tcfg.pages_per_block))
            try:
                with self._progress("Traducción", blocks) as step:
                    stats = translate_chapter(
                        sub,
                        translator,
                        glossary,
                        tcfg.pages_per_block,
                        tcfg.previous_lines,
                        on_block=lambda i, n: step(),
                        pivot=tcfg.pivot_english,
                        previous_chapters=previous,
                    )
                # Dry-run the lettering while the model is still loaded: balloons that do not
                # fit even at the minimum size get one request for a shorter version.
                overflows = find_overflows(sub, ProjectPaths(project_file), self.settings, pages)
                shorter = translator.shorten(system_prompt(project.meta), overflows)
                for region, _ in overflows:
                    if shorter.get(region.id):
                        region.shorter_alternative = clean_translation(shorter[region.id])
                if overflows:
                    log.info(
                        "Versión corta pedida para %d globo(s) que no cabían; recibidas %d",
                        len(overflows),
                        len(shorter),
                    )
                loaded = next(
                    (m for m in runtime.running_models() if m.get("name") == translator.model),
                    {},
                )
                vram, context = loaded.get("size_vram", 0), loaded.get("context_length")
            finally:
                translator.close()  # keep_alive=0: frees the VRAM right away
        if sub is not project:
            project.pending_glossary = sub.pending_glossary
            project.chapter_summary, project.block_summaries = (
                sub.chapter_summary,
                sub.block_summaries,
            )
        pending = remember_chapter(self.settings, project)
        self._record(
            project,
            project_file,
            "translate",
            start,
            backend="ollama",
            model=translator.model,
            blocks=stats.blocks,
            requests=stats.requests,
            regions=stats.regions,
            missing=stats.missing,
            realigned=translator.realigned,
            prompt_tokens=stats.prompt_tokens,
            output_tokens=stats.output_tokens,
            tokens_per_second=round(stats.tokens_per_second, 1),
            context=f"{stats.context_peak}/{context or '?'}",
            vram_gb=round(vram / 2**30, 2),
            pivot=tcfg.pivot_english,
            glossary_pending=pending,
            shorter_requested=len(overflows),
            shorter_received=len(shorter),
        )
        return True

    def run_render(
        self, project: Project, project_file: Path, pages: list[Page], debug_images: bool
    ) -> None:
        from mangatl.stages.render import run_export, run_inpaint, run_typeset

        paths = ProjectPaths(project_file)
        for stage, runner in (("inpaint", run_inpaint), ("typeset", run_typeset)):
            start = time.perf_counter()
            with self._progress(STAGE_LABELS[stage], len(pages)) as step:
                details = runner(project, paths, self.settings, pages, step)
            self._record(project, project_file, stage, start, **details)
        start = time.perf_counter()
        details = run_export(project, paths, self.settings)
        self._record(project, project_file, "export", start, **details)
        write_review_report(project, paths)
        if debug_images:
            self.write_debug(project, paths, pages)

    def write_debug(self, project: Project, paths: ProjectPaths, pages: list[Page]) -> None:
        for page in pages:
            debug.draw_regions(paths, page)
            debug.draw_masks(paths, page)
        self.console.print(
            f"Imágenes de depuración en {paths.debug} (páginas limpias en {paths.clean})"
        )

    # ---------------------------------------------------------------- commands
    def translate(self, opts: TranslateOptions) -> Path:
        s = self.settings
        if opts.src not in ("ja", "en", "auto"):
            raise PipelineError("--src debe ser ja, en o auto")
        if opts.direction not in ("rtl", "ltr"):
            raise PipelineError("--direction debe ser rtl (manga) o ltr (cómic occidental)")
        if opts.model:
            ollama = s.translator.ollama.model_copy(update={"model": opts.model})
            s = s.model_copy(
                update={"translator": s.translator.model_copy(update={"ollama": ollama})}
            )
            self.settings = s
        if opts.from_stage and opts.from_stage not in STAGES:
            raise PipelineError(
                f"Etapa desconocida: {opts.from_stage}. Opciones: {', '.join(STAGES)}"
            )
        out_dir = opts.out_dir or s.resolve(s.paths.output_dir) / slugify(
            f"{opts.series}-{opts.chapter}"
        )
        project_file = project_file_for(out_dir, opts.series, opts.chapter)
        selection = parse_page_range(opts.pages)
        backend = opts.backend or s.translator.backend
        total_start = time.perf_counter()

        if project_file.exists() and opts.from_stage != "ingest":
            project = load_project(project_file)
            if Path(project.meta.input_path) != opts.input_path.resolve():
                log.warning("El proyecto existente usaba otra entrada: %s", project.meta.input_path)
            self.console.print(f"Reanudando proyecto: {project_file}")
        else:
            project = Project(
                meta=ChapterMeta(
                    series=opts.series,
                    chapter=opts.chapter,
                    source_lang=opts.src,  # type: ignore[arg-type]
                    reading_direction=opts.direction,  # type: ignore[arg-type]
                    input_path=str(opts.input_path.resolve()),
                )
            )
        meta = project.meta
        if opts.from_stage in ("ingest", "detect", "ocr"):
            meta.source_lang = opts.src  # type: ignore[assignment]
            meta.reading_direction = opts.direction  # type: ignore[assignment]
        meta.target_variant = s.translator.target_variant
        meta.honorifics = s.translator.honorifics
        meta.sfx_mode = s.translator.sfx_mode
        meta.translator_backend = backend
        meta.translator_model = s.translator.ollama.model if backend == "ollama" else None
        meta.pipeline_version = PIPELINE_VERSION

        if opts.from_stage:
            todo = list(STAGES[STAGES.index(opts.from_stage) :])
        else:
            todo = [
                st
                for st in STAGES
                if project.stages.get(st, StageRecord(status="failed")).status != "done"
            ]
        if not todo:
            self.console.print(
                "Todas las etapas ya estaban hechas. Usa --from-stage para repetir alguna."
            )
            return project_file

        paths = ProjectPaths(project_file)
        if "ingest" in todo:
            start = time.perf_counter()
            with open_count(opts.input_path) as count:
                total = len([n for n in selection if n <= count]) if selection else count
            with self._progress("Ingesta", total or 1) as step:
                kind, pages = ingest(opts.input_path, paths, selection, step)
            meta.input_kind = kind  # type: ignore[assignment]
            project.pages = pages
            project.stages = {}
            self._record(project, project_file, "ingest", start, pages=len(pages), kind=kind)
            selection = None  # ingest already applied it
        save_project(project, project_file)

        vision = [st for st in ("detect", "ocr") if st in todo]
        if vision:
            project = self.run_vision(project_file, vision, opts.pages if selection else None)
            if opts.debug:
                self.write_debug(project, paths, self._select(project, selection))

        pages = self._select(project, selection)
        if "translate" in todo and not self.run_translate_stage(project, project_file, pages):
            self.summary(project, project_file, time.perf_counter() - total_start)
            return project_file
        if any(st in todo for st in ("inpaint", "typeset", "export")):
            self.run_render(project, project_file, pages, opts.debug)
        self.summary(project, project_file, time.perf_counter() - total_start)
        return project_file

    def render(self, project_file: Path, pages_spec: str | None, debug_images: bool) -> None:
        start = time.perf_counter()
        project = load_project(project_file)
        self.run_render(
            project, project_file, self._select(project, parse_page_range(pages_spec)), debug_images
        )
        self.summary(project, project_file, time.perf_counter() - start)

    # ---------------------------------------------------------------- report
    def summary(self, project: Project, project_file: Path, wall_seconds: float) -> None:
        paths = ProjectPaths(project_file)
        table = Table(title=f"{project.meta.series} · capítulo {project.meta.chapter}")
        table.add_column("Etapa")
        table.add_column("Tiempo", justify="right")
        table.add_column("Detalle", overflow="fold")
        stage_total = 0.0
        for stage in STAGES:
            rec = project.stages.get(stage)
            if rec is None:
                continue
            stage_total += rec.seconds
            detail = ", ".join(
                f"{k}={v}"
                for k, v in rec.details.items()
                if k not in ("cbz",) and v not in ([], None)
            )
            status = "" if rec.status == "done" else f" [yellow]({rec.status})[/]"
            table.add_row(STAGE_LABELS[stage] + status, f"{rec.seconds:.1f} s", detail)
        n_pages = max(1, len(project.pages))
        table.add_row(
            "[bold]Total etapas",
            f"[bold]{stage_total:.1f} s",
            f"{stage_total / n_pages:.1f} s por página",
        )
        table.add_row(
            "Esta ejecución (reloj)",
            f"{wall_seconds:.1f} s",
            "incluye arranque de procesos y modelos",
        )
        self.console.print(table)

        counts = Counter(r.status for _, r in project.regions())
        types = Counter(r.type for _, r in project.regions())
        self.console.print(
            f"Regiones: {sum(counts.values())} · auto {counts['auto']} · "
            f"editadas {counts['edited']} · [yellow]a revisar {counts['needs_review']}[/] · "
            f"omitidas {counts['skipped']} "
            f"(globos {types['speech_bubble']}, cuadros {types['narration_box']}, "
            f"texto sobre dibujo {types['text_on_art']})"
        )
        review = [r for _, r in project.regions() if r.status == "needs_review"]
        if review:
            t = Table(title="Regiones a revisar", show_lines=False)
            t.add_column("ID")
            t.add_column("Motivo", overflow="fold")
            for r in review[:25]:
                t.add_row(r.id, "; ".join(r.notes) or "—")
            if len(review) > 25:
                t.add_row("…", f"y {len(review) - 25} más (ver el archivo del proyecto)")
            self.console.print(t)
        if project.stages.get("export"):
            self.console.print(f"Páginas: {paths.output_pages}")
            for label, file in (("CBZ", paths.cbz), ("PDF", paths.pdf)):
                if file.exists():
                    self.console.print(f"{label}: {file}")
            if review:
                self.console.print(f"Informe de revisión: {paths.root / 'revision.html'}")
        self.console.print(f"Proyecto: {project_file}")


@contextmanager
def open_count(input_path: Path) -> Iterator[int]:
    from mangatl.ingest import open_source

    with open_source(input_path) as source:
        yield len(source.pages())
