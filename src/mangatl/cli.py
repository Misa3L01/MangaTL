"""Command-line interface. User-facing text is in Spanish."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

import typer

from mangatl import __version__
from mangatl.config import Settings, load_settings
from mangatl.logging_setup import console, setup_logging
from mangatl.runtime_env import apply_runtime_env

app = typer.Typer(
    name="mangatl",
    help="MangaTL: traductor automático de manga (japonés/inglés → español), local y gratuito.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
ollama_app = typer.Typer(help="Gestiona el servidor Ollama portable.", no_args_is_help=True)
app.add_typer(ollama_app, name="ollama")
glossary_app = typer.Typer(
    help="Glosario por serie: revisar, aprobar y corregir entradas.", no_args_is_help=True
)
app.add_typer(glossary_app, name="glossary")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mangatl {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Ruta a un config.toml alternativo."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Muestra mensajes de depuración.")
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Muestra la versión."
        ),
    ] = None,
) -> None:
    if config is not None:
        # Child processes (vision worker) must load the same configuration.
        os.environ["MANGATL_CONFIG"] = str(config.resolve())
    settings = load_settings(config)
    apply_runtime_env(settings)
    setup_logging(settings, verbose)
    ctx.obj = settings


@app.command()
def setup(
    ctx: typer.Context,
    check: Annotated[
        bool, typer.Option("--check", help="Solo verifica el entorno; no descarga nada.")
    ] = False,
    skip_llm: Annotated[
        bool, typer.Option("--skip-llm", help="No descarga el modelo LLM de Ollama.")
    ] = False,
) -> None:
    """Verifica CUDA y Ollama, y descarga los modelos (todo dentro del proyecto)."""
    from mangatl.bootstrap import SetupRunner

    settings: Settings = ctx.obj
    ok = SetupRunner(settings, console, check_only=check, skip_llm=skip_llm).run()
    raise typer.Exit(0 if ok else 1)


ProjectArg = Annotated[
    Path,
    typer.Argument(
        help="Archivo del proyecto (<capítulo>.mangatl.json).", exists=True, dir_okay=False
    ),
]
PagesOpt = Annotated[str | None, typer.Option("--pages", help="Páginas a procesar, p. ej. 3-7,10.")]
DebugOpt = Annotated[
    bool, typer.Option("--debug", help="Guarda imágenes intermedias en work/debug.")
]


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/] {message}")
    raise typer.Exit(1)


@app.command()
def translate(
    ctx: typer.Context,
    input_path: Annotated[
        Path, typer.Argument(help="Carpeta de imágenes, .zip, .cbz o .pdf.", exists=True)
    ],
    series: Annotated[str, typer.Option("--series", "-s", help="Nombre de la serie.")],
    chapter: Annotated[str, typer.Option("--chapter", "-n", help="Número o nombre del capítulo.")],
    src: Annotated[str, typer.Option("--src", help="Idioma de origen: ja | en | auto.")] = "ja",
    direction: Annotated[
        str,
        typer.Option(
            "--direction",
            help="Sentido de lectura: rtl (manga, también en ediciones en inglés) | ltr (cómic).",
        ),
    ] = "rtl",
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Modelo de Ollama para esta ejecución (p. ej. qwen3.5:4b)."
        ),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", "-b", help="ollama | manual (por defecto, el de config.toml)."),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Carpeta de salida.")] = None,
    pages: PagesOpt = None,
    from_stage: Annotated[
        str | None,
        typer.Option(
            "--from-stage", help="Repite desde: ingest|detect|ocr|translate|inpaint|typeset|export."
        ),
    ] = None,
    debug: DebugOpt = False,
) -> None:
    """Traduce un capítulo de punta a punta (reanuda si el proyecto ya existe)."""
    from mangatl.pipeline import Pipeline, PipelineError, TranslateOptions

    if backend not in (None, "ollama", "manual"):
        _fail("--backend debe ser 'ollama' o 'manual'")
    try:
        Pipeline(ctx.obj, console).translate(
            TranslateOptions(
                input_path=input_path,
                series=series,
                chapter=chapter,
                src=src,
                direction=direction,
                model=model,
                backend=backend,
                out_dir=out,
                pages=pages,
                from_stage=from_stage,
                debug=debug,
            )
        )
    except PipelineError as exc:
        _fail(str(exc))


@app.command("export-prompt")
def export_prompt_cmd(
    ctx: typer.Context,
    project_file: ProjectArg,
    images: Annotated[
        bool,
        typer.Option("--images", help="Genera también las páginas con las regiones numeradas."),
    ] = False,
) -> None:
    """Genera el prompt para traducir el capítulo en el chat de claude.ai."""
    from mangatl.project_io import ProjectPaths, load_project
    from mangatl.translation.glossary import load_glossary, load_previous_summaries
    from mangatl.translation.manual_backend import export_prompt

    settings: Settings = ctx.obj
    project = load_project(project_file)
    if not any(r.ocr_text for _, r in project.regions()):
        _fail(
            "El proyecto aún no tiene texto OCR. Ejecuta primero: "
            "mangatl translate ... --backend manual"
        )
    files = export_prompt(
        project,
        ProjectPaths(project_file),
        load_glossary(settings, project.meta.series),
        settings.translator.manual_part_chars,
        load_previous_summaries(settings, project.meta.series),
        images=images,
    )
    prompts = [f for f in files if f.suffix == ".txt"]
    console.print(f"Se generaron {len(prompts)} parte(s) del prompt:")
    for f in files:
        console.print(f"  • {f}")
    console.print(
        "Pega cada parte en el chat, guarda la respuesta (puede tener texto alrededor del JSON) y "
        f'ejecuta:\n  uv run mangatl import-translation "{project_file}" <respuesta.txt>'
    )


@app.command("import-translation")
def import_translation_cmd(
    ctx: typer.Context,
    project_file: ProjectArg,
    answers: Annotated[
        list[Path],
        typer.Argument(help="Respuesta(s) del chat (.json o .txt).", exists=True, dir_okay=False),
    ],
) -> None:
    """Importa la traducción pegada del chat y reporta IDs faltantes o sobrantes."""
    from rich.table import Table

    from mangatl.models import StageRecord
    from mangatl.project_io import load_project, save_project
    from mangatl.translation.manual_backend import import_translation

    project = load_project(project_file)
    text = "\n".join(a.read_text(encoding="utf-8", errors="replace") for a in answers)
    report = import_translation(project, text)
    if not report.applied:
        _fail(
            f"No se encontró ninguna traducción válida ({report.objects_found} objeto(s) JSON con "
            f"'regions'; {len(report.invalid)} entrada(s) inválida(s))."
        )
    done = not report.missing
    project.stages["translate"] = StageRecord(
        status="done" if done else "partial",
        details={
            "backend": "manual",
            "applied": len(report.applied),
            "missing": len(report.missing),
        },
    )
    save_project(project, project_file)
    from mangatl.translation.glossary import remember_chapter

    remember_chapter(ctx.obj, project)

    table = Table(title="Importación")
    table.add_column("Resultado")
    table.add_column("Cantidad", justify="right")
    table.add_column("IDs", overflow="fold")
    table.add_row("Aplicadas", str(len(report.applied)), "")
    table.add_row("[yellow]Faltantes[/]", str(len(report.missing)), ", ".join(report.missing[:40]))
    table.add_row(
        "[yellow]Sobrantes (no existen)[/]",
        str(len(report.unknown)),
        ", ".join(report.unknown[:40]),
    )
    table.add_row("[red]Inválidas[/]", str(len(report.invalid)), "; ".join(report.invalid[:10]))
    table.add_row("Glosario (pendientes nuevos)", str(report.glossary_added), "")
    console.print(table)
    if report.missing:
        console.print(
            "[yellow]Faltan regiones.[/] Pide al chat las que faltan o impórtalas después; "
            "puedes rotular igual (las faltantes quedan con el texto original)."
        )
    console.print(f'Siguiente paso:  uv run mangatl render "{project_file}"')


@app.command()
def render(
    ctx: typer.Context,
    project_file: ProjectArg,
    pages: PagesOpt = None,
    debug: DebugOpt = False,
) -> None:
    """Vuelve a limpiar, rotular y exportar (tras editar o importar traducciones)."""
    from mangatl.pipeline import Pipeline, PipelineError

    try:
        Pipeline(ctx.obj, console).render(project_file, pages, debug)
    except PipelineError as exc:
        _fail(str(exc))


@app.command()
def ui(
    ctx: typer.Context,
    target: Annotated[
        Path | None,
        typer.Argument(help="Carpeta con proyectos o un .mangatl.json (por defecto, output\\)."),
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Puerto local.")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="No abrir el navegador automáticamente.")
    ] = False,
) -> None:
    """Abre el editor web local para revisar y corregir traducciones."""
    import threading
    import webbrowser

    import uvicorn

    from mangatl.server.app import WEB_DIST, create_app
    from mangatl.server.store import ProjectStore

    settings: Settings = ctx.obj
    root = target or settings.resolve(settings.paths.output_dir)
    if not root.exists():
        _fail(f"No existe {root}")
    store = ProjectStore(settings, root)
    url = f"http://127.0.0.1:{port}/"
    console.print(
        f"Editor de MangaTL en [bold]{url}[/] · proyectos en {store.root} · Ctrl+C para salir"
    )
    if not (WEB_DIST / "index.html").is_file():
        console.print("[yellow]Falta compilar la interfaz:[/] ejecuta .\\scripts\\build-web.ps1")
    if not no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(settings, store), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def review(ctx: typer.Context, project_file: ProjectArg) -> None:
    """Genera revision.html con las regiones a revisar (recortes, texto y motivo)."""
    from mangatl.project_io import ProjectPaths, load_project
    from mangatl.report import review_regions, write_review_report

    project = load_project(project_file)
    out = write_review_report(project, ProjectPaths(project_file))
    console.print(f"{len(review_regions(project))} región(es) a revisar · informe: {out}")


SeriesOpt = Annotated[str, typer.Option("--series", "-s", help="Nombre de la serie.")]


@glossary_app.command("list")
def glossary_list(
    ctx: typer.Context,
    series: SeriesOpt,
    pending: Annotated[
        bool, typer.Option("--pending", help="Solo las entradas pendientes.")
    ] = False,
) -> None:
    """Muestra el glosario de la serie."""
    from rich.table import Table

    from mangatl.translation.glossary import read_glossary, series_dir

    entries = read_glossary(ctx.obj, series)
    if pending:
        entries = [e for e in entries if e.status == "pending"]
    if not entries:
        console.print(f"No hay entradas{' pendientes' if pending else ''} para «{series}».")
        return
    table = Table(title=f"Glosario · {series}  ({series_dir(ctx.obj, series)})")
    for col in ("Original", "Traducción", "Tipo", "Estado", "Notas", "Capítulo"):
        table.add_column(col, overflow="fold")
    labels = {"approved": "[green]aprobada[/]", "pending": "[yellow]pendiente[/]"}
    for e in entries:
        table.add_row(
            e.source, e.target, e.category, labels[e.status], e.notes or "", e.chapter or ""
        )
    console.print(table)


@glossary_app.command("approve")
def glossary_approve(
    ctx: typer.Context,
    series: SeriesOpt,
    terms: Annotated[
        list[str] | None, typer.Argument(help="Términos a aprobar (original).")
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Aprueba todas las pendientes.")] = False,
) -> None:
    """Aprueba entradas pendientes (se usarán como obligatorias en los próximos capítulos)."""
    from mangatl.translation.glossary import approve

    if not terms and not all_:
        _fail("Indica los términos a aprobar o usa --all")
    approved = approve(ctx.obj, series, None if all_ else list(terms or []))
    if not approved:
        console.print("No se aprobó nada (¿los términos existen y estaban pendientes?).")
        return
    console.print(f"Aprobadas {len(approved)}: {', '.join(approved)}")


@glossary_app.command("edit")
def glossary_edit(
    ctx: typer.Context,
    series: SeriesOpt,
    term: Annotated[str, typer.Argument(help="Término original (p. ej. 斉藤).")],
    target: Annotated[str | None, typer.Option("--target", "-t", help="Traducción fija.")] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", "-k", help="character | place | technique | term | other"),
    ] = None,
    notes: Annotated[
        str | None, typer.Option("--notes", help="Forma de hablar, cómo trata a otros, etc.")
    ] = None,
) -> None:
    """Crea o corrige una entrada (queda aprobada)."""
    from mangatl.translation.glossary import upsert

    if category not in (None, "character", "place", "technique", "term", "other"):
        _fail("--category debe ser character, place, technique, term u other")
    try:
        entry = upsert(ctx.obj, series, term, target, category, notes)
    except ValueError as exc:
        _fail(str(exc))
        return
    console.print(f"Guardado: {entry.source} → {entry.target} [{entry.category}] (aprobada)")


@glossary_app.command("remove")
def glossary_remove(
    ctx: typer.Context,
    series: SeriesOpt,
    term: Annotated[str, typer.Argument(help="Término original a eliminar.")],
) -> None:
    """Elimina una entrada del glosario."""
    from mangatl.translation.glossary import remove

    if not remove(ctx.obj, series, term):
        _fail(f"«{term}» no está en el glosario de «{series}»")
    console.print(f"Eliminado: {term}")


@ollama_app.command("status")
def ollama_status(ctx: typer.Context) -> None:
    """Muestra si Ollama está en ejecución y qué modelos tiene."""
    from mangatl.disk import fmt_size
    from mangatl.ollama_runtime import OllamaRuntime

    rt = OllamaRuntime(ctx.obj)
    console.print(f"Instalado: {'sí' if rt.is_installed() else 'no'} ({rt.exe_path})")
    console.print(f"Modelos en: {rt.models_dir}")
    version = rt.version()
    if version is None:
        console.print("Servidor: detenido (MangaTL lo inicia automáticamente cuando lo necesita)")
        return
    console.print(f"Servidor: en ejecución, versión {version}")
    for model in rt.list_models():
        console.print(f"  • {model['name']} ({fmt_size(model.get('size', 0))})")
    for model in rt.running_models():
        console.print(f"  cargado: {model['name']} · {fmt_size(model.get('size_vram', 0))} en VRAM")


@ollama_app.command("pull")
def ollama_pull(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Modelo de Ollama, p. ej. qwen3.5:9b.")],
) -> None:
    """Descarga un modelo a models\\ollama (muestra el tamaño y respeta el espacio mínimo)."""
    from mangatl.bootstrap import download_progress
    from mangatl.disk import InsufficientSpaceError, check_space, fmt_size, free_bytes
    from mangatl.ollama_runtime import OllamaRuntime

    settings: Settings = ctx.obj
    rt = OllamaRuntime(settings)
    size = rt.remote_model_size(model)
    if size is None:
        _fail(f"No se encontró «{model}» en el registro de Ollama")
        return
    try:
        check_space(settings.root, size, settings.setup.min_free_disk_gb)
    except InsufficientSpaceError as exc:
        _fail(str(exc))
    console.print(
        f"Descarga: {model} · {fmt_size(size)} · "
        f"quedarán ~{fmt_size(free_bytes(settings.root) - size)} libres en {settings.root.anchor}"
    )
    with rt.running():
        if rt.has_model(model):
            console.print("Ya estaba descargado.")
            return
        with download_progress(console, model) as progress:
            rt.pull(model, progress)
    console.print(f"Listo: {model}")


@ollama_app.command("rm")
def ollama_rm(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Modelo a borrar.")],
) -> None:
    """Borra un modelo de models\\ollama y libera su espacio."""
    from mangatl.ollama_runtime import OllamaRuntime

    rt = OllamaRuntime(ctx.obj)
    with rt.running():
        if not rt.has_model(model):
            _fail(f"«{model}» no está descargado")
        rt.delete(model)
    console.print(f"Borrado: {model}")


@ollama_app.command("serve")
def ollama_serve(ctx: typer.Context) -> None:
    """Inicia Ollama en primer plano (Ctrl+C para detenerlo y liberar la VRAM)."""
    from mangatl.ollama_runtime import OllamaRuntime

    rt = OllamaRuntime(ctx.obj)
    with rt.running():
        console.print(f"Ollama escuchando en {rt.base_url} · Ctrl+C para detener")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("Deteniendo Ollama...")


if __name__ == "__main__":
    app()
