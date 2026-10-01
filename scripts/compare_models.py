"""Compare local LLMs and translation settings on an already OCR'd project.

    uv run python scripts/compare_models.py run --project <p.mangatl.json> --pages 61-72 \
        --model qwen3.5:4b --tag qwen4b [--pivot] [--set translator.page_images=true ...]
    uv run python scripts/compare_models.py report --tags qwen4b qwen4b-pivot ... \
        --out output/evals/comparacion.md

`--set` overrides any setting for this run only (dotted key, TOML value), so experimental
options can be measured without editing config.toml. Results go to output/evals/<tag>.json
(outside git).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import tomllib
from pathlib import Path

from mangatl.config import Settings, load_settings
from mangatl.runtime_env import apply_runtime_env

EVALS = Path(__file__).resolve().parents[1] / "output" / "evals"
_OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")


def apply_overrides(settings: Settings, overrides: list[str]) -> Settings:
    """Return a copy of `settings` with dotted `key=value` overrides (values parsed as TOML)."""
    data = settings.model_dump()
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set necesita CLAVE=VALOR: {item}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # bare strings: --set translator.ollama.model=qwen3.5:4b
        node = data
        *parents, leaf = key.strip().split(".")
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise SystemExit(f"Clave desconocida: {key}")
            node = node[part]
        if leaf not in node:
            raise SystemExit(f"Clave desconocida: {key}")
        node[leaf] = value
    return Settings.model_validate(data)


def gpu_layers(server_log: str) -> str | None:
    """'18/34' from the last 'offloaded N/M layers to GPU' line of the server log."""
    found = _OFFLOADED.findall(server_log)
    return f"{found[-1][0]}/{found[-1][1]}" if found else None


def run(args: argparse.Namespace) -> None:
    settings = load_settings()
    apply_runtime_env(settings)
    from mangatl.ingest import parse_page_range
    from mangatl.logging_setup import setup_logging
    from mangatl.ollama_runtime import OllamaRuntime
    from mangatl.project_io import ProjectPaths, load_project
    from mangatl.translation.base import translate_chapter
    from mangatl.translation.ollama_backend import OllamaTranslator

    setup_logging(settings)
    overrides = list(args.set or [])
    if args.model:
        overrides.append(f"translator.ollama.model={args.model}")
    settings = apply_overrides(settings, overrides)
    model = settings.translator.ollama.model
    project = load_project(args.project).model_copy(deep=True)
    paths = ProjectPaths(args.project)
    selection = parse_page_range(args.pages)
    project.pages = [p for p in project.pages if selection is None or p.number in selection]
    project.pending_glossary = []
    for page in project.pages:
        for region in page.regions:
            region.translation = region.speaker = region.source_text_corrected = None
            region.status = "auto" if region.status != "skipped" else "skipped"

    runtime = OllamaRuntime(settings)
    with runtime.running():
        translator = OllamaTranslator(settings)
        start = time.perf_counter()
        stats = translate_chapter(
            project,
            translator,
            [],
            settings.translator.pages_per_block,
            settings.translator.previous_lines,
            pivot=args.pivot,
            **_chapter_extras(settings, paths),
        )
        seconds = time.perf_counter() - start
        loaded = next((m for m in runtime.running_models() if m.get("name") == model), {})
        layers = gpu_layers(runtime.session_log())
        translator.close()

    size, vram = loaded.get("size", 0), loaded.get("size_vram", 0)
    result = {
        "tag": args.tag,
        "model": model,
        "mode": "pivote JA→EN→ES" if args.pivot else "directo JA→ES",
        "overrides": overrides,
        "pages": [p.number for p in project.pages],
        "seconds": round(seconds, 1),
        "seconds_per_page": round(seconds / max(1, len(project.pages)), 1),
        "tokens_per_second": round(stats.tokens_per_second, 1),
        "requests": stats.requests,
        "prompt_tokens": stats.prompt_tokens,
        "output_tokens": stats.output_tokens,
        "missing": stats.missing,
        "realigned": translator.realigned,
        "context_peak": stats.context_peak,
        "num_ctx": settings.translator.ollama.num_ctx,
        "size_gb": round(size / 2**30, 2),
        "vram_gb": round(vram / 2**30, 2),
        "gpu_share": round(vram / size, 2) if size else None,
        "gpu_layers": layers,
        "summary": project.chapter_summary,
        "glossary": [(g.source, g.target) for g in project.pending_glossary],
        "regions": {
            r.id: {
                "ja": r.ocr_text,
                "es": r.translation,
                "speaker": r.speaker,
                "style": r.style,
                "confidence": r.translation_confidence,
                "corrected": r.source_text_corrected,
            }
            for p in project.pages
            for r in p.regions
            if r.ocr_text
        },
    }
    EVALS.mkdir(parents=True, exist_ok=True)
    out = EVALS / f"{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{args.tag}: {seconds:.0f} s ({result['seconds_per_page']} s/página), "
        f"{result['tokens_per_second']} tok/s, salida {stats.output_tokens} tokens, "
        f"VRAM {result['vram_gb']} GB de {result['size_gb']} GB (capas {layers}), "
        f"faltantes {len(stats.missing)}, realineadas {translator.realigned} -> {out}"
    )


def _chapter_extras(settings: Settings, paths: object) -> dict[str, object]:
    """Optional translate_chapter arguments that depend on experimental settings."""
    return {}


def report(args: argparse.Namespace) -> None:
    runs = [json.loads((EVALS / f"{t}.json").read_text(encoding="utf-8")) for t in args.tags]
    lines = ["# Comparación de traducción local", ""]
    lines += [
        "| Variante | Modelo | Ajustes | Tiempo | s/página | tok/s | Tokens salida | VRAM "
        "| En GPU | Capas GPU | Faltantes | Realineadas |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        share = f"{r['gpu_share']:.0%}" if r.get("gpu_share") is not None else "?"
        settings = ", ".join(r.get("overrides") or []) or r["mode"]
        lines.append(
            f"| {r['tag']} | {r['model']} | {settings} | {r['seconds']:.0f} s | "
            f"{r['seconds_per_page']} | {r['tokens_per_second']} | "
            f"{r.get('output_tokens', '?')} | {r['vram_gb']} GB | {share} | "
            f"{r.get('gpu_layers') or '?'} | {len(r['missing'])} | {r['realigned']} |"
        )
    lines += ["", "## Traducciones lado a lado", ""]
    ids = list(runs[0]["regions"])
    header = "| ID | Japonés (OCR) | " + " | ".join(r["tag"] for r in runs) + " |"
    lines += [header, "|" + "---|" * (2 + len(runs))]
    for rid in ids:
        ja = runs[0]["regions"][rid]["ja"].replace("|", "¦")
        cells = []
        for r in runs:
            entry = r["regions"].get(rid, {})
            es = (entry.get("es") or "—").replace("|", "¦")
            who = entry.get("speaker")
            cells.append(f"{es} *({who})*" if who and who != "desconocido" else es)
        lines.append(f"| {rid} | {ja} | " + " | ".join(cells) + " |")
    lines += ["", "## Resúmenes", ""]
    for r in runs:
        lines += [f"**{r['tag']}:** {r['summary']}", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Informe: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--project", type=Path, required=True)
    p_run.add_argument("--pages", default=None)
    p_run.add_argument("--model", default=None, help="Modelo de Ollama (por defecto, el de config)")
    p_run.add_argument("--tag", required=True)
    p_run.add_argument("--pivot", action="store_true")
    p_run.add_argument(
        "--set", action="append", metavar="CLAVE=VALOR", help="Ajuste solo para esta ejecución"
    )
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--tags", nargs="+", required=True)
    p_rep.add_argument("--out", default=str(EVALS / "comparacion.md"))
    args = parser.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
