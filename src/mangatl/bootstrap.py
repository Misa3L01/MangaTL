"""`mangatl setup`: verify the environment and download everything the pipeline needs.

Every download states its size first and is refused if it would leave the project drive
below `setup.min_free_disk_gb`.
"""

from __future__ import annotations

import fnmatch
import logging
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from mangatl import gpu
from mangatl.config import GgufImport, Settings
from mangatl.disk import InsufficientSpaceError, check_space, fmt_size, free_bytes
from mangatl.fonts import OPTIONAL_CHARS, font_family_name, missing_glyphs
from mangatl.ollama_runtime import OllamaError, OllamaRuntime
from mangatl.runtime_env import paths_to_audit, same_drive

log = logging.getLogger(__name__)

Status = Literal["ok", "warn", "error"]
_ICONS = {"ok": "[green]✔[/]", "warn": "[yellow]⚠[/]", "error": "[red]✖[/]"}

# Rendered with a Windows Japanese font and read back by manga-ocr as an end-to-end check.
OCR_SAMPLE_TEXT = "今日はいい天気ですね"
JAPANESE_SYSTEM_FONTS = ("YuGothM.ttc", "YuGothR.ttc", "meiryo.ttc", "msgothic.ttc")
LLM_SMOKE_PROMPT = (
    "Traduce al español latinoamericano neutro. Responde solo con la traducción:\n"
    "「おはよう！今日も一日頑張ろうぜ！」"
)


@dataclass
class CheckResult:
    component: str
    status: Status
    detail: str


@dataclass(frozen=True)
class HfModelSpec:
    label: str
    repo: str
    revision: str
    patterns: tuple[str, ...]


@contextmanager
def download_progress(console: Console, label: str) -> Iterator[Callable[[int, int | None], None]]:
    with Progress(
        TextColumn(f"[bold]{label}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(label, total=None)

        def update(done: int, total: int | None) -> None:
            progress.update(task, completed=done, total=total)

        yield update


class SetupRunner:
    def __init__(
        self,
        settings: Settings,
        console: Console,
        *,
        check_only: bool = False,
        skip_llm: bool = False,
    ) -> None:
        self.settings = settings
        self.console = console
        self.check_only = check_only
        self.skip_llm = skip_llm
        self.results: list[CheckResult] = []
        self.min_free_gb = settings.setup.min_free_disk_gb
        self._driver_major: int | None = None
        self._model_dirs: dict[str, Path] = {}

    # ------------------------------------------------------------------ driver
    def run(self) -> bool:
        steps: list[tuple[str, Callable[[], None]]] = [
            ("Configuración", self._config_file),
            ("Rutas", self._paths),
            ("CUDA", self._cuda),
            ("Modelos de visión", self._vision_models),
            ("Prueba de modelos de visión", self._verify_vision),
            ("LaMa (limpieza sobre dibujo)", self._lama),
            ("Ollama", self._ollama),
            ("Fuentes", self._fonts),
            ("Editor web", self._web_ui),
        ]
        for name, step in steps:
            self.console.rule(f"[bold]{name}")
            try:
                step()
            except InsufficientSpaceError as exc:
                self._add(name, "error", str(exc))
            except Exception as exc:
                log.debug("Fallo en el paso %s", name, exc_info=True)
                self._add(name, "error", f"{type(exc).__name__}: {exc}")
        self._summary()
        return not any(r.status == "error" for r in self.results)

    def _add(self, component: str, status: Status, detail: str) -> None:
        self.results.append(CheckResult(component, status, detail))
        log.debug("%s [%s] %s", component, status, detail)

    def _announce_download(
        self, what: str, size: int, peak: int | None = None, final: int | None = None
    ) -> None:
        """Print size and remaining space; refuse if the peak usage breaks the free-space floor."""
        check_space(self.settings.root, peak or size, self.min_free_gb)
        remaining = free_bytes(self.settings.root) - (final or size)
        self.console.print(
            f"Descarga: [bold]{what}[/] · {fmt_size(size)} · "
            f"quedarán ~{fmt_size(remaining)} libres en {self.settings.root.anchor}"
        )

    # ------------------------------------------------------------------ steps
    def _config_file(self) -> None:
        root = self.settings.root
        config, example = root / "config.toml", root / "config.example.toml"
        if config.is_file():
            missing = missing_config_keys(config, example) if example.is_file() else []
            if missing:
                self._add(
                    "config.toml",
                    "warn",
                    f"Le faltan opciones nuevas (se usan los valores por defecto): "
                    f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}. "
                    "Cópialas de config.example.toml.",
                )
            else:
                self._add("config.toml", "ok", str(config))
        elif self.check_only:
            self._add("config.toml", "warn", "No existe; se usan los valores por defecto")
        elif example.is_file():
            shutil.copyfile(example, config)
            self._add("config.toml", "ok", "Creado a partir de config.example.toml")
        else:
            self._add(
                "config.toml", "warn", "Falta config.example.toml; se usan valores por defecto"
            )

    def _paths(self) -> None:
        s = self.settings
        if not self.check_only:
            for p in (s.paths.input_dir, s.paths.output_dir, s.paths.series_dir):
                s.resolve(p).mkdir(parents=True, exist_ok=True)
        off_drive = []
        for label, path in paths_to_audit(s).items():
            if not path.drive:
                off_drive.append(f"{label}: sin definir (carga env.ps1 antes de usar uv)")
            elif not same_drive(path, s.root):
                off_drive.append(f"{label}: {path}")
        if off_drive:
            self._add(
                "Rutas en " + s.root.anchor, "warn", "Fuera de la unidad: " + "; ".join(off_drive)
            )
        else:
            self._add(
                "Rutas en " + s.root.anchor,
                "ok",
                f"Todo dentro de {s.root.anchor} · libre: {fmt_size(free_bytes(s.root))}",
            )

    def _cuda(self) -> None:
        smi = gpu.query_nvidia_smi()
        if smi:
            self._driver_major = smi.driver_major
        info = gpu.torch_cuda_info()
        if not info.available:
            self._add(
                "CUDA",
                "error",
                f"PyTorch {info.torch_version} no ve ninguna GPU CUDA "
                f"(driver: {smi.driver_version if smi else 'no detectado'})",
            )
            return
        error = gpu.cuda_smoke_test()
        detail = (
            f"{info.device_name} · sm_{info.compute_capability.replace('.', '')} · "
            f"driver {smi.driver_version if smi else '?'} · torch {info.torch_version} · "
            f"VRAM libre {info.free_mb}/{info.total_mb} MiB · prueba matmul OK (err {error:.1e})"
        )
        status: Status = "ok" if error < 1e-2 else "error"
        self._add("CUDA", status, detail)
        if info.free_mb is not None and info.free_mb < self.settings.gpu.warn_free_vram_mb:
            self._add(
                "VRAM libre",
                "warn",
                f"Solo {info.free_mb} MiB libres: otras apps (navegador, Steam...) usan la GPU. "
                "Ciérralas antes de traducir un capítulo.",
            )

    def _hf_specs(self) -> list[HfModelSpec]:
        s = self.settings
        return [
            HfModelSpec(
                "Detector (RT-DETR-v2)",
                s.detection.model_repo,
                s.detection.model_revision,
                tuple(s.detection.model_files),
            ),
            HfModelSpec(
                "OCR japonés (manga-ocr)",
                s.ocr.ja_model_repo,
                s.ocr.ja_model_revision,
                ("*.json", "*.bin", "*.safetensors", "*.txt"),
            ),
        ]

    def _vision_models(self) -> None:
        from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

        api = HfApi()
        for spec in self._hf_specs():
            try:
                info = api.model_info(spec.repo, revision=spec.revision, files_metadata=True)
                files = [
                    f
                    for f in info.siblings or []
                    if any(fnmatch.fnmatch(f.rfilename, p) for p in spec.patterns)
                ]
                missing = [
                    f
                    for f in files
                    if not isinstance(
                        try_to_load_from_cache(spec.repo, f.rfilename, revision=spec.revision), str
                    )
                ]
            except Exception as exc:  # offline: fall back to whatever is cached
                log.debug("Sin acceso a Hugging Face para %s: %s", spec.repo, exc)
                files, missing = [], []

            if missing:
                size = sum(f.size or 0 for f in missing)
                if self.check_only:
                    self._add(
                        spec.label, "warn", f"Faltan {len(missing)} archivos ({fmt_size(size)})"
                    )
                    continue
                self._announce_download(spec.label, size)
                snapshot_download(
                    spec.repo, revision=spec.revision, allow_patterns=list(spec.patterns)
                )

            try:
                local = Path(
                    snapshot_download(
                        spec.repo,
                        revision=spec.revision,
                        allow_patterns=list(spec.patterns),
                        local_files_only=True,
                    )
                )
            except Exception:
                self._add(
                    spec.label, "error", "No está descargado y no hay conexión con Hugging Face"
                )
                continue
            size_on_disk = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
            self._model_dirs[spec.repo] = local
            self._add(spec.label, "ok", f"{spec.repo} · {fmt_size(size_on_disk)} · {local}")

    def _verify_vision(self) -> None:
        s = self.settings
        det_dir = self._model_dirs.get(s.detection.model_repo)
        ocr_dir = self._model_dirs.get(s.ocr.ja_model_repo)
        if det_dir is None or ocr_dir is None:
            self._add("Prueba de visión", "warn", "Omitida: faltan modelos")
            return

        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        device = "cuda" if s.gpu.device == "cuda" and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        processor = AutoImageProcessor.from_pretrained(det_dir)
        model = AutoModelForObjectDetection.from_pretrained(det_dir).to(device).eval()
        page = Image.new("RGB", (1100, 1600), "white")
        with torch.inference_mode():
            inputs = processor(images=page, return_tensors="pt").to(device)
            outputs = model(**inputs)
        labels = ", ".join(model.config.id2label.values())
        det_s = time.perf_counter() - t0
        del model, processor, inputs, outputs
        gpu.release_cuda_memory()
        self._add(
            "Detector", "ok", f"Carga + inferencia en {device}: {det_s:.1f} s · clases: {labels}"
        )

        from manga_ocr import MangaOcr

        t0 = time.perf_counter()
        mocr = MangaOcr(pretrained_model_name_or_path=str(ocr_dir), force_cpu=device == "cpu")
        sample = _render_japanese_sample(OCR_SAMPLE_TEXT)
        if sample is None:
            self._add("manga-ocr", "ok", f"Cargado en {device} (sin fuente japonesa para probar)")
        else:
            text = mocr(sample)
            ocr_s = time.perf_counter() - t0
            status: Status = "ok" if text == OCR_SAMPLE_TEXT else "warn"
            self._add(
                "manga-ocr",
                status,
                f"Carga + lectura en {device}: {ocr_s:.1f} s · esperado «{OCR_SAMPLE_TEXT}» "
                f"· leído «{text}»",
            )
        del mocr
        gpu.release_cuda_memory()
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() // 2**20
            self._add("VRAM etapa visión", "ok", f"Pico con detector y OCR: {peak} MiB")
        self._verify_english_ocr()

    def _verify_english_ocr(self) -> None:
        from PIL import Image, ImageDraw, ImageFont

        from mangatl.ocr.english import PaddleRapidOcrEngine

        sample = "WHY DOES EVERYONE THINK DOCTORS ARE RICH?"
        try:
            font = ImageFont.truetype("arial.ttf", 34)
        except OSError:
            font = ImageFont.load_default()
        img = Image.new("RGB", (round(font.getlength(sample)) + 24, 70), "white")
        ImageDraw.Draw(img).text((12, 14), sample, font=font, fill="black")
        t0 = time.perf_counter()
        [result] = PaddleRapidOcrEngine().read([img])
        ok = result.text.replace(" ", "") == sample.replace(" ", "")
        self._add(
            "OCR inglés (PP-OCR)",
            "ok" if ok else "warn",
            f"{time.perf_counter() - t0:.1f} s · esperado «{sample}» · leído «{result.text}»",
        )

    def _lama(self) -> None:
        from mangatl.downloads import download_file, remote_size

        cfg = self.settings.inpaint
        if not cfg.use_lama:
            self._add("LaMa", "ok", "Desactivado (inpaint.use_lama = false)")
            return
        path = self.settings.resolve(cfg.lama_file)
        if not path.is_file():
            if self.check_only:
                self._add("LaMa", "warn", f"No descargado ({path.name})")
                return
            size = remote_size(cfg.lama_url) or 200 * 2**20
            self._announce_download("LaMa anime-manga-big-lama", size)
            with download_progress(self.console, "LaMa") as progress:
                download_file(cfg.lama_url, path, progress)

        import numpy as np
        import torch

        from mangatl.inpainting.lama import LamaInpainter

        device = (
            "cuda" if self.settings.gpu.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        t0 = time.perf_counter()
        lama = LamaInpainter(path, device=device, max_side=cfg.lama_max_side)
        page = np.full((256, 256, 3), 255, np.uint8)
        page[100:140, 60:200] = 0  # a black bar to erase
        labels = np.zeros((256, 256), np.uint8)
        labels[96:144, 56:204] = 1
        from mangatl.models import BBox, Region

        region = Region(
            id="P000-B01", type="text_on_art", bbox=BBox(x0=56, y0=96, x1=204, y1=144), mask_label=1
        )
        out = lama.clean(page, labels, [region])
        lama.close()
        erased = float(out[100:140, 60:200].mean())
        size_mb = path.stat().st_size / 2**20
        status: Status = "ok" if erased > 200 else "warn"
        self._add(
            "LaMa",
            status,
            f"{path.name} · {size_mb:.0f} MB · "
            f"prueba en {device}: {time.perf_counter() - t0:.1f} s "
            f"(zona borrada, gris medio {erased:.0f}/255)",
        )

    def _ollama(self) -> None:
        rt = OllamaRuntime(self.settings)
        cfg = self.settings.translator.ollama

        if not rt.is_installed():
            if self.check_only:
                self._add("Ollama", "error", f"No instalado en {rt.install_dir}")
                return
            zip_size = rt.release_zip_size() or int(1.5 * 2**30)
            # Peak: zip + extracted files (~1.4x the zip). After pruning one CUDA runtime and
            # deleting the zip, roughly 0.6x the zip stays on disk.
            self._announce_download(
                f"Ollama {cfg.version} (portable)",
                zip_size,
                peak=int(zip_size * 2.4),
                final=int(zip_size * 0.6),
            )
            with download_progress(self.console, "Ollama") as progress:
                pruned = rt.install(self._driver_major, progress)
            if pruned:
                self.console.print(
                    f"Eliminado lib/ollama/{', '.join(pruned)} (no lo necesita tu driver)"
                )
        elif not self.check_only:
            rt.prune(self._driver_major)

        size = sum(p.stat().st_size for p in rt.install_dir.rglob("*") if p.is_file())
        self._add("Ollama (binarios)", "ok", f"{cfg.version} · {fmt_size(size)} · {rt.install_dir}")

        external = rt.is_running()
        try:
            rt.start()
        except OllamaError as exc:
            self._add("Servidor Ollama", "error", str(exc))
            return
        try:
            self._ollama_server_checks(rt, external)
        finally:
            rt.stop()

    def _ollama_server_checks(self, rt: OllamaRuntime, external: bool) -> None:
        cfg = self.settings.translator.ollama
        if external:
            self._add(
                "Servidor Ollama",
                "warn",
                f"Ya había un Ollama en ejecución en {cfg.host} (no lo inició MangaTL); "
                "sus modelos podrían no estar en R:",
            )
        else:
            devices = rt.compute_devices()
            cuda = [d for d in devices if d.library.upper().startswith("CUDA")]
            others = [d for d in devices if d not in cuda]
            detail = "; ".join(f"{d.library}: {d.name} ({d.total or '?'})" for d in devices)
            if cuda and not others:
                self._add("Servidor Ollama", "ok", f"v{rt.version()} · {detail}")
            elif cuda:
                self._add("Servidor Ollama", "warn", f"v{rt.version()} · también detecta: {detail}")
            else:
                self._add(
                    "Servidor Ollama",
                    "error",
                    f"v{rt.version()} · no detecta la GPU NVIDIA por CUDA ({detail or 'sin GPUs'})",
                )

        if not rt.has_model(cfg.model):
            if self.check_only or self.skip_llm:
                self._add(f"LLM {cfg.model}", "warn", "No descargado")
                return
            if cfg.gguf is not None and cfg.gguf.name == cfg.model:
                self._import_gguf(rt, cfg.model, cfg.gguf)
            else:
                size = rt.remote_model_size(cfg.model) or int(3.5 * 2**30)
                self._announce_download(f"LLM {cfg.model}", size)
                with download_progress(self.console, cfg.model) as progress:
                    rt.pull(cfg.model, progress)
        model_info = next((m for m in rt.list_models() if m.get("name") == cfg.model), {"size": 0})
        self._add(f"LLM {cfg.model}", "ok", f"{fmt_size(model_info['size'])} · {rt.models_dir}")

        t0 = time.perf_counter()
        reply = rt.chat(cfg.model, LLM_SMOKE_PROMPT, temperature=0.2)
        total_s = time.perf_counter() - t0
        eval_count = reply.get("eval_count", 0)
        eval_s = reply.get("eval_duration", 0) / 1e9
        tps = eval_count / eval_s if eval_s else 0.0
        answer = reply.get("message", {}).get("content", "").strip().replace("\n", " ")
        running = next((m for m in rt.running_models() if m.get("name") == cfg.model), None)
        placement = ""
        if running and running.get("size"):
            share = running.get("size_vram", 0) / running["size"]
            placement = f" · {share:.0%} en GPU ({fmt_size(running.get('size_vram', 0))} VRAM)"
        self._add(
            "Prueba del LLM",
            "ok" if answer else "error",
            f"«{answer[:120]}» · {tps:.0f} tokens/s · {total_s:.1f} s en total "
            f"(num_ctx={cfg.num_ctx}){placement}",
        )

    def _import_gguf(self, rt: OllamaRuntime, name: str, spec: GgufImport) -> None:
        """Create `name` in Ollama from a GGUF: models/gguf/<file> if present, else downloaded
        to tmp/ and deleted once Ollama has copied it into its blob store."""
        from huggingface_hub import HfApi, hf_hub_download

        local = self.settings.resolve(self.settings.paths.models_dir) / "gguf" / spec.file
        download_dir = None
        if local.is_file():
            check_space(self.settings.root, local.stat().st_size, self.min_free_gb)
        else:
            info = HfApi().model_info(spec.repo, revision=spec.revision, files_metadata=True)
            size = next((s.size for s in info.siblings or [] if s.rfilename == spec.file), 0)
            if not size:
                raise OllamaError(f"{spec.file} no está en {spec.repo}@{spec.revision}")
            # Peak: the download plus Ollama's copy; the download is deleted afterwards.
            self._announce_download(f"GGUF {spec.file}", size, peak=2 * size, final=size)
            download_dir = rt.tmp_dir / "gguf"
            local = Path(
                hf_hub_download(
                    spec.repo, spec.file, revision=spec.revision, local_dir=download_dir
                )
            )
        try:
            with self.console.status(f"Creando {name} en Ollama desde {local.name}..."):
                rt.create_from_gguf(name, local, spec)
        finally:
            if download_dir is not None:
                shutil.rmtree(download_dir, ignore_errors=True)

    def _fonts(self) -> None:
        fonts = self.settings.typesetting.fonts
        for style, rel in fonts.model_dump().items():
            path = self.settings.resolve(Path(rel))
            if not path.is_file():
                self._add(f"Fuente «{style}»", "error", f"No existe: {path}")
                continue
            missing = missing_glyphs(path)
            optional = missing_glyphs(path, OPTIONAL_CHARS)
            if missing:
                self._add(
                    f"Fuente «{style}»",
                    "error",
                    f"{path.name}: faltan glifos obligatorios {''.join(missing)}",
                )
                continue
            note = f" (sin {''.join(optional)}: se sustituyen al rotular)" if optional else ""
            self._add(f"Fuente «{style}»", "ok", f"{font_family_name(path)} · {path.name}{note}")

    def _web_ui(self) -> None:
        from mangatl.server.app import WEB_DIST

        if (WEB_DIST / "index.html").is_file():
            self._add("Editor web", "ok", f"Interfaz compilada en {WEB_DIST} · uv run mangatl ui")
        else:
            self._add(
                "Editor web",
                "warn",
                "Interfaz sin compilar: ejecuta .\\scripts\\build-web.ps1 (instala Node portable)",
            )

    # ------------------------------------------------------------------ output
    def _summary(self) -> None:
        table = Table(title="Resumen de mangatl setup", show_lines=False)
        table.add_column("Componente", style="bold")
        table.add_column("", justify="center")
        table.add_column("Detalle", overflow="fold")
        for r in self.results:
            table.add_row(r.component, _ICONS[r.status], r.detail)
        self.console.print(table)
        errors = sum(r.status == "error" for r in self.results)
        warns = sum(r.status == "warn" for r in self.results)
        if errors:
            self.console.print(f"[red]Hay {errors} error(es). Revisa logs/mangatl.log.[/]")
        elif warns:
            self.console.print(f"[yellow]Listo, con {warns} aviso(s).[/]")
        else:
            self.console.print("[green]Todo listo.[/]")


def missing_config_keys(config: Path, example: Path) -> list[str]:
    """Dotted keys present in config.example.toml but absent from the user's config.toml."""
    import tomllib

    def flatten(data: dict, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        for key, value in data.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                keys |= flatten(value, f"{dotted}.")
            else:
                keys.add(dotted)
        return keys

    with config.open("rb") as fh:
        user = flatten(tomllib.load(fh))
    with example.open("rb") as fh:
        reference = flatten(tomllib.load(fh))
    return sorted(reference - user)


def _render_japanese_sample(text: str):
    """Render `text` with a Japanese Windows system font (read-only), or None if unavailable."""
    from PIL import Image, ImageDraw, ImageFont

    fonts_dir = Path("C:/Windows/Fonts")
    font_path = next(
        (fonts_dir / f for f in JAPANESE_SYSTEM_FONTS if (fonts_dir / f).is_file()), None
    )
    if font_path is None:
        return None
    font = ImageFont.truetype(str(font_path), 48)
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("RGB", (right - left + 40, bottom - top + 40), "white")
    ImageDraw.Draw(img).text((20 - left, 20 - top), text, font=font, fill="black")
    return img
