"""Portable Ollama: install from the official zip, start/stop `ollama serve`, pull models.

The server is started with OLLAMA_MODELS pointing inside the project, so no permanent
Windows environment variable is needed and nothing is written to C: except Ollama's
tiny identity key in %USERPROFILE%\\.ollama.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from mangatl.config import GgufImport, Settings

log = logging.getLogger(__name__)

RELEASE_URL = (
    "https://github.com/ollama/ollama/releases/download/{version}/ollama-windows-amd64.zip"
)
RELEASE_API = "https://api.github.com/repos/ollama/ollama/releases/tags/{version}"
REGISTRY_MANIFEST = "https://registry.ollama.ai/v2/{namespace}/{name}/manifests/{tag}"
# Minimum Windows driver branch that runs CUDA 13 binaries.
CUDA13_MIN_DRIVER = 580

ProgressCallback = Callable[[int, int | None], None]


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComputeDevice:
    library: str
    name: str
    total: str | None = None
    available: str | None = None


_INFERENCE_COMPUTE = re.compile(r'msg="inference compute"(?P<rest>.*)')
_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')


def parse_compute_devices(log_text: str) -> list[ComputeDevice]:
    """Extract the GPUs/backends Ollama reports at startup from its server log."""
    devices: list[ComputeDevice] = []
    for match in _INFERENCE_COMPUTE.finditer(log_text):
        fields = {
            m.group(1): (m.group(3) if m.group(3) is not None else m.group(2))
            for m in _KV.finditer(match.group("rest"))
        }
        name = fields.get("description") or fields.get("name") or fields.get("id", "?")
        devices.append(
            ComputeDevice(
                library=fields.get("library", "?"),
                name=name,
                total=fields.get("total"),
                available=fields.get("available"),
            )
        )
    return devices


def split_model_name(model: str) -> tuple[str, str, str]:
    """'qwen3.5:4b' -> ('library', 'qwen3.5', '4b'); 'user/model' -> ('user', 'model', 'latest')."""
    name, _, tag = model.partition(":")
    namespace, _, short = name.rpartition("/")
    return namespace or "library", short, tag or "latest"


def modelfile_text(gguf_path: Path, spec: GgufImport) -> str:
    """Modelfile for `ollama create` from a GGUF, keeping the family's renderer and params."""
    lines = [f"FROM {gguf_path.as_posix()}"]
    if spec.renderer:
        lines.append(f"RENDERER {spec.renderer}")
    if spec.parser:
        lines.append(f"PARSER {spec.parser}")
    lines += [f"PARAMETER {key} {value}" for key, value in spec.parameters.items()]
    return "\n".join(lines) + "\n"


def cuda_dir_to_prune(driver_major: int | None) -> str | None:
    """Pick the CUDA runtime folder the installed driver does not need."""
    if driver_major is None:
        return None
    return "cuda_v12" if driver_major >= CUDA13_MIN_DRIVER else "cuda_v13"


class OllamaRuntime:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.cfg = settings.translator.ollama
        self.install_dir = settings.resolve(self.cfg.install_dir)
        self.models_dir = settings.resolve(self.cfg.models_dir)
        self.tmp_dir = settings.resolve(settings.paths.tmp_dir)
        self.log_file = settings.resolve(settings.paths.logs_dir) / "ollama-serve.log"
        self.base_url = self.cfg.host.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(10.0, read=None))
        self._process: subprocess.Popen[bytes] | None = None
        self._log_offset = 0

    # ---------------------------------------------------------------- install
    @property
    def exe_path(self) -> Path:
        return self.install_dir / "ollama.exe"

    def is_installed(self) -> bool:
        return self.exe_path.is_file()

    def release_zip_size(self) -> int | None:
        try:
            resp = self._client.get(RELEASE_API.format(version=self.cfg.version))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.debug("No se pudo consultar el tamaño del zip de Ollama: %s", exc)
            return None
        for asset in resp.json().get("assets", []):
            if asset.get("name") == "ollama-windows-amd64.zip":
                return int(asset["size"])
        return None

    def install(
        self, driver_major: int | None, progress: ProgressCallback | None = None
    ) -> list[str]:
        """Download and extract the portable build. Returns the pruned folders."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.tmp_dir / f"ollama-windows-amd64-{self.cfg.version}.zip"
        url = RELEASE_URL.format(version=self.cfg.version)
        log.info("Descargando Ollama %s...", self.cfg.version)
        try:
            with self._client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0)) or None
                done = 0
                with zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            log.info("Extrayendo Ollama en %s", self.install_dir)
            self.install_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.install_dir)
        finally:
            zip_path.unlink(missing_ok=True)
        return self.prune(driver_major)

    def prune(self, driver_major: int | None) -> list[str]:
        """Remove the CUDA runtime the driver does not need (saves ~0.7-1.1 GB)."""
        pruned: list[str] = []
        folder = cuda_dir_to_prune(driver_major)
        if folder:
            target = self.install_dir / "lib" / "ollama" / folder
            if target.is_dir():
                shutil.rmtree(target)
                pruned.append(folder)
                log.info("Eliminado lib/ollama/%s (no lo necesita tu driver)", folder)
        return pruned

    # ---------------------------------------------------------------- server
    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        host = urlparse(self.base_url)
        env.update(
            OLLAMA_MODELS=str(self.models_dir),
            OLLAMA_HOST=f"{host.hostname}:{host.port or 11434}",
            # Recent Ollama sizes the context from VRAM (4k-8k on 6 GB) and may ignore a larger
            # num_ctx; the server-level setting is the documented way to fix it.
            OLLAMA_CONTEXT_LENGTH=str(self.cfg.num_ctx),
            OLLAMA_FLASH_ATTENTION="1" if self.cfg.flash_attention else "0",
            OLLAMA_KV_CACHE_TYPE=self.cfg.kv_cache_type,
            TEMP=str(self.tmp_dir),
            TMP=str(self.tmp_dir),
        )
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        return env

    def is_running(self) -> bool:
        return self.version() is not None

    def version(self) -> str | None:
        try:
            resp = self._client.get(f"{self.base_url}/api/version", timeout=2.0)
            resp.raise_for_status()
            return resp.json().get("version")
        except (httpx.HTTPError, ValueError):
            return None

    @property
    def started_by_us(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        if self.is_running():
            log.info("Ollama ya está en ejecución en %s", self.base_url)
            return
        if not self.is_installed():
            raise OllamaError(
                f"No se encontró Ollama en {self.exe_path}. Ejecuta: uv run mangatl setup"
            )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_offset = self.log_file.stat().st_size if self.log_file.exists() else 0
        log.info("Iniciando Ollama (modelos en %s)...", self.models_dir)
        with self.log_file.open("ab") as log_fh:
            self._process = subprocess.Popen(
                [str(self.exe_path), "serve"],
                env=self.server_env(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                tail = self.session_log()[-2000:]
                self._process = None
                raise OllamaError(f"Ollama terminó al arrancar. Últimas líneas del log:\n{tail}")
            if self.is_running():
                log.info("Ollama listo (versión %s)", self.version())
                return
            time.sleep(0.5)
        self.stop()
        raise OllamaError(f"Ollama no respondió en {self.cfg.startup_timeout_s:.0f} s")

    def stop(self) -> None:
        """Unload models (frees VRAM) and stop the server if MangaTL started it."""
        if self._process is None:
            return
        with suppress(httpx.HTTPError):
            self.unload_all()
        # Kill the whole tree: `ollama serve` spawns runner processes that hold VRAM.
        subprocess.run(
            ["taskkill", "/PID", str(self._process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        log.info("Ollama detenido")

    @contextmanager
    def running(self) -> Iterator[OllamaRuntime]:
        """Ensure the server is up for the duration of the block."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def session_log(self) -> str:
        if not self.log_file.exists():
            return ""
        with self.log_file.open("rb") as fh:
            fh.seek(self._log_offset)
            return fh.read().decode("utf-8", errors="replace")

    def compute_devices(self) -> list[ComputeDevice]:
        return parse_compute_devices(self.session_log())

    # ---------------------------------------------------------------- models
    def list_models(self) -> list[dict]:
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def has_model(self, model: str) -> bool:
        wanted = model if ":" in model else f"{model}:latest"
        return any(m.get("name") == wanted or m.get("model") == wanted for m in self.list_models())

    def remote_model_size(self, model: str) -> int | None:
        namespace, name, tag = split_model_name(model)
        url = REGISTRY_MANIFEST.format(namespace=namespace, name=name, tag=tag)
        try:
            resp = self._client.get(
                url, headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
            )
            resp.raise_for_status()
            manifest = json.loads(resp.content)
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("No se pudo leer el manifiesto de %s: %s", model, exc)
            return None
        layers = manifest.get("layers", [])
        return sum(int(layer.get("size", 0)) for layer in layers) + int(
            manifest.get("config", {}).get("size", 0)
        )

    def pull(self, model: str, progress: ProgressCallback | None = None) -> None:
        with self._client.stream(
            "POST", f"{self.base_url}/api/pull", json={"model": model, "stream": True}
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise OllamaError(f"Error al descargar {model}: {event['error']}")
                if progress and "total" in event:
                    progress(int(event.get("completed", 0)), int(event["total"]))

    def create_from_gguf(self, name: str, gguf_path: Path, spec: GgufImport) -> None:
        """`ollama create` from a local GGUF (the server copies it into its blob store)."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        modelfile = self.tmp_dir / "Modelfile"
        modelfile.write_text(modelfile_text(gguf_path, spec), encoding="utf-8")
        try:
            result = subprocess.run(
                [str(self.exe_path), "create", name, "-f", str(modelfile)],
                env=self.server_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        finally:
            modelfile.unlink(missing_ok=True)
        if result.returncode != 0:
            raise OllamaError(f"No se pudo crear {name}: {(result.stderr or result.stdout)[-500:]}")

    def delete(self, model: str) -> None:
        resp = self._client.request("DELETE", f"{self.base_url}/api/delete", json={"model": model})
        resp.raise_for_status()

    def running_models(self) -> list[dict]:
        resp = self._client.get(f"{self.base_url}/api/ps")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def loaded_models(self) -> list[str]:
        return [m["name"] for m in self.running_models()]

    def chat(self, model: str, prompt: str, **options: object) -> dict:
        """Single non-streaming chat turn (used for smoke tests; the translator has its own)."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.cfg.think,
            "options": {"num_ctx": self.cfg.num_ctx, **options},
        }
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    def unload(self, model: str) -> None:
        """keep_alive=0 makes Ollama release the model's VRAM immediately."""
        resp = self._client.post(
            f"{self.base_url}/api/generate", json={"model": model, "keep_alive": 0}
        )
        resp.raise_for_status()

    def unload_all(self) -> None:
        for model in self.loaded_models():
            self.unload(model)
