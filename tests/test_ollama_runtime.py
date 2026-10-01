from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from mangatl.config import Settings
from mangatl.ollama_runtime import (
    OllamaError,
    OllamaRuntime,
    cuda_dir_to_prune,
    parse_compute_devices,
    split_model_name,
)

SERVER_LOG = """\
time=2026-09-25T15:00:00.000-06:00 level=INFO source=routes.go:1500 msg="Listening on 127.0.0.1:11434 (version 0.34.4)"
time=2026-09-25T15:00:01.000-06:00 level=INFO source=types.go:131 msg="inference compute" id=GPU-1234 library=CUDA compute=8.9 name=CUDA0 description="NVIDIA GeForce RTX 4050 Laptop GPU" libdirs=ollama,cuda_v13 driver=13.4 type=discrete total="6.0 GiB" available="4.9 GiB"
time=2026-09-25T15:00:01.000-06:00 level=INFO source=types.go:131 msg="inference compute" id=0 library=Vulkan name=Vulkan1 description="AMD Radeon(TM) 760M Graphics" type=iGPU total="8.0 GiB" available="7.5 GiB"
"""


def make_runtime(settings: Settings, handler) -> OllamaRuntime:
    return OllamaRuntime(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_parse_compute_devices() -> None:
    devices = parse_compute_devices(SERVER_LOG)
    assert [(d.library, d.name) for d in devices] == [
        ("CUDA", "NVIDIA GeForce RTX 4050 Laptop GPU"),
        ("Vulkan", "AMD Radeon(TM) 760M Graphics"),
    ]
    assert devices[0].total == "6.0 GiB"
    assert devices[0].available == "4.9 GiB"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3.5:4b", ("library", "qwen3.5", "4b")),
        ("translategemma", ("library", "translategemma", "latest")),
        ("someone/model:q4", ("someone", "model", "q4")),
    ],
)
def test_split_model_name(model: str, expected: tuple[str, str, str]) -> None:
    assert split_model_name(model) == expected


def test_cuda_dir_to_prune() -> None:
    assert cuda_dir_to_prune(616) == "cuda_v12"  # driver 616.64 runs CUDA 13
    assert cuda_dir_to_prune(560) == "cuda_v13"
    assert cuda_dir_to_prune(None) is None


def test_prune_removes_unneeded_cuda_runtime(settings: Settings) -> None:
    rt = OllamaRuntime(settings)
    lib = rt.install_dir / "lib" / "ollama"
    for name in ("cuda_v12", "cuda_v13"):
        (lib / name).mkdir(parents=True)
        (lib / name / "ggml-cuda.dll").write_bytes(b"x")
    assert rt.prune(616) == ["cuda_v12"]
    assert not (lib / "cuda_v12").exists()
    assert (lib / "cuda_v13" / "ggml-cuda.dll").exists()


def test_server_env_points_models_into_project(settings: Settings, tmp_path: Path) -> None:
    env = OllamaRuntime(settings).server_env()
    assert Path(env["OLLAMA_MODELS"]) == (tmp_path / "models" / "ollama").resolve()
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert Path(env["TEMP"]) == (tmp_path / "tmp").resolve()


def test_version_and_is_running(settings: Settings) -> None:
    rt = make_runtime(settings, lambda req: httpx.Response(200, json={"version": "0.34.4"}))
    assert rt.version() == "0.34.4"
    assert rt.is_running()


def test_not_running_on_connection_error(settings: Settings) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert not make_runtime(settings, refuse).is_running()


def test_has_model_matches_implicit_latest(settings: Settings) -> None:
    tags = {"models": [{"name": "qwen3.5:4b", "size": 1}, {"name": "gemma:latest", "size": 1}]}
    rt = make_runtime(settings, lambda req: httpx.Response(200, json=tags))
    assert rt.has_model("qwen3.5:4b")
    assert rt.has_model("gemma")
    assert not rt.has_model("qwen3.5:9b")


def test_remote_model_size_sums_layers(settings: Settings) -> None:
    manifest = {
        "config": {"size": 475},
        "layers": [{"size": 3_389_971_840}, {"size": 11_355}, {"size": 65}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/library/qwen3.5/manifests/4b"
        return httpx.Response(200, content=json.dumps(manifest).encode())

    assert make_runtime(settings, handler).remote_model_size("qwen3.5:4b") == 3_389_983_735


def test_pull_reports_progress(settings: Settings) -> None:
    events = [
        {"status": "pulling manifest"},
        {"status": "pulling 81fb", "total": 100, "completed": 40},
        {"status": "pulling 81fb", "total": 100, "completed": 100},
        {"status": "success"},
    ]
    body = "\n".join(json.dumps(e) for e in events).encode()
    rt = make_runtime(settings, lambda req: httpx.Response(200, content=body))
    seen: list[tuple[int, int | None]] = []
    rt.pull("qwen3.5:4b", lambda done, total: seen.append((done, total)))
    assert seen == [(40, 100), (100, 100)]


def test_pull_raises_on_error_event(settings: Settings) -> None:
    body = json.dumps({"error": "pull model manifest: file does not exist"}).encode()
    rt = make_runtime(settings, lambda req: httpx.Response(200, content=body))
    with pytest.raises(OllamaError, match="file does not exist"):
        rt.pull("nope:1b")


def test_start_without_install_explains_how_to_fix(settings: Settings) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaError, match="mangatl setup"):
        make_runtime(settings, refuse).start()
