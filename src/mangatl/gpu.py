"""GPU inspection and VRAM housekeeping.

torch is imported lazily so that commands which do not need it stay fast.
"""

from __future__ import annotations

import gc
import logging
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NvidiaSmiInfo:
    name: str
    driver_version: str
    memory_total_mb: int
    memory_used_mb: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def driver_major(self) -> int:
        return int(self.driver_version.split(".")[0])


@dataclass(frozen=True)
class TorchCudaInfo:
    torch_version: str
    cuda_version: str | None
    available: bool
    device_name: str | None = None
    compute_capability: str | None = None
    total_mb: int | None = None
    free_mb: int | None = None


def query_nvidia_smi() -> NvidiaSmiInfo | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("nvidia-smi falló: %s", exc)
        return None
    first = out.strip().splitlines()[0]
    name, driver, total, used = (part.strip() for part in first.split(","))
    return NvidiaSmiInfo(name, driver, int(total), int(used))


def torch_cuda_info() -> TorchCudaInfo:
    import torch

    if not torch.cuda.is_available():
        return TorchCudaInfo(torch.__version__, torch.version.cuda, available=False)
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return TorchCudaInfo(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        available=True,
        device_name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        total_mb=total // 2**20,
        free_mb=free // 2**20,
    )


def cuda_smoke_test(size: int = 2048) -> float:
    """Run a matmul on the GPU and return the max abs error against the CPU result."""
    import torch

    a = torch.randn(size, size)
    b = torch.randn(size, size)
    expected = a @ b
    result = (a.cuda() @ b.cuda()).cpu()
    error = (result - expected).abs().max().item()
    del a, b, expected, result
    release_cuda_memory()
    return error


def release_cuda_memory() -> None:
    """Free cached CUDA blocks. Note: the CUDA context itself stays until the process exits."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
