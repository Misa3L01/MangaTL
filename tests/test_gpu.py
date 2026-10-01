from __future__ import annotations

import pytest

from mangatl import gpu

torch = pytest.importorskip("torch")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")


@pytest.mark.gpu
@requires_cuda
def test_torch_sees_the_nvidia_gpu() -> None:
    info = gpu.torch_cuda_info()
    assert info.available
    assert "NVIDIA" in (info.device_name or "")
    assert info.total_mb and info.total_mb > 0


@pytest.mark.gpu
@requires_cuda
def test_cuda_matmul_matches_cpu() -> None:
    assert gpu.cuda_smoke_test(size=512) < 1e-2


def test_nvidia_smi_parsing_is_optional() -> None:
    info = gpu.query_nvidia_smi()
    if info is not None:
        assert info.driver_major > 0
        assert info.memory_total_mb >= info.memory_used_mb
