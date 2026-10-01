"""Disk-space guard: downloads must never leave the project drive below the configured floor."""

from __future__ import annotations

import shutil
from pathlib import Path

GIB = 2**30


class InsufficientSpaceError(RuntimeError):
    pass


def free_bytes(path: Path) -> int:
    anchor = Path(path.anchor or path)
    return shutil.disk_usage(anchor).free


def fmt_size(n_bytes: int | float) -> str:
    if n_bytes >= GIB:
        return f"{n_bytes / GIB:.2f} GB"
    return f"{n_bytes / 2**20:.0f} MB"


def check_space(path: Path, required: int, min_free_gb: float) -> int:
    """Return the free bytes left after downloading `required`; raise if below the floor."""
    free = free_bytes(path)
    remaining = free - required
    if remaining < min_free_gb * GIB:
        raise InsufficientSpaceError(
            f"Espacio insuficiente en {Path(path).anchor}: la descarga ocupa {fmt_size(required)}, "
            f"hay {fmt_size(free)} libres y deben quedar al menos {min_free_gb:.1f} GB."
        )
    return remaining
