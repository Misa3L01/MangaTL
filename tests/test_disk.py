from __future__ import annotations

from pathlib import Path

import pytest

from mangatl import disk
from mangatl.disk import GIB, InsufficientSpaceError, check_space, fmt_size


def test_check_space_returns_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk, "free_bytes", lambda _: 10 * GIB)
    assert check_space(Path("R:/"), 4 * GIB, min_free_gb=3.0) == 6 * GIB


def test_check_space_refuses_to_break_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk, "free_bytes", lambda _: 10 * GIB)
    with pytest.raises(InsufficientSpaceError, match=r"al menos 3\.0 GB"):
        check_space(Path("R:/"), int(7.5 * GIB), min_free_gb=3.0)


def test_fmt_size() -> None:
    assert fmt_size(164 * 2**20) == "164 MB"
    assert fmt_size(3.39e9) == "3.16 GB"
