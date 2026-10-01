"""Plain HTTP downloads (model files that are not on Hugging Face or Ollama)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx


def remote_size(url: str, client: httpx.Client | None = None) -> int | None:
    http = client or httpx.Client(timeout=30.0)
    try:
        resp = http.head(url, follow_redirects=True)
        resp.raise_for_status()
        return int(resp.headers.get("Content-Length", 0)) or None
    except httpx.HTTPError:
        return None


def download_file(
    url: str,
    dest: Path,
    progress: Callable[[int, int | None], None] | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Stream `url` to `dest` (via a .part file, so an interrupted download never looks done)."""
    http = client or httpx.Client(timeout=httpx.Timeout(30.0, read=300.0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with http.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            done = 0
            with part.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest
