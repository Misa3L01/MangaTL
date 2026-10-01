"""Logging: concise Spanish messages on the console, full detail in logs/mangatl.log."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler

from mangatl.config import Settings

console = Console()

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock", "PIL")


def setup_logging(settings: Settings, verbose: bool = False) -> None:
    logs_dir = settings.resolve(settings.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    console_handler = RichHandler(
        console=console,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="%H:%M:%S",
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    file_handler = RotatingFileHandler(
        logs_dir / "mangatl.log", maxBytes=5 * 2**20, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
