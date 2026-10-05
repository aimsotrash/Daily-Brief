"""Logging configuration, kept out of application logic."""

from __future__ import annotations

import logging
import sys

from .config import Config

_CONFIGURED = False


def setup_logging(config: Config, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_path = config.log_path()
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - permissions/full disk
            root.warning("could not open log file %s: %s", log_path, exc)

    # These are chatty at DEBUG and never useful here.
    for noisy in ("httpx", "httpcore", "apscheduler.executors.default"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
