from __future__ import annotations

"""
Central logging setup.

Usage:
    from apps.logger import get_logger, get_job_logger
    log = get_logger(__name__)          # module-level
    jlog = get_job_logger(job_id)       # per-job file
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(os.environ.get("LOGS_DIR", "./logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / "jobs").mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

# ── Root handler setup (called once from main.py) ─────────────────────────────

def setup_root_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove default handlers
    root.handlers.clear()

    # Console — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(_formatter)
    root.addHandler(console)

    # Main rotating file — DEBUG and above
    main_file = RotatingFileHandler(
        LOGS_DIR / "videosum.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(_formatter)
    root.addHandler(main_file)

    # Errors-only file
    error_file = RotatingFileHandler(
        LOGS_DIR / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(_formatter)
    root.addHandler(error_file)


# ── Per-job logger ────────────────────────────────────────────────────────────

_job_loggers: dict[str, logging.Logger] = {}


def get_job_logger(job_id: str) -> logging.Logger:
    """Returns a logger that writes to logs/jobs/{job_id}.log + root handlers."""
    if job_id in _job_loggers:
        return _job_loggers[job_id]

    logger = logging.getLogger(f"job.{job_id}")
    logger.setLevel(logging.DEBUG)

    job_file = RotatingFileHandler(
        LOGS_DIR / "jobs" / f"{job_id}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    job_file.setLevel(logging.DEBUG)
    job_file.setFormatter(_formatter)
    logger.addHandler(job_file)

    _job_loggers[job_id] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
