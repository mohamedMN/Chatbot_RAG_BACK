# utils/logger.py
from __future__ import annotations
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

# --- internal singleton guard ---
_INITIALIZED = False


def init_logging(
    level: Optional[str] = None,
    file_path: Optional[Path | str] = None,
    *,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: Optional[str] = None,
    capture_warnings: bool = True,
    uvicorn_loggers: Iterable[str] = (
        "uvicorn", "uvicorn.error", "uvicorn.access"),
) -> None:
    """
    Initialize root logging once (idempotent). Safe to call multiple times.
    - level: e.g. "DEBUG", "INFO" (falls back to env LOG_LEVEL or DEBUG)
    - file_path: where to write the log file (falls back to env LOG_FILE or ./log.txt)
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Resolve level / file
    level_name = (level or os.getenv("LOG_LEVEL", "DEBUG")).upper()
    level_val = getattr(logging, level_name, logging.DEBUG)
    file_path = Path(file_path or os.getenv("LOG_FILE", "log.txt")).resolve()

    # Ensure parent exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Handlers
    handlers: list[logging.Handler] = []

    # Rotating file handler (keeps logs from growing forever)
    file_handler = logging.handlers.RotatingFileHandler(
        file_path, mode="a", encoding="utf-8", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    handlers.append(file_handler)

    # Console
    stream_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stream_handler)

    # Root config
    logging.basicConfig(level=level_val, format=fmt,
                        datefmt=datefmt, handlers=handlers)

    # Capture warnings to logging
    if capture_warnings:
        logging.captureWarnings(True)

    # Let uvicorn loggers bubble up to root so they use our handlers
    for name in uvicorn_loggers:
        lg = logging.getLogger(name)
        lg.setLevel(level_val)
        lg.propagate = True

    _INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger. If init_logging() wasn’t called yet, initialize with defaults.
    """
    if not _INITIALIZED:
        init_logging()  # default bootstrap (safe)
    return logging.getLogger(name if name else "app")


def install_global_excepthook(logger_name: str = "unhandled") -> None:
    """
    Route uncaught exceptions to logging.
    """
    lg = get_logger(logger_name)

    def _excepthook(exc_type, exc_value, exc_tb):
        lg.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _excepthook
