"""
Centralized logging configuration for the project.

Usage
-----
In your main/entry script (once, near startup):

    from utils.logging_config import setup_logging
    setup_logging()

In any module:

    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Hello")

This module is idempotent: calling setup_logging() multiple times will not
add duplicate handlers or produce duplicate log lines.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

_LOGGING_CONFIGURED: bool = False


def setup_logging(
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure project-wide logging.

    - Console handler (INFO by default)
    - Rotating file handler logs/app.log (DEBUG by default)
    - Timestamp format: HH:MM only (no date, no seconds)
    - Message format excludes the level name (just time + message)
    - Idempotent: safe to call multiple times without duplicating handlers.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    # Determine log file path
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
    else:
        log_dir = log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)

    # Basic formatter: HH:MM only, unicode-safe, NO level name in output
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%H:%M",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter levels

    # Remove any pre-existing handlers once so we control the configuration.
    # This also prevents duplicate emission on repeated setup in the same process.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Console handler (stream to stderr by default)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,  # ~10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Ensure child loggers propagate to root and don't have their own handlers
    root_logger.propagate = False

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name, using the centralized configuration.

    Note: this does NOT call setup_logging() automatically; callers should
    ensure setup_logging() was invoked once in the main entrypoint.
    """
    logger = logging.getLogger(name)
    # Ensure loggers created elsewhere don't accumulate their own handlers
    # which could cause duplicate messages when combined with the root logger.
    if logger is not logging.getLogger():
        logger.propagate = True
        if logger.handlers:
            # Clear any accidental per-module handlers so everything flows
            # through the centralized root handlers.
            logger.handlers.clear()
    return logger


__all__ = ["setup_logging", "get_logger"]


