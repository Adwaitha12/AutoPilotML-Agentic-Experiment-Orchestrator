from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logger(name: str, log_dir: str = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """Create a reusable file-based logger for the project."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    file_name = f"{name.replace('.', '_')}.log"
    log_file = log_path / file_name
    if not any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_file.resolve())
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)

    return logger
