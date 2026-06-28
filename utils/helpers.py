from __future__ import annotations

<<<<<<< HEAD
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
=======
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return "_".join(part for part in cleaned.split("_") if part).lower()


def truncate_value(value: Any, max_length: int = 140) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."
>>>>>>> origin/main
