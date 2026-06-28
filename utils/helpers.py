from __future__ import annotations

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
