"""Logging setup and the append-only JSONL experiment manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from twist.utils.serialization import to_jsonable

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "twist", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        logging.basicConfig(level=level, format=_FORMAT)
    logger.setLevel(level)
    return logger


def append_manifest(path: str | Path, record: dict[str, Any]) -> None:
    """Append one immutable record to a JSONL manifest (never rewritten in place)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
