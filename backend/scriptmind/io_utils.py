"""Atomic, resumable IO helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .paths import ScriptMindPaths, assert_isolated_path


def stable_hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def shard_for(identifier: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return int(hashlib.sha256(identifier.encode("utf-8")).hexdigest(), 16) % num_shards


def atomic_write_text(path: Path, text: str, roots: ScriptMindPaths | None = None) -> None:
    assert_isolated_path(path, roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, value: Any, roots: ScriptMindPaths | None = None) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", roots)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]], roots: ScriptMindPaths | None = None) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    atomic_write_text(path, text, roots)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
