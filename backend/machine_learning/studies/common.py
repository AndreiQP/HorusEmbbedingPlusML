from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = BACKEND_ROOT / "experiment_results"
TRANSFORMER_STUDIES = RESULTS_ROOT / "transformer" / "studies"
UNSUPERVISED_STUDIES = RESULTS_ROOT / "unsupervised" / "studies"
DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_SEEDS = (42, 52, 62)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mark_complete(directory: Path, payload: dict[str, Any]) -> None:
    atomic_json(directory / "complete.json", payload)


def is_complete(directory: Path, expected_hash: str) -> bool:
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("run_hash") == expected_hash
    except (OSError, json.JSONDecodeError):
        return False


def nested_subset_indices(
    sample_ids: Iterable[str],
    labels: Iterable[int],
    fraction: float,
    seed: int,
    *,
    ham_only: bool = False,
) -> np.ndarray:
    """Deterministic, nested sampling. Sorting by a seeded ID hash makes 5% a prefix of 10%."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction deve estar em (0, 1]")
    ids = np.asarray(list(sample_ids), dtype=str)
    y = np.asarray(list(labels), dtype=int)
    if len(ids) != len(y) or len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids devem ser únicos e alinhados aos labels")
    classes = [0] if ham_only else sorted(np.unique(y).tolist())
    selected: list[int] = []
    for label in classes:
        candidates = np.flatnonzero(y == label)
        ordered = sorted(
            candidates.tolist(),
            key=lambda index: hashlib.sha256(f"{seed}:{ids[index]}".encode()).hexdigest(),
        )
        count = len(ordered) if fraction == 1 else max(1, int(np.ceil(len(ordered) * fraction)))
        selected.extend(ordered[:count])
    return np.asarray(sorted(selected), dtype=int)


def align_prediction_bundles(bundles: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not bundles:
        raise ValueError("Nenhum bundle de predição informado")
    first_name = next(iter(bundles))
    first = bundles[first_name]
    if "sample_ids" not in first:
        raise ValueError(f"Predições de {first_name} não possuem sample_ids")
    ids = np.asarray(first["sample_ids"]).astype(str)
    labels = np.asarray(first["y_true"]).astype(int)
    for name, bundle in bundles.items():
        if "sample_ids" not in bundle:
            raise ValueError(f"Predições de {name} não possuem sample_ids")
        if not np.array_equal(ids, np.asarray(bundle["sample_ids"]).astype(str)):
            raise ValueError(f"sample_ids desalinhados: {first_name} x {name}")
        if not np.array_equal(labels, np.asarray(bundle["y_true"]).astype(int)):
            raise ValueError(f"Rótulos desalinhados: {first_name} x {name}")
    return ids, labels


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    metric_columns = [
        "accuracy", "f1_macro", "f1_scam", "precision_scam", "recall_scam",
        "roc_auc", "pr_auc", "TP", "TN", "FP", "FN", "training_time", "best_epoch",
    ]
    keys = [c for c in ("strategy", "candidate", "model_type", "embedding", "fraction", "n_samples", "split") if c in frame]
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        for metric in metric_columns:
            if metric not in group:
                continue
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if series.empty:
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = 1.96 * std / np.sqrt(len(series)) if len(series) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
