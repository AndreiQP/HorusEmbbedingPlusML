from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..data import get_embedding_dim, get_shared_split_indices, load_sequence_embeddings, load_single_embedding
from ..evaluation import compute_metrics
from ..result import ExperimentResult
from .common import (
    DEFAULT_FRACTIONS, DEFAULT_SEEDS, RESULTS_ROOT, TRANSFORMER_STUDIES,
    UNSUPERVISED_STUDIES, aggregate_metrics, atomic_csv, atomic_json, atomic_npz,
    is_complete, mark_complete, nested_subset_indices, stable_hash,
)
from .manifest import StudyManifest, load_study_manifest


def _manifest(value: StudyManifest | str | Path | None) -> StudyManifest:
    return value if isinstance(value, StudyManifest) else load_study_manifest(value)


def _select(items: list[dict], candidate: str | None) -> list[dict]:
    selected = items if candidate is None else [item for item in items if item["candidate"] == candidate]
    if not selected:
        raise ValueError(f"Candidato não encontrado no manifesto: {candidate}")
    return selected


def _seed_row(candidate: dict, seed: int) -> dict:
    try:
        return next(row for row in candidate["seeds"] if int(row["seed"]) == int(seed))
    except StopIteration as exc:
        raise ValueError(f"Seed {seed} não está disponível para {candidate['candidate']}") from exc


def _metric_row(metric: dict, meta: dict) -> dict:
    return {**meta, **metric}


def _save_run(directory: Path, payload: dict, predictions: dict[str, dict] | None = None) -> None:
    atomic_json(directory / "metrics.json", payload)
    for split, bundle in (predictions or {}).items():
        atomic_npz(directory / f"predictions_{split}.npz", **bundle)
    mark_complete(directory, {"run_hash": payload["run_hash"], "rows": len(payload["metrics"])})


def _reuse_transformer_full(candidate: dict, seed: int, run_hash: str, directory: Path) -> list[dict]:
    from ..cache import ModelCache

    row = _seed_row(candidate, seed)
    bundles = {
        split: ModelCache.load_prediction_bundle("transformer", row["run_id"], split)
        for split in ("internal_validation", "test_internal", "validation")
    }
    metrics = []
    predictions = {}
    for split, bundle in bundles.items():
        metric = compute_metrics(
            bundle["y_true"], bundle["y_pred"], y_prob=bundle["y_prob"],
            model_name=candidate["candidate"], dataset_name=split,
        )
        meta = {
            "strategy": "transformer", "candidate": candidate["candidate"],
            "model_type": "transformer", "embedding": candidate["embedding"],
            "fraction": 1.0, "n_samples": None, "seed": seed, "split": split,
            "training_time": 0.0, "best_epoch": None, "run_id": row["run_id"],
            "cache_reused": True,
        }
        metrics.append(_metric_row(metric, meta))
        predictions[split] = bundle
    payload = {"run_hash": run_hash, "metrics": metrics, "source_run_id": row["run_id"]}
    _save_run(directory, payload, predictions)
    return metrics


def _train_transformer_fraction(candidate: dict, fraction: float, seed: int, run_hash: str, directory: Path) -> list[dict]:
    import torch
    from ..transformer.runner import (
        TransformerScamClassifier, _evaluate_dl, _get_device, _make_single_dataloader,
        _train_transformer_loop,
    )

    config = candidate["configuration"]
    embedding = candidate["embedding"]
    sequences, labels, sample_ids = load_sequence_embeddings("train", embedding, with_augmented=False, return_ids=True)
    sequences = np.asarray(sequences, dtype=object)
    labels = np.asarray(labels, dtype=int)
    sample_ids = np.asarray(sample_ids).astype(str)
    idx_train, idx_val, idx_test = get_shared_split_indices("transformer", sample_ids, labels, seed=42)
    local = nested_subset_indices(sample_ids[idx_train], labels[idx_train], fraction, seed)
    idx_subset = idx_train[local]

    batch_size = int(config.get("batch_size", 32))
    dl_train = _make_single_dataloader(sequences[idx_subset], labels[idx_subset], batch_size=batch_size, shuffle=True)
    dl_val = _make_single_dataloader(sequences[idx_val], labels[idx_val], batch_size=batch_size, shuffle=False)
    dl_test = _make_single_dataloader(sequences[idx_test], labels[idx_test], batch_size=batch_size, shuffle=False)
    ext_sequences, ext_labels, ext_ids = load_sequence_embeddings("validation", embedding, with_augmented=False, return_ids=True)
    dl_external = _make_single_dataloader(ext_sequences, ext_labels, batch_size=batch_size, shuffle=False)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _get_device()
    model = TransformerScamClassifier(
        embedding_dim=get_embedding_dim(embedding),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    started = time.perf_counter()
    model, best_state, history = _train_transformer_loop(
        model, dl_train, dl_val, int(config.get("max_epochs", config.get("num_epochs", 100))),
        float(config.get("lr", 1e-5)), device,
        optimizer_name=config.get("optimizer", "adamw"),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        patience=int(config.get("patience", 8)), min_delta=float(config.get("min_delta", 1e-4)),
        scheduler_patience=int(config.get("scheduler_patience", 3)),
        min_lr=float(config.get("min_lr", 1e-7)),
        mixed_precision=bool(config.get("mixed_precision", True)),
        gradient_clip=float(config.get("gradient_clip", 1.0)),
    )
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    predictions = {}
    metrics = []
    eval_sets = {
        "internal_validation": (dl_val, sample_ids[idx_val]),
        "test_internal": (dl_test, sample_ids[idx_test]),
        "validation": (dl_external, np.asarray(ext_ids).astype(str)),
    }
    best_epoch = int(history.attrs.get("best_epoch", 0))
    for split, (loader, ids) in eval_sets.items():
        y_true, y_pred, y_prob = _evaluate_dl(model, loader, device)
        metric = compute_metrics(y_true, y_pred, y_prob=y_prob, model_name=candidate["candidate"], dataset_name=split)
        meta = {
            "strategy": "transformer", "candidate": candidate["candidate"],
            "model_type": "transformer", "embedding": embedding, "fraction": fraction,
            "n_samples": int(len(idx_subset)), "seed": seed, "split": split,
            "training_time": elapsed, "best_epoch": best_epoch, "run_id": run_hash[:16],
            "cache_reused": False,
        }
        metrics.append(_metric_row(metric, meta))
        predictions[split] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "sample_ids": ids}

    directory.mkdir(parents=True, exist_ok=True)
    history.to_csv(directory / "history.csv", index=False)
    torch.save({"state_dict": model.to("cpu").state_dict(), "config": config}, directory / "model.pth")
    atomic_json(directory / "subset.json", {"sample_ids": sample_ids[idx_subset].tolist(), "fraction": fraction, "seed": seed})
    _save_run(directory, {"run_hash": run_hash, "metrics": metrics}, predictions)
    return metrics


def run_transformer_dataset_size_study(
    manifest: StudyManifest | str | Path | None = None,
    fractions: Iterable[float] = DEFAULT_FRACTIONS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    candidate: str | None = None,
    force: bool = False,
) -> ExperimentResult:
    manifest_obj = _manifest(manifest)
    all_rows = []
    for item in _select(manifest_obj.transformer, candidate):
        for fraction in fractions:
            for seed in seeds:
                config = {"manifest": manifest_obj.manifest_hash, "candidate": item["candidate"], "fraction": float(fraction), "seed": int(seed)}
                run_hash = stable_hash(config)
                directory = TRANSFORMER_STUDIES / "dataset_size" / "runs" / run_hash[:16]
                if not force and is_complete(directory, run_hash):
                    all_rows.extend(json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["metrics"])
                    continue
                if float(fraction) == 1.0:
                    all_rows.extend(_reuse_transformer_full(item, int(seed), run_hash, directory))
                else:
                    all_rows.extend(_train_transformer_fraction(item, float(fraction), int(seed), run_hash, directory))
    frame = pd.DataFrame(all_rows)
    return ExperimentResult(strategy="transformer_studies", config={"type": "dataset_size"}, df_history=frame, artifacts_dir=str(TRANSFORMER_STUDIES / "dataset_size"))


def _reuse_unsupervised_full(candidate: dict, seed: int, run_hash: str, directory: Path) -> list[dict]:
    row = _seed_row(candidate, seed)
    metrics = []
    for split, source_name in (("test_internal", "test"), ("validation", "validation")):
        metric = dict(row[source_name])
        metric["dataset_name"] = split
        meta = {
            "strategy": "unsupervised", "candidate": candidate["candidate"],
            "model_type": candidate["model_type"], "embedding": candidate["embedding"],
            "fraction": 1.0, "n_samples": None, "seed": seed, "split": split,
            "training_time": 0.0, "best_epoch": None, "run_id": row["run_id"], "cache_reused": True,
        }
        metrics.append(_metric_row(metric, meta))
    _save_run(directory, {"run_hash": run_hash, "metrics": metrics, "source_run_id": row["run_id"]})
    return metrics


def _train_unsupervised_fraction(candidate: dict, fraction: float, seed: int, run_hash: str, directory: Path) -> list[dict]:
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler
    from ..unsupervised.runner import _anomaly_predict, _anomaly_score, _build_anomaly_model, _build_reducer

    embedding = candidate["embedding"]
    X, labels, sample_ids = load_single_embedding("train", "all_data", embedding, with_augmented=False, return_ids=True)
    labels = np.asarray(labels, dtype=int)
    sample_ids = np.asarray(sample_ids).astype(str)
    idx_train, idx_val, idx_test = get_shared_split_indices("unsupervised", sample_ids, labels, seed=42)
    fit_pool = np.concatenate([idx_train, idx_val])
    ham_pool = fit_pool[labels[fit_pool] == 0]
    local = nested_subset_indices(sample_ids[ham_pool], labels[ham_pool], fraction, seed, ham_only=True)
    idx_subset = ham_pool[local]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    scaler = StandardScaler().fit(X[idx_subset])
    effective_dim = min(int(candidate["pca_dim"]), len(idx_subset), X.shape[1])
    reducer = _build_reducer(candidate.get("reducer", "pca"), effective_dim)
    reduced_fit = reducer.fit_transform(scaler.transform(X[idx_subset]))
    params = dict(candidate.get("params", {}))
    params.setdefault("random_state", seed)
    model = _build_anomaly_model(candidate["model_type"], **params)
    model.fit(reduced_fit)
    elapsed = time.perf_counter() - started

    X_external, y_external, external_ids = load_single_embedding("validation", "all_data", embedding, with_augmented=False, return_ids=True)
    eval_sets = {
        "test_internal": (X[idx_test], labels[idx_test], sample_ids[idx_test]),
        "validation": (X_external, np.asarray(y_external, dtype=int), np.asarray(external_ids).astype(str)),
    }
    metrics = []
    predictions = {}
    for split, (features, y_true, ids) in eval_sets.items():
        transformed = reducer.transform(scaler.transform(features))
        y_pred = _anomaly_predict(model, transformed, candidate["model_type"])
        y_prob = _anomaly_score(model, transformed, candidate["model_type"])
        metric = compute_metrics(y_true, y_pred, y_score=y_prob, model_name=candidate["candidate"], dataset_name=split)
        meta = {
            "strategy": "unsupervised", "candidate": candidate["candidate"],
            "model_type": candidate["model_type"], "embedding": embedding,
            "fraction": fraction, "n_samples": int(len(idx_subset)), "seed": seed,
            "split": split, "training_time": elapsed, "best_epoch": None,
            "run_id": run_hash[:16], "cache_reused": False, "effective_dim": effective_dim,
        }
        metrics.append(_metric_row(metric, meta))
        predictions[split] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "sample_ids": ids}
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.pkl")
    joblib.dump(reducer, directory / "reducer.pkl")
    joblib.dump(scaler, directory / "scaler.pkl")
    atomic_json(directory / "subset.json", {"sample_ids": sample_ids[idx_subset].tolist(), "fraction": fraction, "seed": seed})
    _save_run(directory, {"run_hash": run_hash, "metrics": metrics}, predictions)
    return metrics


def run_unsupervised_dataset_size_study(
    manifest: StudyManifest | str | Path | None = None,
    fractions: Iterable[float] = DEFAULT_FRACTIONS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    candidate: str | None = None,
    force: bool = False,
) -> ExperimentResult:
    manifest_obj = _manifest(manifest)
    all_rows = []
    for item in _select(manifest_obj.unsupervised, candidate):
        for fraction in fractions:
            for seed in seeds:
                config = {"manifest": manifest_obj.manifest_hash, "candidate": item["candidate"], "fraction": float(fraction), "seed": int(seed)}
                run_hash = stable_hash(config)
                directory = UNSUPERVISED_STUDIES / "dataset_size" / "runs" / run_hash[:16]
                if not force and is_complete(directory, run_hash):
                    all_rows.extend(json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["metrics"])
                    continue
                if float(fraction) == 1.0:
                    all_rows.extend(_reuse_unsupervised_full(item, int(seed), run_hash, directory))
                else:
                    all_rows.extend(_train_unsupervised_fraction(item, float(fraction), int(seed), run_hash, directory))
    frame = pd.DataFrame(all_rows)
    return ExperimentResult(strategy="unsupervised_studies", config={"type": "dataset_size"}, df_history=frame, artifacts_dir=str(UNSUPERVISED_STUDIES / "dataset_size"))


def aggregate_dataset_size_results() -> dict[str, str]:
    outputs = {}
    for strategy, root in (("transformer", TRANSFORMER_STUDIES), ("unsupervised", UNSUPERVISED_STUDIES)):
        rows = []
        for path in (root / "dataset_size" / "runs").glob("*/metrics.json"):
            try:
                rows.extend(json.loads(path.read_text(encoding="utf-8"))["metrics"])
            except (KeyError, json.JSONDecodeError):
                continue
        raw = pd.DataFrame(rows)
        aggregate = aggregate_metrics(raw)
        destination = root / "dataset_size"
        atomic_csv(destination / "metrics.csv", raw)
        atomic_csv(destination / "aggregate.csv", aggregate)
        outputs[strategy] = str(destination / "aggregate.csv")
    return outputs
