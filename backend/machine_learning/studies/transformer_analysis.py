from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import ModelCache
from ..evaluation import compute_metrics
from ..result import ExperimentResult
from .common import TRANSFORMER_STUDIES, align_prediction_bundles, atomic_csv, atomic_json, stable_hash
from .manifest import StudyManifest, load_study_manifest


def _manifest(value: StudyManifest | str | Path | None) -> StudyManifest:
    return value if isinstance(value, StudyManifest) else load_study_manifest(value)


def _seed_run(candidate: dict, seed: int) -> str:
    try:
        return next(row["run_id"] for row in candidate["seeds"] if int(row["seed"]) == int(seed))
    except StopIteration as exc:
        raise ValueError(f"Seed {seed} ausente em {candidate['candidate']}") from exc


def _bundles(manifest: StudyManifest, seed: int, split: str) -> dict[str, dict[str, np.ndarray]]:
    return {
        candidate["embedding"]: ModelCache.load_prediction_bundle(
            "transformer", _seed_run(candidate, seed), split
        )
        for candidate in manifest.transformer
    }


def _temperature(y_true: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(probability / (1 - probability))
    temperatures = np.logspace(-2, 2, 401)
    losses = []
    for temperature in temperatures:
        calibrated = 1 / (1 + np.exp(-np.clip(logits / temperature, -50, 50)))
        losses.append(-np.mean(y_true * np.log(calibrated + 1e-12) + (1 - y_true) * np.log(1 - calibrated + 1e-12)))
    return float(temperatures[int(np.argmin(losses))])


def _ensemble_outputs(
    bundles: dict[str, dict[str, np.ndarray]], temperatures: dict[str, float]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    names = list(bundles)
    predictions = np.stack([bundles[name]["y_pred"] for name in names])
    probabilities = np.stack([bundles[name]["y_prob"] for name in names])
    mean_probability = probabilities.mean(axis=0)
    raw_logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
    divisor = np.asarray([temperatures[name] for name in names])[:, None]
    calibrated = 1 / (1 + np.exp(-np.clip((raw_logits / divisor).sum(axis=0), -50, 50)))
    return {
        "majority": ((predictions.sum(axis=0) >= 2).astype(int), mean_probability),
        "soft": ((mean_probability >= 0.5).astype(int), mean_probability),
        "calibrated_logit_sum": ((calibrated >= 0.5).astype(int), calibrated),
    }


def min_fn_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Largest cutoff that still classifies every calibration positive as positive."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positives = probability[y_true == 1]
    if not len(positives):
        raise ValueError("Conjunto de calibração sem exemplos positivos")
    threshold = float(positives.min())
    if np.any((probability >= threshold)[y_true == 1] == 0):
        raise AssertionError("Threshold min_fn não zerou falsos negativos")
    return threshold


def analyze_transformer_complementarity(
    manifest: StudyManifest | str | Path | None = None,
    *,
    force: bool = False,
) -> ExperimentResult:
    manifest_obj = _manifest(manifest)
    output = TRANSFORMER_STUDIES / "complementarity"
    manifest_path = output / "manifest.json"
    if not force and manifest_path.exists() and all((output / name).exists() for name in ("summary.csv", "cases.csv", "ensemble_metrics.csv")):
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached_manifest.get("study_manifest_hash") == manifest_obj.manifest_hash:
            return ExperimentResult(
                strategy="transformer_studies", config={"type": "complementarity"},
                df_history=pd.read_csv(output / "summary.csv"),
                df_metrics=pd.read_csv(output / "ensemble_metrics.csv"), artifacts_dir=str(output),
            )
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []

    for seed in manifest_obj.seeds:
        internal = _bundles(manifest_obj, seed, "internal_validation")
        _, internal_labels = align_prediction_bundles(internal)
        temperatures = {
            name: _temperature(internal_labels, bundle["y_prob"])
            for name, bundle in internal.items()
        }
        for split in ("test_internal", "validation"):
            bundles = _bundles(manifest_obj, seed, split)
            ids, labels = align_prediction_bundles(bundles)
            ensembles = _ensemble_outputs(bundles, temperatures)
            names = list(bundles)
            correct = {name: np.asarray(bundle["y_pred"]) == labels for name, bundle in bundles.items()}

            for source in names:
                for rescuer in names:
                    if source == rescuer:
                        continue
                    mask = (~correct[source]) & correct[rescuer]
                    for class_name, class_mask in (("all", np.ones(len(labels), dtype=bool)), ("ham", labels == 0), ("scam", labels == 1)):
                        summary_rows.append({
                            "seed": seed, "split": split, "analysis": "pairwise_rescue",
                            "source": source, "rescuer": rescuer, "class": class_name,
                            "count": int((mask & class_mask).sum()),
                        })

            for index, left in enumerate(names):
                for right in names[index + 1:]:
                    left_correct = correct[left].astype(float)
                    right_correct = correct[right].astype(float)
                    correlation = float(np.corrcoef(left_correct, right_correct)[0, 1]) if left_correct.std() and right_correct.std() else np.nan
                    prediction_disagreement = np.asarray(bundles[left]["y_pred"]) != np.asarray(bundles[right]["y_pred"])
                    correlation_rows.append({
                        "seed": seed, "split": split, "left": left, "right": right,
                        "correctness_correlation": correlation,
                        "prediction_disagreement_count": int(prediction_disagreement.sum()),
                        "prediction_disagreement_rate": float(prediction_disagreement.mean()),
                    })

            any_correct = np.stack(list(correct.values())).any(axis=0)
            all_correct = np.stack(list(correct.values())).all(axis=0)
            summary_rows.extend([
                {"seed": seed, "split": split, "analysis": "oracle_correct", "source": "all", "rescuer": "oracle", "class": "all", "count": int(any_correct.sum())},
                {"seed": seed, "split": split, "analysis": "shared_errors", "source": "all", "rescuer": "none", "class": "all", "count": int((~any_correct).sum())},
                {"seed": seed, "split": split, "analysis": "all_correct", "source": "all", "rescuer": "all", "class": "all", "count": int(all_correct.sum())},
            ])
            for name in names:
                only = correct[name] & ~np.stack([correct[other] for other in names if other != name]).any(axis=0)
                summary_rows.append({"seed": seed, "split": split, "analysis": "unique_correct", "source": name, "rescuer": "none", "class": "all", "count": int(only.sum())})

            for ensemble_name, (ensemble_pred, ensemble_prob) in ensembles.items():
                metric = compute_metrics(labels, ensemble_pred, y_prob=ensemble_prob, model_name=ensemble_name, dataset_name=split)
                metric_rows.append({"seed": seed, "split": split, **metric})
                ensemble_correct = ensemble_pred == labels
                for name in names:
                    summary_rows.append({"seed": seed, "split": split, "analysis": "ensemble_rescue", "source": name, "rescuer": ensemble_name, "class": "all", "count": int(((~correct[name]) & ensemble_correct).sum())})
                    summary_rows.append({"seed": seed, "split": split, "analysis": "ensemble_harm", "source": name, "rescuer": ensemble_name, "class": "all", "count": int((correct[name] & ~ensemble_correct).sum())})

            prediction_stack = np.stack([bundles[name]["y_pred"] for name in names])
            disagreement = prediction_stack.max(axis=0) != prediction_stack.min(axis=0)
            for index, sample_id in enumerate(ids):
                row: dict[str, Any] = {
                    "seed": seed, "split": split, "sample_id": sample_id,
                    "label": int(labels[index]), "disagreement": bool(disagreement[index]),
                    "oracle_correct": bool(any_correct[index]),
                }
                for name in names:
                    row[f"pred_{name}"] = int(bundles[name]["y_pred"][index])
                    row[f"prob_{name}"] = float(bundles[name]["y_prob"][index])
                    row[f"correct_{name}"] = bool(correct[name][index])
                for ensemble_name, (ensemble_pred, ensemble_prob) in ensembles.items():
                    row[f"pred_{ensemble_name}"] = int(ensemble_pred[index])
                    row[f"prob_{ensemble_name}"] = float(ensemble_prob[index])
                    row[f"correct_{ensemble_name}"] = bool(ensemble_pred[index] == labels[index])
                detail_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    metrics = pd.DataFrame(metric_rows)
    correlations = pd.DataFrame(correlation_rows)
    atomic_csv(output / "summary.csv", summary)
    atomic_csv(output / "cases.csv", details)
    atomic_csv(output / "ensemble_metrics.csv", metrics)
    atomic_csv(output / "correctness_correlation.csv", correlations)
    atomic_json(output / "manifest.json", {
        "study_manifest_hash": manifest_obj.manifest_hash,
        "analysis_hash": stable_hash({"manifest": manifest_obj.manifest_hash, "type": "complementarity"}),
        "seeds": manifest_obj.seeds, "splits": ["test_internal", "validation"],
    })
    return ExperimentResult(strategy="transformer_studies", config={"type": "complementarity"}, df_history=summary, df_metrics=metrics, artifacts_dir=str(output))


def run_transformer_threshold_study(
    manifest: StudyManifest | str | Path | None = None,
    strategy: str = "min_fn",
    *,
    force: bool = False,
) -> ExperimentResult:
    if strategy != "min_fn":
        raise ValueError("Este protocolo implementa somente strategy='min_fn'")
    manifest_obj = _manifest(manifest)
    output = TRANSFORMER_STUDIES / "threshold"
    manifest_path = output / "manifest.json"
    if not force and manifest_path.exists() and all((output / name).exists() for name in ("metrics.csv", "threshold_curve.csv", "thresholds.json")):
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached_manifest.get("study_manifest_hash") == manifest_obj.manifest_hash and cached_manifest.get("strategy") == strategy:
            return ExperimentResult(
                strategy="transformer_studies", config={"type": "threshold", "strategy": strategy},
                df_history=pd.read_csv(output / "threshold_curve.csv"),
                df_metrics=pd.read_csv(output / "metrics.csv"), artifacts_dir=str(output),
            )
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []

    for seed in manifest_obj.seeds:
        internal = _bundles(manifest_obj, seed, "internal_validation")
        _, labels_internal = align_prediction_bundles(internal)
        probability_internal = np.stack([bundle["y_prob"] for bundle in internal.values()]).mean(axis=0)
        threshold = min_fn_threshold(labels_internal, probability_internal)
        calibrated_pred = (probability_internal >= threshold).astype(int)
        if int(((labels_internal == 1) & (calibrated_pred == 0)).sum()) != 0:
            raise AssertionError("Falha ao garantir FN=0 na validação interna")
        thresholds.append({"seed": seed, "threshold": threshold, "calibration_split": "internal_validation"})

        candidates = np.unique(np.concatenate(([0.0], probability_internal, [1.0])))
        for candidate_threshold in candidates:
            pred = (probability_internal >= candidate_threshold).astype(int)
            metric = compute_metrics(labels_internal, pred, y_prob=probability_internal)
            curve_rows.append({"seed": seed, "threshold": float(candidate_threshold), "FP": metric["FP"], "FN": metric["FN"], "precision_scam": metric["precision_scam"], "recall_scam": metric["recall_scam"]})

        temperatures = {
            name: _temperature(labels_internal, bundle["y_prob"])
            for name, bundle in internal.items()
        }
        for split in ("internal_validation", "test_internal", "validation"):
            bundles = internal if split == "internal_validation" else _bundles(manifest_obj, seed, split)
            _, labels = align_prediction_bundles(bundles)
            outputs = _ensemble_outputs(bundles, temperatures)
            soft_probability = outputs["soft"][1]
            variants = {
                "soft_0.5": ((soft_probability >= 0.5).astype(int), soft_probability, 0.5),
                "soft_min_fn": ((soft_probability >= threshold).astype(int), soft_probability, threshold),
                "majority": (*outputs["majority"], None),
                "calibrated_logit_sum": (*outputs["calibrated_logit_sum"], 0.5),
            }
            for name, (prediction, probability, used_threshold) in variants.items():
                metric = compute_metrics(labels, prediction, y_prob=probability, model_name=name, dataset_name=split)
                metric_rows.append({"seed": seed, "split": split, "variant": name, "threshold": used_threshold, **metric})

    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    atomic_csv(output / "metrics.csv", metrics)
    atomic_csv(output / "threshold_curve.csv", curves)
    atomic_json(output / "thresholds.json", thresholds)
    atomic_json(output / "manifest.json", {
        "study_manifest_hash": manifest_obj.manifest_hash, "strategy": strategy,
        "guarantee": "FN=0 only on internal_validation",
    })
    return ExperimentResult(strategy="transformer_studies", config={"type": "threshold", "strategy": strategy}, df_history=curves, df_metrics=metrics, artifacts_dir=str(output))
