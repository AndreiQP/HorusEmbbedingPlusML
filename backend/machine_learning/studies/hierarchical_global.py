"""Estudo canônico, global, particionado e retomável de explicabilidade BGE.

Usa as primitivas congeladas de perturbação, grava Parquet por shard e agrega
apenas quando a cobertura do split está completa.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ..cache import ModelCache
from .common import (
    TRANSFORMER_STUDIES, atomic_csv, atomic_json, is_complete, mark_complete,
    sha256_file, stable_hash,
)
from .explainability import _dataset
from .hierarchical_perturbation import (
    RANKING_ABSOLUTE_INFLUENCE,
    RANKING_DECISION_SUPPORTING,
    HierarchicalExplanation,
    HierarchicalPerturbationExplainer,
    ReproductionError,
    _flatten_with_metadata,
    _stability_rows,
    class_support_delta,
    encoder_provenance,
    load_bge_encoder,
    load_bge_finalist_model,
    parse_conversation,
    resolve_bge_finalist_checkpoint,
    select_message_evidence,
)


HIERARCHICAL_EXPLAINABILITY_ROOT = TRANSFORMER_STUDIES / "hierarchical_explainability"
STUDY_SCHEMA_VERSION = 2
DEFAULT_SEEDS = (42, 52, 62)
EXPECTED_SPLIT_COUNTS = {"test_internal": 1390, "validation": 1083}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _completed_shard_is_valid(directory: Path, run_hash: str) -> bool:
    if not is_complete(directory, run_hash):
        return False
    try:
        marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        return all(
            (directory / name).exists()
            and sha256_file(directory / name) == expected
            for name, expected in marker.get("checksums", {}).items()
        )
    except (OSError, json.JSONDecodeError):
        return False


def _category(label: int, prediction: int) -> str:
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[
        (int(label), int(prediction))
    ]


def _stable_control_seed(sample_id: str, row: dict[str, Any]) -> int:
    payload = ":".join((
        str(sample_id), str(row["model_index"]), str(row["start_char"]),
        str(row["end_char"]), str(row["ngram_size"]),
    ))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _random_control_rows(
    explanation: HierarchicalExplanation,
    *,
    controls_per_span: int = 5,
) -> list[dict[str, Any]]:
    sample_id = str(explanation.metadata["sample_id"])
    selected_message_indices = {
        int(row["model_index"]) for row in explanation.top_messages
    }
    selected = [
        row for row in explanation.span_ranking
        if int(row.get("rank_within_message", 0)) == 1
        and int(row["model_index"]) in selected_message_indices
        and float(row.get("class_support_delta", 0.0)) > 1e-4
    ]
    output: list[dict[str, Any]] = []
    for span in selected:
        candidates = [
            row for row in explanation._all_span_effects
            if int(row["model_index"]) == int(span["model_index"])
            and int(row["ngram_size"]) == int(span["ngram_size"])
            and (int(row["start_char"]), int(row["end_char"]))
            != (int(span["start_char"]), int(span["end_char"]))
        ]
        count = min(int(controls_per_span), len(candidates))
        rng = np.random.default_rng(_stable_control_seed(sample_id, span))
        chosen = rng.choice(len(candidates), size=count, replace=False).tolist() if count else []
        effects = [abs(float(candidates[index]["delta_logit"])) for index in chosen]
        mean = float(np.mean(effects)) if effects else math.nan
        median = float(np.median(effects)) if effects else math.nan
        variance = float(np.var(effects, ddof=1)) if len(effects) > 1 else 0.0 if effects else math.nan
        std = float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0 if effects else math.nan
        selected_effect = abs(float(span["delta_logit"]))
        ratio = selected_effect / mean if effects and mean != 0.0 else math.nan
        common = {
            "sample_id": sample_id,
            "split": explanation.metadata["split"],
            "seed": int(explanation.metadata["seed"]),
            "prediction_category": explanation.metadata["prediction_category"],
            "predicted_label": int(explanation.baseline["predicted_label"]),
            "model_index": int(span["model_index"]),
            "original_index": int(span["original_index"]),
            "span_text": str(span["span_text"]),
            "start_char": int(span["start_char"]),
            "end_char": int(span["end_char"]),
            "ngram_size": int(span["ngram_size"]),
            "selected_abs_delta_logit": selected_effect,
            "control_count": count,
            "control_mean": mean,
            "control_median": median,
            "control_variance": variance,
            "control_std": std,
            "specificity_ratio": ratio,
            "selected_minus_control_mean": selected_effect - mean if effects else math.nan,
        }
        if not effects:
            output.append({**common, "control_rank": None, "control_abs_delta_logit": math.nan})
        else:
            for rank, value in enumerate(effects, start=1):
                output.append({**common, "control_rank": rank, "control_abs_delta_logit": value})
    return output


def _enrich_fidelity(rows: Iterable[dict[str, Any]], ranking_policy: str) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        if row.get("level") == "cumulative_messages":
            row["comprehensiveness_abs"] = abs(float(row["comprehensiveness"]))
            row["sufficiency_distance"] = abs(float(row["sufficiency_gap"]))
            row["comprehensiveness_negative"] = float(row["comprehensiveness"]) < 0.0
            row["sufficiency_gap_negative"] = float(row["sufficiency_gap"]) < 0.0
            row["selected_message_count"] = int(row["top_k"])
        row["ranking_policy"] = ranking_policy
        output.append(row)
    return output


def _global_stability_rows(
    explanations: Sequence[HierarchicalExplanation],
) -> list[dict[str, Any]]:
    """Completa o helper legado com as métricas exigidas pelo estudo global."""
    if not explanations:
        return []
    primary = next(
        item for item in explanations
        if int(item.metadata["seed"]) == int(item.metadata["candidate_source_seed"])
    )
    split = str(primary.metadata["split"])
    category = str(primary.metadata["prediction_category"])
    primary_prediction = int(primary.baseline["predicted_label"])
    rows = _stability_rows(explanations)
    for row in rows:
        row["split"] = split
        row["prediction_category"] = category
        if row["item_type"] in {"message", "span"}:
            mean = float(row.get("mean_abs_delta_logit", 0.0))
            std = float(row.get("std_abs_delta_logit", 0.0))
            row["impact_coefficient_variation"] = std / mean if mean else math.nan

    summaries = {
        int(row["compared_seed"]): row
        for row in rows if row.get("item_type") == "summary"
    }
    primary_message_values = {
        int(row["model_index"]): float(row["abs_delta_logit"])
        for row in primary.message_ranking
    }
    primary_span_keys = {
        (int(row["model_index"]), int(row["start_char"]), int(row["end_char"]))
        for row in primary.span_ranking
    }
    for candidate in explanations:
        seed = int(candidate.metadata["seed"])
        if candidate is primary:
            continue
        summary = summaries[seed]
        agrees = int(candidate.baseline["predicted_label"]) == primary_prediction
        summary["primary_predicted_label"] = primary_prediction
        summary["compared_predicted_label"] = int(candidate.baseline["predicted_label"])
        summary["prediction_agreement"] = bool(agrees)
        candidate_values = {
            int(row["model_index"]): float(row["abs_delta_logit"])
            for row in candidate.message_ranking
        }
        shared = sorted(set(primary_message_values) & set(candidate_values))
        kendall = kendalltau(
            [primary_message_values[index] for index in shared],
            [candidate_values[index] for index in shared],
        ).statistic if len(shared) > 1 else math.nan
        summary["message_rank_kendall"] = float(kendall) if np.isfinite(kendall) else math.nan
        candidate_span_keys = {
            (int(row["model_index"]), int(row["start_char"]), int(row["end_char"]))
            for row in candidate.span_ranking
        }
        union = primary_span_keys | candidate_span_keys
        summary["span_top_k_overlap"] = len(primary_span_keys & candidate_span_keys)
        summary["span_top_k_jaccard"] = (
            len(primary_span_keys & candidate_span_keys) / len(union) if union else 1.0
        )
        primary_span_values = {
            (int(row["model_index"]), int(row["start_char"]), int(row["end_char"])):
                float(row["abs_delta_logit"])
            for row in primary._all_span_effects
        }
        candidate_span_values = {
            (int(row["model_index"]), int(row["start_char"]), int(row["end_char"])):
                float(row["abs_delta_logit"])
            for row in candidate._all_span_effects
        }
        shared_spans = sorted(set(primary_span_values) & set(candidate_span_values))
        span_kendall = kendalltau(
            [primary_span_values[key] for key in shared_spans],
            [candidate_span_values[key] for key in shared_spans],
        ).statistic if len(shared_spans) > 1 else math.nan
        summary["span_rank_kendall"] = (
            float(span_kendall) if np.isfinite(span_kendall) else math.nan
        )
        if not agrees:
            for key in (
                "message_top_k_overlap", "message_top_k_jaccard",
                "message_direction_agreement", "message_rank_spearman",
                "message_rank_kendall", "span_top_k_overlap", "span_top_k_jaccard",
                "span_direction_agreement", "span_rank_spearman", "span_rank_kendall",
            ):
                summary[key] = math.nan
    return rows


def build_global_inventory(
    *,
    splits: Sequence[str] = ("test_internal", "validation"),
    primary_seed: int = 42,
    output_root: str | Path = HIERARCHICAL_EXPLAINABILITY_ROOT,
    checkpoint_root: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inventory_model, checkpoint_metadata = load_bge_finalist_model(
        seed=primary_seed, device="cpu", checkpoint_root=checkpoint_root,
    )
    del inventory_model
    for split in splits:
        sequences, labels, ids, text_by_id = _dataset("bge", split)
        bundle = ModelCache.load_prediction_bundle(
            "transformer", checkpoint_metadata["run_id"], split,
        )
        bundle_ids = np.asarray(bundle["sample_ids"]).astype(str)
        ids = np.asarray(ids).astype(str)
        if not np.array_equal(bundle_ids, ids):
            raise ReproductionError(f"IDs desalinhados no inventário: {split}")
        if not np.array_equal(np.asarray(bundle["y_true"]).astype(int), np.asarray(labels).astype(int)):
            raise ReproductionError(f"Rótulos desalinhados no inventário: {split}")
        predictions = np.asarray(bundle["y_pred"]).astype(int)
        for index, sample_id in enumerate(ids):
            n_turns = len(parse_conversation(text_by_id[sample_id]))
            embedding_turns = int(len(sequences[index]))
            if n_turns != embedding_turns:
                raise ReproductionError(
                    f"Alinhamento texto/embedding inválido em {split}/{sample_id}: "
                    f"{n_turns} turnos textuais e {embedding_turns} embeddings"
                )
            rows.append({
                "split": split,
                "sample_id": sample_id,
                "dataset_index": index,
                "label": int(labels[index]),
                "predicted_label": int(predictions[index]),
                "prediction_category": _category(labels[index], predictions[index]),
                "n_turns": n_turns,
                "embedding_turns": embedding_turns,
            })
    frame = pd.DataFrame(rows).sort_values(["split", "sample_id"]).reset_index(drop=True)
    observed_counts = frame.groupby("split")["sample_id"].nunique().to_dict()
    for split in splits:
        expected = EXPECTED_SPLIT_COUNTS.get(split)
        if expected is not None and int(observed_counts.get(split, 0)) != expected:
            raise ReproductionError(
                f"Cobertura inesperada em {split}: {observed_counts.get(split, 0)} != {expected}"
            )
    destination = Path(output_root)
    _atomic_parquet(destination / "inventory.parquet", frame)
    atomic_csv(destination / "inventory_summary.csv", (
        frame.groupby(["split", "prediction_category"], as_index=False)
        .agg(conversations=("sample_id", "nunique"), min_turns=("n_turns", "min"), max_turns=("n_turns", "max"))
    ))
    atomic_json(destination / "inventory_manifest.json", {
        "version": STUDY_SCHEMA_VERSION,
        "splits": list(splits),
        "primary_seed": int(primary_seed),
        "checkpoint": checkpoint_metadata,
        "inventory_hash": stable_hash(frame.to_dict(orient="records")),
        "counts": observed_counts,
    })
    return frame


def run_global_shard(
    *,
    split: str,
    shard_index: int,
    shard_count: int,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    primary_seed: int = 42,
    top_messages: int = 6,
    max_ngram: int = 5,
    top_spans_per_message: int = 20,
    controls_per_span: int = 5,
    device=None,
    output_root: str | Path = HIERARCHICAL_EXPLAINABILITY_ROOT,
    checkpoint_root: str | Path | None = None,
    resume: bool = True,
) -> Path:
    if split not in {"test_internal", "validation"}:
        raise ValueError("split deve ser test_internal ou validation")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index/shard_count inválidos")
    seeds = tuple(dict.fromkeys([int(primary_seed), *(int(seed) for seed in seeds)]))
    checkpoint_fingerprints = {}
    for seed in seeds:
        path = resolve_bge_finalist_checkpoint(seed=seed, checkpoint_root=checkpoint_root)
        checkpoint_fingerprints[str(seed)] = {
            "path": str(path), "sha256": sha256_file(path),
        }
    destination = Path(output_root) / split / "shards" / f"shard-{shard_index:05d}"
    run_config = {
        "version": STUDY_SCHEMA_VERSION, "split": split, "shard_index": shard_index,
        "shard_count": shard_count, "seeds": seeds, "primary_seed": primary_seed,
        "top_messages": top_messages, "max_ngram": max_ngram,
        "top_spans_per_message": top_spans_per_message,
        "controls_per_span": controls_per_span,
        "checkpoint_fingerprints": checkpoint_fingerprints,
    }
    run_hash = stable_hash(run_config)
    if resume and _completed_shard_is_valid(destination, run_hash):
        return destination

    sequences, labels, ids, text_by_id = _dataset("bge", split)
    ids = np.asarray(ids).astype(str)
    labels = np.asarray(labels).astype(int)
    ordered = sorted(range(len(ids)), key=lambda index: hashlib.sha256(ids[index].encode()).hexdigest())
    selected_indices = [index for position, index in enumerate(ordered) if position % shard_count == shard_index]
    if not selected_indices:
        raise ValueError(f"Shard vazio: {shard_index}/{shard_count}")

    encoder = load_bge_encoder(device=device)
    explainers: dict[int, HierarchicalPerturbationExplainer] = {}
    bundles: dict[int, dict[str, np.ndarray]] = {}
    model_metadata: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        model, metadata = load_bge_finalist_model(seed=seed, device=device, checkpoint_root=checkpoint_root)
        explainers[seed] = HierarchicalPerturbationExplainer(model, encoder, device=device, seed=seed, model_metadata=metadata)
        bundles[seed] = ModelCache.load_prediction_bundle("transformer", metadata["run_id"], split)
        model_metadata[seed] = metadata
        if not np.array_equal(np.asarray(bundles[seed]["sample_ids"]).astype(str), ids):
            raise ReproductionError(f"Bundle desalinhado: split={split}, seed={seed}")
        if not np.array_equal(np.asarray(bundles[seed]["y_true"]).astype(int), labels):
            raise ReproductionError(f"Rótulos do bundle desalinhados: split={split}, seed={seed}")

    primary = explainers[int(primary_seed)]
    audit_index = selected_indices[0]
    audit = primary.audit_cached_embeddings(
        text_by_id[ids[audit_index]], np.asarray(sequences[audit_index], dtype=np.float32)[-100:],
    )
    neutral = primary._encode(["Innocent:", "Suspect:"])
    neutral_by_speaker = dict(zip(("Innocent", "Suspect"), neutral))

    conversations: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    fidelity: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for index in selected_indices:
        sample_id = ids[index]
        prepared = primary.prepare_perturbations(
            text_by_id[sample_id], top_messages=top_messages, max_ngram=max_ngram,
            original_embeddings=np.asarray(sequences[index], dtype=np.float32)[-100:],
            neutral_by_speaker=neutral_by_speaker,
            ranking_policy=RANKING_DECISION_SUPPORTING,
            include_absolute_candidates=True,
        )
        sample_explanations: list[HierarchicalExplanation] = []
        for seed in seeds:
            explanation = explainers[seed].explain_prepared(
                prepared, top_messages=top_messages,
                top_spans_per_message=top_spans_per_message,
                sample_id=sample_id, label=int(labels[index]), split=split,
                analysis_scope="global",
                expected_probability=float(np.asarray(bundles[seed]["y_prob"])[index]),
                defer_span_fidelity=True,
            )
            sample_explanations.append(explanation)

        sufficient_texts: list[str] = []
        for explanation in sample_explanations:
            selected_message_indices = {
                int(row["model_index"]) for row in explanation.top_messages
            }
            sufficient_texts.extend(
                f"{row['speaker']}: {row['span_text']}"
                for row in explanation.span_ranking
                if int(row.get("rank_within_message", 0)) == 1
                and int(row["model_index"]) in selected_message_indices
            )
        sufficient_texts = list(dict.fromkeys(sufficient_texts))
        sufficient_lookup = dict(zip(sufficient_texts, primary._encode(sufficient_texts))) if sufficient_texts else {}
        for explanation in sample_explanations:
            explainers[int(explanation.metadata["seed"])].append_span_fidelity(
                explanation, prepared, sufficient_lookup,
                random_controls_per_span=controls_per_span,
            )
            prefix = {
                "sample_id": sample_id, "split": split,
                "seed": int(explanation.metadata["seed"]),
                "label": int(labels[index]),
                "predicted_label": int(explanation.baseline["predicted_label"]),
                "prediction_category": explanation.metadata["prediction_category"],
                "candidate_source_seed": int(prepared.candidate_source_seed),
            }
            absolute = select_message_evidence(
                explanation.message_ranking,
                predicted_label=int(explanation.baseline["predicted_label"]),
                top_messages=top_messages,
                ranking_policy=RANKING_ABSOLUTE_INFLUENCE,
            )
            absolute_indices = {int(row["model_index"]) for row in absolute}
            supporting_indices = {int(row["model_index"]) for row in explanation.top_messages}
            for row in explanation.message_ranking:
                messages.append({
                    **prefix, **row,
                    "selected_absolute_influence": int(row["model_index"]) in absolute_indices,
                    "selected_decision_supporting": int(row["model_index"]) in supporting_indices,
                    "available_ranking_policies": (
                        f"{RANKING_ABSOLUTE_INFLUENCE},{RANKING_DECISION_SUPPORTING}"
                    ),
                })
            for row in explanation._all_span_effects:
                spans.append({**prefix, **row})
            fidelity.extend({**prefix, **row} for row in _enrich_fidelity(
                explanation.fidelity, RANKING_DECISION_SUPPORTING,
            ))
            conversations.append({
                **prefix,
                "baseline_logit": float(explanation.baseline["logit"]),
                "baseline_probability_scam": float(explanation.baseline["probability_scam"]),
                "n_turns": int(explanation.baseline["n_turns"]),
                "absolute_top_count": len(absolute),
                "supporting_top_count": len(explanation.top_messages),
                "prediction_matches_primary": int(explanation.baseline["predicted_label"])
                == int(sample_explanations[0].baseline["predicted_label"]),
                "supported_class": "scam" if int(explanation.baseline["predicted_label"]) else "ham",
                "ranking_policy": RANKING_DECISION_SUPPORTING,
            })
        controls.extend(_random_control_rows(sample_explanations[0], controls_per_span=controls_per_span))
        stability.extend(_global_stability_rows(sample_explanations))

    frames = {
        "conversation_metrics": pd.DataFrame(conversations),
        "message_effects": pd.DataFrame(messages),
        "span_effects": pd.DataFrame(spans),
        "fidelity": pd.DataFrame(fidelity),
        "random_controls": pd.DataFrame(controls),
        "stability": pd.DataFrame(stability),
    }
    for name, frame in frames.items():
        _atomic_parquet(destination / f"{name}.parquet", frame)
    checksums = {
        f"{name}.parquet": sha256_file(destination / f"{name}.parquet")
        for name in frames
    }
    dataset_hash = stable_hash([
        {
            "sample_id": ids[index], "label": int(labels[index]),
            "embedding_shape": list(np.asarray(sequences[index]).shape),
            "text_hash": hashlib.sha256(text_by_id[ids[index]].encode("utf-8")).hexdigest(),
        }
        for index in selected_indices
    ])
    atomic_json(destination / "manifest.json", {
        **run_config, "run_hash": run_hash, "sample_count": len(selected_indices),
        "sample_ids": ids[selected_indices].tolist(), "bge_audit": audit,
        "encoder": encoder_provenance(encoder), "models": model_metadata,
        "dataset_hash": dataset_hash, "checksums": checksums,
    })
    mark_complete(destination, {
        "run_hash": run_hash, "sample_count": len(selected_indices), "checksums": checksums,
    })
    return destination


def _bootstrap_ci(values: np.ndarray, *, seed: int = 42, draws: int = 1000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.asarray([
        np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(draws)
    ])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_metric(values: Iterable[float], *, bootstrap_seed: int = 42) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if not n:
        return {key: math.nan for key in (
            "mean", "variance", "std", "sem", "median", "q1", "q3", "iqr",
            "min", "max", "ci95_low", "ci95_high", "negative_rate",
        )} | {"n": 0}
    q1, q3 = np.quantile(array, [0.25, 0.75])
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    low, high = _bootstrap_ci(array, seed=bootstrap_seed)
    return {
        "n": n, "mean": float(np.mean(array)),
        "variance": float(np.var(array, ddof=1)) if n > 1 else 0.0,
        "std": std, "sem": std / math.sqrt(n), "median": float(np.median(array)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(np.min(array)), "max": float(np.max(array)),
        "ci95_low": low, "ci95_high": high,
        "negative_rate": float(np.mean(array < 0.0)),
    }


def _metric_summaries(frame: pd.DataFrame, group_columns: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        prefix = dict(zip(group_columns, keys))
        for metric in metrics:
            if metric in group:
                rows.append({**prefix, "metric": metric, **summarize_metric(group[metric])})
    return pd.DataFrame(rows)


def aggregate_global_results(
    *,
    output_root: str | Path = HIERARCHICAL_EXPLAINABILITY_ROOT,
    expected_shards: int,
    splits: Sequence[str] = ("test_internal", "validation"),
) -> Path:
    root = Path(output_root)
    table_names = (
        "conversation_metrics", "message_effects", "span_effects",
        "fidelity", "random_controls", "stability",
    )
    split_counts: dict[str, int] = {}
    observed_ids_by_split: dict[str, set[str]] = {}
    shard_dirs_by_split: dict[str, list[Path]] = {}
    shard_records: list[dict[str, Any]] = []
    for split in splits:
        shard_root = root / split / "shards"
        shard_dirs = [shard_root / f"shard-{index:05d}" for index in range(expected_shards)]
        missing = [str(path) for path in shard_dirs if not (path / "complete.json").exists()]
        if missing:
            raise RuntimeError(f"Shards incompletos em {split}: {missing[:5]}")
        shard_dirs_by_split[split] = shard_dirs
        seen: set[str] = set()
        for shard in shard_dirs:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
            if not _completed_shard_is_valid(shard, str(manifest["run_hash"])):
                raise RuntimeError(f"Checksum inválido no shard: {shard}")
            shard_records.append({
                "split": split,
                "shard_index": int(manifest["shard_index"]),
                "sample_count": int(manifest["sample_count"]),
                "run_hash": manifest["run_hash"],
                "dataset_hash": manifest["dataset_hash"],
                "checksums": manifest["checksums"],
            })
            duplicate = seen.intersection(manifest["sample_ids"])
            if duplicate:
                raise RuntimeError(f"sample_ids duplicados em {split}: {sorted(duplicate)[:5]}")
            seen.update(manifest["sample_ids"])
        split_counts[split] = len(seen)
        observed_ids_by_split[split] = seen

    inventory_path = root / "inventory.parquet"
    if inventory_path.exists():
        inventory = pd.read_parquet(inventory_path)
        for split in splits:
            expected_ids = set(
                inventory.loc[inventory["split"].eq(split), "sample_id"].astype(str)
            )
            observed_ids = observed_ids_by_split[split]
            if observed_ids != expected_ids:
                missing_ids = sorted(expected_ids - observed_ids)
                extra_ids = sorted(observed_ids - expected_ids)
                raise RuntimeError(
                    f"Cobertura inválida em {split}: ausentes={missing_ids[:5]}, "
                    f"extras={extra_ids[:5]}"
                )
    # Consolida uma tabela por vez. message_effects e span_effects, que são as
    # maiores, nunca coexistem em memória com as demais tabelas globais.
    needed_for_aggregates = {
        "conversation_metrics", "fidelity", "random_controls", "stability",
    }
    consolidated: dict[str, pd.DataFrame] = {}
    for name in table_names:
        combined_splits: list[pd.DataFrame] = []
        for split in splits:
            frame = pd.concat([
                pd.read_parquet(shard / f"{name}.parquet")
                for shard in shard_dirs_by_split[split]
            ], ignore_index=True)
            _atomic_parquet(root / split / f"{name}.parquet", frame)
            if name in needed_for_aggregates:
                combined_splits.append(frame)
            else:
                del frame
        if combined_splits:
            consolidated[name] = pd.concat(combined_splits, ignore_index=True)

    aggregate_root = root / "aggregates"
    fidelity = consolidated["fidelity"]
    cumulative = fidelity[(fidelity["level"] == "cumulative_messages") & (fidelity["seed"] == 42)].copy()
    metrics = ["comprehensiveness", "comprehensiveness_abs", "sufficiency_gap", "sufficiency_distance"]
    population = _metric_summaries(cumulative, ["split", "top_k"], metrics)
    population["aggregation"] = "population"
    by_category = _metric_summaries(cumulative, ["split", "prediction_category", "top_k"], metrics)
    by_category["aggregation"] = "category"
    by_predicted_class = _metric_summaries(
        cumulative, ["split", "predicted_label", "top_k"], metrics,
    )
    by_predicted_class["aggregation"] = "predicted_class"
    category_means = (
        cumulative.groupby(["split", "prediction_category", "top_k"], as_index=False)[metrics].mean()
    )
    balanced = _metric_summaries(category_means, ["split", "top_k"], metrics)
    balanced["aggregation"] = "category_balanced"
    fidelity_curves = pd.concat(
        [population, balanced, by_category, by_predicted_class], ignore_index=True,
    )
    _atomic_parquet(aggregate_root / "fidelity_curves.parquet", fidelity_curves)
    atomic_csv(aggregate_root / "fidelity_curves.csv", fidelity_curves)
    _atomic_parquet(aggregate_root / "metric_distributions.parquet", cumulative)

    primary_conversations = (
        consolidated["conversation_metrics"]
        .loc[lambda frame: frame["seed"].eq(42)]
        .drop_duplicates(["split", "sample_id"])
    )
    coverage_rows = []
    for split in splits:
        split_conversations = primary_conversations[primary_conversations["split"].eq(split)]
        split_fidelity = cumulative[cumulative["split"].eq(split)]
        for top_k in range(1, 7):
            coverage_rows.append({
                "split": split,
                "top_k": top_k,
                "conversation_count": int(split_conversations["sample_id"].nunique()),
                "N_k": int(split_conversations["n_turns"].ge(top_k).sum()),
                "metric_N_k": int(
                    split_fidelity.loc[split_fidelity["top_k"].eq(top_k), "sample_id"].nunique()
                ),
                "ranking_policy": RANKING_DECISION_SUPPORTING,
            })
    coverage = pd.DataFrame(coverage_rows)
    _atomic_parquet(aggregate_root / "coverage.parquet", coverage)
    atomic_csv(aggregate_root / "coverage.csv", coverage)

    controls = consolidated["random_controls"]
    control_unique = controls.drop_duplicates([
        "split", "sample_id", "seed", "model_index", "start_char", "end_char",
    ])
    random_population = _metric_summaries(
        control_unique, ["split"],
        ["selected_abs_delta_logit", "control_mean", "specificity_ratio", "selected_minus_control_mean"],
    )
    random_population["aggregation"] = "population"
    random_category = _metric_summaries(
        control_unique, ["split", "prediction_category"],
        ["selected_abs_delta_logit", "control_mean", "specificity_ratio", "selected_minus_control_mean"],
    )
    random_category["aggregation"] = "category"
    random_summary = pd.concat([random_population, random_category], ignore_index=True)
    _atomic_parquet(aggregate_root / "random_control_summary.parquet", random_summary)
    atomic_csv(aggregate_root / "random_control_summary.csv", random_summary)

    stability = consolidated["stability"]
    summary_stability = stability[stability.get("item_type", pd.Series(dtype=str)) == "summary"].copy()
    stability_metrics = [
        "prediction_agreement", "message_top_k_jaccard", "message_direction_agreement",
        "message_rank_spearman", "message_rank_kendall", "span_top_k_jaccard",
        "span_direction_agreement", "span_rank_spearman", "span_rank_kendall",
    ]
    stability_summary = _metric_summaries(summary_stability, ["split", "compared_seed"], stability_metrics)
    _atomic_parquet(aggregate_root / "stability_summary.parquet", stability_summary)
    atomic_csv(aggregate_root / "stability_summary.csv", stability_summary)
    _atomic_parquet(aggregate_root / "category_summary.parquet", by_category)
    atomic_json(root / "manifest.json", {
        "version": STUDY_SCHEMA_VERSION, "splits": list(splits),
        "expected_shards": expected_shards, "conversation_counts": split_counts,
        "seeds": list(DEFAULT_SEEDS), "primary_seed": 42,
        "ranking_policy_primary": RANKING_DECISION_SUPPORTING,
        "ranking_policy_diagnostic": RANKING_ABSOLUTE_INFLUENCE,
        "parameters": {
            key: value for key, value in manifest.items()
            if key in {
                "top_messages", "max_ngram", "top_spans_per_message",
                "controls_per_span", "shard_count",
            }
        },
        "models": manifest.get("models", {}),
        "encoder": manifest.get("encoder", {}),
        "shards": shard_records,
        "failures": [],
        "complete": True,
    })
    return root
