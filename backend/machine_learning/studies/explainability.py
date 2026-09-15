from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import ModelCache
from ..data import (
    _load_parquet, _parquet_path, _sample_ids_from_df, get_embedding_dim,
    get_shared_split_indices, load_sequence_embeddings,
)
from ..evaluation import compute_metrics
from ..result import ExperimentResult
from .common import RESULTS_ROOT, TRANSFORMER_STUDIES, atomic_csv, atomic_json, atomic_npz, stable_hash
from .manifest import StudyManifest, load_study_manifest


SPEAKER_RE = re.compile(r"(?=Innocent:\s|Suspect:\s)")


def _manifest(value: StudyManifest | str | Path | None) -> StudyManifest:
    return value if isinstance(value, StudyManifest) else load_study_manifest(value)


def _seed_row(candidate: dict, seed: int) -> dict:
    try:
        return next(row for row in candidate["seeds"] if int(row["seed"]) == int(seed))
    except StopIteration as exc:
        raise ValueError(f"Seed {seed} ausente em {candidate['candidate']}") from exc


def _load_model(candidate: dict, seed: int, device):
    import torch
    from ..transformer.runner import TransformerScamClassifier

    row = _seed_row(candidate, seed)
    path = RESULTS_ROOT / "transformer" / row["run_id"] / "model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint ausente: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    config = checkpoint.get("config", candidate["configuration"])
    model = TransformerScamClassifier(
        embedding_dim=get_embedding_dim(candidate["embedding"]),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(state)
    model.eval().to(device)
    return model, row["run_id"], config


def _manual_self_attention(module, x, padding_mask, ablate_head: int | None = None):
    """Exact eval-mode self-attention with exposed per-head weights."""
    import torch
    import torch.nn.functional as F

    if module.in_proj_weight is None:
        raise NotImplementedError("Apenas MultiheadAttention com in_proj_weight combinado é suportada")
    projected = F.linear(x, module.in_proj_weight, module.in_proj_bias)
    query, key, value = projected.chunk(3, dim=-1)
    batch, turns, width = query.shape
    heads = module.num_heads
    head_dim = width // heads

    def reshape(tensor):
        return tensor.reshape(batch, turns, heads, head_dim).transpose(1, 2)

    query, key, value = reshape(query), reshape(key), reshape(value)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    if padding_mask is not None:
        scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if padding_mask is not None:
        weights = weights.masked_fill(padding_mask[:, None, :, None], 0.0)
    head_output = torch.matmul(weights, value)
    if ablate_head is not None:
        head_output = head_output.clone()
        head_output[:, ablate_head] = 0.0
    merged = head_output.transpose(1, 2).contiguous().reshape(batch, turns, width)
    output = F.linear(merged, module.out_proj.weight, module.out_proj.bias)
    return output, weights


def inspect_forward(model, x, padding_mask=None, ablate: tuple[int, int] | None = None):
    """Forward compatible with TransformerScamClassifier that returns [B,H,T,T] per layer."""
    encoded = model.pos_encoder(x)
    attentions = []
    for layer_index, layer in enumerate(model.transformer.layers):
        target_head = ablate[1] if ablate is not None and ablate[0] == layer_index else None
        if layer.norm_first:
            attention_output, weights = _manual_self_attention(layer.self_attn, layer.norm1(encoded), padding_mask, target_head)
            encoded = encoded + layer.dropout1(attention_output)
            encoded = encoded + layer._ff_block(layer.norm2(encoded))
        else:
            attention_output, weights = _manual_self_attention(layer.self_attn, encoded, padding_mask, target_head)
            encoded = layer.norm1(encoded + layer.dropout1(attention_output))
            encoded = layer.norm2(encoded + layer._ff_block(encoded))
        attentions.append(weights)
    if model.transformer.norm is not None:
        encoded = model.transformer.norm(encoded)
    if padding_mask is not None:
        expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)
    else:
        pooled = encoded.mean(dim=1)
    return model.classifier(pooled), attentions


def _pad_batch(sequences: list[np.ndarray], device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    tensors = [torch.as_tensor(sequence[-100:], dtype=torch.float32) for sequence in sequences]
    padded = pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    lengths = torch.as_tensor([len(tensor) for tensor in tensors], device=device)
    mask = torch.arange(padded.shape[1], device=device)[None, :] >= lengths[:, None]
    return padded, mask, lengths


def _dataset(embedding: str, split: str):
    if split == "validation":
        sequences, labels, ids = load_sequence_embeddings("validation", embedding, with_augmented=False, return_ids=True)
        source_split = "validation"
    elif split == "test_internal":
        all_sequences, all_labels, all_ids = load_sequence_embeddings("train", embedding, with_augmented=False, return_ids=True)
        _, _, test_indices = get_shared_split_indices("transformer", all_ids, all_labels, seed=42)
        sequences = [all_sequences[index] for index in test_indices]
        labels = np.asarray(all_labels)[test_indices]
        ids = np.asarray(all_ids)[test_indices]
        source_split = "train"
    else:
        raise ValueError("split deve ser test_internal ou validation")
    frame = _load_parquet(_parquet_path(source_split, None, embedding, transformer=True), with_augmented=False)
    text_by_id = dict(zip(_sample_ids_from_df(frame).astype(str), frame["text"].astype(str)))
    return list(sequences), np.asarray(labels, dtype=int), np.asarray(ids).astype(str), text_by_id


def _turns(text: str, expected: int) -> list[dict[str, Any]]:
    chunks = [chunk.strip() for chunk in SPEAKER_RE.split(text) if chunk.strip()][-100:]
    if len(chunks) != expected:
        raise ValueError(f"Alinhamento texto/embedding inválido: {len(chunks)} turnos textuais, {expected} vetores")
    result = []
    for index, chunk in enumerate(chunks):
        speaker = "Suspect" if chunk.startswith("Suspect:") else "Innocent" if chunk.startswith("Innocent:") else "Unknown"
        result.append({"turn_index": index, "speaker": speaker})
    return result


def _representative_ids(candidate: dict, seed: int, split: str, ids: np.ndarray) -> set[str]:
    bundle = ModelCache.load_prediction_bundle("transformer", _seed_row(candidate, seed)["run_id"], split)
    available = set(ids.tolist())
    selected: set[str] = set()
    bundle_ids = np.asarray(bundle["sample_ids"]).astype(str)
    if not np.array_equal(bundle_ids, ids):
        raise ValueError(f"IDs do checkpoint e do dataset desalinhados para {candidate['candidate']} seed={seed} split={split}")
    labels, preds, probs = bundle["y_true"], bundle["y_pred"], bundle["y_prob"]
    categories = {
        "tp": (labels == 1) & (preds == 1), "tn": (labels == 0) & (preds == 0),
        "fp": (labels == 0) & (preds == 1), "fn": (labels == 1) & (preds == 0),
    }
    for mask in categories.values():
        selected.update(sorted(bundle_ids[mask].tolist())[:5])
    confidence_order = np.argsort(-np.abs(np.asarray(probs) - 0.5))[:5]
    selected.update(bundle_ids[confidence_order].tolist())

    cases_path = TRANSFORMER_STUDIES / "complementarity" / "cases.csv"
    if cases_path.exists():
        cases = pd.read_csv(cases_path, dtype={"sample_id": str})
        cases = cases[(cases["seed"] == seed) & (cases["split"] == split)]
        correct_columns = [column for column in cases if column.startswith("correct_") and column not in {"correct_soft", "correct_majority", "correct_calibrated_logit_sum"}]
        if "correct_soft" in cases:
            rescue = cases[cases["correct_soft"].astype(bool) & ~cases[correct_columns].all(axis=1)]
            harm = cases[~cases["correct_soft"].astype(bool) & cases[correct_columns].any(axis=1)]
            selected.update(rescue["sample_id"].astype(str).head(5))
            selected.update(harm["sample_id"].astype(str).head(5))
        selected.update(cases[cases["disagreement"].astype(bool)]["sample_id"].astype(str).head(5))
    return selected & available


def _attention_stats(weights: np.ndarray, speakers: list[str]) -> dict[str, float]:
    turns = weights.shape[-1]
    eps = 1e-12
    entropy = -(weights * np.log(weights + eps)).sum(axis=-1)
    normalized_entropy = float(entropy.mean() / max(math.log(max(turns, 2)), eps))
    positions = np.arange(turns)
    distance = np.abs(positions[:, None] - positions[None, :]) / max(turns - 1, 1)
    attended_distance = float((weights * distance).sum(axis=-1).mean())
    recent_start = int(np.floor(turns * 0.75))
    result = {
        "attention_entropy": normalized_entropy,
        "attended_distance": attended_distance,
        "recent_attention": float(weights[:, recent_start:].sum(axis=-1).mean()),
    }
    speaker_array = np.asarray(speakers)
    for query in ("Suspect", "Innocent"):
        for key in ("Suspect", "Innocent"):
            query_mask, key_mask = speaker_array == query, speaker_array == key
            name = f"attention_{query.lower()}_to_{key.lower()}"
            if query_mask.any() and key_mask.any():
                result[name] = float(weights[query_mask][:, key_mask].sum(axis=-1).mean())
            else:
                result[name] = np.nan
    return result


def _rollout(attentions: list[np.ndarray]) -> np.ndarray:
    turns = attentions[0].shape[-1]
    result = np.eye(turns)
    for weights in attentions:
        averaged = weights.mean(axis=0) + np.eye(turns)
        averaged /= averaged.sum(axis=-1, keepdims=True)
        result = averaged @ result
    return result


def extract_transformer_head_attention(
    manifest: StudyManifest | str | Path | None = None,
    split: str = "test_internal",
    seed: int = 42,
    *,
    embedding: str | None = None,
    force: bool = False,
) -> ExperimentResult:
    import torch

    manifest_obj = _manifest(manifest)
    candidates = [item for item in manifest_obj.transformer if embedding in {None, item["embedding"]}]
    if not candidates:
        raise ValueError(f"Embedding ausente no manifesto: {embedding}")
    all_summary = []
    all_occlusion = []
    for candidate in candidates:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, run_id, _ = _load_model(candidate, seed, device)
        sequences, labels, ids, text_by_id = _dataset(candidate["embedding"], split)
        representatives = _representative_ids(candidate, seed, split, ids)
        destination = TRANSFORMER_STUDIES / "explainability" / candidate["embedding"] / f"seed_{seed}" / split
        maps_dir = destination / "attention_maps"
        run_hash = stable_hash({
            "manifest": manifest_obj.manifest_hash, "checkpoint": run_id,
            "embedding": candidate["embedding"], "seed": seed, "split": split,
            "analysis": "attention-v1",
        })
        cached_manifest_path = destination / "manifest.json"
        if not force and cached_manifest_path.exists() and (destination / "head_summary.csv").exists() and (destination / "turn_occlusion.csv").exists():
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            if cached_manifest.get("analysis_hash") == run_hash:
                all_summary.append(pd.read_csv(destination / "head_summary.csv"))
                if (destination / "turn_occlusion.csv").stat().st_size > 1:
                    all_occlusion.append(pd.read_csv(destination / "turn_occlusion.csv"))
                continue
        sample_rows = []
        occlusion_rows = []
        turn_attention_rows = []
        connection_rows = []

        # Prove that inspection does not alter the decision before collecting explanations.
        probe, probe_mask, _ = _pad_batch(sequences[: min(4, len(sequences))], device)
        with torch.no_grad():
            original = model(probe, padding_mask=probe_mask)
            inspected, _ = inspect_forward(model, probe, probe_mask)
        max_error = float(torch.max(torch.abs(original - inspected)).cpu())
        if max_error > 1e-4:
            raise AssertionError(f"Inspector alterou logits (erro máximo={max_error:.6g})")

        for sequence, label, sample_id in zip(sequences, labels, ids):
            tensor, mask, lengths = _pad_batch([sequence], device)
            with torch.no_grad():
                logits, attention_tensors = inspect_forward(model, tensor, mask)
            probability = float(torch.sigmoid(logits).squeeze().cpu())
            turns_count = int(lengths[0])
            turn_meta = _turns(text_by_id[sample_id], turns_count)
            speakers = [turn["speaker"] for turn in turn_meta]
            attention_arrays = [value[0, :, :turns_count, :turns_count].cpu().numpy() for value in attention_tensors]
            for layer_index, layer_weights in enumerate(attention_arrays):
                for head_index, head_weights in enumerate(layer_weights):
                    sample_rows.append({
                        "embedding": candidate["embedding"], "seed": seed, "split": split,
                        "sample_id": sample_id, "label": int(label), "probability": probability,
                        "layer": layer_index, "head": head_index, "n_turns": turns_count,
                        **_attention_stats(head_weights, speakers),
                    })
                    if sample_id in representatives:
                        received = head_weights.mean(axis=0)
                        for turn_index, value in enumerate(received):
                            turn_attention_rows.append({
                                "embedding": candidate["embedding"], "seed": seed, "split": split,
                                "sample_id": sample_id, "layer": layer_index, "head": head_index,
                                "turn_index": turn_index, "speaker": speakers[turn_index],
                                "attention_received": float(value),
                            })
                        flat_order = np.argsort(head_weights.ravel())[::-1][: min(10, head_weights.size)]
                        for flat_index in flat_order:
                            query_index, key_index = np.unravel_index(flat_index, head_weights.shape)
                            connection_rows.append({
                                "embedding": candidate["embedding"], "seed": seed, "split": split,
                                "sample_id": sample_id, "layer": layer_index, "head": head_index,
                                "query_turn": int(query_index), "key_turn": int(key_index),
                                "query_speaker": speakers[query_index], "key_speaker": speakers[key_index],
                                "attention_weight": float(head_weights[query_index, key_index]),
                            })
            if sample_id in representatives:
                stack = np.stack(attention_arrays)
                atomic_npz(
                    maps_dir / f"{sample_id.replace(':', '_')}.npz",
                    attention=stack, rollout=_rollout(attention_arrays),
                    turn_indices=np.arange(turns_count), speakers=np.asarray(speakers),
                    sample_id=np.asarray(sample_id),
                )
                baseline_logit = float(logits.squeeze().cpu())
                for turn_index in range(turns_count):
                    occluded = tensor.clone()
                    occluded[0, turn_index] = 0.0
                    with torch.no_grad():
                        changed, _ = inspect_forward(model, occluded, mask)
                    changed_logit = float(changed.squeeze().cpu())
                    occlusion_rows.append({
                        "embedding": candidate["embedding"], "seed": seed, "split": split,
                        "sample_id": sample_id, "label": int(label), "turn_index": turn_index,
                        "speaker": speakers[turn_index], "baseline_logit": baseline_logit,
                        "occluded_logit": changed_logit, "delta_logit": baseline_logit - changed_logit,
                    })

        samples = pd.DataFrame(sample_rows)
        summary = samples.groupby(["embedding", "seed", "split", "label", "layer", "head"], as_index=False).mean(numeric_only=True)
        occlusion = pd.DataFrame(occlusion_rows)
        turn_attention = pd.DataFrame(turn_attention_rows)
        connections = pd.DataFrame(connection_rows)
        correlations = pd.DataFrame()
        if not turn_attention.empty and not occlusion.empty:
            merged = turn_attention.merge(
                occlusion[["sample_id", "turn_index", "delta_logit"]],
                on=["sample_id", "turn_index"], how="inner",
            )
            correlation_rows = []
            for keys, group in merged.groupby(["embedding", "seed", "split", "sample_id", "layer", "head"]):
                correlation = group["attention_received"].corr(group["delta_logit"].abs()) if len(group) > 1 else np.nan
                correlation_rows.append(dict(zip(
                    ["embedding", "seed", "split", "sample_id", "layer", "head"], keys
                )) | {"attention_occlusion_correlation": correlation})
            correlations = pd.DataFrame(correlation_rows)
        atomic_csv(destination / "head_samples.csv", samples)
        atomic_csv(destination / "head_summary.csv", summary)
        atomic_csv(destination / "turn_occlusion.csv", occlusion)
        atomic_csv(destination / "turn_attention.csv", turn_attention)
        atomic_csv(destination / "head_top_connections.csv", connections)
        atomic_csv(destination / "attention_occlusion_correlation.csv", correlations)
        atomic_json(destination / "manifest.json", {
            "study_manifest_hash": manifest_obj.manifest_hash, "checkpoint_run_id": run_id,
            "embedding": candidate["embedding"], "seed": seed, "split": split,
            "logit_equivalence_max_error": max_error,
            "analysis_hash": run_hash,
            "representative_sample_ids": sorted(representatives),
        })
        all_summary.append(summary)
        all_occlusion.append(occlusion)
    combined = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    return ExperimentResult(strategy="transformer_studies", config={"type": "head_attention", "split": split, "seed": seed}, df_history=combined, artifacts_dir=str(TRANSFORMER_STUDIES / "explainability"))


def analyze_transformer_head_ablation(
    manifest: StudyManifest | str | Path | None = None,
    split: str = "test_internal",
    seed: int = 42,
    *,
    embedding: str | None = None,
    batch_size: int = 32,
    force: bool = False,
) -> ExperimentResult:
    import torch

    manifest_obj = _manifest(manifest)
    candidates = [item for item in manifest_obj.transformer if embedding in {None, item["embedding"]}]
    rows = []
    for candidate in candidates:
        destination = TRANSFORMER_STUDIES / "explainability" / candidate["embedding"] / f"seed_{seed}" / split
        cached_path = destination / "head_ablation.csv"
        if not force and cached_path.exists() and cached_path.stat().st_size > 1:
            cached = pd.read_csv(cached_path)
            if not cached.empty and set(cached["checkpoint_run_id"].astype(str)) == {_seed_row(candidate, seed)["run_id"]}:
                rows.extend(cached.to_dict(orient="records"))
                continue
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, run_id, config = _load_model(candidate, seed, device)
        sequences, labels, ids, _ = _dataset(candidate["embedding"], split)
        heads = int(config.get("num_heads", 4))
        layers = int(config.get("num_layers", 2))

        def predict(ablate=None):
            probabilities = []
            for start in range(0, len(sequences), batch_size):
                tensor, mask, _ = _pad_batch(sequences[start:start + batch_size], device)
                with torch.no_grad():
                    logits, _ = inspect_forward(model, tensor, mask, ablate=ablate)
                probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy().tolist())
            return np.asarray(probabilities)

        baseline_probability = predict()
        baseline_prediction = (baseline_probability >= 0.5).astype(int)
        cached_bundle = ModelCache.load_prediction_bundle("transformer", run_id, split)
        cached_ids = np.asarray(cached_bundle["sample_ids"]).astype(str)
        if not np.array_equal(cached_ids, ids):
            raise ValueError(f"IDs desalinhados na ablação: {candidate['candidate']} seed={seed} split={split}")
        probability_error = float(np.max(np.abs(baseline_probability - np.asarray(cached_bundle["y_prob"]))))
        if probability_error > 5e-4:
            raise AssertionError(
                f"Probabilidades instrumentadas diferem do cache (erro máximo={probability_error:.6g})"
            )
        baseline_metrics = compute_metrics(labels, baseline_prediction, y_prob=baseline_probability)
        for layer in range(layers):
            for head in range(heads):
                probability = predict((layer, head))
                prediction = (probability >= 0.5).astype(int)
                metrics = compute_metrics(labels, prediction, y_prob=probability)
                rows.append({
                    "embedding": candidate["embedding"], "seed": seed, "split": split,
                    "layer": layer, "head": head, "checkpoint_run_id": run_id,
                    "baseline_probability_max_error": probability_error,
                    "mean_abs_delta_probability": float(np.mean(np.abs(probability - baseline_probability))),
                    "mean_delta_probability": float(np.mean(baseline_probability - probability)),
                    "f1_macro_drop": baseline_metrics["f1_macro"] - metrics["f1_macro"],
                    "recall_scam_drop": baseline_metrics["recall_scam"] - metrics["recall_scam"],
                    "precision_scam_drop": baseline_metrics["precision_scam"] - metrics["precision_scam"],
                    "fp_delta": metrics["FP"] - baseline_metrics["FP"],
                    "fn_delta": metrics["FN"] - baseline_metrics["FN"],
                    **{f"ablated_{key}": value for key, value in metrics.items() if key in {"f1_macro", "recall_scam", "precision_scam", "FP", "FN"}},
                })
        atomic_csv(destination / "head_ablation.csv", pd.DataFrame([row for row in rows if row["embedding"] == candidate["embedding"]]))
    frame = pd.DataFrame(rows)
    return ExperimentResult(strategy="transformer_studies", config={"type": "head_ablation", "split": split, "seed": seed}, df_history=frame, artifacts_dir=str(TRANSFORMER_STUDIES / "explainability"))


def build_transformer_explainability_report(
    manifest: StudyManifest | str | Path | None = None,
) -> ExperimentResult:
    manifest_obj = _manifest(manifest)
    root = TRANSFORMER_STUDIES / "explainability"
    summaries = [pd.read_csv(path) for path in root.glob("*/seed_*/*/head_summary.csv")]
    ablations = [pd.read_csv(path) for path in root.glob("*/seed_*/*/head_ablation.csv")]
    occlusions = [pd.read_csv(path) for path in root.glob("*/seed_*/*/turn_occlusion.csv") if path.stat().st_size > 1]
    correlations = [pd.read_csv(path) for path in root.glob("*/seed_*/*/attention_occlusion_correlation.csv") if path.stat().st_size > 1]
    connections = [pd.read_csv(path) for path in root.glob("*/seed_*/*/head_top_connections.csv") if path.stat().st_size > 1]
    if not summaries or not ablations:
        raise FileNotFoundError("Resultados de atenção/ablação incompletos; execute --only explainability")
    summary = pd.concat(summaries, ignore_index=True)
    ablation = pd.concat(ablations, ignore_index=True).drop_duplicates()
    occlusion = pd.concat(occlusions, ignore_index=True) if occlusions else pd.DataFrame()
    correlation = pd.concat(correlations, ignore_index=True) if correlations else pd.DataFrame()
    connection = pd.concat(connections, ignore_index=True) if connections else pd.DataFrame()
    stability = ablation.groupby(["embedding", "split", "layer", "head"], as_index=False).agg(
        f1_macro_drop_mean=("f1_macro_drop", "mean"),
        f1_macro_drop_std=("f1_macro_drop", "std"),
        impact_mean=("mean_abs_delta_probability", "mean"),
    )
    class_comparison = summary.pivot_table(
        index=["embedding", "seed", "split", "layer", "head"], columns="label",
        values=["attention_entropy", "attended_distance", "recent_attention"],
    )
    class_comparison.columns = [f"{metric}_{'scam' if label == 1 else 'ham'}" for metric, label in class_comparison.columns]
    class_comparison = class_comparison.reset_index()
    atomic_csv(root / "head_summary.csv", summary)
    atomic_csv(root / "head_class_comparison.csv", class_comparison)
    atomic_csv(root / "head_ablation.csv", ablation)
    atomic_csv(root / "turn_occlusion.csv", occlusion)
    atomic_csv(root / "attention_occlusion_correlation.csv", correlation)
    atomic_csv(root / "head_top_connections.csv", connection)
    atomic_csv(root / "head_stability.csv", stability)
    atomic_json(root / "manifest.json", {
        "study_manifest_hash": manifest_obj.manifest_hash,
        "analysis_hash": stable_hash({"manifest": manifest_obj.manifest_hash, "type": "explainability"}),
        "warning": "Attention is descriptive evidence; ablation and occlusion estimate decision impact.",
    })
    return ExperimentResult(strategy="transformer_studies", config={"type": "explainability_report"}, df_history=summary, df_metrics=ablation, artifacts_dir=str(root))
