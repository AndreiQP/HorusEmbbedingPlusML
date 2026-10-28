"""Comparação uniforme de explicabilidade BGE por unidade contextual.

O módulo mantém a conversa canônica como baseline e acrescenta blocos fixos por
interlocutor e janelas cronológicas deslizantes. Todos os modos compartilham a
mesma representação intermediária, inferência, fidelidade e agregação.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..data import get_shared_split_indices
from ..evaluation import compute_metrics
from ..transformer.runner import (
    TransformerScamClassifier,
    _evaluate_dl,
    _make_dataloaders,
    _make_single_dataloader,
    _train_transformer_loop,
)
from .common import TRANSFORMER_STUDIES, atomic_json, sha256_file, stable_hash
from .hierarchical_global import summarize_metric
from .hierarchical_perturbation import (
    DEFAULT_EFFECT_EPSILON,
    RANKING_ABSOLUTE_INFLUENCE,
    RANKING_DECISION_SUPPORTING,
    ConversationTurn,
    class_support_delta,
    encoder_provenance,
    load_bge_encoder,
    load_bge_finalist_model,
    parse_conversation,
    select_message_evidence,
)


CONTEXT_BLOCK_ROOT = TRANSFORMER_STUDIES / "hierarchical_explainability" / "context_blocks"
REPRESENTATIONS = (
    "canonical_message",
    "speaker_block_4",
    "chronological_window_4",
    "chronological_window_5",
    "chronological_window_6",
    "chronological_window_7",
    "chronological_window_8",
)
MODEL_MODES = ("frozen_probe", "retrained")
SEEDS = (42, 52, 62)
PRIMARY_SEED = 42
MAX_ORIGINAL_MESSAGES = 100
BGE_DIMENSION = 1024
SCHEMA_VERSION = 1

FINALIST_CONFIG = {
    "max_epochs": 100,
    "lr": 1e-6,
    "batch_size": 16,
    "num_heads": 2,
    "num_layers": 4,
    "hidden_dim": 256,
    "dropout": 0.1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "patience": 8,
    "min_delta": 1e-4,
    "scheduler_patience": 3,
    "min_lr": 1e-7,
    "gradient_clip": 1.0,
    "mixed_precision": True,
    "split_seed": 42,
}


@dataclass(frozen=True)
class ContextBlock:
    block_index: int
    representation: str
    text: str
    neutral_text: str
    message_model_indices: tuple[int, ...]
    message_original_indices: tuple[int, ...]
    speakers: tuple[str, ...]
    message_texts: tuple[str, ...]
    message_offsets: tuple[tuple[int, int], ...]
    first_model_index: int
    last_model_index: int
    is_partial: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "representation": self.representation,
            "text": self.text,
            "neutral_text": self.neutral_text,
            "message_model_indices": list(self.message_model_indices),
            "message_original_indices": list(self.message_original_indices),
            "speakers": list(self.speakers),
            "message_texts": list(self.message_texts),
            "message_offsets": [list(value) for value in self.message_offsets],
            "first_model_index": self.first_model_index,
            "last_model_index": self.last_model_index,
            "is_partial": self.is_partial,
        }


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_list(value: Any) -> list:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _stable_ids(frame: pd.DataFrame) -> np.ndarray:
    id_column = next(
        (name for name in ("conversation_id", "chat_id", "id", "uuid") if name in frame),
        None,
    )
    if id_column:
        result = frame[id_column].astype(str)
    else:
        hashes = frame["text"].astype(str).map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        result = hashes + ":" + hashes.groupby(hashes).cumcount().astype(str)
    if result.duplicated().any():
        raise ValueError("IDs não são únicos")
    return result.to_numpy(dtype=str)


def _source_parquet(dataset_name: str) -> Path:
    backend = Path(__file__).resolve().parents[2]
    return backend / "datasets" / dataset_name / "processed" / "messages_transformer" / "bge_chunks.parquet"


def load_source_frame(source_split: str) -> pd.DataFrame:
    """Lê texto/rótulo sem materializar a coluna de embeddings BGE."""
    if source_split not in {"train", "validation"}:
        raise ValueError("source_split deve ser train ou validation")
    path = _source_parquet("dataset_train" if source_split == "train" else "dataset_validation")
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(path).schema.names)
    except Exception:
        available = {"text", "label"}
    id_columns = [name for name in ("conversation_id", "chat_id", "id", "uuid") if name in available]
    columns = ["text", "label", *id_columns]
    if "origin" in available:
        columns.append("origin")
    frame = pd.read_parquet(path, columns=columns)
    if source_split == "train" and "origin" in frame:
        frame = frame[frame.origin.eq("external")].reset_index(drop=True)
    frame = frame.copy()
    frame["sample_id"] = _stable_ids(frame)
    frame["source_split"] = source_split
    return frame


def _source_fingerprint(frame: pd.DataFrame) -> str:
    records = [
        {
            "sample_id": str(row.sample_id),
            "label": int(row.label),
            "text_sha256": hashlib.sha256(str(row.text).encode("utf-8")).hexdigest(),
        }
        for row in frame.itertuples(index=False)
    ]
    return stable_hash(records)


def _compose_block(
    representation: str,
    turns: Sequence[ConversationTurn],
    *,
    block_index: int,
    expected_size: int,
) -> ContextBlock:
    if not turns:
        raise ValueError("Bloco vazio")
    texts = [turn.text for turn in turns]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, text in enumerate(texts):
        start = cursor
        end = start + len(text)
        offsets.append((start, end))
        cursor = end + (1 if index < len(texts) - 1 else 0)
    return ContextBlock(
        block_index=block_index,
        representation=representation,
        text="\n".join(texts),
        neutral_text="\n".join(f"{turn.speaker}:" for turn in turns),
        message_model_indices=tuple(turn.model_index for turn in turns),
        message_original_indices=tuple(turn.original_index for turn in turns),
        speakers=tuple(turn.speaker for turn in turns),
        message_texts=tuple(texts),
        message_offsets=tuple(offsets),
        first_model_index=min(turn.model_index for turn in turns),
        last_model_index=max(turn.model_index for turn in turns),
        is_partial=len(turns) < expected_size,
    )


def build_context_blocks(
    conversation: str,
    representation: str,
    *,
    max_messages: int = MAX_ORIGINAL_MESSAGES,
) -> list[ContextBlock]:
    """Transforma a conversa em uma das sete representações canônicas."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Representação inválida: {representation}")
    turns = parse_conversation(conversation, max_turns=max_messages)
    groups: list[list[ConversationTurn]]
    expected_size = 1
    if representation == "canonical_message":
        groups = [[turn] for turn in turns]
    elif representation == "speaker_block_4":
        expected_size = 4
        groups = []
        for speaker in ("Suspect", "Innocent"):
            speaker_turns = [turn for turn in turns if turn.speaker == speaker]
            groups.extend(
                speaker_turns[start:start + 4]
                for start in range(0, len(speaker_turns), 4)
            )
        groups.sort(key=lambda group: group[0].model_index)
    else:
        expected_size = int(representation.rsplit("_", 1)[1])
        if len(turns) < expected_size:
            groups = [list(turns)]
        else:
            groups = [
                turns[start:start + expected_size]
                for start in range(len(turns) - expected_size + 1)
            ]
    return [
        _compose_block(
            representation, group, block_index=index, expected_size=expected_size,
        )
        for index, group in enumerate(groups)
    ]


def generate_block_word_perturbations(
    block: ContextBlock,
    *,
    max_ngram: int = 5,
) -> list[dict[str, Any]]:
    """Remove palavras dentro de uma única mensagem; nunca cruza fronteiras."""
    import re

    rows: list[dict[str, Any]] = []
    for local_index, (message, (block_start, _)) in enumerate(
        zip(block.message_texts, block.message_offsets)
    ):
        prefix_end = message.find(":") + 1
        content_start = prefix_end
        while content_start < len(message) and message[content_start].isspace():
            content_start += 1
        words = list(re.finditer(r"\S+", message[content_start:]))
        for size in range(1, min(max_ngram, len(words)) + 1):
            for start_word in range(len(words) - size + 1):
                end_word = start_word + size
                start = content_start + words[start_word].start()
                end = content_start + words[end_word - 1].end()
                left = message[:start].rstrip()
                right = message[end:].lstrip()
                perturbed_message = f"{left} {right}" if left and right else left or right
                changed_messages = list(block.message_texts)
                changed_messages[local_index] = perturbed_message
                rows.append({
                    "block_index": block.block_index,
                    "message_local_index": local_index,
                    "message_model_index": block.message_model_indices[local_index],
                    "message_original_index": block.message_original_indices[local_index],
                    "speaker": block.speakers[local_index],
                    "message_text": message,
                    "span_text": message[start:end],
                    "start_char": start,
                    "end_char": end,
                    "block_start_char": block_start + start,
                    "block_end_char": block_start + end,
                    "start_word": start_word,
                    "end_word": end_word,
                    "ngram_size": size,
                    "perturbed_text": "\n".join(changed_messages),
                })
    return rows


def _token_metadata(encoder, texts: Sequence[str]) -> tuple[list[int], list[bool], int]:
    tokenizer = getattr(encoder, "tokenizer", None)
    max_length = int(getattr(encoder, "max_seq_length", 0) or 0)
    if tokenizer is None:
        return [0] * len(texts), [False] * len(texts), max_length
    encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    return lengths, [bool(max_length and length > max_length) for length in lengths], max_length


def _encode_unique(encoder, texts: Sequence[str], batch_size: int = 64) -> dict[str, np.ndarray]:
    unique = list(dict.fromkeys(texts))
    if not unique:
        return {}
    vectors = np.asarray(encoder.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ), dtype=np.float32)
    if vectors.shape != (len(unique), BGE_DIMENSION):
        raise ValueError(f"Shape BGE inválido: {vectors.shape}")
    return dict(zip(unique, vectors))


def build_context_embedding_cache(
    *,
    output_root: str | Path = CONTEXT_BLOCK_ROOT,
    representations: Sequence[str] = REPRESENTATIONS,
    source_splits: Sequence[str] = ("train", "validation"),
    device=None,
    batch_size: int = 64,
    force: bool = False,
) -> Path:
    """Gera caches de blocos para treino e validação externa."""
    root = Path(output_root)
    encoder = load_bge_encoder(device=device)
    for source_split in source_splits:
        source = load_source_frame(source_split)
        source_fingerprint = _source_fingerprint(source)
        for representation in representations:
            destination = root / "cache" / representation / f"{source_split}.parquet"
            metadata_path = root / "cache" / representation / f"{source_split}_blocks.parquet"
            manifest_path = destination.with_suffix(".manifest.json")
            if destination.exists() and metadata_path.exists() and manifest_path.exists() and not force:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("schema_version") == SCHEMA_VERSION
                    and manifest.get("source_fingerprint") == source_fingerprint
                    and manifest.get("sequence_sha256") == sha256_file(destination)
                    and manifest.get("blocks_sha256") == sha256_file(metadata_path)
                ):
                    continue
            prepared: list[tuple[pd.Series, list[ContextBlock]]] = []
            texts: list[str] = []
            for row in source.itertuples(index=False):
                blocks = build_context_blocks(row.text, representation)
                prepared.append((row, blocks))
                texts.extend(block.text for block in blocks)
                texts.extend(block.neutral_text for block in blocks)
            lookup = _encode_unique(encoder, texts, batch_size=batch_size)
            sequence_rows: list[dict[str, Any]] = []
            block_rows: list[dict[str, Any]] = []
            for row, blocks in prepared:
                token_counts, truncated, max_length = _token_metadata(
                    encoder, [block.text for block in blocks],
                )
                embeddings = np.stack([lookup[block.text] for block in blocks])
                neutral = np.stack([lookup[block.neutral_text] for block in blocks])
                sequence_rows.append({
                    "sample_id": str(row.sample_id),
                    "label": int(row.label),
                    "representation": representation,
                    "source_split": source_split,
                    "n_original_messages": len(parse_conversation(row.text, max_turns=MAX_ORIGINAL_MESSAGES)),
                    "n_units": len(blocks),
                    "embeddings": json.dumps(embeddings.tolist(), separators=(",", ":")),
                    "neutral_embeddings": json.dumps(neutral.tolist(), separators=(",", ":")),
                    "text_hash": hashlib.sha256(str(row.text).encode("utf-8")).hexdigest(),
                })
                previous_indices: set[int] = set()
                for block, token_count, was_truncated in zip(blocks, token_counts, truncated):
                    current_indices = set(block.message_model_indices)
                    overlap = len(current_indices & previous_indices)
                    block_rows.append({
                        "sample_id": str(row.sample_id),
                        "label": int(row.label),
                        "source_split": source_split,
                        **block.to_record(),
                        "token_count": int(token_count),
                        "max_token_length": int(max_length),
                        "was_truncated": bool(was_truncated),
                        "overlap_with_previous": overlap,
                        "overlap_fraction_with_previous": overlap / max(len(current_indices), 1),
                    })
                    previous_indices = current_indices
            _atomic_parquet(destination, pd.DataFrame(sequence_rows))
            _atomic_parquet(metadata_path, pd.DataFrame(block_rows))
            atomic_json(manifest_path, {
                "schema_version": SCHEMA_VERSION,
                "representation": representation,
                "source_split": source_split,
                "source_fingerprint": source_fingerprint,
                "encoder": encoder_provenance(encoder),
                "max_original_messages": MAX_ORIGINAL_MESSAGES,
                "conversations": len(sequence_rows),
                "blocks": len(block_rows),
                "sequence_sha256": sha256_file(destination),
                "blocks_sha256": sha256_file(metadata_path),
            })
    return root / "cache"


def load_context_cache(
    representation: str,
    source_split: str,
    *,
    output_root: str | Path = CONTEXT_BLOCK_ROOT,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(output_root) / "cache" / representation / f"{source_split}.parquet"
    frame = pd.read_parquet(path)
    sequences = [np.asarray(json.loads(value), dtype=np.float32) for value in frame.embeddings]
    neutral = [np.asarray(json.loads(value), dtype=np.float32) for value in frame.neutral_embeddings]
    return sequences, neutral, frame.label.to_numpy(int), frame.sample_id.astype(str).to_numpy(), frame


def _context_model_dir(root: Path, representation: str, seed: int) -> Path:
    return root / "models" / representation / f"seed-{int(seed)}"


def train_context_transformer(
    representation: str,
    seed: int,
    *,
    output_root: str | Path = CONTEXT_BLOCK_ROOT,
    force: bool = False,
) -> Path:
    """Treina um Transformer contextual sem alterar o cache dos finalistas."""
    import torch

    if representation == "canonical_message":
        raise ValueError("canonical_message reutiliza os finalistas existentes")
    root = Path(output_root)
    destination = _context_model_dir(root, representation, seed)
    checkpoint = destination / "model.pth"
    train_cache = root / "cache" / representation / "train.parquet"
    validation_cache = root / "cache" / representation / "validation.parquet"
    cache_fingerprints = {
        "train": sha256_file(train_cache),
        "validation": sha256_file(validation_cache),
    }
    metrics_path = destination / "metrics.json"
    if checkpoint.exists() and (destination / "predictions_validation.parquet").exists() and metrics_path.exists() and not force:
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous.get("cache_fingerprints") == cache_fingerprints:
            return destination
    sequences, _, labels, ids, _ = load_context_cache(representation, "train", output_root=root)
    validation, _, validation_labels, validation_ids, _ = load_context_cache(
        representation, "validation", output_root=root,
    )
    sequences_array = np.asarray(sequences, dtype=object)
    split_indices = get_shared_split_indices("transformer", ids, labels, seed=42)
    train_dl, val_dl, test_dl, _, idx_val, idx_test = _make_dataloaders(
        sequences_array, labels,
        batch_size=FINALIST_CONFIG["batch_size"],
        seed=42,
        return_indices=True,
        split_indices=split_indices,
    )
    external_dl = _make_single_dataloader(
        validation, validation_labels,
        batch_size=FINALIST_CONFIG["batch_size"], shuffle=False,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerScamClassifier(
        BGE_DIMENSION,
        num_heads=FINALIST_CONFIG["num_heads"],
        num_layers=FINALIST_CONFIG["num_layers"],
        hidden_dim=FINALIST_CONFIG["hidden_dim"],
        dropout=FINALIST_CONFIG["dropout"],
    ).to(device)
    model, best_state, history = _train_transformer_loop(
        model, train_dl, val_dl,
        FINALIST_CONFIG["max_epochs"], FINALIST_CONFIG["lr"], device,
        optimizer_name=FINALIST_CONFIG["optimizer"],
        weight_decay=FINALIST_CONFIG["weight_decay"],
        patience=FINALIST_CONFIG["patience"],
        min_delta=FINALIST_CONFIG["min_delta"],
        scheduler_patience=FINALIST_CONFIG["scheduler_patience"],
        min_lr=FINALIST_CONFIG["min_lr"],
        mixed_precision=FINALIST_CONFIG["mixed_precision"],
        gradient_clip=FINALIST_CONFIG["gradient_clip"],
    )
    model.load_state_dict(best_state)
    predictions = {}
    for name, loader, sample_ids in (
        ("internal_validation", val_dl, ids[idx_val]),
        ("test_internal", test_dl, ids[idx_test]),
        ("validation", external_dl, validation_ids),
    ):
        y_true, y_pred, y_prob = _evaluate_dl(model, loader, device)
        predictions[name] = pd.DataFrame({
            "sample_id": sample_ids.astype(str), "label": y_true.astype(int),
            "predicted_label": y_pred.astype(int), "probability_scam": y_prob,
        })
    destination.mkdir(parents=True, exist_ok=True)
    config = {**FINALIST_CONFIG, "representation": representation, "seed": int(seed)}
    torch.save({"state_dict": model.to("cpu").state_dict(), "config": config}, checkpoint)
    _atomic_parquet(destination / "history.parquet", history)
    metrics = {}
    for split, frame in predictions.items():
        _atomic_parquet(destination / f"predictions_{split}.parquet", frame)
        metrics[split] = compute_metrics(
            frame.label.to_numpy(), frame.predicted_label.to_numpy(),
            y_prob=frame.probability_scam.to_numpy(),
            model_name=f"Transformer_bge_{representation}", dataset_name=split,
        )
    atomic_json(metrics_path, {
        "schema_version": SCHEMA_VERSION, "config": config, "metrics": metrics,
        "cache_fingerprints": cache_fingerprints,
        "best_epoch": int(history.attrs.get("best_epoch", 0)),
        "stopped_epoch": int(history.attrs.get("stopped_epoch", len(history))),
    })
    return destination


def load_context_model(
    representation: str,
    seed: int,
    model_mode: str,
    *,
    output_root: str | Path = CONTEXT_BLOCK_ROOT,
    device=None,
    checkpoint_root: str | Path | None = None,
):
    import torch

    if model_mode == "frozen_probe" or representation == "canonical_message":
        return load_bge_finalist_model(seed=seed, device=device, checkpoint_root=checkpoint_root)
    path = _context_model_dir(Path(output_root), representation, seed) / "model.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = TransformerScamClassifier(
        BGE_DIMENSION,
        num_heads=int(config["num_heads"]), num_layers=int(config["num_layers"]),
        hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model = model.eval().to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, {"representation": representation, "seed": seed, "model_mode": model_mode, "checkpoint": str(path)}


def _infer_same_length(model, arrays: np.ndarray, device, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    import torch

    logits = []
    for start in range(0, len(arrays), batch_size):
        values = torch.as_tensor(arrays[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits.append(model(values).reshape(-1).detach().cpu().numpy())
    logit = np.concatenate(logits).astype(float)
    return logit, 1.0 / (1.0 + np.exp(-np.clip(logit, -80, 80)))


def _infer_replacements(model, original, indices, replacements, device, batch_size: int = 32):
    indices = list(indices)
    replacements = np.asarray(replacements, dtype=np.float32)
    if len(indices) != len(replacements):
        raise ValueError("Índices e substituições desalinhados")
    logits = []; probabilities = []
    for start in range(0, len(indices), batch_size):
        batch_indices = np.asarray(indices[start:start + batch_size], dtype=int)
        batch_replacements = replacements[start:start + batch_size]
        batch = np.repeat(original[None, :, :], len(batch_indices), axis=0)
        batch[np.arange(len(batch_indices)), batch_indices] = batch_replacements
        batch_logits, batch_probabilities = _infer_same_length(
            model, batch, device, batch_size=batch_size,
        )
        logits.append(batch_logits); probabilities.append(batch_probabilities)
    if not logits:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.concatenate(logits), np.concatenate(probabilities)


def _category(label: int, prediction: int) -> str:
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(int(label), int(prediction))]


def _effect_direction(delta_logit: float) -> str:
    if delta_logit > DEFAULT_EFFECT_EPSILON:
        return "scam"
    if delta_logit < -DEFAULT_EFFECT_EPSILON:
        return "ham"
    return "neutral"


def _context_blocks_from_frame(frame: pd.DataFrame, representation: str) -> list[ContextBlock]:
    frame = frame.sort_values("block_index")
    return [ContextBlock(
        block_index=int(row.block_index), representation=representation,
        text=str(row.text), neutral_text=str(row.neutral_text),
        message_model_indices=tuple(map(int, _json_list(row.message_model_indices))),
        message_original_indices=tuple(map(int, _json_list(row.message_original_indices))),
        speakers=tuple(map(str, _json_list(row.speakers))),
        message_texts=tuple(map(str, _json_list(row.message_texts))),
        message_offsets=tuple(tuple(map(int, value)) for value in _json_list(row.message_offsets)),
        first_model_index=int(row.first_model_index), last_model_index=int(row.last_model_index),
        is_partial=bool(row.is_partial),
    ) for row in frame.itertuples(index=False)]


def explain_context_sequence(
    *,
    model,
    encoder,
    device,
    representation: str,
    model_mode: str,
    seed: int,
    split: str,
    sample_id: str,
    label: int,
    blocks: Sequence[ContextBlock],
    original: np.ndarray,
    neutral: np.ndarray,
    top_units: int = 6,
    max_ngram: int = 5,
    controls_per_span: int = 5,
    include_micro: bool = True,
) -> dict[str, pd.DataFrame]:
    """Calcula macro, micro e fidelidade para uma sequência contextual."""
    baseline_logit, baseline_probability = _infer_same_length(model, original[None], device)
    baseline_logit = float(baseline_logit[0]); baseline_probability = float(baseline_probability[0])
    predicted = int(baseline_probability >= 0.5)
    macro_logits, macro_probabilities = _infer_replacements(
        model, original, range(len(blocks)), neutral, device,
    )
    prefix = {
        "sample_id": str(sample_id), "split": split, "representation": representation,
        "model_mode": model_mode, "seed": int(seed), "label": int(label),
        "predicted_label": predicted, "prediction_category": _category(label, predicted),
        "baseline_logit": baseline_logit, "baseline_probability_scam": baseline_probability,
        "total_units": len(blocks),
        "total_original_messages": len(set(index for block in blocks for index in block.message_model_indices)),
    }
    unit_rows = []
    for block, changed_logit, changed_probability in zip(blocks, macro_logits, macro_probabilities):
        delta_logit = baseline_logit - float(changed_logit)
        delta_probability = baseline_probability - float(changed_probability)
        unit_rows.append({
            **prefix, **block.to_record(), "model_index": block.block_index,
            "original_index": block.first_model_index,
            "perturbed_logit": float(changed_logit),
            "perturbed_probability": float(changed_probability),
            "delta_logit": delta_logit, "abs_delta_logit": abs(delta_logit),
            "delta_probability": delta_probability,
            "abs_delta_probability": abs(delta_probability),
            "class_support_delta": class_support_delta(delta_logit, predicted),
            "direction": _effect_direction(delta_logit),
            "prediction_changed": int(float(changed_probability) >= 0.5) != predicted,
            "supported_class": "scam" if predicted else "ham",
        })
    absolute = select_message_evidence(
        unit_rows, predicted_label=predicted, top_messages=top_units,
        ranking_policy=RANKING_ABSOLUTE_INFLUENCE,
    )
    supporting = select_message_evidence(
        unit_rows, predicted_label=predicted, top_messages=top_units,
        ranking_policy=RANKING_DECISION_SUPPORTING,
    )
    absolute_indices = {int(row["model_index"]) for row in absolute}
    supporting_indices = {int(row["model_index"]) for row in supporting}
    for row in unit_rows:
        row["selected_absolute_influence"] = int(row["model_index"]) in absolute_indices
        row["selected_decision_supporting"] = int(row["model_index"]) in supporting_indices

    span_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    if include_micro:
        candidate_blocks = [blocks[index] for index in sorted(absolute_indices | supporting_indices)]
        candidates = [row for block in candidate_blocks for row in generate_block_word_perturbations(block, max_ngram=max_ngram)]
        if candidates:
            vectors = np.asarray(encoder.encode(
                [row["perturbed_text"] for row in candidates], batch_size=64,
                show_progress_bar=False, convert_to_numpy=True,
            ), dtype=np.float32)
            changed_logits, changed_probabilities = _infer_replacements(
                model, original, [row["block_index"] for row in candidates], vectors, device,
            )
            for candidate, changed_logit, changed_probability in zip(candidates, changed_logits, changed_probabilities):
                delta_logit = baseline_logit - float(changed_logit)
                span_rows.append({
                    **prefix, **candidate,
                    "perturbed_logit": float(changed_logit),
                    "perturbed_probability": float(changed_probability),
                    "delta_logit": delta_logit, "abs_delta_logit": abs(delta_logit),
                    "delta_probability": baseline_probability - float(changed_probability),
                    "class_support_delta": class_support_delta(delta_logit, predicted),
                    "direction": _effect_direction(delta_logit),
                    "prediction_changed": int(float(changed_probability) >= 0.5) != predicted,
                })
            span_frame = pd.DataFrame(span_rows)
            selected_spans = (
                span_frame[span_frame.class_support_delta.gt(DEFAULT_EFFECT_EPSILON)]
                .sort_values(["block_index", "class_support_delta", "ngram_size"], ascending=[True, False, True])
                .groupby("block_index", group_keys=False).head(1)
            )
            for selected in selected_spans.itertuples(index=False):
                pool = span_frame[
                    span_frame.message_model_index.eq(selected.message_model_index)
                    & span_frame.ngram_size.eq(selected.ngram_size)
                    & ~(
                        span_frame.start_char.eq(selected.start_char)
                        & span_frame.end_char.eq(selected.end_char)
                    )
                ]
                rng = np.random.RandomState(int(hashlib.sha256(
                    f"{sample_id}:{selected.block_index}:{selected.start_char}:{selected.end_char}".encode()
                ).hexdigest()[:8], 16))
                if len(pool) > controls_per_span:
                    pool = pool.iloc[rng.choice(len(pool), controls_per_span, replace=False)]
                for control in pool.itertuples(index=False):
                    control_rows.append({
                        **prefix, "block_index": selected.block_index,
                        "message_model_index": selected.message_model_index,
                        "span_text": selected.span_text,
                        "selected_abs_delta_logit": abs(selected.delta_logit),
                        "control_span_text": control.span_text,
                        "control_abs_delta_logit": abs(control.delta_logit),
                        "ngram_size": selected.ngram_size,
                    })

    fidelity_rows: list[dict[str, Any]] = []
    for policy in (RANKING_ABSOLUTE_INFLUENCE, RANKING_DECISION_SUPPORTING):
        ranked = select_message_evidence(
            unit_rows, predicted_label=predicted, top_messages=len(unit_rows),
            ranking_policy=policy,
        )
        if not ranked:
            continue
        selected_so_far: list[int] = []
        covered: set[int] = set()
        cumulative_mass = 0.0
        removed_sequences = []; sufficient_sequences = []
        coverage_rows = []
        for rank, row in enumerate(ranked, start=1):
            index = int(row["model_index"]); selected_so_far.append(index)
            covered.update(blocks[index].message_model_indices)
            cumulative_mass += abs(float(row["delta_logit"])) if policy == RANKING_ABSOLUTE_INFLUENCE else max(0.0, float(row["class_support_delta"]))
            removed = original.copy(); removed[selected_so_far] = neutral[selected_so_far]
            sufficient = neutral.copy(); sufficient[selected_so_far] = original[selected_so_far]
            removed_sequences.append(removed); sufficient_sequences.append(sufficient)
            occurrences = sum(
                len(set(blocks[selected].message_model_indices)) for selected in selected_so_far
            )
            coverage_rows.append((rank, len(covered), occurrences, cumulative_mass))
        removed_logits, removed_prob = _infer_same_length(model, np.stack(removed_sequences), device)
        sufficient_logits, sufficient_prob = _infer_same_length(model, np.stack(sufficient_sequences), device)
        original_score = baseline_probability if predicted else 1 - baseline_probability
        for (rank, unique_count, occurrences, mass), rem_logit, rem_prob, suff_logit, suff_prob in zip(
            coverage_rows, removed_logits, removed_prob, sufficient_logits, sufficient_prob,
        ):
            removed_score = float(rem_prob) if predicted else 1 - float(rem_prob)
            sufficient_score = float(suff_prob) if predicted else 1 - float(suff_prob)
            comp_raw = baseline_logit - float(rem_logit)
            suff_raw = baseline_logit - float(suff_logit)
            comp_aligned = class_support_delta(comp_raw, predicted)
            fidelity_rows.append({
                **prefix, "ranking_policy": policy, "top_k": rank,
                "unique_messages_covered": unique_count,
                "coverage_fraction": unique_count / max(prefix["total_original_messages"], 1),
                "message_occurrences_covered": occurrences,
                "redundancy_ratio": max(occurrences - unique_count, 0) / max(unique_count, 1),
                "removed_logit": float(rem_logit), "removed_probability_scam": float(rem_prob),
                "sufficient_logit": float(suff_logit), "sufficient_probability_scam": float(suff_prob),
                "comprehensiveness": original_score - removed_score,
                "comprehensiveness_abs": abs(original_score - removed_score),
                "comprehensiveness_logit_raw": comp_raw,
                "comprehensiveness_logit_aligned": comp_aligned,
                "comprehensiveness_logit_abs": abs(comp_raw),
                "sufficiency_gap": original_score - sufficient_score,
                "sufficiency_distance": abs(original_score - sufficient_score),
                "sufficiency_gap_logit_aligned": class_support_delta(suff_raw, predicted),
                "sufficiency_distance_logit": abs(suff_raw),
                "removed_prediction_changed": int(float(rem_prob) >= 0.5) != predicted,
                "sufficient_prediction_changed": int(float(suff_prob) >= 0.5) != predicted,
                "cumulative_evidence_mass_logit": mass,
                "redundancy_gap_logit": mass - comp_aligned,
            })
    return {
        "conversation_metrics": pd.DataFrame([{**prefix}]),
        "unit_effects": pd.DataFrame(unit_rows),
        "span_effects": pd.DataFrame(span_rows),
        "fidelity": pd.DataFrame(fidelity_rows),
        "random_controls": pd.DataFrame(control_rows),
    }


def run_context_explanation_shard(
    *,
    split: str,
    representation: str,
    model_mode: str,
    shard_index: int,
    shard_count: int,
    output_root: str | Path = CONTEXT_BLOCK_ROOT,
    device=None,
    checkpoint_root: str | Path | None = None,
    resume: bool = True,
) -> Path:
    import torch

    root = Path(output_root)
    destination = root / "explanations" / representation / model_mode / split / f"shard-{shard_index:05d}"
    complete = destination / "complete.json"
    source_split = "train" if split == "test_internal" else "validation"
    cache_paths = {
        "sequence": root / "cache" / representation / f"{source_split}.parquet",
        "blocks": root / "cache" / representation / f"{source_split}_blocks.parquet",
    }
    cache_fingerprints = {name: sha256_file(path) for name, path in cache_paths.items()}
    if resume and complete.exists():
        previous = json.loads(complete.read_text(encoding="utf-8"))
        if (
            previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("cache_fingerprints") == cache_fingerprints
        ):
            return destination
    sequences, neutral, labels, ids, _ = load_context_cache(representation, source_split, output_root=root)
    block_frame = pd.read_parquet(
        root / "cache" / representation / f"{source_split}_blocks.parquet"
    )
    block_frame["sample_id"] = block_frame.sample_id.astype(str)
    block_groups = {
        str(sample_id): group
        for sample_id, group in block_frame.groupby("sample_id", sort=False)
    }
    if split == "test_internal":
        _, _, test_indices = get_shared_split_indices("transformer", ids, labels, seed=42)
        chosen = list(map(int, test_indices))
    else:
        chosen = list(range(len(ids)))
    chosen = sorted(chosen, key=lambda index: hashlib.sha256(ids[index].encode()).hexdigest())
    chosen = [index for position, index in enumerate(chosen) if position % shard_count == shard_index]
    encoder = load_bge_encoder(device=device)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output: dict[str, list[pd.DataFrame]] = {
        "conversation_metrics": [], "unit_effects": [], "span_effects": [],
        "fidelity": [], "random_controls": [], "stability": [],
    }
    per_seed_models = {
        seed: load_context_model(
            representation, seed, model_mode, output_root=root,
            device=device, checkpoint_root=checkpoint_root,
        )[0].eval().to(device)
        for seed in SEEDS
    }
    for index in chosen:
        sample_id = ids[index]
        blocks = _context_blocks_from_frame(block_groups[str(sample_id)], representation)
        seed_unit_frames = {}
        for seed, model in per_seed_models.items():
            frames = explain_context_sequence(
                model=model, encoder=encoder, device=device,
                representation=representation, model_mode=model_mode, seed=seed,
                split=split, sample_id=sample_id, label=int(labels[index]),
                blocks=blocks, original=sequences[index], neutral=neutral[index],
                include_micro=seed == PRIMARY_SEED,
            )
            seed_unit_frames[seed] = frames["unit_effects"]
            for name, frame in frames.items():
                if not frame.empty:
                    output[name].append(frame)
        base = seed_unit_frames[PRIMARY_SEED]
        base_prediction = int(base.predicted_label.iloc[0])
        base_top = set(base.nlargest(6, "abs_delta_logit").model_index.astype(int))
        for seed in (52, 62):
            other = seed_unit_frames[seed]
            agrees = base_prediction == int(other.predicted_label.iloc[0])
            other_top = set(other.nlargest(6, "abs_delta_logit").model_index.astype(int))
            union = base_top | other_top
            comparison = base[["model_index", "delta_logit", "abs_delta_logit"]].merge(
                other[["model_index", "delta_logit", "abs_delta_logit"]],
                on="model_index", suffixes=("_base", "_other"), validate="one_to_one",
            )
            base_direction = comparison.delta_logit_base.map(_effect_direction)
            other_direction = comparison.delta_logit_other.map(_effect_direction)
            impacts = comparison[["abs_delta_logit_base", "abs_delta_logit_other"]].to_numpy(float)
            impact_mean = float(np.mean(impacts)) if impacts.size else np.nan
            impact_std = float(np.std(impacts, ddof=1)) if impacts.size > 1 else 0.0
            stability_row = {
                "sample_id": sample_id, "split": split, "representation": representation,
                "model_mode": model_mode, "base_seed": 42, "compared_seed": seed,
                "prediction_agreement": agrees,
                "top_k_overlap": len(base_top & other_top),
                "top_k_jaccard": len(base_top & other_top) / max(len(union), 1),
                "direction_agreement": float((base_direction == other_direction).mean()),
                "rank_spearman": float(comparison.abs_delta_logit_base.corr(
                    comparison.abs_delta_logit_other, method="spearman",
                )),
                "rank_kendall": float(comparison.abs_delta_logit_base.corr(
                    comparison.abs_delta_logit_other, method="kendall",
                )),
                "impact_mean": impact_mean,
                "impact_std": impact_std,
                "impact_cv": impact_std / impact_mean if impact_mean > 0 else np.nan,
            }
            if not agrees:
                for key in (
                    "top_k_overlap", "top_k_jaccard", "direction_agreement",
                    "rank_spearman", "rank_kendall", "impact_mean",
                    "impact_std", "impact_cv",
                ):
                    stability_row[key] = np.nan
            output["stability"].append(pd.DataFrame([stability_row]))
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, frames in output.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = destination / f"{name}.parquet"
        _atomic_parquet(path, frame)
        checksums[path.name] = sha256_file(path)
    atomic_json(destination / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "split": split,
        "representation": representation, "model_mode": model_mode,
        "shard_index": shard_index, "shard_count": shard_count,
        "sample_count": len(chosen), "checksums": checksums,
        "cache_fingerprints": cache_fingerprints,
    })
    atomic_json(complete, {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "checksums": checksums, "cache_fingerprints": cache_fingerprints,
    })
    return destination


def _k_points(fidelity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "representation", "model_mode", "sample_id", "seed", "ranking_policy"]
    for values, group in fidelity.groupby(keys, dropna=False):
        group = group.sort_values("top_k").copy()
        envelope = group.comprehensiveness_logit_aligned.cummax()
        maximum = max(0.0, float(envelope.max()))
        threshold = 0.95 * maximum
        k95_candidates = group[envelope.ge(threshold)] if maximum > 0 else group.iloc[0:0]
        minimum_suff = float(group.sufficiency_distance_logit.min())
        initial_suff = float(group.sufficiency_distance_logit.iloc[0])
        suff_target = minimum_suff + 0.05 * max(initial_suff - minimum_suff, 0.0)
        joint = group[envelope.ge(threshold) & group.sufficiency_distance_logit.le(suff_target)]
        base = dict(zip(keys, values))
        for name, candidates in (("k95", k95_candidates), ("k_joint", joint)):
            if candidates.empty:
                rows.append({**base, "point_type": name, "status": "not_reached"})
            else:
                point = candidates.iloc[0]
                rows.append({
                    **base, "point_type": name, "status": "reached",
                    "units": int(point.top_k),
                    "unique_messages": int(point.unique_messages_covered),
                    "fraction": float(point.coverage_fraction),
                    "comprehensiveness_logit_aligned": float(point.comprehensiveness_logit_aligned),
                    "sufficiency_distance_logit": float(point.sufficiency_distance_logit),
                    "redundancy_ratio": float(point.redundancy_ratio),
                })
    return pd.DataFrame(rows)


def aggregate_context_results(
    *,
    output_root: str | Path = CONTEXT_BLOCK_ROOT,
    expected_shards: int = 32,
) -> Path:
    root = Path(output_root)
    aggregate = root / "aggregates"
    aggregate.mkdir(parents=True, exist_ok=True)
    tables = ["conversation_metrics", "unit_effects", "span_effects", "fidelity", "random_controls", "stability"]
    consolidated = {}
    for name in tables:
        frames = []
        for representation in REPRESENTATIONS:
            for mode in MODEL_MODES:
                for split in ("test_internal", "validation"):
                    for index in range(expected_shards):
                        shard = root / "explanations" / representation / mode / split / f"shard-{index:05d}"
                        complete_path = shard / "complete.json"
                        if not complete_path.exists():
                            raise FileNotFoundError(complete_path)
                        complete = json.loads(complete_path.read_text(encoding="utf-8"))
                        if complete.get("status") != "complete":
                            raise RuntimeError(f"Shard incompleto: {shard}")
                        path = shard / f"{name}.parquet"
                        if not path.exists():
                            raise FileNotFoundError(path)
                        expected_checksum = complete.get("checksums", {}).get(path.name)
                        if expected_checksum != sha256_file(path):
                            raise RuntimeError(f"Checksum inválido: {path}")
                        frame = pd.read_parquet(path)
                        if not frame.empty:
                            frames.append(frame)
        consolidated[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        _atomic_parquet(aggregate / f"{name}.parquet", consolidated[name])
    fidelity = consolidated["fidelity"]
    points = _k_points(fidelity)
    _atomic_parquet(aggregate / "quantity_points.parquet", points)
    metric_rows = []
    metrics = [
        "comprehensiveness", "comprehensiveness_logit_aligned",
        "sufficiency_distance", "sufficiency_distance_logit",
        "cumulative_evidence_mass_logit", "redundancy_gap_logit",
        "unique_messages_covered", "coverage_fraction", "redundancy_ratio",
    ]
    conversations = consolidated["conversation_metrics"]
    conversation_keys = ["split", "representation", "model_mode", "seed", "sample_id"]
    if conversations.duplicated(conversation_keys).any():
        duplicates = conversations.loc[
            conversations.duplicated(conversation_keys, keep=False), conversation_keys
        ].head(10)
        raise RuntimeError(f"Conversas duplicadas entre shards:\n{duplicates}")
    for split in ("test_internal", "validation"):
        expected_ids = None
        for representation in REPRESENTATIONS:
            ids = set(conversations[
                conversations.split.eq(split)
                & conversations.representation.eq(representation)
                & conversations.model_mode.eq("frozen_probe")
                & conversations.seed.eq(PRIMARY_SEED)
            ].sample_id.astype(str))
            expected_ids = ids if expected_ids is None else expected_ids
            if ids != expected_ids:
                raise RuntimeError(
                    f"Cobertura desigual em {split}/{representation}: "
                    f"esperado={len(expected_ids)} observado={len(ids)}"
                )
    paired_frames = []
    common_counts = {}
    for split in ("test_internal", "validation"):
        for mode in MODEL_MODES:
            common_ids = None
            for representation in REPRESENTATIONS:
                ids = set(conversations[
                    conversations.split.eq(split)
                    & conversations.model_mode.eq(mode)
                    & conversations.representation.eq(representation)
                    & conversations.seed.eq(PRIMARY_SEED)
                ].sample_id.astype(str))
                common_ids = ids if common_ids is None else common_ids & ids
            common_ids = common_ids or set()
            common_counts[f"{split}:{mode}"] = len(common_ids)
            paired_frames.append(fidelity[
                fidelity.split.eq(split)
                & fidelity.model_mode.eq(mode)
                & fidelity.sample_id.astype(str).isin(common_ids)
            ])
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    _atomic_parquet(aggregate / "paired_common_fidelity.parquet", paired)

    for population, frame in (("all_available", fidelity), ("paired_common", paired)):
        for keys, group in frame.groupby(
            ["split", "representation", "model_mode", "seed", "ranking_policy", "top_k"]
        ):
            base = dict(zip(
                ["split", "representation", "model_mode", "seed", "ranking_policy", "top_k"], keys,
            ))
            for metric in metrics:
                values = group[metric].dropna().to_numpy()
                if len(values):
                    metric_rows.append({
                        **base, "population": population, "metric": metric,
                        **summarize_metric(values),
                    })
    summary = pd.DataFrame(metric_rows)
    _atomic_parquet(aggregate / "metric_summary.parquet", summary)

    category_rows = []
    for keys, group in fidelity.groupby([
        "split", "representation", "model_mode", "seed", "ranking_policy",
        "prediction_category", "top_k",
    ]):
        base = dict(zip([
            "split", "representation", "model_mode", "seed", "ranking_policy",
            "prediction_category", "top_k",
        ], keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy()
            if len(values):
                category_rows.append({**base, "metric": metric, **summarize_metric(values)})
    _atomic_parquet(aggregate / "category_summary.parquet", pd.DataFrame(category_rows))

    classification_rows = []
    for keys, group in conversations.groupby(["split", "representation", "model_mode", "seed"]):
        split, representation, mode, seed = keys
        metric = compute_metrics(
            group.label.to_numpy(int), group.predicted_label.to_numpy(int),
            y_prob=group.baseline_probability_scam.to_numpy(float),
            model_name=f"Transformer_bge_{representation}_{mode}", dataset_name=split,
        )
        classification_rows.append({
            "split": split, "representation": representation,
            "model_mode": mode, "seed": int(seed), **metric,
        })
    _atomic_parquet(aggregate / "classification_metrics.parquet", pd.DataFrame(classification_rows))

    agreement_rows = []
    for (split, mode, seed), group in conversations.groupby(["split", "model_mode", "seed"]):
        canonical = group[group.representation.eq("canonical_message")][
            ["sample_id", "predicted_label", "baseline_logit"]
        ].rename(columns={
            "predicted_label": "canonical_prediction",
            "baseline_logit": "canonical_logit",
        })
        for representation in REPRESENTATIONS:
            view = group[group.representation.eq(representation)].merge(canonical, on="sample_id")
            if view.empty:
                continue
            agreement_rows.append({
                "split": split, "model_mode": mode, "seed": int(seed),
                "representation": representation, "n": len(view),
                "prediction_agreement": float((view.predicted_label == view.canonical_prediction).mean()),
                "mean_abs_logit_shift": float((view.baseline_logit - view.canonical_logit).abs().mean()),
            })
    _atomic_parquet(aggregate / "canonical_agreement.parquet", pd.DataFrame(agreement_rows))
    atomic_json(root / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "representations": list(REPRESENTATIONS), "model_modes": list(MODEL_MODES),
        "seeds": list(SEEDS), "expected_shards": expected_shards,
        "rows": {name: len(frame) for name, frame in consolidated.items()},
        "paired_common_conversations": common_counts,
    })
    return aggregate
