"""Rebuild BGE message embeddings for the corrected validation split.

The final Parquet keeps the historical public contract unchanged:
``text``, ``label`` and ``text_embedded``.  Each ``text_embedded`` value is a
JSON list with shape ``(number_of_messages, 1024)`` so existing Transformer,
notebook and explainability loaders continue to work without code changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
RAW_PATH = REPOSITORY_ROOT / "backend/datasets/dataset_validation/raw/all_data.csv"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "backend/datasets/dataset_validation/processed/messages_transformer/bge_chunks.parquet"
)
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_DIMENSION = 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_messages(text: str) -> list[str]:
    """Apply the same message boundary contract used by the trained pipeline."""
    import re

    chunks = re.split(r"(?=Innocent: |Suspect: )", str(text))
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not chunks:
        raise ValueError("Conversa sem mensagens")
    invalid = [
        chunk for chunk in chunks
        if not chunk.startswith(("Innocent: ", "Suspect: "))
    ]
    if invalid:
        raise ValueError(f"Mensagem sem prefixo canônico: {invalid[0][:120]!r}")
    return chunks


def resolve_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA foi exigida, mas não está disponível")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_source(frame: pd.DataFrame) -> list[list[str]]:
    if list(frame.columns) != ["text", "label"]:
        raise ValueError(f"Colunas inesperadas no raw: {list(frame.columns)}")
    if len(frame) != 1083:
        raise ValueError(f"Esperadas 1083 conversas; encontradas {len(frame)}")
    distribution = frame["label"].value_counts().to_dict()
    if distribution != {0: 1000, 1: 83}:
        raise ValueError(f"Distribuição de classes inesperada: {distribution}")
    return [split_messages(text) for text in frame["text"]]


def _validate_embedding_json(value: str, expected_messages: int) -> None:
    array = np.asarray(json.loads(value), dtype=np.float32)
    if array.shape != (expected_messages, BGE_DIMENSION):
        raise ValueError(
            f"Embedding com shape {array.shape}; esperado "
            f"({expected_messages}, {BGE_DIMENSION})"
        )
    if not np.isfinite(array).all():
        raise ValueError("Embedding contém NaN ou infinito")


def _load_valid_shard(path: Path, source: pd.DataFrame, message_counts: list[int]) -> pd.DataFrame:
    shard = pd.read_parquet(path)
    expected_columns = ["row_index", "text", "label", "text_embedded"]
    if list(shard.columns) != expected_columns:
        raise ValueError(f"Shard inválido {path}: {list(shard.columns)}")
    for row in shard.itertuples(index=False):
        index = int(row.row_index)
        if row.text != source.at[index, "text"] or int(row.label) != int(source.at[index, "label"]):
            raise ValueError(f"Shard {path} não corresponde ao raw na linha {index}")
        _validate_embedding_json(row.text_embedded, message_counts[index])
    return shard


def build_bge_message_parquet(
    raw_path: Path = RAW_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    device: str = "auto",
    encode_batch_size: int = 32,
    conversations_per_shard: int = 20,
    force: bool = False,
    reproduction_audit: bool = True,
) -> dict:
    from sentence_transformers import SentenceTransformer

    raw_path = raw_path.resolve()
    output_path = output_path.resolve()
    source = pd.read_csv(raw_path)
    messages_by_conversation = _validate_source(source)
    message_counts = [len(messages) for messages in messages_by_conversation]
    raw_hash = file_sha256(raw_path)
    selected_device = resolve_device(device)

    work_dir = output_path.parent / ".bge_rebuild" / raw_hash[:16]
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(BGE_MODEL_NAME, device=str(selected_device))
    audit_result = None
    if reproduction_audit and output_path.exists():
        old = pd.read_parquet(output_path, columns=["text", "label", "text_embedded"])
        audit_index = int(source.index[source["label"] == 0][0])
        if old.at[audit_index, "text"] != source.at[audit_index, "text"]:
            raise ValueError(
                "A conversa Ham da auditoria mudou entre o Parquet antigo e o raw"
            )
        old_value = old.at[audit_index, "text_embedded"]
        old_array = np.asarray(
            json.loads(old_value) if isinstance(old_value, str) else old_value,
            dtype=np.float32,
        )
        new_array = np.asarray(model.encode(
            messages_by_conversation[audit_index],
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ), dtype=np.float32)
        if old_array.shape != new_array.shape:
            raise ValueError(
                f"Auditoria BGE falhou: shape antigo={old_array.shape}, novo={new_array.shape}"
            )
        max_abs_difference = float(np.max(np.abs(old_array - new_array)))
        if not np.allclose(old_array, new_array, atol=1e-5, rtol=1e-4):
            raise ValueError(
                "Auditoria BGE falhou para uma conversa Ham inalterada: "
                f"max_abs_difference={max_abs_difference:.8g}. "
                "Confira revisão do checkpoint e versões antes de prosseguir."
            )
        audit_result = {
            "row_index": audit_index,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_abs_difference": max_abs_difference,
        }
        print(f"[audit] BGE reproduzido: {audit_result}", flush=True)
    shard_paths: list[Path] = []
    shard_count = math.ceil(len(source) / conversations_per_shard)
    for shard_index in range(shard_count):
        start = shard_index * conversations_per_shard
        stop = min(len(source), start + conversations_per_shard)
        shard_path = work_dir / f"shard-{shard_index:04d}.parquet"
        shard_paths.append(shard_path)
        if shard_path.exists() and not force:
            _load_valid_shard(shard_path, source, message_counts)
            print(f"[resume] shard {shard_index + 1}/{shard_count}", flush=True)
            continue

        rows = []
        for row_index in range(start, stop):
            messages = messages_by_conversation[row_index]
            embeddings = model.encode(
                messages,
                batch_size=encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.shape != (len(messages), BGE_DIMENSION):
                raise ValueError(
                    f"Linha {row_index}: shape BGE {embeddings.shape}; "
                    f"esperado ({len(messages)}, {BGE_DIMENSION})"
                )
            if not np.isfinite(embeddings).all():
                raise ValueError(f"Linha {row_index}: BGE produziu NaN ou infinito")
            rows.append({
                "row_index": row_index,
                "text": source.at[row_index, "text"],
                "label": int(source.at[row_index, "label"]),
                "text_embedded": json.dumps(
                    embeddings.tolist(), separators=(",", ":"), allow_nan=False
                ),
            })
        shard_frame = pd.DataFrame(rows, columns=[
            "row_index", "text", "label", "text_embedded",
        ])
        temporary = shard_path.with_suffix(".tmp.parquet")
        shard_frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(shard_path)
        print(
            f"[encode] shard {shard_index + 1}/{shard_count} "
            f"conversas={start}:{stop}",
            flush=True,
        )

    rebuilt = pd.concat(
        [_load_valid_shard(path, source, message_counts) for path in shard_paths],
        ignore_index=True,
    ).sort_values("row_index")
    expected_indices = list(range(len(source)))
    if rebuilt["row_index"].tolist() != expected_indices:
        raise ValueError("Shards ausentes, duplicados ou fora de ordem")
    rebuilt = rebuilt[["text", "label", "text_embedded"]].reset_index(drop=True)

    backup_path = output_path.with_name("bge_chunks.pre_speaker_reconstruction.parquet")
    if output_path.exists() and not backup_path.exists():
        shutil.copy2(output_path, backup_path)
        print(f"[backup] {backup_path}", flush=True)

    with tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix="bge_chunks_rebuilt_",
        dir=output_path.parent, delete=False,
    ) as handle:
        temporary_output = Path(handle.name)
    rebuilt.to_parquet(temporary_output, index=False, compression="zstd")
    verified = pd.read_parquet(temporary_output)
    if len(verified) != len(source) or not verified["text"].equals(source["text"]):
        raise ValueError("Parquet final não reproduz o texto e a ordem do raw")
    temporary_output.replace(output_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(raw_path),
        "source_sha256": raw_hash,
        "output": str(output_path),
        "model": BGE_MODEL_NAME,
        "embedding_dimension": BGE_DIMENSION,
        "normalize_embeddings": False,
        "pooling": "SentenceTransformer checkpoint configuration",
        "device": str(selected_device),
        "conversation_count": len(source),
        "message_count": int(sum(message_counts)),
        "message_count_min": int(min(message_counts)),
        "message_count_max": int(max(message_counts)),
        "class_counts": {str(k): int(v) for k, v in source["label"].value_counts().items()},
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "sentence-transformers")
        },
        "reproduction_audit": audit_result,
    }
    manifest_path = output_path.with_name("bge_chunks.manifest.json")
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[output] {output_path}", flush=True)
    print(f"[manifest] {manifest_path}", flush=True)
    return metadata


def revalidate_bge_finalists(
    *,
    seeds: tuple[int, ...] = (42, 52, 62),
    device: str = "auto",
    checkpoint_root: Path | None = None,
    batch_size: int = 32,
) -> dict:
    """Re-evaluate frozen finalist checkpoints; no model is retrained."""
    import torch

    from machine_learning.cache import ModelCache
    from machine_learning.data import load_sequence_embeddings
    from machine_learning.evaluation import compute_metrics
    from machine_learning.studies.common import RESULTS_ROOT
    from machine_learning.studies.hierarchical_perturbation import load_bge_finalist_model
    from machine_learning.transformer.runner import _evaluate_dl, _make_single_dataloader

    selected_device = resolve_device(device)
    sequences, labels, sample_ids = load_sequence_embeddings(
        "validation", "bge", with_augmented=False, return_ids=True,
    )
    loader = _make_single_dataloader(sequences, labels, batch_size=batch_size, shuffle=False)
    rows = []
    for seed in seeds:
        model, model_metadata = load_bge_finalist_model(
            seed=seed, device=selected_device, checkpoint_root=checkpoint_root,
        )
        model = model.to(selected_device)
        y_true, y_pred, y_probability = _evaluate_dl(model, loader, selected_device)
        metrics = compute_metrics(
            y_true, y_pred, y_prob=y_probability,
            model_name="Transformer_bge", dataset_name="validation",
        )
        run_id = model_metadata["run_id"]
        artifacts = Path(ModelCache.artifacts_dir("transformer", run_id))
        metrics_path = artifacts / "metrics.json"
        predictions_path = artifacts / "predictions_validation.npz"
        for path in (metrics_path, predictions_path):
            if path.exists():
                backup = path.with_name(f"{path.stem}.pre_speaker_reconstruction{path.suffix}")
                if not backup.exists():
                    shutil.copy2(path, backup)
        saved_metrics = ModelCache.load_metrics("transformer", run_id) or {}
        saved_metrics["val"] = metrics
        saved_metrics["validation_rebuilt_from_corrected_speakers"] = True
        ModelCache.save_metrics("transformer", run_id, saved_metrics)
        ModelCache.save_predictions(
            "transformer", run_id, "validation",
            y_true, y_pred, y_probability, sample_ids=sample_ids,
        )
        rows.append({"seed": int(seed), "run_id": run_id, "validation": metrics})
        print(
            f"[validation] seed={seed} f1_macro={metrics['f1_macro']:.4f} "
            f"recall_scam={metrics['recall_scam']:.4f}",
            flush=True,
        )
        del model
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = RESULTS_ROOT / "transformer" / "finalists" / "bge" / "summary.json"
    if summary_path.exists():
        backup = summary_path.with_name("summary.pre_speaker_reconstruction.json")
        if not backup.exists():
            shutil.copy2(summary_path, backup)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        by_seed = {int(row["seed"]): row for row in rows}
        for seed_row in summary.get("seeds", []):
            seed = int(seed_row["seed"])
            if seed in by_seed:
                seed_row["validation"] = by_seed[seed]["validation"]
        values = [row["validation"] for row in rows]
        summary["val_f1_macro_mean"] = float(np.mean([x["f1_macro"] for x in values]))
        summary["val_f1_macro_std"] = float(np.std([x["f1_macro"] for x in values]))
        summary["val_pr_auc_mean"] = float(np.mean([x["pr_auc"] for x in values]))
        summary["val_recall_scam_mean"] = float(np.mean([x["recall_scam"] for x in values]))
        summary["validation_rebuilt_from_corrected_speakers"] = True
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(selected_device),
        "seeds": rows,
    }
    report_path = OUTPUT_PATH.parent / "bge_validation_metrics_rebuilt.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[metrics] {report_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--conversations-per-shard", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-reproduction-audit", action="store_true")
    parser.add_argument("--evaluate-finalists", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.encode_batch_size < 1 or args.conversations_per_shard < 1:
        parser.error("Tamanhos de lote e shard devem ser positivos")

    metadata = build_bge_message_parquet(
        raw_path=args.raw,
        output_path=args.output,
        device=args.device,
        encode_batch_size=args.encode_batch_size,
        conversations_per_shard=args.conversations_per_shard,
        force=args.force,
        reproduction_audit=not args.skip_reproduction_audit,
    )
    if args.evaluate_finalists:
        revalidate_bge_finalists(
            seeds=tuple(args.seed or (42, 52, 62)),
            device=args.device,
            checkpoint_root=args.checkpoint_root,
            batch_size=args.evaluation_batch_size,
        )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
