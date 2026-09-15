"""Read-only source ingestion and isolated ScriptMind inventory generation."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io_utils import atomic_write_json, file_sha256, shard_for, stable_hash
from .paths import ScriptMindPaths, assert_isolated_path


SPEAKER_RE = re.compile(r"(?i)(?<![A-Za-z])(Innocent|Suspect)\s*:\s*")
SPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{6,}\d)(?!\w)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


@dataclass(frozen=True)
class ParsedTurn:
    speaker: str
    text: str


def normalize_text(text: object) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def anonymize_text(text: str) -> str:
    value = EMAIL_RE.sub("[EMAIL]", text)
    value = SSN_RE.sub("[SSN]", value)
    value = PHONE_RE.sub("[PHONE]", value)
    value = LONG_NUMBER_RE.sub("[NUMBER]", value)
    value = URL_RE.sub("[URL]", value)
    return value


def parse_turns(text: object) -> list[ParsedTurn]:
    value = normalize_text(text)
    matches = list(SPEAKER_RE.finditer(value))
    turns: list[ParsedTurn] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        body = normalize_text(value[match.end():end]).strip(" .")
        if body:
            turns.append(ParsedTurn(match.group(1).title(), body))
    return turns


def _normalized_fingerprint(text: str) -> str:
    value = anonymize_text(normalize_text(text).lower())
    value = re.sub(r"\[[a-z_]+\]", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return normalize_text(value)


def _near_duplicate_groups(df: pd.DataFrame, threshold: float = 0.93) -> list[str]:
    """Link exact/near duplicate conversations without changing source rows."""
    fingerprints = df["text"].map(_normalized_fingerprint).tolist()
    exact = [stable_hash(value or f"empty:{idx}") for idx, value in enumerate(fingerprints)]
    if len(df) < 2:
        return exact
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 5), min_df=2, max_features=25_000,
            dtype=np.float32,
        ).fit_transform(fingerprints)
        if matrix.shape[1] == 0:
            return exact
        neighbors = NearestNeighbors(n_neighbors=min(3, len(df)), metric="cosine", n_jobs=-1).fit(matrix)
        distances, indices = neighbors.kneighbors(matrix)
        parent = list(range(len(df)))

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

        exact_first: dict[str, int] = {}
        for index, fingerprint in enumerate(exact):
            if fingerprint in exact_first:
                union(index, exact_first[fingerprint])
            else:
                exact_first[fingerprint] = index
        for row_index, (row_distances, row_neighbors) in enumerate(zip(distances, indices)):
            for distance, neighbor in zip(row_distances[1:], row_neighbors[1:]):
                if 1.0 - float(distance) >= threshold and int(df.iloc[row_index]["label"]) == int(df.iloc[neighbor]["label"]):
                    union(row_index, int(neighbor))
        return [stable_hash(f"near-duplicate:{find(index)}:{exact[find(index)]}") for index in range(len(df))]
    except (ImportError, ValueError):
        return exact


def _assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Approximately stratified, deterministic 70/10/20 split by duplicate group."""
    group_meta = df.groupby("duplicate_group", sort=True).agg(
        label=("label", "first"), dataset=("dataset", "first"), origin=("origin", "first")
    ).reset_index()
    mapping: dict[str, str] = {}
    for _, stratum in group_meta.groupby(["label", "dataset", "origin"], dropna=False, sort=True):
        groups = sorted(
            stratum["duplicate_group"].tolist(),
            key=lambda value: stable_hash(f"{seed}:{value}", length=64),
        )
        total = len(groups)
        train_end = int(round(total * 0.70))
        val_end = train_end + int(round(total * 0.10))
        for index, group in enumerate(groups):
            mapping[group] = "train" if index < train_end else "internal_validation" if index < val_end else "test"
    return df["duplicate_group"].map(mapping)


def _turn_rows(conversations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    quarantined: list[dict] = []
    for conversation in conversations.itertuples(index=False):
        turns = parse_turns(conversation.text)
        if not turns:
            quarantined.append({
                "conversation_id": conversation.conversation_id,
                "source": conversation.source,
                "reason": "no_speaker_markers",
                "text": conversation.text,
            })
            continue
        for position, turn in enumerate(turns):
            rows.append({
                "conversation_id": conversation.conversation_id,
                "turn_id": f"{conversation.conversation_id}:t{position:04d}",
                "turn_index": position,
                "speaker": turn.speaker,
                "text": turn.text,
                "text_anonymized": anonymize_text(turn.text),
                "label": int(conversation.label),
                "split": conversation.split,
                "source": conversation.source,
                "origin": conversation.origin,
                "dataset": conversation.dataset,
            })
    return pd.DataFrame(rows), pd.DataFrame(quarantined)


def _load_train(path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    required = {"origin", "dataset", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    df = df.copy()
    df["source_row_id"] = [f"train:{index}" for index in range(len(df))]
    df["source"] = "train"
    df["text"] = df["text"].map(normalize_text)
    df["conversation_id"] = [stable_hash(f"train:{index}:{text}") for index, text in enumerate(df["text"])]
    return df


def _load_external(path: Path, limit: int | None = None) -> pd.DataFrame:
    prefixes = pd.read_parquet(path)
    if limit is not None:
        # Limit by reconstructed conversation, not prefix rows.
        group_number = prefixes["n_turns"].eq(1).cumsum()
        prefixes = prefixes[group_number <= limit].copy()
    prefixes = prefixes.copy()
    prefixes["external_group"] = prefixes["n_turns"].eq(1).cumsum()
    rows: list[dict] = []
    for group_number, group in prefixes.groupby("external_group", sort=False):
        representative = group.sort_values("n_turns").iloc[-1]
        text = normalize_text(representative["text_last_n_turns"])
        rows.append({
            "origin": "external",
            "dataset": "dataset_validation",
            "personality": None,
            "motivation": None,
            "label": int(representative["label"]),
            "text": text,
            "source_row_id": f"external:{int(group_number) - 1}",
            "source": "external_validation",
            "conversation_id": stable_hash(f"external:{int(group_number)}:{text}"),
            "turn_count_reported": int(representative["turn_count"]),
            "prefixes_available": int(len(group)),
        })
    return pd.DataFrame(rows)


def audit_data(
    paths: ScriptMindPaths | None = None,
    *,
    limit: int | None = None,
    seed: int = 42,
    similarity_threshold: float = 0.93,
    force: bool = False,
) -> dict:
    paths = (paths or ScriptMindPaths.defaults()).ensure()
    conversations_path = paths.data_root / "inventory" / "conversations.parquet"
    turns_path = paths.data_root / "inventory" / "turns.parquet"
    report_path = paths.data_root / "inventory" / "report.json"
    for output in (conversations_path, turns_path, report_path):
        assert_isolated_path(output, paths)
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))

    train = _load_train(paths.source_train, limit=limit)
    external = _load_external(paths.source_external, limit=limit)
    train["duplicate_group"] = _near_duplicate_groups(train, threshold=similarity_threshold)
    train["split"] = _assign_splits(train, seed=seed)
    external["duplicate_group"] = external["conversation_id"]
    external["split"] = "external_validation"
    conversations = pd.concat([train, external], ignore_index=True, sort=False)
    turns, quarantine = _turn_rows(conversations)

    conversations.to_parquet(conversations_path, index=False)
    turns.to_parquet(turns_path, index=False)
    quarantine_path = paths.data_root / "inventory" / "quarantine.parquet"
    (quarantine if not quarantine.empty else pd.DataFrame(columns=["conversation_id", "source", "reason", "text"])).to_parquet(quarantine_path, index=False)

    split_counts = train.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
    pii_counts = Counter()
    for text in conversations["text"]:
        pii_counts["email"] += len(EMAIL_RE.findall(text))
        pii_counts["phone"] += len(PHONE_RE.findall(text))
        pii_counts["ssn"] += len(SSN_RE.findall(text))
        pii_counts["url"] += len(URL_RE.findall(text))
    report = {
        "protocol_version": 1,
        "seed": seed,
        "similarity_threshold": similarity_threshold,
        "limited_run": limit is not None,
        "source_hashes": {
            "train": file_sha256(paths.source_train),
            "external_validation": file_sha256(paths.source_external),
        },
        "conversation_counts": {
            "train_source": int(len(train)),
            "external_validation": int(len(external)),
            "total": int(len(conversations)),
        },
        "turns": int(len(turns)),
        "quarantined_conversations": int(len(quarantine)),
        "duplicate_groups": int(train["duplicate_group"].nunique()),
        "split_counts": {key: {str(label): int(count) for label, count in value.items()} for key, value in split_counts.items()},
        "labels": {str(key): int(value) for key, value in conversations["label"].value_counts().items()},
        "origins": {str(key): int(value) for key, value in conversations["origin"].value_counts(dropna=False).items()},
        "datasets": {str(key): int(value) for key, value in conversations["dataset"].value_counts(dropna=False).items()},
        "possible_pii_matches": dict(pii_counts),
        "outputs": {
            "conversations": str(conversations_path),
            "turns": str(turns_path),
            "quarantine": str(quarantine_path),
        },
    }
    atomic_write_json(report_path, report, paths)
    manifest = {
        "stage": "audit-data",
        "source_hashes": report["source_hashes"],
        "outputs": report["outputs"],
        "parameters": {"seed": seed, "similarity_threshold": similarity_threshold, "limit": limit},
    }
    atomic_write_json(paths.data_root / "manifests" / "audit-data.json", manifest, paths)
    return report


def build_annotation_items(
    paths: ScriptMindPaths | None = None,
    *,
    shard_index: int | None = None,
    num_shards: int = 32,
) -> pd.DataFrame:
    paths = paths or ScriptMindPaths.defaults()
    turns = pd.read_parquet(paths.data_root / "inventory" / "turns.parquet")
    items: list[dict] = []
    current_conversation: str | None = None
    context_lines: list[str] = []
    for row in turns.sort_values(["conversation_id", "turn_index"]).itertuples(index=False):
        if row.conversation_id != current_conversation:
            current_conversation = row.conversation_id
            context_lines = []
        if int(row.label) != 1:
            continue
        context_lines.append(f"[{row.turn_id}] {row.speaker}: {row.text_anonymized}")
        if row.speaker != "Suspect":
            continue
        if shard_index is not None and shard_for(row.turn_id, num_shards) != shard_index:
            continue
        value = row._asdict()
        items.append({**value, "context": "\n".join(context_lines), "annotation_id": row.turn_id})
    return pd.DataFrame(items)
