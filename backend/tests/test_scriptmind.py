from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from scriptmind.csid import build_csid
from scriptmind.annotation import Annotator, annotate_dialogues
from scriptmind.data import anonymize_text, parse_turns
from scriptmind.io_utils import atomic_write_jsonl, shard_for, stable_hash
from scriptmind.paths import ScriptMindPaths, assert_isolated_path
from scriptmind.schemas import ScriptMindPrediction, TurnAnnotation
from scriptmind.statistics import holm_adjust, mcnemar_exact


def test_parser_and_anonymizer() -> None:
    turns = parse_turns("Innocent: Hello. Suspect: Call 1-800-555-1234 or visit https://bad.test now")
    assert [(turn.speaker, turn.text[:5]) for turn in turns] == [("Innocent", "Hello"), ("Suspect", "Call ")]
    safe = anonymize_text(turns[1].text)
    assert "1-800-555-1234" not in safe
    assert "https://bad.test" not in safe
    assert "[PHONE]" in safe and "[URL]" in safe


def test_prediction_schema_enforces_non_scam_nulls() -> None:
    assert ScriptMindPrediction(label="non_scam").to_dict()["next_intent"] is None
    with pytest.raises(ValueError):
        ScriptMindPrediction(label="non_scam", rationale="not allowed")
    annotation = TurnAnnotation(
        primary_intent="request_payment", stage="victim_action", rationale="Requests a transfer."
    )
    assert annotation.primary_intent == "request_payment"
    with pytest.raises(ValueError):
        TurnAnnotation(primary_intent="request_payment", stage="pressure", rationale="Wrong stage")


def test_shards_are_stable() -> None:
    identifier = stable_hash("conversation:turn")
    assert shard_for(identifier, 32) == shard_for(identifier, 32)
    assert 0 <= shard_for(identifier, 32) < 32


def test_isolation_guard(tmp_path: Path) -> None:
    paths = ScriptMindPaths(tmp_path / "data", tmp_path / "results").ensure()
    assert_isolated_path(paths.data_root / "inventory" / "ok.json", paths)
    with pytest.raises(ValueError):
        assert_isolated_path(tmp_path / "source" / "must-not-write.json", paths)


def _minimal_inventory(paths: ScriptMindPaths) -> None:
    rows = [
        {
            "conversation_id": "scam-1", "turn_id": "scam-1:t0000", "turn_index": 0,
            "speaker": "Innocent", "text": "Hello", "text_anonymized": "Hello", "label": 1,
            "split": "train", "source": "train", "origin": "external", "dataset": "fixture",
        },
        {
            "conversation_id": "scam-1", "turn_id": "scam-1:t0001", "turn_index": 1,
            "speaker": "Suspect", "text": "I am calling from your bank", "text_anonymized": "I am calling from your bank",
            "label": 1, "split": "train", "source": "train", "origin": "external", "dataset": "fixture",
        },
        {
            "conversation_id": "scam-1", "turn_id": "scam-1:t0002", "turn_index": 2,
            "speaker": "Innocent", "text": "Why?", "text_anonymized": "Why?", "label": 1,
            "split": "train", "source": "train", "origin": "external", "dataset": "fixture",
        },
        {
            "conversation_id": "scam-1", "turn_id": "scam-1:t0003", "turn_index": 3,
            "speaker": "Suspect", "text": "Send money immediately", "text_anonymized": "Send money immediately",
            "label": 1, "split": "train", "source": "train", "origin": "external", "dataset": "fixture",
        },
        {
            "conversation_id": "ham-1", "turn_id": "ham-1:t0000", "turn_index": 0,
            "speaker": "Innocent", "text": "Hello", "text_anonymized": "Hello", "label": 0,
            "split": "train", "source": "train", "origin": "external", "dataset": "fixture",
        },
        {
            "conversation_id": "ham-1", "turn_id": "ham-1:t0001", "turn_index": 1,
            "speaker": "Suspect", "text": "Your appointment is confirmed", "text_anonymized": "Your appointment is confirmed",
            "label": 0, "split": "train", "source": "train", "origin": "external", "dataset": "fixture",
        },
    ]
    pd.DataFrame(rows).to_parquet(paths.data_root / "inventory" / "turns.parquet", index=False)
    annotations = [
        {
            "annotation_id": "scam-1:t0001", "turn_id": "scam-1:t0001", "conversation_id": "scam-1",
            "agreement": True,
            "annotation": {
                "primary_intent": "impersonation", "stage": "credibility",
                "rationale": "Claims to represent a bank.", "evidence_turn_ids": ["scam-1:t0001"],
            },
        },
        {
            "annotation_id": "scam-1:t0003", "turn_id": "scam-1:t0003", "conversation_id": "scam-1",
            "agreement": True,
            "annotation": {
                "primary_intent": "request_payment", "stage": "victim_action",
                "rationale": "Requests money.", "evidence_turn_ids": ["scam-1:t0003"],
            },
        },
    ]
    atomic_write_jsonl(paths.data_root / "adjudicated" / "shard-00-of-01.jsonl", annotations, paths)


def test_csid_has_no_future_target_in_prefix_and_balances_train(tmp_path: Path) -> None:
    paths = ScriptMindPaths(tmp_path / "data", tmp_path / "results").ensure()
    _minimal_inventory(paths)
    summary = build_csid(paths, force=True, allow_partial=True, minimum_annotation_coverage=1.0)
    frame = pd.read_parquet(paths.data_root / "csid" / "csid.parquet")
    assert summary["annotation_coverage"] == 1.0
    scam = frame[frame["label"] == 1]
    assert not scam.empty
    assert all(row.target_turn_id not in row.prefix_turn_ids for row in scam.itertuples())
    train_counts = frame[frame["split"] == "train"]["label"].value_counts().to_dict()
    assert train_counts[0] == train_counts[1]
    target = scam.iloc[-1]["target"]
    assert target["next_intent"] == "request_payment"


class _FakeAnnotator(Annotator):
    provider = "local"
    model_id = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0

    def annotate(self, prompt: str):
        self.calls += 1
        return TurnAnnotation(
            primary_intent="other_or_insufficient", stage="benign_or_other",
            rationale="Fixture annotation.", evidence_turn_ids=[],
        ), {"total_tokens": 1}


def test_annotation_resume_uses_completed_shard(tmp_path: Path) -> None:
    paths = ScriptMindPaths(tmp_path / "data", tmp_path / "results").ensure()
    _minimal_inventory(paths)
    worker = _FakeAnnotator()
    output = annotate_dialogues("local", paths=paths, num_shards=1, annotator=worker)
    assert worker.calls == 2
    second_worker = _FakeAnnotator()
    assert annotate_dialogues("local", paths=paths, num_shards=1, annotator=second_worker) == output
    assert second_worker.calls == 0


def test_notebook_is_valid_json() -> None:
    notebook = BACKEND / "notebooks" / "ScriptMind_metrics.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert any("métricas" in "".join(cell.get("source", [])).lower() for cell in payload["cells"])


def test_scriptmind_slurm_root_is_spool_safe() -> None:
    slurm_dir = BACKEND / "slurm" / "scriptmind"
    submit = (slurm_dir / "submit_scriptmind_pipeline.sh").read_text(encoding="utf-8")
    assert "HORUS_PROJECT_ROOT=$PROJECT_ROOT" in submit
    for name in (
        "run_scriptmind_cpu.sbatch",
        "run_scriptmind_annotation.sbatch",
        "run_scriptmind_train.sbatch",
        "run_scriptmind_evaluate.sbatch",
    ):
        script = (slurm_dir / name).read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="${HORUS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in script


def test_predeclared_paired_statistics() -> None:
    result = mcnemar_exact([0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1])
    assert result["discordant"] == 3
    adjusted = holm_adjust([0.01, 0.04, 0.20])
    assert adjusted == pytest.approx([0.03, 0.08, 0.20])
