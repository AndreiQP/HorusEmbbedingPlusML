from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from machine_learning.studies.hierarchical_global import summarize_metric
import machine_learning.studies.hierarchical_global as global_study
from machine_learning.studies.hierarchical_perturbation import (
    RANKING_ABSOLUTE_INFLUENCE,
    RANKING_DECISION_SUPPORTING,
    HierarchicalPerturbationExplainer,
    class_support_delta,
    parse_conversation,
    select_message_evidence,
)


def _rows():
    return [
        {"model_index": 0, "original_index": 0, "delta_logit": 2.0, "delta_probability": .3, "abs_delta_logit": 2.0, "abs_delta_probability": .3},
        {"model_index": 1, "original_index": 1, "delta_logit": -3.0, "delta_probability": -.4, "abs_delta_logit": 3.0, "abs_delta_probability": .4},
        {"model_index": 2, "original_index": 2, "delta_logit": 5e-5, "delta_probability": 1e-5, "abs_delta_logit": 5e-5, "abs_delta_probability": .1},
    ]


def test_class_support_delta_tracks_the_predicted_class():
    assert class_support_delta(0.7, 1) == pytest.approx(0.7)
    assert class_support_delta(-0.7, 0) == pytest.approx(0.7)
    assert class_support_delta(-0.7, 1) == pytest.approx(-0.7)


def test_decision_supporting_never_returns_contrary_evidence():
    scam = select_message_evidence(
        _rows(), predicted_label=1, top_messages=6,
        ranking_policy=RANKING_DECISION_SUPPORTING,
    )
    ham = select_message_evidence(
        _rows(), predicted_label=0, top_messages=6,
        ranking_policy=RANKING_DECISION_SUPPORTING,
    )
    assert [row["model_index"] for row in scam] == [0]
    assert [row["model_index"] for row in ham] == [1]
    assert all(row["class_support_delta"] > 1e-4 for row in [*scam, *ham])


def test_absolute_influence_keeps_both_directions():
    selected = select_message_evidence(
        _rows(), predicted_label=1, top_messages=2,
        ranking_policy=RANKING_ABSOLUTE_INFLUENCE,
    )
    assert [row["model_index"] for row in selected] == [1, 0]


def test_global_summary_has_requested_dispersion_and_negative_rate():
    summary = summarize_metric([-1.0, 1.0, 2.0, 2.0])
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["negative_rate"] == pytest.approx(0.25)
    assert summary["variance"] == pytest.approx(2.0)
    assert summary["ci95_low"] <= summary["mean"] <= summary["ci95_high"]


def test_sufficiency_distance_definition_is_never_negative():
    original_class_probability = 0.91
    sufficient_class_probability = 0.62
    distance = abs(original_class_probability - sufficient_class_probability)
    assert distance == pytest.approx(0.29)
    assert distance >= 0.0


def test_union_candidates_cover_absolute_and_decision_supporting_messages():
    torch = pytest.importorskip("torch")

    class Encoder:
        def encode(self, texts, **kwargs):
            vectors = np.zeros((len(texts), 1024), dtype=np.float32)
            for index, text in enumerate(texts):
                vectors[index, 0] = text.count("scam")
                vectors[index, 1] = text.count("legit")
            return vectors

    class Model(torch.nn.Module):
        def forward(self, values, padding_mask=None):
            return (values[:, :, 0].sum(1) - values[:, :, 1].sum(1)).unsqueeze(1)

    conversation = (
        "Suspect: scam scam scam "
        "Innocent: legit legit "
        "Suspect: contexto "
        "Innocent: legit legit"
    )
    explainer = HierarchicalPerturbationExplainer(Model(), Encoder(), device="cpu", seed=42)
    prepared = explainer.prepare_perturbations(
        conversation,
        top_messages=1,
        max_ngram=1,
        ranking_policy=RANKING_DECISION_SUPPORTING,
        include_absolute_candidates=True,
    )
    # A classe prevista é Ham: "legit legit" a sustenta; "scam" é a maior
    # influência contrária e deve continuar disponível para o explorador científico.
    assert set(prepared.candidate_message_indices) == {0, 1}
    assert {row["model_index"] for row in prepared.source_top_messages} == {1}
    assert len(parse_conversation(conversation)) == 4


def test_global_shard_and_aggregation_write_complete_parquets(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    class Encoder:
        def encode(self, texts, **kwargs):
            vectors = np.zeros((len(texts), 1024), dtype=np.float32)
            for index, text in enumerate(texts):
                vectors[index, 0] = text.count("scam")
                vectors[index, 1] = text.count("legit")
            return vectors

    class Model(torch.nn.Module):
        def forward(self, values, padding_mask=None):
            return (values[:, :, 0].sum(1) - values[:, :, 1].sum(1)).unsqueeze(1)

    ids = np.asarray(["scam-case", "ham-case"])
    labels = np.asarray([1, 0])
    texts = {"scam-case": "Suspect: scam", "ham-case": "Innocent: legit"}
    encoder = Encoder()
    sequences = [
        encoder.encode([turn.text for turn in parse_conversation(texts[sample_id])])
        for sample_id in ids
    ]
    probabilities = np.asarray([1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    predictions = (probabilities >= .5).astype(int)

    monkeypatch.setattr(
        global_study, "_dataset",
        lambda embedding, split: (sequences, labels, ids, texts),
    )
    monkeypatch.setattr(global_study, "load_bge_encoder", lambda device=None: encoder)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"dummy-checkpoint")
    monkeypatch.setattr(
        global_study, "resolve_bge_finalist_checkpoint",
        lambda seed=42, checkpoint_root=None: checkpoint,
    )
    monkeypatch.setattr(
        global_study, "load_bge_finalist_model",
        lambda seed=42, device=None, checkpoint_root=None: (
            Model(), {"run_id": f"run-{seed}", "seed": seed},
        ),
    )
    monkeypatch.setattr(
        global_study.ModelCache, "load_prediction_bundle",
        staticmethod(lambda strategy, run_id, split: {
            "sample_ids": ids, "y_true": labels, "y_pred": predictions,
            "y_prob": probabilities,
        }),
    )
    monkeypatch.setattr(
        global_study, "encoder_provenance",
        lambda encoder: {"model_name": "dummy", "revision": "test"},
    )

    shard = global_study.run_global_shard(
        split="test_internal", shard_index=0, shard_count=1,
        output_root=tmp_path, device="cpu", top_messages=1,
        max_ngram=1, top_spans_per_message=1, controls_per_span=1,
    )
    assert (shard / "complete.json").is_file()
    assert global_study._completed_shard_is_valid(
        shard,
        __import__("json").loads((shard / "manifest.json").read_text())["run_hash"],
    )

    destination = global_study.aggregate_global_results(
        output_root=tmp_path, expected_shards=1, splits=("test_internal",),
    )
    coverage = __import__("pandas").read_parquet(destination / "aggregates" / "coverage.parquet")
    assert coverage.loc[coverage.top_k.eq(1), "N_k"].item() == 2
    assert (destination / "test_internal" / "message_effects.parquet").is_file()
