from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from machine_learning.studies.hierarchical_perturbation import (
    HierarchicalPerturbationExplainer,
    ReproductionError,
    encoder_provenance,
    generate_ngram_perturbations,
    parse_conversation,
    resolve_bge_finalist_checkpoint,
    save_hierarchical_explanations,
)
import machine_learning.studies.hierarchical_perturbation as hierarchical_module


class DummyEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            tokens = text.lower().split()
            vectors[index, 0] = sum(token.strip(".,!?") == "fraude" for token in tokens)
            vectors[index, 1] = sum(token.strip(".,!?") == "verdadeiro" for token in tokens)
        return vectors


def _dummy_model(capture: list[np.ndarray] | None = None):
    torch = pytest.importorskip("torch")

    class DummyModel(torch.nn.Module):
        def forward(self, values, padding_mask=None):
            if capture is not None:
                capture.append(values.detach().cpu().numpy().copy())
            return (values[:, :, 0].sum(dim=1) - values[:, :, 1].sum(dim=1)).unsqueeze(1)

    return DummyModel()


def _conversation() -> str:
    return " ".join([
        "Innocent: conteúdo verdadeiro confirmado",
        "Suspect: existe fraude no pagamento",
        "Innocent: mensagem comum um",
        "Suspect: mensagem comum dois",
        "Innocent: mensagem comum três",
        "Suspect: mensagem comum quatro",
        "Innocent: mensagem comum cinco",
        "Suspect: mensagem comum seis",
    ])


def test_parser_preserves_offsets_and_last_100_turns():
    text = " ".join(
        f"{'Innocent' if index % 2 == 0 else 'Suspect'}: mensagem {index}"
        for index in range(105)
    )
    turns = parse_conversation(text)
    assert len(turns) == 100
    assert turns[0].original_index == 5
    assert turns[0].model_index == 0
    assert text[turns[0].conversation_start:turns[0].conversation_end] == turns[0].text
    assert turns[-1].original_index == 104


def test_ngram_perturbations_never_remove_speaker_prefix():
    turn = parse_conversation("Suspect: clique no link de pagamento")[0]
    rows = generate_ngram_perturbations(turn, max_ngram=3)
    assert rows
    assert all(row["perturbed_text"].startswith("Suspect:") for row in rows)
    target = next(row for row in rows if row["span_text"] == "link de pagamento")
    assert turn.text[target["start_char"]:target["end_char"]] == "link de pagamento"
    assert target["perturbed_text"] == "Suspect: clique no"


def test_macro_reuses_neutral_embeddings_and_finds_both_directions():
    encoder = DummyEncoder()
    explainer = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=42, model_metadata={"run_id": "dummy"},
    )
    result = explainer.explain(
        _conversation(), top_messages=6, max_ngram=3, top_spans_per_message=5,
    )

    assert len(result.top_messages) == 6
    assert {row["original_index"] for row in result.top_messages[:2]} == {0, 1}
    assert result.strongest_scam_span["span_text"] == "fraude"
    assert result.strongest_ham_span["span_text"] == "verdadeiro"
    assert result.strongest_scam_span["delta_logit"] > 0
    assert result.strongest_ham_span["delta_logit"] < 0
    assert len([row for row in result.fidelity if row["level"] == "cumulative_messages"]) == 6

    # original, dois baselines em uma chamada, todas as perturbações e sufficiency.
    assert len(encoder.calls) == 4
    assert set(encoder.calls[1]) == {"Innocent:", "Suspect:"}


def test_replacement_changes_only_the_target_vector():
    capture: list[np.ndarray] = []
    explainer = HierarchicalPerturbationExplainer(
        _dummy_model(capture), DummyEncoder(), device="cpu", inference_batch_size=8,
    )
    original = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024)
    replacements = np.stack([
        np.full(1024, -1.0, dtype=np.float32),
        np.full(1024, -2.0, dtype=np.float32),
    ])
    explainer._infer_replacements(original, [1, 2], replacements)
    batch = capture[-1]
    assert np.array_equal(batch[0, 0], original[0])
    assert np.array_equal(batch[0, 2], original[2])
    assert np.array_equal(batch[0, 1], replacements[0])
    assert np.array_equal(batch[1, 0], original[0])
    assert np.array_equal(batch[1, 1], original[1])
    assert np.array_equal(batch[1, 2], replacements[1])


def test_cached_original_embeddings_are_not_reencoded_and_align_with_text():
    conversation = _conversation()
    reference_encoder = DummyEncoder()
    original = reference_encoder.encode([
        turn.text for turn in parse_conversation(conversation)
    ])
    encoder = DummyEncoder()
    explainer = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=42,
    )
    neutral = {
        "Innocent": np.zeros(1024, dtype=np.float32),
        "Suspect": np.zeros(1024, dtype=np.float32),
    }

    prepared = explainer.prepare_perturbations(
        conversation,
        top_messages=6,
        max_ngram=2,
        original_embeddings=original,
        neutral_by_speaker=neutral,
    )

    assert prepared.embedding_source == "cached_dataset_embeddings"
    assert len(encoder.calls) == 1
    assert encoder.calls[0] == [row["perturbed_text"] for row in prepared.candidates]
    assert not any(call == [turn.text for turn in parse_conversation(conversation)] for call in encoder.calls)

    with pytest.raises(ReproductionError, match="Alinhamento texto/embedding inválido"):
        explainer.prepare_perturbations(
            conversation,
            original_embeddings=original[:-1],
            neutral_by_speaker=neutral,
        )


def test_prepared_perturbations_are_reused_across_seeds_without_bge_calls():
    conversation = _conversation()
    reference_encoder = DummyEncoder()
    original = reference_encoder.encode([
        turn.text for turn in parse_conversation(conversation)
    ])
    encoder = DummyEncoder()
    primary = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=42,
    )
    secondary = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=52,
    )
    neutral_vectors = reference_encoder.encode(["Innocent:", "Suspect:"])
    prepared = primary.prepare_perturbations(
        conversation,
        top_messages=6,
        max_ngram=2,
        original_embeddings=original,
        neutral_by_speaker=dict(zip(("Innocent", "Suspect"), neutral_vectors)),
    )
    calls_after_prepare = len(encoder.calls)

    explanation_42 = primary.explain_prepared(
        prepared, top_messages=6, defer_span_fidelity=True,
    )
    explanation_52 = secondary.explain_prepared(
        prepared, top_messages=6, analysis_scope="stability",
        defer_span_fidelity=True,
    )

    assert len(encoder.calls) == calls_after_prepare
    assert explanation_42.metadata["candidate_source_seed"] == 42
    assert explanation_52.metadata["candidate_source_seed"] == 42
    assert [row["perturbed_text"] for row in explanation_42._all_span_effects] == [
        row["perturbed_text"] for row in explanation_52._all_span_effects
    ]


def test_bge_audit_encodes_the_reference_conversation_once():
    conversation = _conversation()
    reference = DummyEncoder().encode([
        turn.text for turn in parse_conversation(conversation)
    ])
    encoder = DummyEncoder()
    explainer = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=42,
    )

    audit = explainer.audit_cached_embeddings(conversation, reference)

    assert audit["status"] == "passed"
    assert audit["policy"] == "once_per_study_job"
    assert len(encoder.calls) == 1


def test_reproduction_gate_accepts_equal_vectors_and_rejects_mismatch():
    conversation = "Innocent: verdadeiro Suspect: fraude"
    encoder = DummyEncoder()
    expected = encoder.encode([turn.text for turn in parse_conversation(conversation)])
    expected_logit = 0.0
    explainer = HierarchicalPerturbationExplainer(_dummy_model(), encoder, device="cpu")
    result = explainer.explain(
        conversation,
        top_messages=2,
        max_ngram=1,
        expected_embeddings=expected,
        expected_logit=expected_logit,
    )
    assert result.metadata["reproduction"]["status"] == "passed"

    different = expected.copy()
    different[0, 10] = 1.0
    with pytest.raises(ReproductionError, match="não reproduzem"):
        explainer.explain(
            conversation,
            top_messages=2,
            max_ngram=1,
            expected_embeddings=different,
            expected_logit=expected_logit,
        )


def test_artifacts_include_full_span_table_and_stability(tmp_path):
    encoder = DummyEncoder()
    primary = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=42,
        model_metadata={"run_id": "dummy-42"},
    )
    secondary = HierarchicalPerturbationExplainer(
        _dummy_model(), encoder, device="cpu", seed=52,
        model_metadata={"run_id": "dummy-52"},
    )
    prepared = primary.prepare_perturbations(
        _conversation(), top_messages=6, max_ngram=2,
    )
    explanations = [
        primary.explain_prepared(
            prepared, top_messages=6, top_spans_per_message=3,
            sample_id="sample-1", label=1, split="test_internal",
            defer_span_fidelity=True,
        ),
        secondary.explain_prepared(
            prepared, top_messages=6, top_spans_per_message=3,
            sample_id="sample-1", label=1, split="test_internal",
            analysis_scope="stability", defer_span_fidelity=True,
        ),
    ]
    destination = save_hierarchical_explanations(explanations, output_dir=tmp_path)
    for name in (
        "manifest.json", "message_occlusion.csv", "span_occlusion.csv",
        "fidelity.csv", "stability.csv",
    ):
        assert (destination / name).is_file()
    assert (destination / "explanations" / "sample-1" / "seed_42.json").is_file()
    assert (destination / "explanations" / "sample-1" / "seed_52.json").is_file()
    stability = __import__("pandas").read_csv(destination / "stability.csv")
    assert "summary" in set(stability.item_type)
    assert set(stability.candidate_source_seed.dropna().astype(int)) == {42}


def test_primary_only_artifacts_keep_readable_empty_stability_csv(tmp_path):
    explanation = HierarchicalPerturbationExplainer(
        _dummy_model(), DummyEncoder(), device="cpu", seed=42,
    ).explain(
        _conversation(), top_messages=2, max_ngram=1,
        sample_id="primary-only", label=1, split="test_internal",
    )

    destination = save_hierarchical_explanations([explanation], output_dir=tmp_path)
    stability = __import__("pandas").read_csv(destination / "stability.csv")

    assert stability.empty
    assert {"item_type", "sample_id", "candidate_source_seed"}.issubset(stability.columns)


def test_encoder_provenance_records_checkpoint_pooling_and_normalization():
    Pooling = type("Pooling", (), {
        "pooling_mode": "cls",
        "get_config_dict": lambda self: {"pooling_mode": "cls", "embedding_dimension": 1024},
    })
    Normalize = type("Normalize", (), {})
    encoder = type("Encoder", (), {"_modules": {"1": Pooling(), "2": Normalize()}})()
    provenance = encoder_provenance(encoder)
    assert provenance["pooling"]["pooling_mode"] == "cls"
    assert provenance["pooling"]["embedding_dimension"] == 1024
    assert provenance["normalize_module"] is True


def test_plan_is_saved_with_top_six_decision():
    plan = (BACKEND / "machine_learning" / "studies" / "HIERARCHICAL_EXPLAINABILITY_PLAN.md")
    content = plan.read_text(encoding="utf-8")
    assert "seis mensagens" in content
    assert "abs(delta_logit)" in content


def test_checkpoint_resolver_selects_only_registered_finalist_run(tmp_path):
    expected_run = "batch_size=16__dropout=0.1__embedding=bge__gradient_clip=1.0__hidden_d_c89c623e"
    expected = tmp_path / "mounted" / "transformer" / expected_run / "model.pth"
    expected.parent.mkdir(parents=True)
    expected.touch()
    distractor = tmp_path / "mounted" / "transformer" / "some_other_bge_run" / "model.pth"
    distractor.parent.mkdir(parents=True)
    distractor.touch()

    resolved = resolve_bge_finalist_checkpoint(seed=42, checkpoint_root=tmp_path)

    assert resolved == expected.resolve()


def test_checkpoint_resolver_reports_missing_registered_run(tmp_path):
    distractor = tmp_path / "some_other_bge_run" / "model.pth"
    distractor.parent.mkdir(parents=True)
    distractor.touch()

    with pytest.raises(FileNotFoundError, match="Run esperado.*c89c623e"):
        resolve_bge_finalist_checkpoint(seed=42, checkpoint_root=tmp_path)
