from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from machine_learning.studies.context_block_explainability import (
    build_context_blocks,
    _source_fingerprint,
    explain_context_sequence,
    generate_block_word_perturbations,
)


def _conversation(turns: int = 10) -> str:
    return " ".join(
        f"{'Suspect' if index % 2 == 0 else 'Innocent'}: mensagem {index}"
        for index in range(turns)
    )


def test_canonical_representation_has_one_block_per_message():
    blocks = build_context_blocks(_conversation(10), "canonical_message")
    assert len(blocks) == 10
    assert [block.message_model_indices for block in blocks] == [(index,) for index in range(10)]
    assert not any(block.is_partial for block in blocks)


def test_speaker_blocks_are_fixed_keep_remainders_and_return_to_timeline():
    blocks = build_context_blocks(_conversation(10), "speaker_block_4")
    assert [block.message_model_indices for block in blocks] == [
        (0, 2, 4, 6), (1, 3, 5, 7), (8,), (9,),
    ]
    assert [block.is_partial for block in blocks] == [False, False, True, True]
    assert blocks[0].text.splitlines() == [
        "Suspect: mensagem 0", "Suspect: mensagem 2",
        "Suspect: mensagem 4", "Suspect: mensagem 6",
    ]


@pytest.mark.parametrize("size", [4, 5, 6, 7, 8])
def test_chronological_windows_have_stride_one(size):
    blocks = build_context_blocks(_conversation(10), f"chronological_window_{size}")
    assert len(blocks) == 10 - size + 1
    assert blocks[0].message_model_indices == tuple(range(size))
    assert blocks[1].message_model_indices == tuple(range(1, size + 1))
    assert not any(block.is_partial for block in blocks)


def test_short_chronological_conversation_is_one_partial_block():
    blocks = build_context_blocks(_conversation(3), "chronological_window_8")
    assert len(blocks) == 1
    assert blocks[0].message_model_indices == (0, 1, 2)
    assert blocks[0].is_partial


def test_context_representations_apply_last_100_before_grouping():
    blocks = build_context_blocks(_conversation(105), "chronological_window_4")
    assert len(blocks) == 97
    assert blocks[0].message_original_indices == (5, 6, 7, 8)
    assert blocks[-1].message_original_indices == (101, 102, 103, 104)


def test_source_fingerprint_rejects_changed_ids_text_or_labels():
    source = pd.DataFrame({
        "sample_id": ["a", "b"], "label": [0, 1], "text": ["Innocent: oi", "Suspect: pix"],
    })
    baseline = _source_fingerprint(source)
    for column, value in (("sample_id", "novo"), ("label", 0), ("text", "Suspect: outro")):
        changed = source.copy()
        changed.loc[1, column] = value
        assert _source_fingerprint(changed) != baseline


def test_word_ngrams_never_cross_message_boundaries():
    block = build_context_blocks(_conversation(4), "chronological_window_4")[0]
    rows = generate_block_word_perturbations(block, max_ngram=5)
    assert rows
    assert all("\n" not in row["span_text"] for row in rows)
    assert all(row["message_model_index"] in (0, 1, 2, 3) for row in rows)
    assert all(row["perturbed_text"].count("\n") == 3 for row in rows)


def test_context_fidelity_tracks_unique_coverage_and_redundancy():
    torch = pytest.importorskip("torch")

    class Encoder:
        def encode(self, texts, **kwargs):
            vectors = np.zeros((len(texts), 1024), dtype=np.float32)
            for index, text in enumerate(texts):
                vectors[index, 0] = text.count("mensagem")
            return vectors

    class Model(torch.nn.Module):
        def forward(self, values, padding_mask=None):
            return values[:, :, 0].sum(1, keepdim=True)

    blocks = build_context_blocks(_conversation(6), "chronological_window_4")
    encoder = Encoder()
    original = encoder.encode([block.text for block in blocks])
    neutral = np.zeros_like(original)
    frames = explain_context_sequence(
        model=Model(), encoder=encoder, device=torch.device("cpu"),
        representation="chronological_window_4", model_mode="frozen_probe",
        seed=42, split="validation", sample_id="case", label=1,
        blocks=blocks, original=original, neutral=neutral,
        top_units=2, max_ngram=1, include_micro=False,
    )
    fidelity = frames["fidelity"]
    supporting = fidelity[fidelity.ranking_policy.eq("decision_supporting")].sort_values("top_k")
    assert supporting.iloc[0].unique_messages_covered == 4
    assert supporting.iloc[1].unique_messages_covered == 5
    assert supporting.iloc[1].message_occurrences_covered == 8
    assert supporting.iloc[1].redundancy_ratio == pytest.approx(3 / 5)
