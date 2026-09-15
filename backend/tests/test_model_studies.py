from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from machine_learning.studies.common import align_prediction_bundles, nested_subset_indices
from machine_learning.studies.explainability import inspect_forward
from machine_learning.studies.transformer_analysis import min_fn_threshold


def test_nested_subsets_are_deterministic_stratified_and_nested():
    ids = np.asarray([f"sample-{index}" for index in range(100)])
    labels = np.asarray([0] * 60 + [1] * 40)
    small = nested_subset_indices(ids, labels, 0.10, 42)
    medium = nested_subset_indices(ids, labels, 0.25, 42)
    repeated = nested_subset_indices(ids, labels, 0.10, 42)
    assert np.array_equal(small, repeated)
    assert set(small).issubset(set(medium))
    assert set(labels[small]) == {0, 1}


def test_ham_only_subset_never_contains_scam():
    ids = np.asarray([f"sample-{index}" for index in range(20)])
    labels = np.asarray([0] * 10 + [1] * 10)
    selected = nested_subset_indices(ids, labels, 0.5, 52, ham_only=True)
    assert len(selected) == 5
    assert np.all(labels[selected] == 0)


def test_prediction_alignment_rejects_different_ids():
    base = {"sample_ids": np.asarray(["a", "b"]), "y_true": np.asarray([0, 1])}
    align_prediction_bundles({"a": base, "b": dict(base)})
    mismatch = {"sample_ids": np.asarray(["b", "a"]), "y_true": np.asarray([1, 0])}
    with pytest.raises(ValueError, match="desalinhados"):
        align_prediction_bundles({"a": base, "b": mismatch})


def test_min_fn_threshold_is_largest_positive_boundary():
    labels = np.asarray([0, 0, 1, 1, 1])
    probability = np.asarray([0.1, 0.8, 0.2, 0.7, 0.5])
    threshold = min_fn_threshold(labels, probability)
    prediction = (probability >= threshold).astype(int)
    assert threshold == pytest.approx(0.2)
    assert int(((labels == 1) & (prediction == 0)).sum()) == 0


def test_attention_inspector_preserves_logits_and_masks_padding():
    torch = pytest.importorskip("torch")
    from machine_learning.transformer.runner import TransformerScamClassifier

    torch.manual_seed(7)
    model = TransformerScamClassifier(
        embedding_dim=8, num_heads=2, num_layers=2, hidden_dim=16, dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 5, 8)
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])
    with torch.no_grad():
        expected = model(inputs, padding_mask=mask)
        actual, attention = inspect_forward(model, inputs, mask)
        ablated, _ = inspect_forward(model, inputs, mask, ablate=(0, 0))
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
    assert len(attention) == 2
    assert tuple(attention[0].shape) == (2, 2, 5, 5)
    assert torch.all(attention[0][1, :, :, 3:] == 0)
    assert not torch.allclose(actual, ablated)


def test_notebook_study_cells_are_cache_only():
    for name in ("judge_decision.ipynb", "unsupervised.ipynb"):
        notebook = json.loads((BACKEND / "notebooks" / name).read_text(encoding="utf-8"))
        study_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "Estudo" in "".join(cell.get("source", [])) or "STUDY_ROOT" in "".join(cell.get("source", []))
        )
        assert "submit_cached_model_studies.sh" in study_sources


def test_standalone_submit_does_not_call_main_pipeline():
    script = (BACKEND / "slurm" / "studies" / "submit_cached_model_studies.sh").read_text(encoding="utf-8")
    assert "submit_full_pipeline.sh" not in script
    assert "array_grid.sh" not in script
    assert script.index('if [[ "$DRY_RUN" == "1" ]]') < script.index("machine_learning.studies prepare")


def test_unsupervised_selection_uses_only_internal_validation(tmp_path, monkeypatch):
    from machine_learning.studies import manifest as manifest_module

    results = tmp_path / "experiment_results"
    protocol = results / "protocol"
    protocol.mkdir(parents=True)
    (protocol / "unsupervised_split_manifest.json").write_text('{"fingerprint":"fixed"}', encoding="utf-8")
    scores = {"ocsvm": 0.91, "iforest": 0.89, "lof": 0.87, "lunar": 0.4, "svdd": 0.3}
    for model, score in scores.items():
        directory = results / "unsupervised" / "pipeline" / model
        directory.mkdir(parents=True)
        payload = {
            "winner": {"embedding": "external-winner", "val_f1_macro_mean": 1.0},
            "finalists": [
                {
                    "embedding": "internal-winner", "pca_dim": 25, "params": {},
                    "internal_val_f1_macro": score,
                    "val_f1_macro_mean": 0.0,
                    "grid": [{"internal_val_pr_auc": score, "internal_val_recall_scam": score, "_complexity": 1}],
                    "seeds": [],
                },
                {
                    "embedding": "external-winner", "pca_dim": 25, "params": {},
                    "internal_val_f1_macro": 0.1,
                    "val_f1_macro_mean": 1.0,
                    "grid": [{"internal_val_pr_auc": 0.1, "internal_val_recall_scam": 0.1, "_complexity": 1}],
                    "seeds": [],
                },
            ],
        }
        (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(manifest_module, "RESULTS_ROOT", results)
    monkeypatch.setattr(manifest_module.ModelCache, "git_commit", staticmethod(lambda: "abc"))
    first = manifest_module.select_study_finalists(
        "unsupervised", strict=False, output_path=tmp_path / "first.json"
    )
    second = manifest_module.select_study_finalists(
        "unsupervised", strict=False, output_path=tmp_path / "second.json"
    )
    assert [row["model_type"] for row in first.unsupervised] == ["ocsvm", "iforest", "lof"]
    assert all(row["embedding"] == "internal-winner" for row in first.unsupervised)
    assert first.manifest_hash == second.manifest_hash
