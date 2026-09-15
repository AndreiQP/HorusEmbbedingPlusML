from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cache import ModelCache
from .common import (
    DEFAULT_FRACTIONS, DEFAULT_SEEDS, RESULTS_ROOT, TRANSFORMER_STUDIES,
    atomic_json, sha256_file, stable_hash,
)


@dataclass
class StudyManifest:
    protocol_version: int = 1
    git_commit: str = "unknown"
    created_at: str = ""
    fractions: list[float] = field(default_factory=lambda: list(DEFAULT_FRACTIONS))
    seeds: list[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    transformer: list[dict[str, Any]] = field(default_factory=list)
    unsupervised: list[dict[str, Any]] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    split_hashes: dict[str, str] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artefato obrigatório ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _transformer_candidates(source_hashes: dict[str, str]) -> list[dict[str, Any]]:
    candidates = []
    for embedding in ("voyage", "bge", "openai"):
        path = RESULTS_ROOT / "transformer" / "finalists" / embedding / "summary.json"
        summary = _read_json(path)
        source_hashes[str(path)] = sha256_file(path)
        candidates.append({
            "candidate": f"transformer_{embedding}",
            "embedding": embedding,
            "configuration": summary["configuration"],
            "seeds": summary["seeds"],
            "source_summary": str(path),
        })
    return candidates


def _unsupervised_candidates(source_hashes: dict[str, str], top_k: int) -> list[dict[str, Any]]:
    algorithm_winners = []
    for model_type in ("ocsvm", "iforest", "lof", "lunar", "svdd"):
        path = RESULTS_ROOT / "unsupervised" / "pipeline" / model_type / "summary.json"
        summary = _read_json(path)
        source_hashes[str(path)] = sha256_file(path)
        finalists = summary.get("finalists", [])
        if not finalists:
            raise ValueError(f"Resumo sem finalistas: {path}")
        # External metrics are deliberately absent from this key.
        best = sorted(
            finalists,
            key=lambda row: (
                -float(row["internal_val_f1_macro"]),
                -float((row.get("grid") or [{}])[0].get("internal_val_pr_auc") or -1),
                -float((row.get("grid") or [{}])[0].get("internal_val_recall_scam") or -1),
                int(row["pca_dim"]),
                float((row.get("grid") or [{}])[0].get("_complexity") or 0),
                row["embedding"],
            ),
        )[0]
        best_grid = (best.get("grid") or [{}])[0]
        algorithm_winners.append({
            "candidate": f"{model_type}_{best['embedding']}",
            "model_type": model_type,
            "embedding": best["embedding"],
            "pca_dim": int(best["pca_dim"]),
            "reducer": "pca",
            "params": best.get("params", {}),
            "internal_val_f1_macro": float(best["internal_val_f1_macro"]),
            "internal_val_pr_auc": best_grid.get("internal_val_pr_auc"),
            "internal_val_recall_scam": best_grid.get("internal_val_recall_scam"),
            "complexity": best_grid.get("_complexity"),
            "seeds": best.get("seeds", []),
            "source_summary": str(path),
        })
    algorithm_winners.sort(
        key=lambda row: (
            -row["internal_val_f1_macro"],
            -float(row["internal_val_pr_auc"] or -1),
            -float(row["internal_val_recall_scam"] or -1),
            row["pca_dim"], float(row["complexity"] or 0), row["model_type"],
        )
    )
    return algorithm_winners[:top_k]


def _split_hashes(families: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for family in families:
        path = RESULTS_ROOT / "protocol" / f"{family}_split_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Manifesto de split ausente: {path}")
        result[str(path)] = sha256_file(path)
    return result


def _validate_cache(candidates: list[dict[str, Any]], strategy: str) -> None:
    missing: list[Path] = []
    splits = ("internal_validation", "test_internal", "validation") if strategy == "transformer" else ("test_internal", "validation_external")
    for candidate in candidates:
        for seed in candidate.get("seeds", []):
            run_dir = RESULTS_ROOT / strategy / seed["run_id"]
            for split in splits:
                path = run_dir / f"predictions_{split}.npz"
                if not path.exists():
                    missing.append(path)
            if strategy == "transformer":
                checkpoint = run_dir / "model.pth"
                if not checkpoint.exists():
                    missing.append(checkpoint)
    if missing:
        listing = "\n  ".join(str(path) for path in missing[:20])
        suffix = f"\n  ... e mais {len(missing) - 20}" if len(missing) > 20 else ""
        raise FileNotFoundError(
            "Caches finais incompletos; a pipeline de estudos não executa a pipeline principal.\n  "
            + listing + suffix
        )


def select_study_finalists(
    strategy: str = "all",
    top_k: int = 3,
    *,
    strict: bool = True,
    output_path: str | Path | None = None,
) -> StudyManifest:
    if strategy not in {"all", "transformer", "unsupervised"}:
        raise ValueError("strategy deve ser all, transformer ou unsupervised")
    sources: dict[str, str] = {}
    transformer = _transformer_candidates(sources) if strategy in {"all", "transformer"} else []
    unsupervised = _unsupervised_candidates(sources, top_k) if strategy in {"all", "unsupervised"} else []
    if strict:
        if transformer:
            _validate_cache(transformer, "transformer")
        if unsupervised:
            _validate_cache(unsupervised, "unsupervised")
    families = tuple(
        family for family, present in (("transformer", bool(transformer)), ("unsupervised", bool(unsupervised)))
        if present
    )
    split_hashes = _split_hashes(families)
    payload = {
        "protocol_version": 1,
        "git_commit": ModelCache.git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fractions": list(DEFAULT_FRACTIONS),
        "seeds": list(DEFAULT_SEEDS),
        "transformer": transformer,
        "unsupervised": unsupervised,
        "source_hashes": sources,
        "split_hashes": split_hashes,
        "selection_policy": {
            "transformer": "one internal-validation winner per embedding",
            "unsupervised": "best embedding per algorithm, then top-k algorithms by internal_val_f1_macro",
            "external_validation_used": False,
        },
    }
    payload["manifest_hash"] = stable_hash({key: value for key, value in payload.items() if key != "created_at"})
    manifest = StudyManifest(**payload)
    destination = Path(output_path) if output_path else TRANSFORMER_STUDIES / "study_manifest.json"
    atomic_json(destination, manifest.to_dict())
    return manifest


def load_study_manifest(path: str | Path | None = None) -> StudyManifest:
    source = Path(path) if path else TRANSFORMER_STUDIES / "study_manifest.json"
    return StudyManifest(**_read_json(source))
