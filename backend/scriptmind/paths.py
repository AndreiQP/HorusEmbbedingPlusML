"""All ScriptMind writes are routed through these isolated paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ScriptMindPaths:
    data_root: Path
    results_root: Path

    @classmethod
    def defaults(cls) -> "ScriptMindPaths":
        data = Path(os.getenv("SCRIPTMIND_DATA_ROOT", BACKEND_ROOT / "datasets" / "dataset_scriptmind"))
        results = Path(os.getenv("SCRIPTMIND_RESULTS_ROOT", BACKEND_ROOT / "experiment_results" / "scriptmind"))
        return cls(data.resolve(), results.resolve())

    def ensure(self) -> "ScriptMindPaths":
        for path in (
            self.data_root / "inventory",
            self.data_root / "annotations" / "gemini",
            self.data_root / "annotations" / "local",
            self.data_root / "adjudicated",
            self.data_root / "csid",
            self.data_root / "manifests",
            self.results_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def source_train(self) -> Path:
        return BACKEND_ROOT / "datasets" / "dataset_train" / "raw" / "all_data.csv"

    @property
    def source_external(self) -> Path:
        return BACKEND_ROOT / "datasets" / "dataset_validation" / "raw" / "all_data_1_to_20_turns_lines.parquet"


def assert_isolated_path(path: Path, roots: ScriptMindPaths | None = None) -> None:
    roots = roots or ScriptMindPaths.defaults()
    resolved = path.resolve()
    allowed = (roots.data_root.resolve(), roots.results_root.resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError(f"ScriptMind refused to write outside isolated roots: {resolved}")
