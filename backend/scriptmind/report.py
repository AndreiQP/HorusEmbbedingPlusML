"""Consolidate annotations, ScriptMind runs and read-only Horus baselines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .annotation import annotation_quality
from .io_utils import atomic_write_json, atomic_write_text
from .paths import BACKEND_ROOT, ScriptMindPaths


def _read_metric_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("metrics.json") if root.exists() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append({**item, "metrics_path": str(path)})
        elif isinstance(value, dict):
            rows.append({**value, "metrics_path": str(path)})
    return rows


def _best_existing_baselines() -> list[dict[str, Any]]:
    result_root = BACKEND_ROOT / "experiment_results"
    rows = []
    for strategy in ("classical", "fcnn", "transformer", "unsupervised"):
        candidates = _read_metric_files(result_root / strategy)
        candidates = [row for row in candidates if row.get("f1_macro") is not None]
        if candidates:
            best = max(candidates, key=lambda row: float(row.get("f1_macro", -1)))
            rows.append({"strategy": strategy, **best})
    return rows


def generate_report(paths: ScriptMindPaths | None = None) -> dict[str, Any]:
    paths = (paths or ScriptMindPaths.defaults()).ensure()
    scriptmind_rows = _read_metric_files(paths.results_root / "evaluation")
    metrics_frame = pd.DataFrame(scriptmind_rows)
    metrics_csv = paths.results_root / "report" / "scriptmind_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(metrics_csv, index=False)
    try:
        quality = annotation_quality(paths)
    except RuntimeError:
        quality = {"status": "annotations_not_available"}
    baselines = _best_existing_baselines()
    judge_summaries = []
    for judge_path in sorted((paths.results_root / "evaluation").rglob("judge_summary.json")):
        try:
            judge_summaries.append({**json.loads(judge_path.read_text(encoding="utf-8")), "path": str(judge_path)})
        except (json.JSONDecodeError, OSError):
            continue
    atomic_write_json(paths.results_root / "report" / "existing_baselines.json", baselines, paths)
    report = {
        "annotation_quality": quality,
        "scriptmind_evaluations": len(scriptmind_rows),
        "existing_baselines": len(baselines),
        "judge_evaluations": len(judge_summaries),
        "metrics_csv": str(metrics_csv),
    }
    lines = [
        "# ScriptMind experiment report", "", "## Summary", "",
        f"- ScriptMind evaluation records: {len(scriptmind_rows)}",
        f"- Existing strategy baselines discovered read-only: {len(baselines)}",
        f"- LLM judge summaries: {len(judge_summaries)}",
        f"- Annotation quality: `{json.dumps(quality, ensure_ascii=False)}`", "",
        "## Artifacts", "", f"- Metrics: `{metrics_csv}`",
        f"- Baselines: `{paths.results_root / 'report' / 'existing_baselines.json'}`", "",
        "The cognitive participant study is intentionally outside this computational report.", "",
    ]
    atomic_write_text(paths.results_root / "report" / "README.md", "\n".join(lines), paths)
    atomic_write_json(paths.results_root / "report" / "summary.json", report, paths)
    return report
