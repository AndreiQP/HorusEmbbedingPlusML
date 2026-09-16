"""Relatórios cache-first para os detectores de texto bruto CVDD e DATE.

Este módulo não treina modelos. Quando solicitado, apenas carrega o checkpoint
persistido para obter predições e scores da validação externa e então grava
tabelas e gráficos em ``experiment_results/unsupervised/text_reports/``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


BACKEND_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BACKEND_DIR / "experiment_results" / "unsupervised"
REPORTS_DIR = RESULTS_DIR / "text_reports"
SPLIT_ORDER = ["train", "val", "test", "validation_external"]
SPLIT_LABELS = {
    "train": "Treino interno",
    "val": "Validação interna",
    "test": "Teste interno",
    "validation_external": "Validação externa",
}


def _metrics_candidates(model_type: str) -> list[Path]:
    pattern = f"embedding=bge__model_type={model_type}__strategy=unsupervised*"
    candidates = []
    for directory in RESULTS_DIR.glob(pattern):
        metrics = directory / "metrics.json"
        if metrics.exists():
            candidates.append(metrics)
    return candidates


def load_text_model_metrics(model_type: str) -> tuple[dict[str, Any], Path]:
    """Lê o conjunto de métricas mais completo disponível para o modelo."""
    candidates = _metrics_candidates(model_type)
    if not candidates:
        raise FileNotFoundError(
            f"Não encontrei métricas cacheadas de {model_type.upper()} em {RESULTS_DIR}."
        )

    parsed: list[tuple[int, float, dict[str, Any], Path]] = []
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        coverage = sum(split in payload for split in SPLIT_ORDER)
        parsed.append((coverage, path.stat().st_mtime, payload, path))
    _, _, metrics, source = max(parsed, key=lambda item: (item[0], item[1]))
    return metrics, source


def metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        if split in metrics:
            row = dict(metrics[split])
            row["split"] = split
            row["split_label"] = SPLIT_LABELS[split]
            rows.append(row)
    if not rows:
        raise ValueError("O arquivo de métricas não contém splits reconhecidos.")
    return pd.DataFrame(rows)


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_metrics_by_split(frame: pd.DataFrame, output: Path, model_type: str) -> None:
    metrics = ["f1_macro", "precision_scam", "recall_scam", "pr_auc", "roc_auc"]
    available = [name for name in metrics if name in frame]
    ax = frame.set_index("split_label")[available].plot(
        kind="bar", figsize=(12, 5), ylim=(0, 1), rot=0
    )
    ax.set(title=f"{model_type.upper()} — métricas por split", ylabel="Valor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncols=3)
    _save_figure(output)


def _plot_confusion_counts(frame: pd.DataFrame, output: Path, model_type: str) -> None:
    counts = [name for name in ["TP", "TN", "FP", "FN"] if name in frame]
    ax = frame.set_index("split_label")[counts].plot(kind="bar", figsize=(12, 5), rot=0)
    ax.set(title=f"{model_type.upper()} — contagens da matriz de confusão", ylabel="Número de conversas")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(output)


def _plot_external_from_metrics(frame: pd.DataFrame, output: Path, model_type: str) -> None:
    external = frame.loc[frame["split"] == "validation_external"].iloc[0]
    matrix = np.array([[external["TN"], external["FP"]], [external["FN"], external["TP"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["Ham", "Scam"]).plot(
        ax=ax, colorbar=False, values_format="d"
    )
    ax.set_title(f"{model_type.upper()} — validação externa")
    _save_figure(output)


def _plot_external_scores(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, plots_dir: Path, model_type: str
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Ham", "Scam"], colorbar=False, ax=ax)
    ax.set_title(f"{model_type.upper()} — validação externa")
    _save_figure(plots_dir / "confusion_matrix_external.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"{model_type.upper()} — ROC externa")
    _save_figure(plots_dir / "roc_external.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"{model_type.upper()} — precisão-recall externa")
    _save_figure(plots_dir / "pr_external.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name, color in [(0, "Ham", "#1f77b4"), (1, "Scam", "#d62728")]:
        ax.hist(y_score[y_true == label], bins=30, alpha=0.55, label=name, color=color)
    ax.set(title=f"{model_type.upper()} — distribuição do score externo", xlabel="Score de anomalia", ylabel="Conversas")
    ax.legend()
    _save_figure(plots_dir / "score_distribution_external.png")


def _load_cached_external_predictions(model_type: str, device: str):
    """Carrega o checkpoint e faz inferência externa, sem atualizar pesos."""
    from . import text_runner
    from ..cache import ModelCache

    # Não delegamos a decisão ao runner: se o checkpoint não existe, a chamada
    # normal dele iniciaria um treino. Relatórios devem falhar de forma explícita
    # nesse caso, preservando a garantia cache-first.
    if not ModelCache.text_model_exists(model_type, "bge", params_hash="default"):
        raise FileNotFoundError(
            f"Checkpoint cacheado de {model_type.upper()} não encontrado. "
            "Este relatório não treina modelos; recupere o model.pt do experimento."
        )
    train_fn = text_runner.train_cvdd if model_type == "cvdd" else text_runner.train_date
    result = train_fn(force_retrain=False, device=device)
    if result.y_true is None or result.y_pred is None or result.y_prob is None:
        raise RuntimeError(f"{model_type.upper()} não retornou predições externas cacheadas.")
    return np.asarray(result.y_true), np.asarray(result.y_pred), np.asarray(result.y_prob)


def build_text_model_report(model_type: str, device: str = "cpu", metrics_only: bool = False) -> Path:
    if model_type not in {"cvdd", "date"}:
        raise ValueError("model_type deve ser 'cvdd' ou 'date'.")

    metrics, source_metrics = load_text_model_metrics(model_type)
    frame = metrics_frame(metrics)
    report_dir = REPORTS_DIR / model_type
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "metrics_by_split.csv", index=False)
    _plot_metrics_by_split(frame, plots_dir / "metrics_by_split.png", model_type)
    _plot_confusion_counts(frame, plots_dir / "confusion_counts_by_split.png", model_type)

    prediction_mode = "metrics_only"
    if metrics_only:
        _plot_external_from_metrics(frame, plots_dir / "confusion_matrix_external.png", model_type)
    else:
        y_true, y_pred, y_score = _load_cached_external_predictions(model_type, device)
        _plot_external_scores(y_true, y_pred, y_score, plots_dir, model_type)
        prediction_mode = "checkpoint_inference"

    manifest = {
        "model_type": model_type,
        "metrics_source": str(source_metrics.relative_to(BACKEND_DIR)),
        "prediction_mode": prediction_mode,
        "device": device,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plots": sorted(str(path.relative_to(report_dir)) for path in plots_dir.glob("*.png")),
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera gráficos cache-first de CVDD e DATE.")
    parser.add_argument("--model", choices=["cvdd", "date", "all"], default="all")
    parser.add_argument("--device", default="cpu", help="cpu ou cuda; não altera o cache do modelo.")
    parser.add_argument("--metrics-only", action="store_true", help="Não carrega checkpoints nem gera ROC/PR/histograma.")
    args = parser.parse_args(argv)

    models = ["cvdd", "date"] if args.model == "all" else [args.model]
    for model_type in models:
        destination = build_text_model_report(model_type, device=args.device, metrics_only=args.metrics_only)
        print(f"[text_reports] {model_type.upper()}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
