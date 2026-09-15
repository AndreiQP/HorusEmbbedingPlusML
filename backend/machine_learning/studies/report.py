from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import TRANSFORMER_STUDIES, UNSUPERVISED_STUDIES, atomic_json
from .dataset_size import aggregate_dataset_size_results
from .explainability import build_transformer_explainability_report


def _plot_size(path: Path, title: str) -> list[str]:
    import matplotlib.pyplot as plt

    if not path.exists() or path.stat().st_size <= 1:
        return []
    frame = pd.read_csv(path)
    outputs = []
    plots = path.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for metric in ("f1_macro", "recall_scam", "precision_scam", "pr_auc", "FP", "FN"):
        column = f"{metric}_mean"
        if column not in frame:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for (candidate, split), group in frame.groupby(["candidate", "split"]):
            group = group.sort_values("fraction")
            ax.plot(group["fraction"] * 100, group[column], marker="o", label=f"{candidate} — {split}")
            ci = group.get(f"{metric}_ci95", pd.Series(np.zeros(len(group)), index=group.index))
            ax.fill_between(group["fraction"] * 100, group[column] - ci, group[column] + ci, alpha=0.12)
        ax.set_title(f"{title}: {metric}")
        ax.set_xlabel("Percentual do conjunto de ajuste")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        output = plots / f"dataset_size_{metric}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(str(output))
    return outputs


def _plot_transformer_analyses() -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    outputs = []
    plots = TRANSFORMER_STUDIES / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    threshold = TRANSFORMER_STUDIES / "threshold" / "metrics.csv"
    if threshold.exists():
        frame = pd.read_csv(threshold)
        view = frame[frame["split"].isin(["test_internal", "validation"])]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=view, x="variant", y="FN", hue="split", ax=axes[0])
        sns.barplot(data=view, x="variant", y="FP", hue="split", ax=axes[1])
        for ax, metric in zip(axes, ("FN", "FP")):
            ax.set_title(f"Threshold Transformer — {metric}")
            ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        output = plots / "threshold_fp_fn.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(str(output))
    ablation = TRANSFORMER_STUDIES / "explainability" / "head_stability.csv"
    if ablation.exists():
        frame = pd.read_csv(ablation)
        for embedding, group in frame.groupby("embedding"):
            pivot = group.groupby(["layer", "head"])["impact_mean"].mean().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="magma", ax=ax)
            ax.set_title(f"Impacto por ablação — {embedding}")
            fig.tight_layout()
            output = plots / f"head_ablation_{embedding}.png"
            fig.savefig(output, dpi=160)
            plt.close(fig)
            outputs.append(str(output))
    return outputs


def build_studies_report(*, include_explainability: bool = True) -> dict:
    aggregate_dataset_size_results()
    if include_explainability:
        try:
            build_transformer_explainability_report()
        except FileNotFoundError:
            pass
    plots = []
    plots.extend(_plot_size(TRANSFORMER_STUDIES / "dataset_size" / "aggregate.csv", "Transformer"))
    plots.extend(_plot_size(UNSUPERVISED_STUDIES / "dataset_size" / "aggregate.csv", "Unsupervised"))
    plots.extend(_plot_transformer_analyses())
    payload = {"plots": plots, "cache_only": True}
    atomic_json(TRANSFORMER_STUDIES / "report.json", payload)
    return payload
