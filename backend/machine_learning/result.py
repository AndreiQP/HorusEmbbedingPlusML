"""
machine_learning/result.py
--------------------------
Classe ExperimentResult: núcleo do sistema.
Retornada por todos os runners (train_*, evaluate_*, learning_curve_*).
Permite acesso ao modelo, métricas e plots diretamente no notebook, célula a célula.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd


class ExperimentResult:
    """
    Objeto rico retornado por todos os runners.
    Mantém o modelo, o histórico, as métricas e oferece métodos de plot inline.
    """

    def __init__(
        self,
        strategy: str,                      # 'classical' | 'fcnn' | 'transformer' | 'unsupervised'
        config: Dict[str, Any],             # parâmetros que geraram este resultado
        model=None,                         # objeto modelo (sklearn, torch, joblib, etc.)
        df_history: Optional[pd.DataFrame] = None,   # histórico por época/fold/fração
        df_metrics: Optional[pd.DataFrame] = None,   # métricas finais (uma linha)
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        artifacts_dir: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,      # dados extras livres por estratégia
    ):
        self.strategy = strategy
        self.config = config
        self.model = model
        self.df_history = df_history if df_history is not None else pd.DataFrame()
        self.df_metrics = df_metrics if df_metrics is not None else pd.DataFrame()
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.threshold = threshold
        self.artifacts_dir = artifacts_dir
        self.extra = extra or {}

    # ──────────────────────────────────────────────────────────────
    # Prints de texto
    # ──────────────────────────────────────────────────────────────

    def print_metrics(self):
        """Imprime tabela de métricas formatada."""
        if self.df_metrics.empty:
            print("Sem métricas disponíveis.")
            return
        from machine_learning.evaluation import print_metrics_table
        print_metrics_table(self.df_metrics)

    def print_classification_report(self):
        """Imprime classification report do sklearn."""
        if self.y_true is None or self.y_pred is None:
            print("Sem predições disponíveis.")
            return
        from sklearn.metrics import classification_report
        print(classification_report(
            self.y_true, self.y_pred,
            target_names=["Ham (0)", "Scam (1)"],
            zero_division=0
        ))

    # ──────────────────────────────────────────────────────────────
    # Plots inline (exibe no notebook E salva PNG se artifacts_dir existir)
    # ──────────────────────────────────────────────────────────────

    def plot_confusion_matrix(self, title: str = "", ax=None, save: bool = True):
        """Heatmap da matriz de confusão."""
        if self.y_true is None or self.y_pred is None:
            print("Sem predições para plotar.")
            return
        from machine_learning.evaluation import plot_confusion_matrix
        save_path = self._artifact_path("confusion_matrix.png") if save else None
        plot_confusion_matrix(self.y_true, self.y_pred, title=title, ax=ax, save_path=save_path)

    def plot_learning_curves(self, metrics: List[str] = None, ax=None, save: bool = True):
        """Curvas de treino/validação por época (loss, f1, etc.)."""
        if self.df_history.empty:
            print("Sem histórico de épocas disponível.")
            return
        from machine_learning.evaluation import plot_epoch_history
        
        df_hist = self.df_history.copy()
        col_map = {c.lower(): c for c in df_hist.columns}
        pct_col = col_map.get("percentual_dataset", col_map.get("percentual", None))
        if pct_col and pct_col in df_hist.columns:
            df_hist = df_hist[df_hist[pct_col] == 100].reset_index(drop=True)

        if metrics is None:
            metrics = ["loss", "f1"]

        save_path = self._artifact_path("learning_curves.png") if save else None
        plot_epoch_history(df_hist, metrics=metrics, ax=ax, save_path=save_path)


    def plot_dataset_size_curve(self, metric: str = "f1_macro", ax=None, save: bool = True):
        """Curva de aprendizado por % do dataset."""
        if self.df_history.empty or "frac" not in self.df_history.columns:
            print("Sem dados de curva por tamanho de dataset.")
            return
        from machine_learning.evaluation import plot_dataset_size_curve
        save_path = self._artifact_path("dataset_size_curve.png") if save else None
        plot_dataset_size_curve(self.df_history, metric=metric, ax=ax, save_path=save_path)

    def plot_confusion_matrix_evolution(self, ax=None, save: bool = True):
        """Evolução percentual da Matriz de Confusão por % do dataset."""
        if self.df_history.empty or "frac" not in self.df_history.columns:
            print("Sem dados de curva por tamanho de dataset.")
            return
        from machine_learning.evaluation import plot_confusion_matrix_evolution
        save_path = self._artifact_path("confusion_matrix_evolution.png") if save else None
        plot_confusion_matrix_evolution(self.df_history, ax=ax, save_path=save_path)

    def plot_confusion_matrices_by_fraction(self, save: bool = True):
        """Grade de Matrizes de Confusão (%) por fração do dataset."""
        if self.df_history.empty or "frac" not in self.df_history.columns:
            print("Sem dados de curva por tamanho de dataset.")
            return
        from machine_learning.evaluation import plot_confusion_matrices_by_fraction
        save_path = self._artifact_path("confusion_matrices_by_fraction.png") if save else None
        plot_confusion_matrices_by_fraction(self.df_history, save_path=save_path)

    def plot_precision_recall_curve(self, ax=None, save: bool = True):
        """Curva Precision-Recall."""
        if self.y_prob is None:
            print("Sem probabilidades para plotar curva PR.")
            return
        from machine_learning.evaluation import plot_precision_recall_curve
        save_path = self._artifact_path("pr_curve.png") if save else None
        plot_precision_recall_curve(self.y_true, self.y_prob, ax=ax, save_path=save_path)

    def plot_roc_curve(self, ax=None, save: bool = True):
        """Curva ROC."""
        if self.y_prob is None:
            print("Sem probabilidades para plotar curva ROC.")
            return
        from machine_learning.evaluation import plot_roc_curve
        save_path = self._artifact_path("roc_curve.png") if save else None
        plot_roc_curve(self.y_true, self.y_prob, ax=ax, save_path=save_path)

    def plot_pca_variance(self, ax=None, save: bool = True):
        """(Unsupervised) Variância explicada pelo PCA por número de componentes."""
        if "pca_variance" not in self.extra:
            print("Sem dados de variância PCA. Execute train_anomaly com store_pca_variance=True.")
            return
        from machine_learning.evaluation import plot_pca_variance
        save_path = self._artifact_path("pca_variance.png") if save else None
        plot_pca_variance(self.extra["pca_variance"], ax=ax, save_path=save_path)

    def plot_umap(self, ax=None, save: bool = True):
        """(Unsupervised) Visualização UMAP 2D com labels coloridas."""
        if "umap_coords" not in self.extra:
            print("Sem coordenadas UMAP disponíveis.")
            return
        from machine_learning.evaluation import plot_umap_2d
        save_path = self._artifact_path("umap.png") if save else None
        plot_umap_2d(
            self.extra["umap_coords"], self.extra.get("umap_labels"),
            ax=ax, save_path=save_path
        )

    # ──────────────────────────────────────────────────────────────
    # Operações funcionais
    # ──────────────────────────────────────────────────────────────

    def evaluate_on_validation(self, path_data: str = None, embedding: str = None) -> "ExperimentResult":
        """
        Avalia o modelo no dataset de validação externo.
        Retorna um novo ExperimentResult com as métricas de validação.
        """
        from machine_learning.evaluation import evaluate_model_on_external
        path_data = path_data or self.config.get("path_data", "all_data")
        if embedding is None:
            emb_cfg = self.config.get("embedding")
            if emb_cfg:
                embedding = emb_cfg
            elif isinstance(self.config.get("embedders"), list):
                embedding = self.config.get("embedders")[0]
            else:
                embedding = "voyage"
        return evaluate_model_on_external(self, path_data=path_data, embedding=embedding)


    def calibrate_threshold(self, strategy: str = "target_recall", **kwargs) -> "ThresholdCalibrator":
        """Retorna um ThresholdCalibrator já ajustado neste resultado."""
        from machine_learning.thresholds import ThresholdCalibrator
        if self.y_prob is None:
            raise ValueError("Probabilidades (y_prob) necessárias para calibrar threshold.")
        calibrator = ThresholdCalibrator(strategy=strategy, **kwargs)
        calibrator.fit(self.y_true, self.y_prob)
        return calibrator

    def compare_with(self, other: "ExperimentResult", metric: str = "f1_macro"):
        """Exibe comparação de métricas entre dois resultados."""
        from machine_learning.evaluation import compare_results
        compare_results(self, other, metric=metric)

    # ──────────────────────────────────────────────────────────────
    # Utilitários internos
    # ──────────────────────────────────────────────────────────────

    def _artifact_path(self, filename: str) -> Optional[str]:
        if self.artifacts_dir:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            return os.path.join(self.artifacts_dir, filename)
        return None

    def __repr__(self):
        metrics_str = ""
        if not self.df_metrics.empty and "f1_macro" in self.df_metrics.columns:
            f1 = self.df_metrics["f1_macro"].iloc[0]
            metrics_str = f", f1_macro={f1:.4f}"
        return (
            f"ExperimentResult(strategy='{self.strategy}', "
            f"config={self.config}{metrics_str})"
        )
