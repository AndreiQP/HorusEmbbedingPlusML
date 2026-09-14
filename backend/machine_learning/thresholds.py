"""
machine_learning/thresholds.py
--------------------------------
Calibração de threshold para classificação binária.

Estratégias disponíveis:
- 'target_recall'  -> Encontra o menor threshold que garante Recall >= target
- 'min_fn'         -> Minimiza FN com menor FP possível (threshold mais alto com FN=0)
- 'f_beta'         -> Maximiza F-beta (peso beta no Recall)
- 'youden'         -> Maximiza TPR - FPR (ponto de Youden na curva ROC)

Uso típico no notebook:
    calibrator = result.calibrate_threshold(strategy='target_recall', target_recall=0.98)
    calibrator.plot_fp_fn_tradeoff()
    calibrator.plot_precision_recall_curve()
    y_pred_calibrated = calibrator.apply(y_prob_val)
    calibrator.apply_and_show(y_true_val, y_prob_val)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    f1_score, recall_score, precision_score,
)


class ThresholdCalibrator:
    """
    Calibra e aplica um threshold ótimo para classificação binária.
    Calibrado num conjunto (ex: teste interno) e aplicável em outro (ex: validação externa).
    """

    STRATEGIES = ("target_recall", "min_fn", "f_beta", "youden")

    def __init__(self, strategy: str = "target_recall", **kwargs):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy deve ser um de {self.STRATEGIES}, recebido: '{strategy}'")
        self.strategy = strategy
        self.kwargs = kwargs

        self.threshold_: Optional[float] = None
        self.y_true_fit_: Optional[np.ndarray] = None
        self.y_prob_fit_: Optional[np.ndarray] = None
        self._candidates: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────────────────────────

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ThresholdCalibrator":
        """
        Encontra o threshold ótimo a partir das probabilidades do conjunto de calibração.
        
        Args:
            y_true: labels verdadeiros (0 ou 1)
            y_prob: probabilidade da classe positiva (Scam)
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        self.y_true_fit_ = y_true
        self.y_prob_fit_ = y_prob

        candidates = np.unique(y_prob)
        candidates = np.sort(np.concatenate([[0.0], candidates, [1.0]]))
        self._candidates = candidates

        if self.strategy == "target_recall":
            self.threshold_ = self._fit_target_recall(y_true, y_prob, candidates)
        elif self.strategy == "min_fn":
            self.threshold_ = self._fit_min_fn(y_true, y_prob, candidates)
        elif self.strategy == "f_beta":
            self.threshold_ = self._fit_f_beta(y_true, y_prob, candidates)
        elif self.strategy == "youden":
            self.threshold_ = self._fit_youden(y_true, y_prob)

        print(f"[ThresholdCalibrator] Estratégia='{self.strategy}' -> threshold={self.threshold_:.6f}")
        return self

    def _fit_target_recall(self, y_true, y_prob, candidates, target_recall=None):
        target = target_recall or self.kwargs.get("target_recall", 0.98)
        best_thr = 0.0
        best_fp = float("inf")
        for thr in candidates:
            y_pred = (y_prob >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            if rec >= target and fp < best_fp:
                best_fp = fp
                best_thr = float(thr)
        return best_thr

    def _fit_min_fn(self, y_true, y_prob, candidates):
        """Maior threshold que garante FN=0 no conjunto de calibração."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        pos_probs = y_prob[y_true == 1]
        if len(pos_probs) == 0:
            return 0.0
        # O menor score entre os positivos é o limiar exato para zerar FN
        return float(np.min(pos_probs))


    def _fit_f_beta(self, y_true, y_prob, candidates):
        beta = self.kwargs.get("beta", 2.0)  # beta>1 favorece recall
        best_thr = 0.5
        best_score = -1.0
        for thr in candidates:
            y_pred = (y_prob >= thr).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if prec + rec == 0:
                continue
            score = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        return best_thr

    def _fit_youden(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden = tpr - fpr
        idx = np.argmax(youden)
        return float(thresholds[idx])

    # ──────────────────────────────────────────────────────────────
    # Apply
    # ──────────────────────────────────────────────────────────────

    def apply(self, y_prob: np.ndarray) -> np.ndarray:
        """Aplica o threshold calibrado a novas probabilidades."""
        self._check_fitted()
        return (np.asarray(y_prob) >= self.threshold_).astype(int)

    def apply_and_show(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Aplica o threshold, imprime classification_report e exibe matriz de confusão."""
        self._check_fitted()
        from sklearn.metrics import classification_report
        from machine_learning.evaluation import plot_confusion_matrix

        y_pred = self.apply(y_prob)
        print(f"\n[Threshold={self.threshold_:.6f}] Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Ham (0)", "Scam (1)"], zero_division=0))
        plot_confusion_matrix(y_true, y_pred,
                              title=f"Threshold={self.threshold_:.4f} ({self.strategy})")


    # ──────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────

    def plot_fp_fn_tradeoff(self, ax=None, save_path: Optional[str] = None):
        """Plota FP e FN em função do threshold, marcando o ponto escolhido."""
        self._check_fitted()
        candidates = self._candidates
        fps, fns = [], []
        for thr in candidates:
            y_pred = (self.y_prob_fit_ >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true_fit_, y_pred, labels=[0, 1]).ravel()
            fps.append(fp); fns.append(fn)

        ax_provided = ax is not None
        if not ax_provided:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.get_figure()

        ax.plot(candidates, fps, "b-", label="Falsos Positivos (FP)", lw=2)
        ax.plot(candidates, fns, "r-", label="Falsos Negativos (FN)", lw=2)
        ax.axvline(self.threshold_, color="green", linestyle="--", lw=2,
                   label=f"Threshold escolhido ({self.threshold_:.4f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Contagem")
        ax.set_title(f"Trade-off FP vs FN por Threshold\nEstratégia: {self.strategy}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        if save_path:
            import os; os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if not ax_provided:
            plt.show(); plt.close(fig)

    def plot_precision_recall_curve(self, ax=None, save_path: Optional[str] = None):
        """Curva PR marcando o ponto do threshold escolhido."""
        self._check_fitted()
        from machine_learning.evaluation import plot_precision_recall_curve as _plot_pr
        _plot_pr(self.y_true_fit_, self.y_prob_fit_,
                 threshold_marker=self.threshold_, ax=ax, save_path=save_path)

    # ──────────────────────────────────────────────────────────────
    # Utils
    # ──────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.threshold_ is None:
            raise RuntimeError("Chame .fit() antes de usar o calibrador.")

    def summary(self) -> pd.DataFrame:
        """Retorna DataFrame com métricas para cada threshold candidato."""
        self._check_fitted()
        rows = []
        for thr in self._candidates:
            y_pred = (self.y_prob_fit_ >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true_fit_, y_pred, labels=[0, 1]).ravel()
            rows.append({"threshold": thr, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                         "recall": tp / (tp + fn) if (tp + fn) > 0 else 0})
        return pd.DataFrame(rows)

    def __repr__(self):
        status = f"threshold={self.threshold_:.6f}" if self.threshold_ is not None else "não calibrado"
        return f"ThresholdCalibrator(strategy='{self.strategy}', {status})"
