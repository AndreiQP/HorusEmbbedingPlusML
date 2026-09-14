"""
machine_learning/evaluation.py
-------------------------------
Métricas, plots padronizados e utilitários de avaliação.

Todos os plots têm assinatura (ax=None, save_path=None):
- ax=None -> cria figura própria e exibe inline no notebook
- ax=Axes -> desenha na figura externa (para subplots compostos)
- save_path -> se fornecido, salva PNG em disco além de exibir
"""
from __future__ import annotations
import os
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    model_name: str = "",
    dataset_name: str = "",
) -> Dict:
    """
    Calcula o conjunto completo de métricas para classificação binária.

    Args:
        y_score: score contínuo de anomalia/probabilidade (maior = mais provável Scam=1).
                  Usado para AUROC/AUPRC. Aceita também via `y_prob` (alias mantido por
                  compatibilidade retroativa).

    Returns:
        dict com: accuracy, f1_macro, f1_scam, precision_macro, recall_macro,
                  precision_scam, recall_scam, roc_auc, pr_auc, TP, TN, FP, FN,
                  model_name, dataset_name
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = y_score if y_score is not None else y_prob

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_scam = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    prec_scam = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    rec_scam = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)

    roc_auc_v, pr_auc_v = None, None
    if score is not None and len(np.unique(y_true)) > 1:
        score = np.asarray(score)
        try:
            roc_auc_v = round(float(roc_auc_score(y_true, score)), 4)
            pr_auc_v = round(float(average_precision_score(y_true, score)), 4)
        except ValueError:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name":      model_name,
        "dataset_name":    dataset_name,
        "accuracy":        round(float(acc), 4),
        "f1_macro":        round(float(f1_macro), 4),
        "f1_scam":         round(float(f1_scam), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro":    round(float(rec_macro), 4),
        "precision_scam":  round(float(prec_scam), 4),
        "recall_scam":     round(float(rec_scam), 4),
        "roc_auc":         roc_auc_v,
        "pr_auc":          pr_auc_v,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def print_metrics_table(df_metrics: pd.DataFrame):
    """Imprime métricas de forma formatada."""
    cols = ["model_name", "dataset_name", "accuracy", "f1_macro", "f1_scam",
            "recall_scam", "precision_scam", "roc_auc", "pr_auc", "TP", "TN", "FP", "FN"]
    cols_available = [c for c in cols if c in df_metrics.columns]
    print(df_metrics[cols_available].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _save_and_show(fig, save_path: Optional[str], ax_was_provided: bool):
    """Salva o PNG e/ou exibe o plot conforme contexto."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  [plot] Salvo em: {save_path}")
    if not ax_was_provided:
        plt.show()
        plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de Confusão",
    ax=None,
    save_path: Optional[str] = None,
):
    """Heatmap padronizado da matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Previsto Ham", "Previsto Scam"],
        yticklabels=["Real Ham", "Real Scam"],
    )
    ax.set_title(title)
    ax.set_ylabel("Classe Real")
    ax.set_xlabel("Predição")

    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(f"Predição\n\nTP={tp}  TN={tn}  FP={fp}  FN={fn}")

    _save_and_show(fig, save_path, ax_provided)


def plot_epoch_history(
    df_history: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Curva de Aprendizado",
    ax=None,
    save_path: Optional[str] = None,
):
    """
    Plota métricas por época (train vs. val).
    
    df_history deve ter colunas como:
        epoch, train_loss, val_loss, train_f1, val_f1, val_f1_std, val_loss_std
    """
    if metrics is None:
        metrics = ["loss", "f1"]

    n = len(metrics)
    ax_provided = ax is not None
    if not ax_provided:
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]
    else:
        # ax foi fornecido externamente -> assume que é uma lista ou array
        if n == 1:
            axes = [ax]
            fig = ax.get_figure()
        else:
            raise ValueError("Forneça ax como lista de Axes quando metrics tem mais de 1 elemento.")

    col_map = {c.lower(): c for c in df_history.columns}
    epoch_col = col_map.get("epoch", col_map.get("epoca", None))
    if epoch_col and epoch_col in df_history.columns:
        epochs = df_history[epoch_col]
    else:
        epochs = range(1, len(df_history) + 1)

    for i, metric in enumerate(metrics):
        m_lower = metric.lower()
        train_col = col_map.get(f"train_{m_lower}", None)
        val_col = col_map.get(f"val_{m_lower}", None)
        std_col = col_map.get(f"val_{m_lower}_std", None)

        if train_col and train_col in df_history.columns:
            axes[i].plot(epochs, df_history[train_col], "b-", label="Treino")
        if val_col and val_col in df_history.columns:
            axes[i].plot(epochs, df_history[val_col], "r--", label="Validação")
            if std_col and std_col in df_history.columns:
                val = df_history[val_col]
                std = df_history[std_col]
                axes[i].fill_between(epochs, val - std, val + std, alpha=0.15, color="red")

        axes[i].set_title(f"{title}: {metric.upper()}")
        axes[i].set_xlabel("Época")
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True, linestyle="--", alpha=0.4)


    if not ax_provided:
        plt.tight_layout()
    _save_and_show(fig, save_path, ax_provided)


def plot_dataset_size_curve(
    df_lc: pd.DataFrame,
    metric: str = "f1_macro",
    title: str = "Curva de Aprendizado por Tamanho do Dataset",
    ax=None,
    save_path: Optional[str] = None,
):
    """
    Curva de aprendizado em função da fração de dados de treino.
    
    df_lc deve ter colunas:
        frac, {metric}_mean_train, {metric}_std_train, {metric}_mean_val, {metric}_std_val
    ou:
        frac, acc_mean, acc_std, val_acc_mean, val_acc_std, f1_mean, f1_std, val_f1_mean, val_f1_std
    """
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    pct = df_lc["frac"] * 100

    def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    base = metric.replace("val_", "").replace("_mean", "").replace("_std", "")
    train_mean_col = _find_col(df_lc, [
        f"{base}_mean", f"{base}_mean_train", f"train_{base}_mean", f"train_{base}", f"{base}",
        "train_f1", "train_f1_macro", "train_precision", "train_recall", "f1_mean", "f1_macro_mean", "acc_mean", "train_acc"
    ])
    train_std_col = _find_col(df_lc, [
        f"{base}_std", f"{base}_std_train", f"train_{base}_std", "f1_std", "f1_macro_std", "acc_std"
    ])
    val_mean_col = _find_col(df_lc, [
        f"val_{base}_mean", f"validation_{base}", f"{base}_mean_val", f"{base}_val", f"val_{base}",
        "validation_f1", "validation_f1_macro", "validation_precision", "validation_recall", "val_f1_mean", "val_f1_macro_mean", "val_acc_mean", "validation_acc"
    ])
    val_std_col = _find_col(df_lc, [
        f"val_{base}_std", f"validation_{base}_std", f"{base}_std_val", f"{base}_val_std", f"val_{base}_std", "val_f1_std", "val_f1_macro_std", "val_acc_std"
    ])

    if train_mean_col and train_mean_col in df_lc.columns:
        ax.plot(pct, df_lc[train_mean_col], "o-", label="Teste interno")
        if train_std_col and train_std_col in df_lc.columns:
            ax.fill_between(pct,
                df_lc[train_mean_col] - df_lc[train_std_col],
                df_lc[train_mean_col] + df_lc[train_std_col],
                alpha=0.12)

    if val_mean_col and val_mean_col in df_lc.columns:
        ax.plot(pct, df_lc[val_mean_col], "s--", label="Validação externa")
        if val_std_col and val_std_col in df_lc.columns:
            ax.fill_between(pct,
                df_lc[val_mean_col] - df_lc[val_std_col],
                df_lc[val_mean_col] + df_lc[val_std_col],
                alpha=0.12)

    ax.set_title(f"{title}\n({metric})")
    ax.set_xlabel("% de dados de treino utilizados")
    ax.set_ylabel(metric)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    _save_and_show(fig, save_path, ax_provided)


def plot_loss_evolution_all_embeddings(
    csv_path: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plota a evolução de Train Loss e Validation Loss por Época para cada percentual de dataset,
    gerando uma figura com subplots lado a lado para cada embedding (Voyage, BGE, OpenAI).
    """
    _BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = csv_path or os.path.join(_BACKEND_DIR, "experiment_results", "legacy_metrics", "historico_epocas.csv")

    if not os.path.exists(csv_path):
        print(f"Histórico de épocas não encontrado em {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_cols = {c.lower(): c for c in df.columns}
    modelo_col = df_cols.get("modelo", "Modelo")
    pct_col = df_cols.get("percentual_dataset", df_cols.get("percentual", "Percentual_Dataset"))
    epoca_col = df_cols.get("epoca", "Epoca")
    train_loss_col = df_cols.get("train_loss", "Train_Loss")
    val_loss_col = df_cols.get("val_loss", "Val_Loss")

    modelos = df[modelo_col].unique()

    for modelo in modelos:
        df_mod = df[df[modelo_col] == modelo]
        percentuais = sorted(df_mod[pct_col].unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.suptitle(f"Evolução do Loss: {str(modelo).capitalize()}", fontsize=16, fontweight="bold")

        for pct in percentuais:
            df_pct = df_mod[df_mod[pct_col] == pct].sort_values(epoca_col)
            axes[0].plot(df_pct[epoca_col], df_pct[train_loss_col], label=f"{pct}%", linewidth=2)

        axes[0].set_title("Train Loss", fontsize=12)
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Loss")
        axes[0].legend(title="Tamanho do Dataset")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        for pct in percentuais:
            df_pct = df_mod[df_mod[pct_col] == pct].sort_values(epoca_col)
            axes[1].plot(df_pct[epoca_col], df_pct[val_loss_col], label=f"{pct}%", linewidth=2, linestyle="--")

        axes[1].set_title("Validation Loss", fontsize=12)
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Loss")
        axes[1].legend(title="Tamanho do Dataset")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()



def plot_confusion_matrix_evolution(
    df_lc: pd.DataFrame,
    title: str = "Evolução da Matriz de Confusão por Tamanho do Dataset",
    ax=None,
    save_path: Optional[str] = None,
    use_percentage: bool = False,
):
    """
    Plota a evolução de TN, FP, FN, TP em função do % de dados de treino.
    Por padrão (use_percentage=False), utiliza valores absolutos (contagens de exemplos).
    """
    pct = df_lc["frac"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    tp_col = "tp_pct" if use_percentage else "tp_mean"
    tn_col = "tn_pct" if use_percentage else "tn_mean"
    fp_col = "fp_pct" if use_percentage else "fp_mean"
    fn_col = "fn_pct" if use_percentage else "fn_mean"

    val_tp_col = "val_tp_pct" if use_percentage else "val_tp_mean"
    val_tn_col = "val_tn_pct" if use_percentage else "val_tn_mean"
    val_fp_col = "val_fp_pct" if use_percentage else "val_fp_mean"
    val_fn_col = "val_fn_pct" if use_percentage else "val_fn_mean"

    unit = "% dos Exemplos" if use_percentage else "Quantidade de Exemplos"

    # 1. Teste Interno
    ax_in = axes[0]
    if tp_col in df_lc.columns:
        ax_in.plot(pct, df_lc[tp_col], "g-o", linewidth=2, label="TP (Scam detectados)")
        ax_in.plot(pct, df_lc[tn_col], "b-s", linewidth=2, label="TN (Legítimos corretos)")
        ax_in.plot(pct, df_lc[fp_col], "m-^", linewidth=2, label="FP (Falsos alarmes)")
        ax_in.plot(pct, df_lc[fn_col], "r-v", linewidth=2, label="FN (Scams não detectados)")
    ax_in.set_title("Teste Interno", fontsize=12)
    ax_in.set_xlabel("% de dados de treino utilizados")
    ax_in.set_ylabel(unit)
    ax_in.legend()
    ax_in.grid(True, linestyle="--", alpha=0.4)

    # 2. Validação Externa
    ax_out = axes[1]
    if val_tp_col in df_lc.columns:
        ax_out.plot(pct, df_lc[val_tp_col], "g-o", linewidth=2, label="TP (Scam detectados)")
        ax_out.plot(pct, df_lc[val_tn_col], "b-s", linewidth=2, label="TN (Legítimos corretos)")
        ax_out.plot(pct, df_lc[val_fp_col], "m-^", linewidth=2, label="FP (Falsos alarmes)")
        ax_out.plot(pct, df_lc[val_fn_col], "r-v", linewidth=2, label="FN (Scams não detectados)")
    ax_out.set_title("Validação Externa", fontsize=12)
    ax_out.set_xlabel("% de dados de treino utilizados")
    ax_out.set_ylabel(unit)
    ax_out.legend()
    ax_out.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path, False)


def plot_confusion_matrices_by_fraction(
    df_lc: pd.DataFrame,
    title: str = "Matrizes de Confusão (Valores Absolutos) por Fração do Dataset",
    save_path: Optional[str] = None,
):
    """
    Exibe uma grade 2 x N de matrizes de confusão com valores absolutos (contagens) para cada fração do dataset.
    Linha 1: Teste Interno. Linha 2: Validação Externa.
    """
    n_fracs = len(df_lc)
    fig, axes = plt.subplots(2, n_fracs, figsize=(3.5 * n_fracs, 7))
    if n_fracs == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, (_, row) in enumerate(df_lc.iterrows()):
        frac_pct = int(row["frac"] * 100)
        
        # Teste Interno
        ax_in = axes[0, i]
        if "tp_mean" in row:
            tn, fp, fn, tp = int(round(row["tn_mean"])), int(round(row["fp_mean"])), int(round(row["fn_mean"])), int(round(row["tp_mean"]))
            cm_counts = np.array([[tn, fp], [fn, tp]])
            annot = np.array([
                [f"{tn}\n(TN)", f"{fp}\n(FP)"],
                [f"{fn}\n(FN)", f"{tp}\n(TP)"]
            ])
            sns.heatmap(cm_counts, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax_in)
        ax_in.set_title(f"Teste Interno ({frac_pct}%)")
        ax_in.set_xticklabels(["Legit", "Scam"])
        ax_in.set_yticklabels(["Legit", "Scam"])
        if i == 0:
            ax_in.set_ylabel("Real")
            
        # Validação Externa
        ax_out = axes[1, i]
        if "val_tp_mean" in row:
            val_tn, val_fp, val_fn, val_tp = int(round(row["val_tn_mean"])), int(round(row["val_fp_mean"])), int(round(row["val_fn_mean"])), int(round(row["val_tp_mean"]))
            cm_val_counts = np.array([[val_tn, val_fp], [val_fn, val_tp]])
            annot_val = np.array([
                [f"{val_tn}\n(TN)", f"{val_fp}\n(FP)"],
                [f"{val_fn}\n(FN)", f"{val_tp}\n(TP)"]
            ])
            sns.heatmap(cm_val_counts, annot=annot_val, fmt="", cmap="Oranges", cbar=False, ax=ax_out)
        ax_out.set_title(f"Validação ({frac_pct}%)")
        ax_out.set_xticklabels(["Legit", "Scam"])
        ax_out.set_yticklabels(["Legit", "Scam"])
        if i == 0:
            ax_out.set_ylabel("Real")
        ax_out.set_xlabel("Predito")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    _save_and_show(fig, save_path, False)



def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Curva Precision-Recall",
    threshold_marker: Optional[float] = None,
    ax=None,
    save_path: Optional[str] = None,
):
    """Curva Precision-Recall com marcador opcional do threshold escolhido."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.plot(recall, precision, "b-", lw=2, label="PR Curve")

    if threshold_marker is not None:
        # Encontra o ponto mais próximo do threshold
        diffs = np.abs(thresholds - threshold_marker)
        idx = np.argmin(diffs)
        ax.scatter(recall[idx], precision[idx], color="red", zorder=5, s=100,
                   label=f"Threshold={threshold_marker:.3f}\n(Recall={recall[idx]:.3f}, Prec={precision[idx]:.3f})")

    ax.set_title(title)
    ax.set_xlabel("Recall (Scam detectado)")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    _save_and_show(fig, save_path, ax_provided)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Curva ROC",
    ax=None,
    save_path: Optional[str] = None,
):
    """Curva ROC com AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("FPR (Falso Alarme)")
    ax.set_ylabel("TPR (Recall)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    _save_and_show(fig, save_path, ax_provided)


def plot_pca_variance(
    explained_variance: np.ndarray,
    dimensions: List[int] = None,
    title: str = "Variância Explicada pelo PCA",
    ax=None,
    save_path: Optional[str] = None,
):
    """Variância explicada acumulada por número de componentes PCA."""
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    cumvar = np.cumsum(explained_variance)
    x = dimensions if dimensions else range(1, len(cumvar) + 1)
    ax.plot(x, cumvar * 100, "o-", lw=2)
    ax.axhline(95, color="red", linestyle="--", alpha=0.6, label="95%")
    ax.axhline(90, color="orange", linestyle="--", alpha=0.6, label="90%")
    ax.set_title(title)
    ax.set_xlabel("Número de Componentes")
    ax.set_ylabel("Variância Explicada Acumulada (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    _save_and_show(fig, save_path, ax_provided)


def plot_dimension_study(
    df_dim: pd.DataFrame,
    metric: str = "val_f1_macro",
    title: str = "Estudo de Dimensionalidade (PCA)",
    ax=None,
    save_path: Optional[str] = None,
):
    """
    Plota o desempenho (ex: F1 Macro ou Recall) em função do número de dimensões PCA para cada embedding.
    """
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for emb in df_dim["embedding"].unique():
        df_emb = df_dim[df_dim["embedding"] == emb].sort_values("dim")
        if metric in df_emb.columns:
            ax.plot(df_emb["dim"], df_emb[metric], "o-", label=emb, lw=2)

    ax.set_title(f"{title}\n({metric})")
    ax.set_xlabel("Número de Dimensões (PCA)")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    _save_and_show(fig, save_path, ax_provided)


def plot_umap_2d(
    coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Visualização UMAP 2D",
    ax=None,
    save_path: Optional[str] = None,
):
    """Scatter UMAP 2D colorido por classe."""
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    if labels is not None:
        colors = ["steelblue" if l == 0 else "tomato" for l in labels]
        for label, color, name in [(0, "steelblue", "Ham"), (1, "tomato", "Scam")]:
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, alpha=0.4, s=8, label=name)
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.4, s=8)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    _save_and_show(fig, save_path, ax_provided)


def compare_results(r1, r2, metric: str = "f1_macro"):
    """Compara dois ExperimentResult lado a lado."""
    rows = []
    for r in [r1, r2]:
        if not r.df_metrics.empty and metric in r.df_metrics.columns:
            row = r.df_metrics.iloc[0].to_dict()
        else:
            row = {"model_name": str(r.config)}
        row["_source"] = str(r.config)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def evaluate_model_on_external(result, path_data: str, embedding: str):
    """
    Avalia o modelo de um ExperimentResult no dataset de validação externo.
    Retorna novo ExperimentResult com as métricas de validação.
    """
    from machine_learning.data import load_single_embedding
    from machine_learning.result import ExperimentResult

    strategy = result.strategy
    model = result.model
    config_val = {**result.config, "split": "validation", "dataset": "validation_external"}

    if strategy == "classical":
        X_val, y_val = load_single_embedding(
            split="validation",
            path_data=path_data,
            embedding=embedding,
            with_augmented=False,
        )
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        if y_prob is None and hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_val)

    elif strategy == "fcnn":
        import torch
        from machine_learning.data import load_concat_embeddings
        embedders = result.config.get("embedders")
        if isinstance(embedders, str):
            embedders = None if embedders == "all" else [embedders]
        elif embedders is None and embedding is not None:
            embedders = [embedding]
            
        X_val, y_val = load_concat_embeddings(
            split="validation", path_data=path_data, embedders=embedders, with_augmented=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_t)
            y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_pred = (y_prob >= result.threshold).astype(int)

    else:
        raise NotImplementedError(f"evaluate_on_external não implementado para strategy='{strategy}'")

    metrics = compute_metrics(y_val, y_pred, y_prob=y_prob,
                               model_name=config_val.get("model_name", ""),
                               dataset_name="validation_external")
    df_metrics = pd.DataFrame([metrics])

    return ExperimentResult(
        strategy=strategy,
        config=config_val,
        model=model,
        df_metrics=df_metrics,
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=result.threshold,
        artifacts_dir=result.artifacts_dir,
    )
