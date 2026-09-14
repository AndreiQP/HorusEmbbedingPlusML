"""
machine_learning/unsupervised/text_runner.py
------------------------------------------------
Runners para os modelos baseados em texto bruto + BGE: CVDD e DATE.
Seguem a mesma convenção de train_anomaly() em runner.py: split 3-way
(treino/validação/teste) dentro do dataset_train, treino somente com Ham, e
avaliação final na validação externa (dataset_validation), com AUROC/AUPRC
incluídos nas métricas.

Observação: diferente dos modelos em runner.py (que operam sobre embeddings
pré-computados + PCA/UMAP), CVDD/DATE consomem o texto bruto diretamente e usam
exclusivamente o embedding BGE (BAAI/bge-m3), conforme definido no planejamento.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..cache import ModelCache, _run_id
from ..evaluation import compute_metrics
from ..result import ExperimentResult
from .text_data import load_raw_text


def _split_ham(texts, y):
    return [t for t, label in zip(texts, y) if label == 0]


def _run_text_model(
    model_type: str,
    model_cls,
    test_size: float = 0.20,
    val_size: float = 0.10,
    force_retrain: bool = False,
    **model_kwargs,
) -> ExperimentResult:
    params_hash = ModelCache.params_hash(model_kwargs)
    config = {
        "strategy": "unsupervised", "embedding": "bge", "model_type": model_type,
        "test_size": test_size, "val_size": val_size,
        "params": "_".join(f"{k}={v}" for k, v in sorted(model_kwargs.items())),
    }
    run_id = _run_id(config)

    if not force_retrain and ModelCache.text_model_exists(model_type, "bge", params_hash):
        print(f"[unsupervised/text] Cache hit: {model_type}_bge (params={params_hash})")
        state = ModelCache.text_model_load(model_type, "bge", params_hash=params_hash)
        model = model_cls.load_state_dict(state)
        saved_metrics = ModelCache.load_metrics("unsupervised", run_id)
        y_val, y_pred_val, y_score_val = ModelCache.load_predictions("unsupervised", run_id, "validation_external")

        if saved_metrics:
            if isinstance(saved_metrics, dict):
                rows = list(saved_metrics.values()) if not ("dataset_name" in saved_metrics) else [saved_metrics]
                df_metrics = pd.DataFrame(rows)
            elif isinstance(saved_metrics, list):
                df_metrics = pd.DataFrame(saved_metrics)
            else:
                df_metrics = pd.DataFrame()
        else:
            df_metrics = pd.DataFrame()

        if df_metrics.empty or y_val is None:
            texts_val, y_val = load_raw_text("validation", with_augmented=False)
            y_score_val = model.score(texts_val)
            y_pred_val = model.predict(texts_val)
            metrics_val = compute_metrics(
                y_val, y_pred_val, y_score=y_score_val,
                model_name=f"{model_type.upper()}_bge", dataset_name="validation_external"
            )
            if df_metrics.empty:
                df_metrics = pd.DataFrame([metrics_val])
                ModelCache.save_metrics("unsupervised", run_id, {"validation_external": metrics_val})
            ModelCache.save_predictions("unsupervised", run_id, "validation_external", y_val, y_pred_val, y_score_val)

        return ExperimentResult(
            strategy="unsupervised", config=config, model=model, df_metrics=df_metrics,
            y_true=y_val, y_pred=y_pred_val, y_prob=y_score_val,
            artifacts_dir=ModelCache.artifacts_dir("unsupervised", run_id),
        )

    print(f"\n[unsupervised/text] Treinando {model_type.upper()} (BGE, texto bruto)")
    texts_all, y_all = load_raw_text("train", with_augmented=False)

    # Split 3-way estratificado: teste primeiro, depois treino/validação do restante
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        texts_all, y_all, test_size=test_size, random_state=42, stratify=y_all
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=val_size / (1 - test_size), random_state=42, stratify=y_tr_full
    )
    ham_tr = _split_ham(X_tr, y_tr)

    model = model_cls(**model_kwargs)
    model.fit(ham_tr)

    def _eval(texts, y, dataset_name):
        y_score = model.score(texts)
        y_pred = model.predict(texts)
        return compute_metrics(
            y, y_pred, y_score=y_score,
            model_name=f"{model_type.upper()}_bge", dataset_name=dataset_name
        ), y_pred, y_score

    metrics_tr, _, _ = _eval(X_tr, y_tr, "train")
    metrics_va, _, _ = _eval(X_va, y_va, "val")
    metrics_te, _, _ = _eval(X_te, y_te, "test")

    texts_val, y_val = load_raw_text("validation", with_augmented=False)
    metrics_val, y_pred_val, y_score_val = _eval(texts_val, y_val, "validation_external")

    print(f"  [train]              acc={metrics_tr['accuracy']:.4f}  f1_macro={metrics_tr['f1_macro']:.4f}  roc_auc={metrics_tr['roc_auc']}  pr_auc={metrics_tr['pr_auc']}")
    print(f"  [val interno]        acc={metrics_va['accuracy']:.4f}  f1_macro={metrics_va['f1_macro']:.4f}  roc_auc={metrics_va['roc_auc']}  pr_auc={metrics_va['pr_auc']}")
    print(f"  [test interno]       acc={metrics_te['accuracy']:.4f}  f1_macro={metrics_te['f1_macro']:.4f}  roc_auc={metrics_te['roc_auc']}  pr_auc={metrics_te['pr_auc']}")
    print(f"  [validação externa]  acc={metrics_val['accuracy']:.4f}  f1_macro={metrics_val['f1_macro']:.4f}  roc_auc={metrics_val['roc_auc']}  pr_auc={metrics_val['pr_auc']}")

    df_metrics = pd.DataFrame([metrics_tr, metrics_va, metrics_te, metrics_val])

    ModelCache.text_model_save(model.state_dict(), model_type, "bge", params_hash=params_hash)
    ModelCache.save_metrics("unsupervised", run_id, {"train": metrics_tr, "val": metrics_va, "test": metrics_te, "validation_external": metrics_val})
    ModelCache.save_predictions("unsupervised", run_id, "validation_external", y_val, y_pred_val, y_score_val)

    artifacts = ModelCache.artifacts_dir("unsupervised", run_id)
    result = ExperimentResult(
        strategy="unsupervised", config=config, model=model,
        df_metrics=df_metrics,
        y_true=y_val, y_pred=y_pred_val, y_prob=y_score_val,
        artifacts_dir=artifacts,
    )
    result.plot_confusion_matrix(title=f"{model_type.upper()} (BGE) — Validação")
    return result


def train_cvdd(test_size: float = 0.20, val_size: float = 0.10, force_retrain: bool = False, **model_kwargs) -> ExperimentResult:
    """Treina CVDD (Ruff et al., 2019) sobre texto bruto com BGE. Ver cvdd.py p/ hiperparâmetros."""
    from .cvdd import CVDD
    return _run_text_model("cvdd", CVDD, test_size=test_size, val_size=val_size, force_retrain=force_retrain, **model_kwargs)


def train_date(test_size: float = 0.20, val_size: float = 0.10, force_retrain: bool = False, **model_kwargs) -> ExperimentResult:
    """Treina DATE (Manolache et al., 2021) sobre texto bruto com BGE. Ver date_model.py p/ hiperparâmetros."""
    from .date_model import DATE
    return _run_text_model("date", DATE, test_size=test_size, val_size=val_size, force_retrain=force_retrain, **model_kwargs)
