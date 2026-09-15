"""
machine_learning/unsupervised/runner.py
-----------------------------------------
Runner para modelos de detecção de anomalias sobre embeddings + PCA/UMAP
(OCSVM, Isolation Forest, LOF, LUNAR, Deep SVDD).

Funções principais:
  train_anomaly()            -> treina um modelo + PCA/UMAP, retorna ExperimentResult
  grid_search_anomaly()      -> Grid Search de hiperparâmetros (genérico por model_type)
  learning_curve_anomaly()   -> curva de aprendizado por % do dataset
  compare_dimensions_anomaly()-> compara diferentes dimensões de PCA/UMAP

CVDD e DATE (modelos baseados em texto bruto + BGE) estão em cvdd.py e date_model.py.
"""
from __future__ import annotations
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from ..data import load_ham_only, load_single_embedding, get_shared_split_indices
from ..evaluation import compute_metrics
from ..cache import ModelCache, _run_id
from ..result import ExperimentResult

PCA_DIMS = [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]
FINAL_SEEDS = [42, 52, 62]
UNSUPERVISED_MODELS = ["ocsvm", "iforest", "lof", "lunar", "svdd"]
UNSUPERVISED_EMBEDDINGS = ["voyage", "openai", "e5", "bge", "minilm"]


def _build_anomaly_model(model_type: str, **kwargs):
    """Instancia o modelo de detecção de anomalias com os parâmetros padrão testados."""
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    if model_type == "ocsvm":
        return OneClassSVM(
            kernel=kwargs.get("kernel", "linear"),
            gamma=kwargs.get("gamma", 0.001),
            nu=kwargs.get("nu", 0.15),
        )
    if model_type == "iforest":
        return IsolationForest(
            n_estimators=kwargs.get("n_estimators", 500),
            contamination=kwargs.get("contamination", 0.1),
            random_state=kwargs.get("random_state", 42),
            n_jobs=kwargs.get("n_jobs", -1),
        )
    if model_type == "lof":
        try:
            from pyod.models.lof import LOF
        except ImportError:
            raise ImportError("Instale pyod: pip install pyod")
        return LOF(
            n_neighbors=kwargs.get("n_neighbors", 20),
            metric=kwargs.get("metric", "minkowski"),
            contamination=kwargs.get("contamination", 0.1),
            n_jobs=kwargs.get("n_jobs", -1),
        )
    if model_type == "lunar":
        try:
            from pyod.models.lunar import LUNAR
        except ImportError:
            raise ImportError("Instale pyod (com suporte a torch): pip install pyod")
        return LUNAR(
            model_type=kwargs.get("lunar_model_type", "WEIGHT"),
            n_neighbours=kwargs.get("n_neighbors", 5),
            epsilon=kwargs.get("epsilon", 0.1),
            lr=kwargs.get("lr", 0.001),
            wd=kwargs.get("weight_decay", 0.1),
            n_epochs=kwargs.get("n_epochs", 200),
            contamination=kwargs.get("contamination", 0.1),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=kwargs.get("random_state", 42),
        )
    if model_type == "svdd":
        from .svdd_torch import DeepSVDD
        try:
            import torch
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            default_device = "cpu"
        return DeepSVDD(
            latent_dim=kwargs.get("latent_dim", 32),
            hidden_dims=kwargs.get("hidden_dims", [128, 64]),
            nu=kwargs.get("nu", 0.1),
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-6),
            n_epochs=kwargs.get("n_epochs", 100),
            batch_size=kwargs.get("batch_size", 128),
            random_state=kwargs.get("random_state", 42),
            device=kwargs.get("device", default_device),
        )

    raise ValueError(
        f"model_type deve ser um de 'ocsvm', 'iforest', 'lof', 'lunar', 'svdd', recebido: '{model_type}'"
    )


# Modelos com convenção sklearn: predict -> +1(normal)/-1(anomalia); decision_function -> maior=normal
_SKLEARN_CONVENTION = {"ocsvm", "iforest", "svdd"}
# Modelos com convenção PyOD: predict -> 0(normal)/1(anomalia); decision_function -> maior=anomalia
_PYOD_CONVENTION = {"lof", "lunar"}


def _build_reducer(reducer: str, dim: int, **kwargs):
    """Instancia PCA ou UMAP com a dimensionalidade desejada."""
    if reducer == "pca":
        return PCA(n_components=dim, random_state=42)
    if reducer == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("Instale umap-learn: pip install umap-learn")
        return umap.UMAP(
            n_components=dim,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            random_state=42,
        )
    raise ValueError(f"reducer deve ser 'pca' ou 'umap', recebido: '{reducer}'")


def _anomaly_predict(model, X: np.ndarray, model_type: str = "ocsvm") -> np.ndarray:
    """
    Converte a saída do modelo para labels binários (0=Ham, 1=Scam), tratando as
    duas convenções existentes:
      - sklearn (OCSVM/IForest/SVDD): +1=normal/Ham, -1=anomalia/Scam
      - PyOD (LOF/LUNAR):              0=normal/Ham,  1=anomalia/Scam
    """
    raw = model.predict(X)
    if model_type in _PYOD_CONVENTION:
        return np.asarray(raw).astype(int)
    return np.where(raw == -1, 1, 0)


def _anomaly_score(model, X: np.ndarray, model_type: str = "ocsvm") -> np.ndarray:
    """
    Score contínuo de anomalia (maior = mais provável Scam), usado para AUROC/AUPRC.
    Unifica a convenção de decision_function entre sklearn (maior=normal) e PyOD (maior=anomalia).
    """
    if model_type in _PYOD_CONVENTION:
        return np.asarray(model.decision_function(X))
    return -np.asarray(model.decision_function(X))


def train_anomaly(
    embedding: str,
    model_type: str = "ocsvm",
    pca_dim: int = 500,
    path_data: str = "all_data",
    reducer: str = "pca",
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    split_seed: int = 42,
    force_retrain: bool = False,
    **model_kwargs,
) -> ExperimentResult:
    """
    Treina um modelo de detecção de anomalias (OCSVM, IForest, LOF, LUNAR ou SVDD) com redução dimensional.

    - Realiza split 3-way (treino / validação / teste, ex: 70%/10%/20%) dentro do
      dataset_train, na mesma convenção usada por classical/fcnn/transformer.
    Esta função é a etapa final, posterior ao grid: ajusta scaler, redutor e modelo
    nas amostras Ham de treino + validação interna e avalia o mesmo artefato no
    teste interno e no dataset_validation. O grid é responsabilidade exclusiva de
    grid_search_anomaly().
    """
    config = {
        "strategy": "unsupervised",
        "embedding": embedding, "model_type": model_type,
        "pca_dim": pca_dim, "reducer": reducer, "path_data": path_data,
        "test_size": test_size, "val_size": val_size,
        "seed": seed, "split_seed": split_seed,
        "params": "_".join(f"{k}={v}" for k, v in sorted(model_kwargs.items())),
    }
    run_id = _run_id(config)
    cache_params = {
        **model_kwargs, "seed": seed, "split_seed": split_seed,
        "test_size": test_size, "val_size": val_size, "path_data": path_data,
    }
    params_hash = ModelCache.params_hash(cache_params)

    # ── Cache check ──────────────────────────────────────────────
    # O cache leve é suficiente para análise/notebook e pode vir do Git sem os
    # artefatos pesados do modelo. Um retrain só é necessário se esses resultados
    # também estiverem ausentes ou se force_retrain=True.
    if not force_retrain:
        saved_metrics = ModelCache.load_metrics("unsupervised", run_id)
        y_val, y_pred_val, y_score_val = ModelCache.load_predictions(
            "unsupervised", run_id, "validation_external"
        )
        y_test, _, _ = ModelCache.load_predictions("unsupervised", run_id, "test_internal")
        if saved_metrics and y_val is not None and y_test is not None:
            print(f"[unsupervised] Cache leve: {embedding}_{model_type}_{pca_dim} (seed={seed})")
            return ExperimentResult(
                strategy="unsupervised", config=config, model=None,
                df_metrics=pd.DataFrame(list(saved_metrics.values())),
                y_true=y_val, y_pred=y_pred_val, y_prob=y_score_val,
                artifacts_dir=ModelCache.artifacts_dir("unsupervised", run_id),
            )

    # ── Dados de treino/val/teste ────────────────────────────────
    print(f"\n[unsupervised] Treinando: {embedding}_{model_type}_{pca_dim}d_{reducer}")
    X_all, y_all, sample_ids = load_single_embedding(
        "train", path_data, embedding, with_augmented=False, return_ids=True
    )
    idx_tr, idx_va, idx_te = get_shared_split_indices(
        "unsupervised", sample_ids, y_all, test_size, val_size, split_seed
    )
    X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
    X_va, y_va = X_all[idx_va], y_all[idx_va]
    X_te, y_te = X_all[idx_te], y_all[idx_te]

    # Finalista: refit em treino + validação interna, somente com Ham.
    X_fit = np.concatenate([X_tr, X_va])
    y_fit = np.concatenate([y_tr, y_va])
    X_ham_fit = X_fit[y_fit == 0]

    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # ── Redução dimensional (fit no Ham do treino) ───────────────
    scaler = StandardScaler()
    X_ham_tr_scaled = scaler.fit_transform(X_ham_fit)

    effective_dim = min(pca_dim, len(X_ham_fit), X_ham_fit.shape[1])
    reducer_obj = _build_reducer(reducer, effective_dim)
    X_ham_tr_reduced = reducer_obj.fit_transform(X_ham_tr_scaled)

    # ── Treino do Modelo de Anomalias ─────────────────────────────
    print(f"[unsupervised] Fit params ({model_type}/{embedding}): {model_kwargs or 'defaults'}")
    fit_kwargs = dict(model_kwargs)
    fit_kwargs.setdefault("random_state", seed)
    anom_model = _build_anomaly_model(model_type, **fit_kwargs)
    anom_model.fit(X_ham_tr_reduced)

    # ── Avaliação 1: Treino (Ham + Scam do split de treino) ──────
    X_tr_red     = reducer_obj.transform(scaler.transform(X_tr))
    y_pred_tr    = _anomaly_predict(anom_model, X_tr_red, model_type)
    y_score_tr   = _anomaly_score(anom_model, X_tr_red, model_type)
    metrics_tr   = compute_metrics(y_tr, y_pred_tr, y_score=y_score_tr, model_name=f"{model_type.upper()}_{embedding}", dataset_name="train")

    # ── Avaliação 2: Teste Interno (holdout final, dentro do dataset_train) ─
    X_te_red     = reducer_obj.transform(scaler.transform(X_te))
    y_pred_te    = _anomaly_predict(anom_model, X_te_red, model_type)
    y_score_te   = _anomaly_score(anom_model, X_te_red, model_type)
    metrics_te   = compute_metrics(y_te, y_pred_te, y_score=y_score_te, model_name=f"{model_type.upper()}_{embedding}", dataset_name="test")

    # ── Avaliação 3: Validação Externa (dataset_validation — vida real) ─
    X_val, y_val = load_single_embedding("validation", path_data, embedding, with_augmented=False)
    X_val_red    = reducer_obj.transform(scaler.transform(X_val))
    y_pred_val   = _anomaly_predict(anom_model, X_val_red, model_type)
    y_score_val  = _anomaly_score(anom_model, X_val_red, model_type)
    metrics_val  = compute_metrics(y_val, y_pred_val, y_score=y_score_val, model_name=f"{model_type.upper()}_{embedding}", dataset_name="validation_external")

    print(f"  [train]              acc={metrics_tr['accuracy']:.4f}  f1_macro={metrics_tr['f1_macro']:.4f}  recall_scam={metrics_tr['recall_scam']:.4f}  roc_auc={metrics_tr['roc_auc']}  pr_auc={metrics_tr['pr_auc']}")
    print(f"  [test interno]       acc={metrics_te['accuracy']:.4f}  f1_macro={metrics_te['f1_macro']:.4f}  recall_scam={metrics_te['recall_scam']:.4f}  roc_auc={metrics_te['roc_auc']}  pr_auc={metrics_te['pr_auc']}")
    print(f"  [validação externa]  acc={metrics_val['accuracy']:.4f}  f1_macro={metrics_val['f1_macro']:.4f}  recall_scam={metrics_val['recall_scam']:.4f}  roc_auc={metrics_val['roc_auc']}  pr_auc={metrics_val['pr_auc']}")

    df_metrics = pd.DataFrame([metrics_tr, metrics_te, metrics_val])

    # ── Variância PCA (para plot) ─────────────────────────────────
    extra = {"pca_reducer": reducer_obj, "scaler": scaler}
    if reducer == "pca" and hasattr(reducer_obj, "explained_variance_ratio_"):
        extra["pca_variance"] = reducer_obj.explained_variance_ratio_

    # ── Salvar ────────────────────────────────────────────────────
    ModelCache.unsupervised_save(anom_model, reducer_obj, scaler, embedding, model_type, pca_dim, reducer, params_hash)
    ModelCache.save_metrics("unsupervised", run_id, {"train": metrics_tr, "test": metrics_te, "validation_external": metrics_val})
    ModelCache.save_predictions("unsupervised", run_id, "test_internal", y_te, y_pred_te, y_score_te)
    ModelCache.save_predictions("unsupervised", run_id, "validation_external", y_val, y_pred_val, y_score_val)

    artifacts = ModelCache.artifacts_dir("unsupervised", run_id)
    result = ExperimentResult(
        strategy="unsupervised", config=config, model=anom_model,
        df_metrics=df_metrics,
        y_true=y_val, y_pred=y_pred_val, y_prob=y_score_val,
        extra=extra,
        artifacts_dir=artifacts,
    )
    result.plot_confusion_matrix(title=f"{model_type.upper()} {embedding} {pca_dim}d — Validação")
    return result


def learning_curve_anomaly(
    embedding: str,
    model_type: str = "ocsvm",
    pca_dim: int = 500,
    train_sizes: Optional[List[float]] = None,
    n_repeats: int = 3,
    path_data: str = "all_data",
    reducer: str = "pca",
    force_recompute: bool = False,
    **model_kwargs,
) -> ExperimentResult:
    """
    Curva de aprendizado por % do dataset.
    
    Para cada fração: treina com frac% das amostras Ham e avalia com todas as classes.
    
    Returns:
        ExperimentResult com df_history contendo colunas:
            frac, n_ham_samples, f1_macro_mean, f1_macro_std, val_f1_mean, val_f1_std,
            recall_scam_mean, recall_scam_std, val_recall_scam_mean, val_recall_scam_std
    """
    train_sizes = train_sizes or [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    config = {
        "strategy": "unsupervised_lc",
        "embedding": embedding, "model_type": model_type,
        "pca_dim": pca_dim, "reducer": reducer, "train_sizes": str(train_sizes),
    }
    run_id = _run_id(config)

    if not force_recompute:
        saved = ModelCache.load_history("unsupervised", f"lc_{run_id}")
        if saved is not None:
            print(f"[unsupervised_lc] Cache hit — curva de aprendizado")
            return ExperimentResult(
                strategy="unsupervised", config=config, df_history=saved,
                artifacts_dir=ModelCache.artifacts_dir("unsupervised", f"lc_{run_id}"),
            )

    # ── Dados ────────────────────────────────────────────────────
    X_ham = load_ham_only("train", path_data, embedding)
    X_tr, y_tr = load_single_embedding("train", path_data, embedding, with_augmented=False)
    X_val, y_val = load_single_embedding("validation", path_data, embedding, with_augmented=False)

    # Teste interno: 20% FIXO do dataset de treino (Ham + Scam)
    _, X_test, _, y_test = train_test_split(X_tr, y_tr, test_size=0.20, random_state=42, stratify=y_tr)

    # Scaler e Redutor globais (fit no Ham completo)
    scaler = StandardScaler()
    X_ham_scaled = scaler.fit_transform(X_ham)

    effective_dim = min(pca_dim, len(X_ham), X_ham.shape[1])
    reducer_obj = _build_reducer(reducer, effective_dim)
    X_ham_red_all = reducer_obj.fit_transform(X_ham_scaled)

    X_test_red = reducer_obj.transform(scaler.transform(X_test))
    X_val_red  = reducer_obj.transform(scaler.transform(X_val))

    n_test = len(y_test)
    n_val = len(y_val)
    rows = []
    for frac in train_sizes:
        n_ham = max(4, int(len(X_ham) * frac))
        f1s, f1_vals, rec_scam, rec_scam_val = [], [], [], []
        roc_aucs, roc_aucs_val, pr_aucs, pr_aucs_val = [], [], [], []
        tns, fps, fns, tps = [], [], [], []
        val_tns, val_fps, val_fns, val_tps = [], [], [], []

        for seed in range(n_repeats):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_ham), size=n_ham, replace=False)
            X_sub_red = X_ham_red_all[idx]

            # Treino do modelo de detecção de anomalias na fração do Ham
            model = _build_anomaly_model(model_type, **model_kwargs)
            model.fit(X_sub_red)

            # Teste interno (FIXO)
            y_pred_test = _anomaly_predict(model, X_test_red, model_type)
            y_score_test = _anomaly_score(model, X_test_red, model_type)
            m_test = compute_metrics(y_test, y_pred_test, y_score=y_score_test)
            f1s.append(m_test["f1_macro"])
            rec_scam.append(m_test["recall_scam"])
            roc_aucs.append(m_test["roc_auc"] or 0.0)
            pr_aucs.append(m_test["pr_auc"] or 0.0)
            tns.append(m_test["TN"]); fps.append(m_test["FP"]); fns.append(m_test["FN"]); tps.append(m_test["TP"])

            # Validação externa
            y_pred_val = _anomaly_predict(model, X_val_red, model_type)
            y_score_val = _anomaly_score(model, X_val_red, model_type)
            m_val = compute_metrics(y_val, y_pred_val, y_score=y_score_val)
            f1_vals.append(m_val["f1_macro"])
            rec_scam_val.append(m_val["recall_scam"])
            roc_aucs_val.append(m_val["roc_auc"] or 0.0)
            pr_aucs_val.append(m_val["pr_auc"] or 0.0)
            val_tns.append(m_val["TN"]); val_fps.append(m_val["FP"]); val_fns.append(m_val["FN"]); val_tps.append(m_val["TP"])

        tn_m, fp_m, fn_m, tp_m = np.mean(tns), np.mean(fps), np.mean(fns), np.mean(tps)
        val_tn_m, val_fp_m, val_fn_m, val_tp_m = np.mean(val_tns), np.mean(val_fps), np.mean(val_fns), np.mean(val_tps)

        rows.append({
            "frac":               frac,
            "n_ham_samples":      n_ham,
            "f1_macro_mean":      np.mean(f1s),
            "f1_macro_std":       np.std(f1s),
            "val_f1_mean":        np.mean(f1_vals),
            "val_f1_macro_mean":  np.mean(f1_vals),
            "val_f1_std":         np.std(f1_vals),
            "recall_scam_mean":   np.mean(rec_scam),
            "recall_scam_std":    np.std(rec_scam),
            "val_recall_scam_mean": np.mean(rec_scam_val),
            "val_recall_scam_std":  np.std(rec_scam_val),
            "roc_auc_mean":       np.mean(roc_aucs),
            "roc_auc_std":        np.std(roc_aucs),
            "val_roc_auc_mean":   np.mean(roc_aucs_val),
            "val_roc_auc_std":    np.std(roc_aucs_val),
            "pr_auc_mean":        np.mean(pr_aucs),
            "pr_auc_std":         np.std(pr_aucs),
            "val_pr_auc_mean":    np.mean(pr_aucs_val),
            "val_pr_auc_std":     np.std(pr_aucs_val),
            "tn_mean": tn_m, "fp_mean": fp_m, "fn_mean": fn_m, "tp_mean": tp_m,
            "tn_pct": (tn_m / n_test) * 100,
            "fp_pct": (fp_m / n_test) * 100,
            "fn_pct": (fn_m / n_test) * 100,
            "tp_pct": (tp_m / n_test) * 100,
            "val_tn_mean": val_tn_m, "val_fp_mean": val_fp_m, "val_fn_mean": val_fn_m, "val_tp_mean": val_tp_m,
            "val_tn_pct": (val_tn_m / n_val) * 100,
            "val_fp_pct": (val_fp_m / n_val) * 100,
            "val_fn_pct": (val_fn_m / n_val) * 100,
            "val_tp_pct": (val_tp_m / n_val) * 100,
        })
        print(f"  [lc] frac={frac:.0%} n_ham={n_ham} -> f1={np.mean(f1s):.4f} | val_f1={np.mean(f1_vals):.4f} | val_roc_auc={np.mean(roc_aucs_val):.4f} | val_pr_auc={np.mean(pr_aucs_val):.4f}")

    df_history = pd.DataFrame(rows)
    ModelCache.save_history("unsupervised", f"lc_{run_id}", df_history)

    artifacts = ModelCache.artifacts_dir("unsupervised", f"lc_{run_id}")
    result = ExperimentResult(
        strategy="unsupervised", config=config, df_history=df_history, artifacts_dir=artifacts
    )
    result.plot_dataset_size_curve(metric="val_f1_mean", save=True)
    result.plot_confusion_matrix_evolution(save=True)
    result.plot_confusion_matrices_by_fraction(save=True)
    return result


def _default_param_grid(model_type: str) -> Dict[str, List]:
    """Grade de hiperparâmetros padrão por modelo, usada quando `param_grid` não é informado."""
    if model_type == "ocsvm":
        return {"kernel": ["linear", "rbf", "poly"], "nu": [0.05, 0.10, 0.15, 0.20], "gamma": ["scale", "auto", 0.001]}
    if model_type == "iforest":
        return {"n_estimators": [200, 500, 800], "contamination": [0.05, 0.10, 0.15, 0.20]}
    if model_type == "lof":
        return {"n_neighbors": [10, 20, 35, 50], "contamination": [0.05, 0.10, 0.15, 0.20]}
    if model_type == "lunar":
        return {"n_neighbors": [5, 10, 20], "epsilon": [0.05, 0.1, 0.2], "contamination": [0.05, 0.10, 0.15]}
    if model_type == "svdd":
        return {"nu": [0.05, 0.10, 0.15, 0.20], "latent_dim": [16, 32, 64], "lr": [1e-3, 5e-4]}
    raise ValueError(f"Sem grade padrão para model_type='{model_type}'")


def _configuration_complexity(model_type: str, params: dict) -> float:
    """Escalar determinístico usado apenas no último critério de desempate."""
    if model_type == "ocsvm":
        return {"linear": 0.0, "rbf": 1.0, "poly": 2.0}.get(str(params.get("kernel")), 3.0)
    if model_type == "iforest":
        return float(params.get("n_estimators", 0))
    if model_type in {"lof", "lunar"}:
        return float(params.get("n_neighbors", 0))
    if model_type == "svdd":
        return float(params.get("latent_dim", 0))
    return 0.0


def grid_search_anomaly(
    embedding: str,
    model_type: str = "ocsvm",
    pca_dim: int = 1000,
    pca_dims: Optional[List[int]] = None,
    path_data: str = "all_data",
    param_grid: Optional[Dict[str, List]] = None,
    test_size: float = 0.20,
    val_size: float = 0.10,
    force_recompute: bool = False,
) -> ExperimentResult:
    """
    Grid Search de hiperparâmetros para qualquer modelo suportado
    ('ocsvm', 'iforest', 'lof', 'lunar', 'svdd').

    A seleção cruza dimensão PCA e hiperparâmetros usando exclusivamente a
    validação interna. Teste interno e dataset_validation não são carregados
    durante o grid search.

    Args:
        param_grid: dict {hiperparametro: [valores]}. Se None, usa a grade padrão do model_type.

    Returns:
        ExperimentResult com df_history contendo resultados de todas as combinações
        (colunas internal_val_* usadas para ranquear) e df_metrics com o melhor combo.
    """
    import itertools

    param_grid = param_grid or _default_param_grid(model_type)
    pca_dims = pca_dims or [pca_dim]
    grid_token = repr(sorted((key, tuple(str(value) for value in values)) for key, values in param_grid.items()))
    config = {
        "strategy": "unsupervised_gs",
        "embedding": embedding, "model_type": model_type, "pca_dims": str(pca_dims),
        "path_data": path_data, "reducer": "pca", "split_seed": 42,
        "test_size": test_size, "val_size": val_size,
        "param_grid": grid_token,
    }
    run_id = _run_id(config)

    if not force_recompute:
        saved = ModelCache.load_history("unsupervised", f"gs_{run_id}")
        if saved is not None and not saved.empty:
            print("[unsupervised_gs] Cache hit — grid search")
            best_saved = saved.sort_values(
                ["internal_val_f1_macro", "internal_val_pr_auc", "internal_val_recall_scam", "pca_dim", "_complexity"],
                ascending=[False, False, False, True, True],
            ).iloc[0]
            return ExperimentResult(
                strategy="unsupervised", config=config, df_history=saved,
                df_metrics=pd.DataFrame([best_saved.to_dict()]),
                artifacts_dir=ModelCache.artifacts_dir("unsupervised", f"gs_{run_id}"),
            )

    param_names = list(param_grid.keys())

    X_all, y_all, sample_ids = load_single_embedding(
        "train", path_data, embedding, with_augmented=False, return_ids=True
    )
    idx_tr, idx_va, _ = get_shared_split_indices(
        "unsupervised", sample_ids, y_all, test_size, val_size, 42
    )
    X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
    X_va, y_va = X_all[idx_va], y_all[idx_va]
    X_ham_tr = X_tr[y_tr == 0]
    scaler = StandardScaler()
    X_ham_sc = scaler.fit_transform(X_ham_tr)

    rows = []
    param_combos = list(itertools.product(*param_grid.values()))
    if model_type == "ocsvm":
        param_combos = [
            combo for combo in param_combos
            if not (
                dict(zip(param_names, combo)).get("kernel") == "linear"
                and dict(zip(param_names, combo)).get("gamma") != "scale"
            )
        ]
    total = len(pca_dims) * len(param_combos)
    completed = 0
    for dim in pca_dims:
        effective_dim = min(dim, len(X_ham_tr), X_ham_tr.shape[1])
        reducer = PCA(n_components=effective_dim, random_state=42)
        X_ham_red = reducer.fit_transform(X_ham_sc)
        X_va_red = reducer.transform(scaler.transform(X_va))
        for combo in param_combos:
            kwargs = dict(zip(param_names, combo))
            completed += 1
            try:
                np.random.seed(42)
                try:
                    import torch
                    torch.manual_seed(42)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(42)
                except ImportError:
                    pass
                model = _build_anomaly_model(model_type, **kwargs)
                model.fit(X_ham_red)
                y_pred_va = _anomaly_predict(model, X_va_red, model_type)
                y_score_va = _anomaly_score(model, X_va_red, model_type)
                m_va = compute_metrics(y_va, y_pred_va, y_score=y_score_va)
                row = {
                    "pca_dim": dim,
                    "_complexity": _configuration_complexity(model_type, kwargs),
                    **{k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in kwargs.items()},
                }
                row.update({
                    "internal_val_f1_macro": m_va["f1_macro"],
                    "internal_val_recall_scam": m_va["recall_scam"],
                    "internal_val_roc_auc": m_va["roc_auc"],
                    "internal_val_pr_auc": m_va["pr_auc"],
                })
                rows.append(row)
                if completed % 10 == 0:
                    print(f"  [gs] {completed}/{total} | dim={dim} {kwargs} -> internal_val_f1_macro={m_va['f1_macro']:.4f}")
            except Exception as e:
                print(f"  [gs] Erro: dim={dim} {kwargs} -> {e}")

    if not rows:
        raise RuntimeError(
            f"grid_search_anomaly: todas as combinações falharam para "
            f"model_type='{model_type}' embedding='{embedding}' — veja os erros '[gs] Erro: ...' acima."
        )

    df_gs = pd.DataFrame(rows).sort_values(
        ["internal_val_f1_macro", "internal_val_pr_auc", "internal_val_recall_scam", "pca_dim", "_complexity"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    best = df_gs.iloc[0]
    best_params = {k: best[k] for k in param_names}
    print(f"\n[gs] Melhores parâmetros ({model_type}), escolhidos pela validação interna: {best_params}")
    print(f"     dim={int(best['pca_dim'])} internal_val_f1_macro={best['internal_val_f1_macro']:.4f}")

    ModelCache.save_history("unsupervised", f"gs_{run_id}", df_gs)

    artifacts = ModelCache.artifacts_dir("unsupervised", f"gs_{run_id}")
    return ExperimentResult(
        strategy="unsupervised", config=config, df_history=df_gs,
        df_metrics=pd.DataFrame([best.to_dict()]),
        artifacts_dir=artifacts,
    )


def _plain_value(value):
    """Converte escalares numpy/pandas em valores serializáveis e estáveis."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _pipeline_summary_path(model_type: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "experiment_results", "unsupervised", "pipeline", model_type, "summary.json",
    )


def finalize_anomaly_model(
    model_type: str,
    embeddings: Optional[List[str]] = None,
    pca_dims: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    path_data: str = "all_data",
    force_retrain: bool = False,
) -> dict:
    """Materializa e resume os cinco finalistas de um algoritmo.

    O grid já deve existir no cache. Para cada embedding, esta função lê o
    vencedor da validação interna, treina/refaz as três seeds e agrega teste
    interno e dataset_validation. O campeão é definido por val_f1_macro médio.
    """
    embeddings = embeddings or UNSUPERVISED_EMBEDDINGS
    pca_dims = pca_dims or PCA_DIMS
    seeds = seeds or FINAL_SEEDS
    finalists = []

    for embedding in embeddings:
        grid_result = grid_search_anomaly(
            embedding=embedding, model_type=model_type, pca_dims=pca_dims,
            path_data=path_data, force_recompute=force_retrain,
        )
        best = grid_result.df_metrics.iloc[0].to_dict()
        dim = int(best.pop("pca_dim"))
        param_names = _default_param_grid(model_type).keys()
        integer_params = {"n_neighbors", "n_estimators", "n_epochs", "latent_dim", "batch_size", "warmup_epochs"}
        params = {
            name: (int(best[name]) if name in integer_params else _plain_value(best[name]))
            for name in param_names
        }
        seed_rows = []

        for seed in seeds:
            result = train_anomaly(
                embedding=embedding, model_type=model_type, pca_dim=dim,
                path_data=path_data, seed=seed, force_retrain=force_retrain,
                **params,
            )
            test_metrics = _get_metrics_by_split(result.df_metrics, "test")
            val_metrics = _get_metrics_by_split(result.df_metrics, "validation_external")
            seed_rows.append({
                "seed": seed,
                "run_id": _run_id(result.config),
                "test": {k: _plain_value(v) for k, v in test_metrics.items()},
                "validation": {k: _plain_value(v) for k, v in val_metrics.items()},
            })

        def aggregate(split: str, metric: str) -> Tuple[float, float]:
            values = [float(row[split][metric]) for row in seed_rows]
            return float(np.mean(values)), float(np.std(values))

        test_f1_mean, test_f1_std = aggregate("test", "f1_macro")
        val_f1_mean, val_f1_std = aggregate("validation", "f1_macro")
        val_pr_mean, _ = aggregate("validation", "pr_auc")
        val_recall_mean, _ = aggregate("validation", "recall_scam")
        finalists.append({
            "embedding": embedding,
            "pca_dim": dim,
            "params": params,
            "internal_val_f1_macro": _plain_value(best["internal_val_f1_macro"]),
            "test_f1_macro_mean": test_f1_mean,
            "test_f1_macro_std": test_f1_std,
            "val_f1_macro_mean": val_f1_mean,
            "val_f1_macro_std": val_f1_std,
            "val_pr_auc_mean": val_pr_mean,
            "val_recall_scam_mean": val_recall_mean,
            "grid": json.loads(grid_result.df_history.to_json(orient="records")),
            "seeds": seed_rows,
        })

    finalists.sort(
        key=lambda row: (
            -row["val_f1_macro_mean"], -row["val_pr_auc_mean"],
            -row["val_recall_scam_mean"], row["pca_dim"],
        )
    )
    summary = {
        "protocol_version": 2,
        "git_commit": ModelCache.git_commit(),
        "selection": {
            "hyperparameters": "internal_val_f1_macro",
            "embedding": "val_f1_macro_mean",
            "tie_breakers": ["pr_auc", "recall_scam", "lower_complexity"],
        },
        "model_type": model_type,
        "pca_dims": pca_dims,
        "seeds": seeds,
        "finalists": finalists,
        "winner": finalists[0],
    }
    path = _pipeline_summary_path(model_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[unsupervised] Resumo final salvo: {path}")
    return summary


def load_anomaly_summary(model_type: str) -> dict:
    """Loader estritamente cache-only usado pelo notebook."""
    path = _pipeline_summary_path(model_type)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache final ausente para {model_type}: {path}. "
            "Execute backend/slurm/submit_full_pipeline.sh --unsupervised."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_unsupervised_leaderboard() -> pd.DataFrame:
    """Compara os campeões dos cinco algoritmos usando somente seus resumos."""
    rows = []
    for model_type in UNSUPERVISED_MODELS:
        winner = load_anomaly_summary(model_type)["winner"]
        rows.append({"model_type": model_type, **{
            key: winner[key] for key in (
                "embedding", "pca_dim", "val_f1_macro_mean", "val_f1_macro_std",
                "val_pr_auc_mean", "val_recall_scam_mean",
            )
        }})
    return pd.DataFrame(rows).sort_values(
        ["val_f1_macro_mean", "val_pr_auc_mean", "val_recall_scam_mean", "pca_dim"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def finalize_unsupervised_pipeline() -> dict:
    """Consolida os cinco campeões depois que os jobs por algoritmo terminarem."""
    leaderboard = build_unsupervised_leaderboard()
    summary = {
        "protocol_version": 2,
        "git_commit": ModelCache.git_commit(),
        "selection_metric": "val_f1_macro_mean",
        "leaderboard": json.loads(leaderboard.to_json(orient="records")),
        "winner": json.loads(leaderboard.iloc[0].to_json()),
    }
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "experiment_results", "unsupervised", "pipeline_summary.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[unsupervised] Consolidação final salva: {path}")
    return summary


def _get_metrics_by_split(df_metrics: pd.DataFrame, split: str = "test") -> dict:
    """Extrai métricas da partição desejada ('test', 'validation' ou 'train')."""
    if df_metrics.empty:
        return {}
    if "dataset_name" in df_metrics.columns:
        sub = df_metrics[df_metrics["dataset_name"] == split]
        if not sub.empty:
            return sub.iloc[0].to_dict()
    return df_metrics.iloc[-1].to_dict()


def train_all_anomaly(
    model_type: str = "ocsvm",
    pca_dim: int = 500,
    embeddings: Optional[List[str]] = None,
    path_data: str = "all_data",
    reducer: str = "pca",
    force_retrain: bool = False,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Treina modelo de detecção de anomalias para TODOS os embeddings fornecidos.
    
    Returns:
        pd.DataFrame comparativo contendo métricas do teste interno (20% holdout) e da validação externa.
    """
    from ..data import EMBEDDERS as ALL_EMBEDDERS
    embeddings = embeddings or list(ALL_EMBEDDERS.keys())
    rows = []
    for emb in embeddings:
        res = train_anomaly(
            embedding=emb, model_type=model_type, pca_dim=pca_dim,
            path_data=path_data, reducer=reducer, force_retrain=force_retrain, **model_kwargs
        )
        test_m = _get_metrics_by_split(res.df_metrics, "test")
        val_m  = _get_metrics_by_split(res.df_metrics, "validation_external")
        rows.append({
            "embedding": emb,
            "model_type": model_type.upper(),
            "pca_dim": pca_dim,
            # Métricas do Teste Interno (20% Holdout)
            "test_acc": test_m.get("accuracy", 0.0),
            "test_f1_macro": test_m.get("f1_macro", 0.0),
            "test_f1_scam": test_m.get("f1_scam", 0.0),
            "test_recall_scam": test_m.get("recall_scam", 0.0),
            "test_precision_scam": test_m.get("precision_scam", 0.0),
            "test_roc_auc": test_m.get("roc_auc"),
            "test_pr_auc": test_m.get("pr_auc"),
            # Métricas da Validação Externa
            "val_acc": val_m.get("accuracy", 0.0),
            "val_f1_macro": val_m.get("f1_macro", 0.0),
            "val_f1_scam": val_m.get("f1_scam", 0.0),
            "val_recall_scam": val_m.get("recall_scam", 0.0),
            "val_precision_scam": val_m.get("precision_scam", 0.0),
            "val_roc_auc": val_m.get("roc_auc"),
            "val_pr_auc": val_m.get("pr_auc"),
        })
    return pd.DataFrame(rows)


def compare_dimensions_anomaly(
    model_type: str = "ocsvm",
    embeddings: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    path_data: str = "all_data",
    reducer: str = "pca",
    split: str = "test",
    force_retrain: bool = False,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Estudo de Dimensionalidade (PCA/UMAP): compara métricas do conjunto especificado (default='test')
    ao variar o número de dimensões (ex: [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]).
    """
    from ..data import EMBEDDERS as ALL_EMBEDDERS
    from ..evaluation import plot_dimension_study
    embeddings = embeddings or list(ALL_EMBEDDERS.keys())
    dimensions = dimensions or [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]

    rows = []
    for emb in embeddings:
        for dim in dimensions:
            try:
                res = train_anomaly(
                    embedding=emb,
                    model_type=model_type,
                    pca_dim=dim,
                    path_data=path_data,
                    reducer=reducer,
                    force_retrain=force_retrain,
                    **model_kwargs,
                )
                test_m = _get_metrics_by_split(res.df_metrics, "test")
                val_m  = _get_metrics_by_split(res.df_metrics, "validation_external")
                rows.append({
                    "embedding": emb,
                    "model": model_type.upper(),
                    "dim": dim,
                    # Métricas do Teste Interno (20% Holdout)
                    "test_acc": test_m.get("accuracy", 0.0),
                    "test_f1_macro": test_m.get("f1_macro", 0.0),
                    "test_f1_scam": test_m.get("f1_scam", 0.0),
                    "test_recall_scam": test_m.get("recall_scam", 0.0),
                    "test_precision_scam": test_m.get("precision_scam", 0.0),
                    "test_roc_auc": test_m.get("roc_auc"),
                    "test_pr_auc": test_m.get("pr_auc"),
                    # Métricas da Validação Externa
                    "val_acc": val_m.get("accuracy", 0.0),
                    "val_f1_macro": val_m.get("f1_macro", 0.0),
                    "val_f1_scam": val_m.get("f1_scam", 0.0),
                    "val_recall_scam": val_m.get("recall_scam", 0.0),
                    "val_precision_scam": val_m.get("precision_scam", 0.0),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_pr_auc": val_m.get("pr_auc"),
                })
            except Exception as e:
                print(f"  [dim_study] Erro em {emb} {model_type} {dim}d: {e}")

    df_dim = pd.DataFrame(rows)
    if not df_dim.empty:
        run_id = f"dim_study_{model_type}_{reducer}_{split}"
        ModelCache.save_history("unsupervised", run_id, df_dim)
        save_path = os.path.join(ModelCache.artifacts_dir("unsupervised", run_id), "dimension_study.png")
        plot_metric = f"{split}_f1_macro" if f"{split}_f1_macro" in df_dim.columns else "test_f1_macro"
        plot_dimension_study(df_dim, metric=plot_metric, title=f"Estudo de Dimensionalidade ({model_type.upper()} — {split.upper()})", save_path=save_path)
    return df_dim
