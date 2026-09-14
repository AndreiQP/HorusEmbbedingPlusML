"""
machine_learning/classical/runner.py
--------------------------------------
Runner para modelos de ML clássico (SVM, KNN, RF, NB, XGBoost).

Funções principais:
  train_classical()             -> treina 1 modelo, retorna ExperimentResult
  train_all_classical()         -> treina todos os modelos × todos os embedders
  learning_curve_classical()    -> curva de aprendizado por % do dataset
  calibrate_threshold_classical() -> calibra threshold via ThresholdCalibrator
"""
from __future__ import annotations
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from ..data import load_single_embedding, EMBEDDERS as ALL_EMBEDDERS
from ..evaluation import compute_metrics, plot_confusion_matrix
from ..cache import ModelCache, _run_id
from ..result import ExperimentResult
from .classifier import get_all_classifiers


def train_classical(
    embedding: str,
    path_data: str,
    model_name: str,
    with_augmented: bool = True,
    test_size: float = 0.2,
    n_cv_folds: int = 5,
    force_retrain: bool = False,
) -> ExperimentResult:
    """
    Treina um modelo clássico de ML.
    
    - Se o modelo já existir em trained_models/ e force_retrain=False, carrega do cache.
    - Treina com cross-validation (n_cv_folds) no conjunto de treino.
    - Reserva test_size para avaliação final.
    - Salva o modelo e as métricas automaticamente.
    
    Args:
        embedding:     chave do embedder ('voyage', 'openai', 'bge', 'e5', 'minilm')
        path_data:     'all_data' ou 'suspect_turns'
        model_name:    nome do modelo de get_all_classifiers() (ex: 'SVM', 'KNN', 'RF')
        with_augmented: se True usa dados aumentados (augmented), se False usa external_only
        test_size:     fração reservada para teste final
        n_cv_folds:    número de folds para validação cruzada
        force_retrain: se True ignora o cache e retreina
    
    Returns:
        ExperimentResult com modelo, métricas e histórico de CV
    """
    origin = "augmented" if with_augmented else "external_only"
    model_map = get_all_classifiers()

    if model_name not in model_map:
        raise ValueError(f"model_name='{model_name}' não encontrado. Disponíveis: {list(model_map.keys())}")

    config = {
        "strategy":      "classical",
        "embedding":     embedding,
        "path_data":     path_data,
        "model_name":    model_name,
        "with_augmented": with_augmented,
        "origin":        origin,
    }
    run_id = _run_id(config)
    model_class_name = type(model_map[model_name]).__name__

    # ── Cache check ──────────────────────────────────────────────
    if not force_retrain and ModelCache.classical_exists(origin, path_data, embedding, model_class_name):
        print(f"[classical] Cache hit: {origin}_{path_data}_{embedding}_{model_class_name}")
        model = ModelCache.classical_load(origin, path_data, embedding, model_class_name)
        # Tenta carregar métricas salvas
        saved_metrics = ModelCache.load_metrics("classical", run_id)
        saved_history = ModelCache.load_history("classical", run_id)
        df_metrics = pd.DataFrame([saved_metrics]) if saved_metrics else pd.DataFrame()
        df_history = saved_history if saved_history is not None else pd.DataFrame()
        return ExperimentResult(
            strategy="classical", config=config, model=model,
            df_metrics=df_metrics, df_history=df_history,
            artifacts_dir=ModelCache.artifacts_dir("classical", run_id),
        )

    # ── Carregar dados ───────────────────────────────────────────
    print(f"\n[classical] Treinando: {origin}_{path_data}_{embedding}_{model_name}")
    X, y = load_single_embedding("train", path_data, embedding, with_augmented=with_augmented)
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Cross-Validation ─────────────────────────────────────────
    model = model_map[model_name]
    from sklearn.base import clone
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

    history_rows = []
    for fold, (idx_tr, idx_val) in enumerate(cv.split(X_train_all, y_train_all), 1):
        m = clone(model)
        m.fit(X_train_all[idx_tr], y_train_all[idx_tr])
        y_pred_val = m.predict(X_train_all[idx_val])
        fold_metrics = compute_metrics(y_train_all[idx_val], y_pred_val)
        fold_metrics["fold"] = fold
        history_rows.append(fold_metrics)

    df_history = pd.DataFrame(history_rows)
    print(f"  [classical] CV f1_macro: {df_history['f1_macro'].mean():.4f} ± {df_history['f1_macro'].std():.4f}")

    # ── Treino final (todos os dados de treino) ───────────────────
    t0 = time.time()
    model.fit(X_train_all, y_train_all)
    print(f"  [classical] Treino final em {time.time()-t0:.1f}s")

    # ── Avaliação no conjunto de teste ────────────────────────────
    y_pred_test = model.predict(X_test)
    y_prob_test = None
    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob_test = model.decision_function(X_test)

    metrics = compute_metrics(y_test, y_pred_test, y_prob=y_prob_test,
                               model_name=f"{model_name}_{embedding}", dataset_name="test_internal")
    df_metrics = pd.DataFrame([metrics])
    print(f"  [classical] Teste -> acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}  "
          f"f1_scam={metrics['f1_scam']:.4f}")

    # ── Salvar ────────────────────────────────────────────────────
    ModelCache.classical_save(model, origin, path_data, embedding)
    ModelCache.save_metrics("classical", run_id, metrics)
    ModelCache.save_history("classical", run_id, df_history)

    artifacts = ModelCache.artifacts_dir("classical", run_id)
    plot_confusion_matrix(y_test, y_pred_test,
                          title=f"{model_name} | {embedding} | {path_data}",
                          save_path=os.path.join(artifacts, "confusion_matrix_test.png"))

    return ExperimentResult(
        strategy="classical", config=config, model=model,
        df_metrics=df_metrics, df_history=df_history,
        y_true=y_test, y_pred=y_pred_test, y_prob=y_prob_test,
        artifacts_dir=artifacts,
    )


def train_all_classical(
    path_data: str = "all_data",
    with_augmented: bool = True,
    model_names: Optional[List[str]] = None,
    embedders: Optional[List[str]] = None,
    force_retrain: bool = False,
) -> Dict[str, ExperimentResult]:
    """
    Treina todos os modelos x todos os embedders.
    Equivalente ao loop principal do ML_models.ipynb.
    
    Returns:
        Dict com chave '{model_name}_{embedding}' -> ExperimentResult
    """
    model_names = model_names or list(get_all_classifiers().keys())
    embedders   = embedders   or list(ALL_EMBEDDERS.keys())

    results = {}
    for emb in embedders:
        for mname in model_names:
            key = f"{mname}_{emb}"
            print(f"\n{'='*60}\n[classical] {key}")
            try:
                results[key] = train_classical(
                    embedding=emb, path_data=path_data, model_name=mname,
                    with_augmented=with_augmented, force_retrain=force_retrain
                )
            except Exception as e:
                print(f"  [ERROR] {key}: {e}")
    return results


def learning_curve_classical(
    embedding: str,
    path_data: str,
    model_name: str,
    fractions: List[float] = None,
    n_repeats: int = 3,
    with_augmented: bool = True,
    force_recompute: bool = False,
) -> ExperimentResult:
    """
    Curva de aprendizado por fração do dataset.
    
    Para cada fração:
    - Treina com frac% dos dados de treino.
    - Avalia no conjunto de teste interno (20% fixo do dataset de treino).
    - Avalia no dataset de validação externa (dataset_validation/).
    
    Returns:
        ExperimentResult com df_history contendo colunas:
            frac, acc_mean, acc_std, f1_macro_mean, f1_macro_std,
            val_acc_mean, val_acc_std, val_f1_macro_mean, val_f1_macro_std
    """
    from sklearn.base import clone
    from .classifier import get_all_classifiers
    from ..data import load_single_embedding as load

    fractions = fractions or [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    config = {
        "strategy": "classical_lc",
        "embedding": embedding, "path_data": path_data,
        "model_name": model_name, "with_augmented": with_augmented,
        "fractions": str(fractions), "n_repeats": n_repeats,
    }
    run_id = _run_id(config)

    if not force_recompute:
        saved = ModelCache.load_history("classical", f"lc_{run_id}")
        if saved is not None:
            print(f"[classical_lc] Cache hit — carregando curva de aprendizado")
            return ExperimentResult(
                strategy="classical", config=config, df_history=saved,
                artifacts_dir=ModelCache.artifacts_dir("classical", f"lc_{run_id}"),
            )

    model_template = get_all_classifiers()[model_name]
    X_train_all, y_train_all = load("train", path_data, embedding, with_augmented=with_augmented)
    X_val, y_val = load("validation", path_data, embedding, with_augmented=False)

    # Separa o conjunto de teste FIXO (20%) antes de variar frações
    X_tr_pool, X_test, y_tr_pool, y_test = train_test_split(
        X_train_all, y_train_all, test_size=0.20, random_state=42, stratify=y_train_all
    )

    n_test = len(y_test)
    n_val = len(y_val)
    rows = []
    for frac in fractions:
        accs, f1s, val_accs, val_f1s = [], [], [], []
        tns, fps, fns, tps = [], [], [], []
        val_tns, val_fps, val_fns, val_tps = [], [], [], []
        n_samples = max(1, int(len(X_tr_pool) * frac))

        for seed in range(n_repeats):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_tr_pool), size=n_samples, replace=False)
            m = clone(model_template)
            m.fit(X_tr_pool[idx], y_tr_pool[idx])

            # Teste interno (conjunto FIXO de 20%)
            y_pred_test = m.predict(X_test)
            m_test = compute_metrics(y_test, y_pred_test)
            accs.append(m_test["accuracy"]); f1s.append(m_test["f1_macro"])
            tns.append(m_test["TN"]); fps.append(m_test["FP"]); fns.append(m_test["FN"]); tps.append(m_test["TP"])

            # Validação externa (dataset_validation)
            y_pred_val = m.predict(X_val)
            m_val = compute_metrics(y_val, y_pred_val)
            val_accs.append(m_val["accuracy"]); val_f1s.append(m_val["f1_macro"])
            val_tns.append(m_val["TN"]); val_fps.append(m_val["FP"]); val_fns.append(m_val["FN"]); val_tps.append(m_val["TP"])

        tn_m, fp_m, fn_m, tp_m = np.mean(tns), np.mean(fps), np.mean(fns), np.mean(tps)
        val_tn_m, val_fp_m, val_fn_m, val_tp_m = np.mean(val_tns), np.mean(val_fps), np.mean(val_fns), np.mean(val_tps)

        rows.append({
            "frac":                frac,
            "n_samples":          n_samples,
            "acc_mean":           np.mean(accs),
            "acc_std":            np.std(accs),
            "f1_macro_mean":      np.mean(f1s),
            "f1_macro_std":       np.std(f1s),
            "val_acc_mean":       np.mean(val_accs),
            "val_acc_std":        np.std(val_accs),
            "val_f1_macro_mean":  np.mean(val_f1s),
            "val_f1_macro_std":   np.std(val_f1s),
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
        print(f"  [lc] frac={frac:.0%} -> f1_macro={np.mean(f1s):.4f} | val_f1={np.mean(val_f1s):.4f}")

    df_history = pd.DataFrame(rows)
    ModelCache.save_history("classical", f"lc_{run_id}", df_history)

    artifacts = ModelCache.artifacts_dir("classical", f"lc_{run_id}")
    result = ExperimentResult(
        strategy="classical", config=config, df_history=df_history, artifacts_dir=artifacts
    )
    result.plot_dataset_size_curve(metric="f1_macro_mean", save=True)
    result.plot_confusion_matrix_evolution(save=True)
    result.plot_confusion_matrices_by_fraction(save=True)
    return result
