"""
machine_learning/fcnn/runner.py
--------------------------------
Runner para a rede neural FCNN (Fully Connected Neural Network).

Funções principais:
  train_fcnn()          -> treina FCNN, retorna ExperimentResult
  learning_curve_fcnn() -> curva de aprendizado por % do dataset
"""
from __future__ import annotations
import os
import sys
import time
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data import load_concat_embeddings
from ..evaluation import compute_metrics
from ..cache import ModelCache, _run_id
from ..result import ExperimentResult
from .fcnn_classifier import ScamClassifierFCNN


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataloaders(X, y, test_size=0.2, batch_size=64):
    """Cria DataLoaders de treino e teste a partir de arrays numpy."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    return dl_train, dl_test, X_test, y_test


def _train_one_epoch(model, loader, optimizer, criterion, device):
    """Treina por uma época e retorna loss e f1_macro médios."""
    import torch
    from sklearn.metrics import f1_score
    model.train()
    losses, preds_all, labels_all = [], [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(y_b.cpu().numpy())
    return np.mean(losses), f1_score(labels_all, preds_all, average="macro", zero_division=0)


def _eval_epoch(model, loader, criterion, device):
    """Avalia no loader sem atualizar pesos. Retorna loss, f1, y_true, y_pred, y_prob."""
    import torch
    from sklearn.metrics import f1_score
    model.eval()
    losses, preds_all, labels_all, probs_all = [], [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            loss = criterion(out, y_b)
            losses.append(loss.item())
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs_all.extend(prob)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(y_b.cpu().numpy())
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return np.mean(losses), f1, np.array(labels_all), np.array(preds_all), np.array(probs_all)


def train_fcnn(
    path_data: str = "all_data",
    embedders: Optional[List[str]] = None,
    with_augmented: bool = True,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    weight_decay: float = 1e-4,
    seed: int = 42,
    test_size: float = 0.2,
    force_retrain: bool = False,
) -> ExperimentResult:

    """
    Treina a FCNN com Early Stopping e salva automaticamente.
    
    - Usa todos os embedders concatenados como entrada.
    - Divide em treino/teste estratificado.
    - Aplica Early Stopping: salva o melhor checkpoint.
    
    Args:
        path_data:     'all_data' ou 'suspect_turns'
        embedders:     lista de embedders a concatenar (default: todos)
        with_augmented: se True usa dados aumentados
        epochs:        máximo de épocas
        lr:            learning rate
        batch_size:    tamanho do batch
        patience:      épocas sem melhora para parar cedo
        test_size:     fração de teste interno
        force_retrain: se True ignora cache e re-treina
    
    Returns:
        ExperimentResult com modelo, histórico por época e métricas
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    config = {
        "strategy":      "fcnn",
        "path_data":     path_data,
        "embedders":     embedders or "all",
        "with_augmented": with_augmented,
        "epochs":        epochs,
        "lr":            lr,
        "batch_size":    batch_size,
    }
    run_id = _run_id(config)

    # ── Cache check ──────────────────────────────────────────────
    if not force_retrain and ModelCache.fcnn_exists(with_augmented, path_data):
        print(f"[fcnn] Cache hit: carregando FCNN existente")
        X_all, y_all = load_concat_embeddings("train", path_data, embedders, with_augmented=with_augmented)
        dl_train, dl_test, X_test, y_test = _build_dataloaders(X_all, y_all, test_size=test_size, batch_size=batch_size)
        model = ModelCache.fcnn_load(with_augmented, path_data, input_size=X_all.shape[1])
        device = _get_device()
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        _, _, y_true, y_pred, y_prob = _eval_epoch(model, dl_test, criterion, device)

        saved_history = ModelCache.load_history("fcnn", run_id)
        saved_metrics = ModelCache.load_metrics("fcnn", run_id)
        return ExperimentResult(
            strategy="fcnn", config=config, model=model,
            df_history=saved_history if saved_history is not None else pd.DataFrame(),
            df_metrics=pd.DataFrame([saved_metrics]) if saved_metrics else pd.DataFrame(),
            y_true=y_true, y_pred=y_pred, y_prob=y_prob,
            artifacts_dir=ModelCache.artifacts_dir("fcnn", run_id),
        )



    # ── Dados ────────────────────────────────────────────────────
    print(f"\n[fcnn] Carregando dados: {path_data} | augmented={with_augmented}")
    X, y = load_concat_embeddings("train", path_data, embedders, with_augmented=with_augmented)
    dl_train, dl_test, X_test, y_test = _build_dataloaders(X, y, test_size=test_size, batch_size=batch_size)

    # ── Modelo ───────────────────────────────────────────────────
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _get_device()
    model = ScamClassifierFCNN(input_size=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()



    print(f"[fcnn] Device: {device} | Input size: {X.shape[1]} | Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    # ── Treino com Early Stopping ─────────────────────────────────
    best_val_f1 = -1.0
    best_state  = None
    patience_counter = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = _train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_loss, val_f1, _, _, _ = _eval_epoch(model, dl_test, criterion, device)

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_f1": train_f1, "val_f1": val_f1,
        })

        print(f"  Época {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [fcnn] Early stopping na época {epoch}. Melhor val_f1={best_val_f1:.4f}")
                break

    # ── Restaura melhor checkpoint ────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)

    # ── Avaliação final ───────────────────────────────────────────
    _, _, y_true, y_pred, y_prob = _eval_epoch(model, dl_test, criterion, device)
    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob,
                               model_name="FCNN", dataset_name="test_internal")
    df_metrics = pd.DataFrame([metrics])
    df_history = pd.DataFrame(history_rows)

    print(f"\n[fcnn] Resultado final -> acc={metrics['accuracy']:.4f}  "
          f"f1_macro={metrics['f1_macro']:.4f}  f1_scam={metrics['f1_scam']:.4f}")

    # ── Salvar ────────────────────────────────────────────────────
    ModelCache.fcnn_save(model, with_augmented, path_data)
    ModelCache.save_metrics("fcnn", run_id, metrics)
    ModelCache.save_history("fcnn", run_id, df_history)

    artifacts = ModelCache.artifacts_dir("fcnn", run_id)
    result = ExperimentResult(
        strategy="fcnn", config=config, model=model,
        df_metrics=df_metrics, df_history=df_history,
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        artifacts_dir=artifacts,
    )
    result.plot_confusion_matrix(title="FCNN — Teste Interno")
    result.plot_learning_curves()
    return result


def learning_curve_fcnn(
    path_data: str = "all_data",
    embedders: Optional[List[str]] = None,
    fractions: Optional[List[float]] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    n_repeats: int = 3,
    with_augmented: bool = True,
    force_recompute: bool = False,
) -> ExperimentResult:
    """
    Curva de aprendizado FCNN por % do dataset.
    
    Para cada fração: treina com frac% do dataset e avalia no:
    - Conjunto de teste interno (20% fixo)
    - Dataset de validação externa
    
    Returns:
        ExperimentResult com df_history contendo colunas:
            frac, val_f1_mean, val_f1_std, f1_mean, f1_std
    """
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from .fcnn_classifier import ScamClassifierFCNN
    from machine_learning.data import load_concat_embeddings as load_cat

    fractions = fractions or [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    config = {
        "strategy": "fcnn_lc",
        "path_data": path_data, "embedders": str(embedders),
        "with_augmented": with_augmented, "fractions": str(fractions),
    }
    run_id = _run_id(config)

    if not force_recompute:
        saved = ModelCache.load_history("fcnn", f"lc_{run_id}")
        if saved is not None:
            print("[fcnn_lc] Cache hit — curva de aprendizado")
            return ExperimentResult(
                strategy="fcnn", config=config, df_history=saved,
                artifacts_dir=ModelCache.artifacts_dir("fcnn", f"lc_{run_id}"),
            )

    X_all, y_all = load_cat("train", path_data, embedders, with_augmented=with_augmented)
    X_val, y_val = load_cat("validation", path_data, embedders, with_augmented=False)
    X_tr_pool, X_test, y_tr_pool, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    device = _get_device()
    criterion = nn.CrossEntropyLoss()

    rows = []
    for frac in fractions:
        n_samples = max(32, int(len(X_tr_pool) * frac))
        f1s, val_f1s = [], []

        for seed in range(n_repeats):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_tr_pool), size=n_samples, replace=False)
            X_sub, y_sub = X_tr_pool[idx], y_tr_pool[idx]

            ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32),
                               torch.tensor(y_sub, dtype=torch.long))
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = ScamClassifierFCNN(input_size=X_all.shape[1]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for _ in range(epochs):
                _train_one_epoch(model, dl, optimizer, criterion, device)

            # Teste interno (conjunto fixo)
            dl_t = DataLoader(
                TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                              torch.tensor(y_test, dtype=torch.long)),
                batch_size=256)
            _, f1_test, _, _, _ = _eval_epoch(model, dl_t, criterion, device)
            f1s.append(f1_test)

            # Validação externa
            dl_v = DataLoader(
                TensorDataset(torch.tensor(X_val.astype(np.float32)),
                              torch.tensor(y_val, dtype=torch.long)),
                batch_size=256)
            _, f1_val, _, _, _ = _eval_epoch(model, dl_v, criterion, device)
            val_f1s.append(f1_val)

        rows.append({
            "frac": frac, "n_samples": n_samples,
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            "val_f1_mean": np.mean(val_f1s), "val_f1_std": np.std(val_f1s),
        })
        print(f"  [fcnn_lc] frac={frac:.0%} -> f1={np.mean(f1s):.4f} | val_f1={np.mean(val_f1s):.4f}")

    df_history = pd.DataFrame(rows)
    ModelCache.save_history("fcnn", f"lc_{run_id}", df_history)

    artifacts = ModelCache.artifacts_dir("fcnn", f"lc_{run_id}")
    result = ExperimentResult(
        strategy="fcnn", config=config, df_history=df_history, artifacts_dir=artifacts
    )
    result.plot_dataset_size_curve(metric="val_f1_mean")
    return result


def evaluate_fcnn_cross_turns(
    model: object,
    path_data: str = "suspect_turns",
    dataset_name: str = "validation",
    embedders: Optional[List[str]] = None,
    with_augmented: bool = True,
    batch_size: int = 64,
) -> ExperimentResult:
    """
    Avalia um modelo FCNN treinado sobre um conjunto com tipo de turnos específico
    (ex: modelo treinado em 'all_data' testado em 'suspect_turns').
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    print(f"\n[fcnn] Avaliação Cross-Turns: testando em {path_data} | dataset={dataset_name} | augmented={with_augmented}")
    X, y = load_concat_embeddings(
        "validation" if dataset_name in ["validation", "validacao_externa"] else "train",
        path_data, embedders,
        with_augmented=with_augmented if dataset_name == "test_internal" else False
    )

    if dataset_name == "test_internal":
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X, y = X_test, y_test

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    device = _get_device()
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    _, _, y_true, y_pred, y_prob = _eval_epoch(model, dl, criterion, device)
    metrics = compute_metrics(
        y_true, y_pred, y_prob=y_prob,
        model_name=f"FCNN_Cross_{path_data}", dataset_name=dataset_name
    )
    df_metrics = pd.DataFrame([metrics])

    print(f"  [fcnn_cross] acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}  f1_scam={metrics['f1_scam']:.4f}")

    return ExperimentResult(
        strategy="fcnn",
        config={"cross_eval": True, "eval_path_data": path_data, "dataset": dataset_name},
        model=model,
        df_metrics=df_metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def grid_search_fcnn(
    path_data: str = "all_data",
    embedders: Optional[List[str]] = None,
    with_augmented: bool = True,
    lrs: Optional[List[float]] = None,
    batch_sizes: Optional[List[int]] = None,
    patiences: Optional[List[int]] = None,
    weight_decays: Optional[List[float]] = None,
    epochs: int = 40,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Executa busca em grade (Grid Search) para encontrar a melhor combinação de:
    - learning_rate (ex: [1e-4, 3e-4, 1e-3])
    - batch_size (ex: [32, 64, 128, 256])
    - patience (ex: [5, 10, 15])
    - weight_decay (ex: [1e-4, 1e-3, 1e-2])

    Retorna um DataFrame ordenado pelas melhores métricas na Validação Externa e no Teste Interno.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import f1_score, precision_score, recall_score
    from machine_learning.thresholds import ThresholdCalibrator

    lrs = lrs or [1e-4, 3e-4, 1e-3]
    batch_sizes = batch_sizes or [32, 64, 128]
    patiences = patiences or [5, 10, 15]
    weight_decays = weight_decays or [1e-4, 1e-3, 1e-2]

    config_grid = {
        "strategy": "fcnn_grid_search",
        "path_data": path_data,
        "with_augmented": with_augmented,
        "lrs": str(lrs),
        "batch_sizes": str(batch_sizes),
        "patiences": str(patiences),
        "weight_decays": str(weight_decays),
    }
    run_id = _run_id(config_grid)

    if not force_recompute:
        saved = ModelCache.load_history("fcnn", f"grid_{run_id}")
        if saved is not None:
            print(f"[fcnn_grid] Cache hit — grid search ({len(saved)} combinações)")
            return saved

    print(f"\n[fcnn_grid] Iniciando Grid Search com 5 Embeddings (6528d) | path_data={path_data}")
    
    # Carrega dados de treino (interno) e validação (externo) uma única vez
    X_train_full, y_train_full = load_concat_embeddings("train", path_data, embedders, with_augmented=with_augmented)
    X_val_ext, y_val_ext = load_concat_embeddings("validation", path_data, embedders, with_augmented=False)

    results_list = []
    total_combos = len(lrs) * len(batch_sizes) * len(patiences) * len(weight_decays)
    combo_idx = 0

    for lr in lrs:
        for bs in batch_sizes:
            for pat in patiences:
                for wd in weight_decays:
                    combo_idx += 1
                    print(f"[{combo_idx:02d}/{total_combos:02d}] Testando: lr={lr} | batch_size={bs} | patience={pat} | weight_decay={wd}")
                    
                    # Constrói dataloaders para esta combinação
                    dl_train, dl_test, X_test, y_test = _build_dataloaders(X_train_full, y_train_full, test_size=0.2, batch_size=bs)
                    
                    device = _get_device()
                    model = ScamClassifierFCNN(input_size=X_train_full.shape[1]).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    criterion = nn.CrossEntropyLoss()

                    best_val_f1 = -1.0
                    best_state = None
                    patience_counter = 0

                    for epoch in range(1, epochs + 1):
                        _train_one_epoch(model, dl_train, optimizer, criterion, device)
                        _, val_f1, _, _, _ = _eval_epoch(model, dl_test, criterion, device)

                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= pat:
                                break

                    # Avaliação no checkpoint selecionado
                    model.load_state_dict(best_state)
                    model.to(device).eval()

                    # Teste Interno
                    _, _, y_test_true, y_test_pred, y_test_prob = _eval_epoch(model, dl_test, criterion, device)
                    test_f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
                    
                    # Calibração Zero-FN no Teste Interno
                    calibrator = ThresholdCalibrator(strategy="min_fn").fit(y_test_true, y_test_prob)
                    best_thr = calibrator.threshold_

                    # Validação Externa (Thr=0.5)
                    X_val_t = torch.tensor(X_val_ext, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        val_logits = model(X_val_t)
                        val_prob = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                    
                    val_pred_05 = (val_prob >= 0.5).astype(int)
                    val_f1_05 = f1_score(y_val_ext, val_pred_05, average="macro", zero_division=0)
                    val_prec_scam_05 = precision_score(y_val_ext, val_pred_05, pos_label=1, zero_division=0)
                    val_rec_scam_05 = recall_score(y_val_ext, val_pred_05, pos_label=1, zero_division=0)

                    # Validação Externa (Thr Calibrado no Teste)
                    val_pred_cal = (val_prob >= best_thr).astype(int)
                    val_f1_cal = f1_score(y_val_ext, val_pred_cal, average="macro", zero_division=0)
                    val_prec_scam_cal = precision_score(y_val_ext, val_pred_cal, pos_label=1, zero_division=0)
                    val_rec_scam_cal = recall_score(y_val_ext, val_pred_cal, pos_label=1, zero_division=0)

                    results_list.append({
                        "lr": lr,
                        "batch_size": bs,
                        "patience": pat,
                        "weight_decay": wd,
                        "best_val_f1_train": best_val_f1,
                        "test_f1_macro": test_f1,
                        "zero_fn_threshold": best_thr,
                        "ext_val_f1_05": val_f1_05,
                        "ext_val_prec_scam_05": val_prec_scam_05,
                        "ext_val_rec_scam_05": val_rec_scam_05,
                        "ext_val_f1_cal": val_f1_cal,
                        "ext_val_prec_scam_cal": val_prec_scam_cal,
                        "ext_val_rec_scam_cal": val_rec_scam_cal,
                    })

    df_results = pd.DataFrame(results_list).sort_values("ext_val_f1_cal", ascending=False).reset_index(drop=True)
    ModelCache.save_history("fcnn", f"grid_{run_id}", df_results)
    
    print("\n[fcnn_grid] Grid Search Concluído com sucesso!")
    print(df_results.head(10).to_string(index=False))
    return df_results


