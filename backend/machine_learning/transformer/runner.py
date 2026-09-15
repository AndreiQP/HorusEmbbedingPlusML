"""
machine_learning/transformer/runner.py
----------------------------------------
Runner para Transformer Ensemble (3 modelos: voyage, bge, openai), com busca
em etapas, treino final multi-seed e consolidação cache-only.

Funções principais:
  train_transformer()            → treina 1 Transformer, retorna ExperimentResult
  load_trained_transformers()    → carrega todos os modelos já treinados
  evaluate_ensemble()            → votação entre os 3 Transformers, retorna ExperimentResult
  run_transformer_search()       → busca em três etapas pela validação interna
  run_transformer_finalist()     → treina as três seeds do vencedor
"""
from __future__ import annotations
import os
import json
import itertools
import sys
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..data import load_sequence_embeddings, get_embedding_dim, get_shared_split_indices
from ..evaluation import compute_metrics
from ..cache import ModelCache, _run_id
from ..result import ExperimentResult


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITETURA
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        pe = self._generate_pe(max_len, d_model)
        self.register_buffer('pe', pe)

    @staticmethod
    def _generate_pe(max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            pe = self._generate_pe(seq_len, self.d_model).to(x.device)
            x = x + pe[:, :seq_len, :]
        else:
            x = x + self.pe[:, :seq_len, :]
        return x


class TransformerScamClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 4, num_layers: int = 2, hidden_dim: int = 128, dropout: float = 0.1, max_turnos: int = 1000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, max_len=max_turnos)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, padding_mask=None):
        x = self.pos_encoder(x)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            sum_embeddings = (out * mask_expanded).sum(dim=1)
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_out = sum_embeddings / valid_counts
        else:
            pooled_out = out.mean(dim=1)
        return self.classifier(pooled_out)


def _get_transformer_class():
    return TransformerScamClassifier


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_single_dataloader(sequences, labels, batch_size=32, shuffle=False):
    """Cria um DataLoader a partir de uma lista de sequências e rótulos."""
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence

    class ScamDataset(Dataset):
        def __init__(self, seqs, lbls):
            self.seqs = [
                torch.tensor(s[-100:], dtype=torch.float32) if len(s) > 100 else torch.tensor(s, dtype=torch.float32)
                for s in seqs
            ]
            self.lbls = torch.tensor(lbls, dtype=torch.float32)
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i): return self.seqs[i], self.lbls[i]

    def collate_fn(batch):
        seqs, lbls = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([s.shape[0] for s in seqs])
        max_len = padded.shape[1]
        mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        return padded, torch.stack(list(lbls)), mask

    return DataLoader(
        ScamDataset(sequences, labels),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def _make_dataloaders(sequences, labels, test_size=0.2, val_size=0.1, batch_size=32, seed=42, return_indices=False, split_indices=None):
    """
    Divide sequências em treino/val/teste e cria DataLoaders com padding.
    """
    from sklearn.model_selection import train_test_split

    if split_indices is None:
        idx = np.arange(len(sequences))
        idx_tr, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
        idx_tr, idx_val = train_test_split(
            idx_tr, test_size=val_size/(1-test_size), random_state=seed, stratify=labels[idx_tr]
        )
    else:
        idx_tr, idx_val, idx_test = split_indices

    dl_tr = _make_single_dataloader([sequences[i] for i in idx_tr], labels[idx_tr], batch_size=batch_size, shuffle=True)
    dl_val = _make_single_dataloader([sequences[i] for i in idx_val], labels[idx_val], batch_size=batch_size, shuffle=False)
    dl_test = _make_single_dataloader([sequences[i] for i in idx_test], labels[idx_test], batch_size=batch_size, shuffle=False)

    if return_indices:
        return dl_tr, dl_val, dl_test, idx_tr, idx_val, idx_test
    return dl_tr, dl_val, dl_test


def _train_transformer_loop(
    model, dl_train, dl_val, num_epochs, lr, device,
    optimizer_name="adamw", weight_decay=1e-4, patience=8,
    min_delta=1e-4, scheduler_patience=3, min_lr=1e-7,
    mixed_precision=True, gradient_clip=1.0,
):
    """Train with real early stopping and return the best validation checkpoint."""
    from sklearn.metrics import f1_score, accuracy_score

    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = optim.AdamW if optimizer_name.lower() == "adamw" else optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience, min_lr=min_lr
    )
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_f1, best_state, best_epoch = -1.0, None, 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses, train_preds, train_labels = [], [], []
        for X_b, y_b, mask in dl_train:
            X_b, y_b, mask = X_b.to(device), y_b.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(X_b, padding_mask=mask).squeeze(1)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            prob = torch.sigmoid(out)
            train_preds.extend((prob >= 0.5).int().cpu().numpy())
            train_labels.extend(y_b.cpu().numpy())

        # Validação
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for X_b, y_b, mask in dl_val:
                X_b, y_b, mask = X_b.to(device), y_b.to(device), mask.to(device)
                out = model(X_b, padding_mask=mask).squeeze(1)
                val_losses.append(criterion(out, y_b).item())
                prob = torch.sigmoid(out)
                val_preds.extend((prob >= 0.5).int().cpu().numpy())
                val_labels.extend(y_b.cpu().numpy())

        t_f1  = f1_score(train_labels, train_preds, average="macro", zero_division=0)
        v_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        t_acc = accuracy_score(train_labels, train_preds)
        v_acc = accuracy_score(val_labels, val_preds)

        scheduler.step(v_f1)
        history.append({
            "epoch": epoch,
            "train_loss": np.mean(train_losses), "val_loss": np.mean(val_losses),
            "train_f1": t_f1, "val_f1": v_f1,
            "train_acc": t_acc, "val_acc": v_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if v_f1 > best_val_f1 + min_delta:
            best_val_f1 = v_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Época {epoch:3d}/{num_epochs} | "
                  f"train_loss={np.mean(train_losses):.4f} train_f1={t_f1:.4f} | "
                  f"val_loss={np.mean(val_losses):.4f} val_f1={v_f1:.4f}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch}; best epoch={best_epoch}, val_f1={best_val_f1:.4f}")
            break

    history_df = pd.DataFrame(history)
    history_df.attrs["best_epoch"] = best_epoch
    history_df.attrs["stopped_epoch"] = len(history)
    return model, best_state, history_df


def train_transformer(
    embedding: str,
    num_epochs: int = 100,
    lr: float = 1e-5,
    batch_size: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    patience: int = 8,
    min_delta: float = 1e-4,
    scheduler_patience: int = 3,
    min_lr: float = 1e-7,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    seed: int = 42,
    split_seed: int = 42,
    with_augmented: bool = False,
    pct: Optional[int] = None,
    force_retrain: bool = False,
) -> ExperimentResult:
    """
    Treina um Transformer para um embedding e salva modelo, histórico, métricas e predições
    (teste interno e validação externa) no disco. Se já existir no cache e force_retrain=False,
    carrega do disco sem re-treinar.
    """
    config = {
        "strategy": "transformer",
        "embedding": embedding, "num_epochs": num_epochs,
        "lr": lr, "pct": pct, "with_augmented": with_augmented,
        "batch_size": batch_size, "num_heads": num_heads, "num_layers": num_layers,
        "hidden_dim": hidden_dim, "dropout": dropout, "optimizer": optimizer_name,
        "weight_decay": weight_decay, "patience": patience, "min_delta": min_delta,
        "scheduler_patience": scheduler_patience, "min_lr": min_lr,
        "mixed_precision": mixed_precision, "gradient_clip": gradient_clip, "seed": seed,
        "split_seed": split_seed,
    }
    run_id = _run_id(config)
    emb_dim = get_embedding_dim(embedding)

    # ── Cache check ──────────────────────────────────────────────
    checkpoint_path = os.path.join(ModelCache.artifacts_dir("transformer", run_id), "model.pth")
    if not force_retrain:
        saved_metrics = ModelCache.load_metrics("transformer", run_id) or {}
        saved_history = ModelCache.load_history("transformer", run_id)
        y_true, y_pred, y_prob = ModelCache.load_predictions("transformer", run_id, "test_internal")
        internal_y, _, _ = ModelCache.load_predictions("transformer", run_id, "internal_validation")
        validation_y, _, _ = ModelCache.load_predictions("transformer", run_id, "validation")
        if saved_metrics.get("test") and y_true is not None and internal_y is not None and validation_y is not None:
            print(f"[transformer] Cache leve: {embedding} (seed={seed})")
            return ExperimentResult(
                strategy="transformer", config=config, model=None,
                df_metrics=pd.DataFrame([saved_metrics["test"]]),
                df_history=saved_history if saved_history is not None else pd.DataFrame(),
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                artifacts_dir=ModelCache.artifacts_dir("transformer", run_id),
            )
    # ── Dados ────────────────────────────────────────────────────
    print(f"\n[transformer] Treinando: {embedding} | epochs={num_epochs} | lr={lr}")
    sequences, labels, sample_ids = load_sequence_embeddings(
        "train", embedding, with_augmented=with_augmented, return_ids=True
    )
    sequences = np.array(sequences, dtype=object)

    if pct is not None and pct < 100:
        n = max(4, int(len(sequences) * pct / 100))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sequences), size=n, replace=False)
        sequences, labels, sample_ids = sequences[idx], labels[idx], sample_ids[idx]
        print(f"  [transformer] Usando {pct}% -> {n} conversas")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    shared_indices = get_shared_split_indices(
        "transformer", sample_ids, labels, seed=split_seed
    )
    dl_train, dl_val, dl_test, idx_train, idx_val, idx_test = _make_dataloaders(
        sequences, labels, batch_size=batch_size, seed=split_seed, return_indices=True,
        split_indices=shared_indices,
    )

    # Dataset de validação externa (dataset_validation)
    sequences_val, labels_val, validation_ids = load_sequence_embeddings(
        "validation", embedding, with_augmented=False, return_ids=True
    )
    dl_val_ext = _make_single_dataloader(sequences_val, labels_val, batch_size=batch_size, shuffle=False)

    # ── Treino ───────────────────────────────────────────────────
    device = _get_device()
    if emb_dim % num_heads != 0:
        raise ValueError(f"num_heads={num_heads} must divide embedding_dim={emb_dim}")
    model = TransformerScamClassifier(
        embedding_dim=emb_dim, num_heads=num_heads, num_layers=num_layers,
        hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)
    model, best_state, df_history = _train_transformer_loop(
        model, dl_train, dl_val, num_epochs, lr, device,
        optimizer_name=optimizer_name, weight_decay=weight_decay,
        patience=patience, min_delta=min_delta, scheduler_patience=scheduler_patience,
        min_lr=min_lr, mixed_precision=mixed_precision, gradient_clip=gradient_clip,
    )
    model.load_state_dict(best_state)

    # ── Avaliação final (Teste Interno + Validação Externa) ───────
    y_true_internal_val, y_pred_internal_val, y_prob_internal_val = _evaluate_dl(model, dl_val, device)
    y_true_test, y_pred_test, y_prob_test = _evaluate_dl(model, dl_test, device)
    metrics_test = compute_metrics(
        y_true_test, y_pred_test, y_prob=y_prob_test,
        model_name=f"Transformer_{embedding}", dataset_name="test_internal"
    )

    y_true_val, y_pred_val, y_prob_val = _evaluate_dl(model, dl_val_ext, device)
    metrics_val = compute_metrics(
        y_true_val, y_pred_val, y_prob=y_prob_val,
        model_name=f"Transformer_{embedding}", dataset_name="validation"
    )

    df_metrics = pd.DataFrame([metrics_test, metrics_val])
    print(f"\n[transformer] Teste Interno    -> acc={metrics_test['accuracy']:.4f}  f1_macro={metrics_test['f1_macro']:.4f}")
    print(f"[transformer] Validação Externa -> acc={metrics_val['accuracy']:.4f}  f1_macro={metrics_val['f1_macro']:.4f}")

    # ── Salvar (Sobrescreve no treino mais recente) ───────────────
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"state_dict": model.to("cpu").state_dict(), "config": config}, checkpoint_path)
    ModelCache.save_metrics("transformer", run_id, {
        "test": metrics_test, "val": metrics_val,
        "training": {
            "best_epoch": int(df_history.attrs.get("best_epoch", 0)),
            "stopped_epoch": int(df_history.attrs.get("stopped_epoch", len(df_history))),
            "max_epochs": num_epochs,
        },
    })
    ModelCache.save_history("transformer", run_id, df_history)

    # Salva predições de teste interno e validação externa separadamente
    ModelCache.save_predictions(
        "transformer", run_id, "test_internal", y_true_test, y_pred_test, y_prob_test,
        sample_ids=sample_ids[idx_test],
    )
    ModelCache.save_predictions(
        "transformer", run_id, "internal_validation",
        y_true_internal_val, y_pred_internal_val, y_prob_internal_val,
        sample_ids=sample_ids[idx_val],
    )
    ModelCache.save_predictions(
        "transformer", run_id, "validation", y_true_val, y_pred_val, y_prob_val,
        sample_ids=validation_ids,
    )

    artifacts = ModelCache.artifacts_dir("transformer", run_id)
    result = ExperimentResult(
        strategy="transformer", config=config, model=model,
        df_metrics=pd.DataFrame([metrics_test]), df_history=df_history,
        y_true=y_true_test, y_pred=y_pred_test, y_prob=y_prob_test,
        artifacts_dir=artifacts,
    )
    return result


TRANSFORMER_EMBEDDINGS = ["voyage", "bge", "openai"]
TRANSFORMER_LR = {"voyage": 2e-6, "bge": 1e-6, "openai": 1e-6}
TRANSFORMER_SEEDS = [42, 52, 62]


def _transformer_results_path(*parts: str) -> str:
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(backend_dir, "experiment_results", "transformer", *parts)


def _candidate_sort_key(row: dict) -> tuple:
    config = row["config"]
    complexity = (
        int(config["num_layers"]), int(config["hidden_dim"]),
        int(config["num_heads"]), float(config["dropout"]),
    )
    return (
        -float(row["internal_val_f1_macro"]),
        -float(row["internal_val_pr_auc"]),
        -float(row["internal_val_recall_scam"]),
        complexity,
    )


def train_transformer_candidate(
    embedding: str,
    stage: str,
    *,
    num_heads: int,
    num_layers: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    optimizer_name: str,
    weight_decay: float,
    lr: Optional[float] = None,
    force_retrain: bool = False,
) -> dict:
    """Treina uma configuração vendo somente treino e validação interna."""
    lr = TRANSFORMER_LR[embedding] if lr is None else lr
    emb_dim = get_embedding_dim(embedding)
    if emb_dim % num_heads != 0:
        raise ValueError(f"num_heads={num_heads} não divide embedding_dim={emb_dim}")
    config = {
        "strategy": "transformer_search", "stage": stage, "embedding": embedding,
        "max_epochs": 100, "lr": lr, "batch_size": batch_size,
        "num_heads": num_heads, "num_layers": num_layers,
        "hidden_dim": hidden_dim, "dropout": dropout,
        "optimizer": optimizer_name, "weight_decay": weight_decay,
        "patience": 8, "min_delta": 1e-4, "scheduler_patience": 3,
        "min_lr": 1e-7, "gradient_clip": 1.0,
        "mixed_precision": True, "seed": 42, "split_seed": 42,
    }
    run_id = _run_id(config)
    metrics_path = os.path.join(ModelCache.artifacts_dir("transformer", run_id), "metrics.json")
    if not force_retrain and os.path.exists(metrics_path):
        cached = ModelCache.load_metrics("transformer", run_id)
        if cached and "internal_val" in cached:
            metric = cached["internal_val"]
            return {
                "run_id": run_id, "config": config,
                "internal_val_f1_macro": metric["f1_macro"],
                "internal_val_pr_auc": metric["pr_auc"],
                "internal_val_recall_scam": metric["recall_scam"],
            }

    sequences, labels, sample_ids = load_sequence_embeddings(
        "train", embedding, with_augmented=False, return_ids=True
    )
    sequences = np.asarray(sequences, dtype=object)
    shared_indices = get_shared_split_indices("transformer", sample_ids, labels, seed=42)
    dl_train, dl_val, _, _, idx_val, _ = _make_dataloaders(
        sequences, labels, batch_size=batch_size, seed=42, return_indices=True,
        split_indices=shared_indices,
    )
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = _get_device()
    model = TransformerScamClassifier(
        embedding_dim=emb_dim, num_heads=num_heads, num_layers=num_layers,
        hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)
    model, best_state, history = _train_transformer_loop(
        model, dl_train, dl_val, 100, lr, device,
        optimizer_name=optimizer_name, weight_decay=weight_decay,
        patience=8, min_delta=1e-4, scheduler_patience=3, min_lr=1e-7,
        mixed_precision=True, gradient_clip=1.0,
    )
    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = _evaluate_dl(model, dl_val, device)
    metric = compute_metrics(
        y_true, y_pred, y_prob=y_prob,
        model_name=f"Transformer_{embedding}", dataset_name="internal_validation",
    )
    artifact_dir = ModelCache.artifacts_dir("transformer", run_id)
    torch.save({"state_dict": model.to("cpu").state_dict(), "config": config}, os.path.join(artifact_dir, "model.pth"))
    ModelCache.save_metrics("transformer", run_id, {
        "internal_val": metric,
        "training": {
            "best_epoch": int(history.attrs.get("best_epoch", 0)),
            "stopped_epoch": int(history.attrs.get("stopped_epoch", len(history))),
            "max_epochs": 100,
        },
    })
    ModelCache.save_history("transformer", run_id, history)
    ModelCache.save_predictions(
        "transformer", run_id, "internal_validation", y_true, y_pred, y_prob,
        sample_ids=sample_ids[idx_val],
    )
    return {
        "run_id": run_id, "config": config,
        "internal_val_f1_macro": metric["f1_macro"],
        "internal_val_pr_auc": metric["pr_auc"],
        "internal_val_recall_scam": metric["recall_scam"],
    }


def run_transformer_search(embedding: str, force_retrain: bool = False) -> dict:
    """Executa as três etapas de busca, sempre por validação interna."""
    common = {
        "embedding": embedding, "lr": TRANSFORMER_LR[embedding],
        "force_retrain": force_retrain,
    }
    stages = []

    architecture = []
    for heads, layers in itertools.product([2, 4, 8], [1, 2, 3, 4]):
        if get_embedding_dim(embedding) % heads == 0:
            architecture.append(train_transformer_candidate(
                stage="architecture", num_heads=heads, num_layers=layers,
                hidden_dim=128, dropout=0.1, batch_size=32,
                optimizer_name="adamw", weight_decay=1e-4, **common,
            ))
    architecture.sort(key=_candidate_sort_key)
    stages.append({"name": "architecture", "candidates": architecture, "winner": architecture[0]})

    arch = architecture[0]["config"]
    capacity = [train_transformer_candidate(
        stage="capacity", num_heads=arch["num_heads"], num_layers=arch["num_layers"],
        hidden_dim=hidden, dropout=dropout, batch_size=batch,
        optimizer_name="adamw", weight_decay=1e-4, **common,
    ) for hidden, dropout, batch in itertools.product([128, 256, 512], [0.1, 0.2, 0.3], [16, 32])]
    capacity.sort(key=_candidate_sort_key)
    stages.append({"name": "capacity", "candidates": capacity, "winner": capacity[0]})

    cap = capacity[0]["config"]
    optimizers = [
        ("adam", 0.0), ("adamw", 1e-4), ("adamw", 1e-2),
    ]
    optimization = [train_transformer_candidate(
        stage="optimizer", num_heads=cap["num_heads"], num_layers=cap["num_layers"],
        hidden_dim=cap["hidden_dim"], dropout=cap["dropout"], batch_size=cap["batch_size"],
        optimizer_name=optimizer_name, weight_decay=weight_decay, **common,
    ) for optimizer_name, weight_decay in optimizers]
    optimization.sort(key=_candidate_sort_key)
    stages.append({"name": "optimizer", "candidates": optimization, "winner": optimization[0]})

    summary = {
        "protocol_version": 2, "git_commit": ModelCache.git_commit(), "embedding": embedding,
        "selection_metric": "internal_val_f1_macro", "stages": stages,
        "winner": optimization[0],
    }
    path = _transformer_results_path("search", embedding, "summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def run_transformer_finalist(embedding: str, force_retrain: bool = False) -> dict:
    """Busca a configuração do embedding e avalia suas três seeds finais."""
    search = run_transformer_search(embedding, force_retrain=force_retrain)
    winner = search["winner"]["config"]
    seed_rows = []
    for seed in TRANSFORMER_SEEDS:
        result = train_transformer(
            embedding=embedding, num_epochs=100, lr=winner["lr"],
            batch_size=winner["batch_size"], num_heads=winner["num_heads"],
            num_layers=winner["num_layers"], hidden_dim=winner["hidden_dim"],
            dropout=winner["dropout"], optimizer_name=winner["optimizer"],
            weight_decay=winner["weight_decay"], seed=seed, split_seed=42,
            patience=8, min_delta=1e-4, scheduler_patience=3, min_lr=1e-7,
            mixed_precision=True, gradient_clip=1.0,
            force_retrain=force_retrain,
        )
        metrics = ModelCache.load_metrics("transformer", _run_id(result.config))
        seed_rows.append({
            "seed": seed, "run_id": _run_id(result.config),
            "test": metrics["test"], "validation": metrics["val"],
        })

    def mean_std(split: str, metric: str) -> tuple[float, float]:
        values = [float(row[split][metric]) for row in seed_rows]
        return float(np.mean(values)), float(np.std(values))

    test_mean, test_std = mean_std("test", "f1_macro")
    val_mean, val_std = mean_std("validation", "f1_macro")
    val_pr_mean, _ = mean_std("validation", "pr_auc")
    val_recall_mean, _ = mean_std("validation", "recall_scam")
    summary = {
        "protocol_version": 2, "git_commit": ModelCache.git_commit(), "embedding": embedding,
        "configuration": winner,
        "test_f1_macro_mean": test_mean, "test_f1_macro_std": test_std,
        "val_f1_macro_mean": val_mean, "val_f1_macro_std": val_std,
        "val_pr_auc_mean": val_pr_mean, "val_recall_scam_mean": val_recall_mean,
        "seeds": seed_rows,
    }
    path = _transformer_results_path("finalists", embedding, "summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _temperature(y_true: np.ndarray, probability: np.ndarray) -> float:
    logits = np.log(np.clip(probability, 1e-7, 1 - 1e-7) / np.clip(1 - probability, 1e-7, 1))
    temperatures = np.logspace(-2, 2, 401)
    losses = []
    for temperature in temperatures:
        calibrated = 1 / (1 + np.exp(-np.clip(logits / temperature, -50, 50)))
        losses.append(-np.mean(y_true * np.log(calibrated + 1e-12) + (1 - y_true) * np.log(1 - calibrated + 1e-12)))
    return float(temperatures[int(np.argmin(losses))])


def finalize_transformer_pipeline() -> dict:
    """Consolida finalistas e ensembles apenas a partir dos caches produzidos."""
    finalists = []
    bundles = {}
    for embedding in TRANSFORMER_EMBEDDINGS:
        path = _transformer_results_path("finalists", embedding, "summary.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Finalista ausente: {path}")
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        finalists.append(summary)
        seed42 = next(row for row in summary["seeds"] if row["seed"] == 42)
        bundles[embedding] = {
            "internal": ModelCache.load_prediction_bundle("transformer", seed42["run_id"], "internal_validation"),
            "validation": ModelCache.load_prediction_bundle("transformer", seed42["run_id"], "validation"),
        }

    reference_ids = bundles[TRANSFORMER_EMBEDDINGS[0]]["validation"].get("sample_ids")
    reference_y = bundles[TRANSFORMER_EMBEDDINGS[0]]["validation"]["y_true"]
    internal_reference_ids = bundles[TRANSFORMER_EMBEDDINGS[0]]["internal"].get("sample_ids")
    internal_reference_y = bundles[TRANSFORMER_EMBEDDINGS[0]]["internal"]["y_true"]
    for embedding in TRANSFORMER_EMBEDDINGS[1:]:
        candidate = bundles[embedding]["validation"]
        if reference_ids is None or "sample_ids" not in candidate or not np.array_equal(reference_ids, candidate["sample_ids"]):
            raise ValueError(f"IDs desalinhados no dataset_validation: {embedding}")
        if not np.array_equal(reference_y, candidate["y_true"]):
            raise ValueError(f"Rótulos desalinhados no dataset_validation: {embedding}")
        internal = bundles[embedding]["internal"]
        if internal_reference_ids is None or "sample_ids" not in internal or not np.array_equal(internal_reference_ids, internal["sample_ids"]):
            raise ValueError(f"IDs desalinhados na validação interna: {embedding}")
        if not np.array_equal(internal_reference_y, internal["y_true"]):
            raise ValueError(f"Rótulos desalinhados na validação interna: {embedding}")

    predictions = np.stack([bundles[e]["validation"]["y_pred"] for e in TRANSFORMER_EMBEDDINGS])
    probabilities = np.stack([bundles[e]["validation"]["y_prob"] for e in TRANSFORMER_EMBEDDINGS])
    raw_logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
    temperatures = []
    for embedding in TRANSFORMER_EMBEDDINGS:
        internal = bundles[embedding]["internal"]
        temperatures.append(_temperature(internal["y_true"], internal["y_prob"]))

    ensemble_outputs = {
        "majority": ((predictions.sum(axis=0) >= 2).astype(int), probabilities.mean(axis=0)),
        "soft": ((probabilities.mean(axis=0) >= 0.5).astype(int), probabilities.mean(axis=0)),
    }
    calibrated_probability = 1 / (1 + np.exp(-np.clip((raw_logits / np.asarray(temperatures)[:, None]).sum(axis=0), -50, 50)))
    ensemble_outputs["calibrated_logit_sum"] = (
        (calibrated_probability >= 0.5).astype(int), calibrated_probability,
    )

    ensembles = []
    for name, (y_pred, y_prob) in ensemble_outputs.items():
        metric = compute_metrics(reference_y, y_pred, y_prob=y_prob, model_name=name, dataset_name="validation")
        ensembles.append({"name": name, **metric})
        ModelCache.save_predictions(
            "transformer", "ensemble", name, reference_y, y_pred, y_prob,
            sample_ids=reference_ids,
        )

    ranking = [
        {"name": row["embedding"], "kind": "individual", "val_f1_macro": row["val_f1_macro_mean"],
         "val_pr_auc": row["val_pr_auc_mean"], "val_recall_scam": row["val_recall_scam_mean"],
         "std": row["val_f1_macro_std"]}
        for row in finalists
    ] + [
        {"name": row["name"], "kind": "ensemble", "val_f1_macro": row["f1_macro"],
         "val_pr_auc": row["pr_auc"], "val_recall_scam": row["recall_scam"], "std": 0.0}
        for row in ensembles
    ]
    ranking.sort(key=lambda row: (-row["val_f1_macro"], -row["val_pr_auc"], -row["val_recall_scam"]))
    final = {
        "protocol_version": 2, "git_commit": ModelCache.git_commit(),
        "temperatures": dict(zip(TRANSFORMER_EMBEDDINGS, temperatures)),
        "finalists": finalists, "ensembles": ensembles, "ranking": ranking,
        "winner": ranking[0],
    }
    path = _transformer_results_path("pipeline_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    return final


def load_transformer_pipeline_summary() -> dict:
    """Loader cache-only para o notebook final."""
    path = _transformer_results_path("pipeline_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache final ausente: {path}. Execute backend/slurm/submit_full_pipeline.sh --transformer."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_transformer_result(
    embedding: str,
    dataset_name: str = "test_internal",
    pct: Optional[int] = None,
    with_augmented: bool = True,
    batch_size: int = 32,
) -> ExperimentResult:
    """Carrega o finalista seed 42 sem abrir checkpoint ou dataset."""
    if pct is not None:
        raise ValueError("A nova pipeline final não usa resultados parciais por pct")
    summary_path = _transformer_results_path("finalists", embedding, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Finalista ausente: {summary_path}. Execute o pipeline SLURM transformer."
        )
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    seed_result = next(row for row in summary["seeds"] if row["seed"] == 42)
    run_id = seed_result["run_id"]
    config = {"strategy": "transformer", "embedding": embedding, "seed": 42}
    saved_history = ModelCache.load_history("transformer", run_id)
    saved_metrics = ModelCache.load_metrics("transformer", run_id)
    df_history = saved_history if (saved_history is not None and not saved_history.empty) else pd.DataFrame()

    metrics_key = "val" if dataset_name in ["validation", "validacao_externa"] else "test"
    if saved_metrics and isinstance(saved_metrics, dict):
        if metrics_key in saved_metrics:
            df_metrics = pd.DataFrame([saved_metrics[metrics_key]])
        else:
            df_metrics = pd.DataFrame([saved_metrics])
    else:
        df_metrics = pd.DataFrame()

    bundle = ModelCache.load_prediction_bundle("transformer", run_id, dataset_name)
    y_true, y_pred, y_prob = bundle["y_true"], bundle["y_pred"], bundle["y_prob"]

    return ExperimentResult(
        strategy="transformer", config=config, model=None,
        df_history=df_history, df_metrics=df_metrics,
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        artifacts_dir=ModelCache.artifacts_dir("transformer", run_id),
    )


def load_all_transformer_results(
    embeddings: Optional[List[str]] = None,
    dataset_name: str = "test_internal",
    pct: Optional[int] = None,
) -> List[ExperimentResult]:
    """
    Carrega a lista de ExperimentResult de todos os embeddings salvos para um dataset específico.
    """
    embeddings = embeddings or ["voyage", "bge", "openai"]
    return [load_transformer_result(emb, dataset_name=dataset_name, pct=pct) for emb in embeddings]


def evaluate_ensemble_from_results(
    results_list: List[ExperimentResult],
    dataset_name: str = "test",
    voting: str = "majority",
) -> ExperimentResult:
    """
    Avalia o Ensemble combinando os resultados de uma lista de ExperimentResult.
    
    Args:
        results_list: lista de ExperimentResult (voyage, bge, openai).
        dataset_name: nome do dataset avaliado.
        voting: 'majority' (votação por maioria 2/3, padrão do backup) ou 'soft' (média de probabilidades).
    """
    all_preds = {}
    all_probs = {}
    y_true_global = None

    for res in results_list:
        emb = res.config.get("embedding", "unknown")
        if res.y_prob is None or res.y_true is None:
            raise ValueError(f"Resultado para {emb} não possui probabilidades ou rótulos salvos.")
        
        all_preds[emb] = res.y_pred
        all_probs[emb] = res.y_prob

        if y_true_global is None:
            y_true_global = res.y_true
        else:
            if not np.array_equal(y_true_global, res.y_true):
                print(f"  [AVISO] Os rótulos de {emb} possuem ordem diferente de referência! ({len(res.y_true)} vs {len(y_true_global)})")

    preds_stack = np.stack(list(all_preds.values()), axis=0)
    probs_stack = np.stack(list(all_probs.values()), axis=0)

    y_prob_ensemble = probs_stack.mean(axis=0)

    if voting == "majority":
        # Votação por maioria simples (2 ou mais modelos votando positivo = 1)
        y_pred_ensemble = ((preds_stack.sum(axis=0)) >= 2).astype(int)
    else:
        # Votação por média de probabilidades (soft voting)
        y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)

    metrics_ensemble = compute_metrics(
        y_true_global, y_pred_ensemble, y_prob=y_prob_ensemble,
        model_name="Ensemble", dataset_name=dataset_name
    )

    print(f"\n  [Ensemble ({voting.capitalize()} Voting - {dataset_name})] acc={metrics_ensemble['accuracy']:.4f}  "
          f"f1_macro={metrics_ensemble['f1_macro']:.4f}  f1_scam={metrics_ensemble['f1_scam']:.4f}")

    extra = {
        "individual_preds": all_preds,
        "individual_probs": all_probs,
        "voting_mode": voting,
    }

    result = ExperimentResult(
        strategy="transformer",
        config={"ensemble": [r.config.get("embedding") for r in results_list], "dataset": dataset_name, "voting": voting},
        df_metrics=pd.DataFrame([metrics_ensemble]),
        y_true=y_true_global,
        y_pred=y_pred_ensemble,
        y_prob=y_prob_ensemble,
        extra=extra,
    )
    return result


def plot_ensemble_diagnostics(
    results_list: List[ExperimentResult],
    ensemble_result: ExperimentResult,
) -> pd.DataFrame:
    """
    Gera as 3 análises diagnósticas replicadas do notebook backup:
    1. Relatório Detalhado de Desempenho por Modelo e Gráfico Comparativo de F1-Score.
    2. Gráfico de Quantidade de vezes em que o modelo errou, mas foi 'salvo' pelos outros dois.
    3. Mapa de Correlação / Concordância de Acertos/Erros entre os Transformers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = ensemble_result.y_true
    y_pred_ens = ensemble_result.y_pred

    preds_dict = {}
    for res in results_list:
        emb = res.config.get("embedding", "").capitalize() or "Modelo"
        preds_dict[emb] = res.y_pred
    preds_dict["Ensemble"] = y_pred_ens

    # 1. Relatório Detalhado
    rows = []
    f1_scores = {}
    for nome, pred in preds_dict.items():
        p = precision_score(y_true, pred, zero_division=0) * 100
        r = recall_score(y_true, pred, zero_division=0) * 100
        f1 = f1_score(y_true, pred, zero_division=0) * 100
        f1_macro = f1_score(y_true, pred, average="macro", zero_division=0) * 100
        f1_scores[nome] = f1
        rows.append({
            "Modelo": nome,
            "Precisão (%)": p,
            "Recall (%)": r,
            "F1 Scam (%)": f1,
            "F1 Macro (%)": f1_macro
        })

    df_report = pd.DataFrame(rows)
    print("=== RELATÓRIO DETALHADO (Foco na Classe 1: Scam) ===\n")
    for _, row in df_report.iterrows():
        print(f"--- {row['Modelo']} ---")
        print(f"Precisão (Alarmes verdadeiros): {row['Precisão (%)']:.1f}%")
        print(f"Recall   (Scams capturados):   {row['Recall (%)']:.1f}%")
        print(f"F1-Score (Scam):               {row['F1 Scam (%)']:.1f}%\n")

    # Gráfico 1: Comparativo F1-Score
    fig, ax = plt.subplots(figsize=(8, 5))
    cores = ['#6baed6', '#3182bd', '#08519c', '#756bb1']
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), hue=list(f1_scores.keys()), palette=cores[:len(f1_scores)], legend=False, ax=ax)
    ax.set_title("Comparativo de Desempenho: F1-Score (Classe Scam)", fontsize=14, pad=15)

    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 105)
    for i, v in enumerate(f1_scores.values()):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.show()

    # 2. Amostras Salvas pelo Ensemble
    salvos = {}
    for res in results_list:
        emb = res.config.get("embedding", "").capitalize()
        salvos[emb] = ((res.y_pred != y_true) & (y_pred_ens == y_true)).sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=list(salvos.keys()), y=list(salvos.values()), color='lightcoral', ax=ax)
    ax.set_title('Quantas vezes o modelo errou, mas foi "salvo" pelos outros dois?')
    ax.set_ylabel('Quantidade de Amostras')
    for i, v in enumerate(salvos.values()):
        ax.text(i, v + 0.5, f"{v}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3. Mapa de Correlação de Acertos
    df_acertos = pd.DataFrame({
        res.config.get("embedding", "").capitalize(): (res.y_pred == y_true)
        for res in results_list
    })
    matriz_concordancia = df_acertos.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matriz_concordancia, annot=True, fmt=".3f", cmap='coolwarm', vmin=0, vmax=1, ax=ax)
    ax.set_title("Mapa de Correlação de Acertos entre Transformers")
    plt.tight_layout()
    plt.show()

    return df_report



def _evaluate_dl(model, dl, device):
    """Roda inferência em um DataLoader, retorna y_true, y_pred, y_prob."""
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    with torch.no_grad():
        for X_b, y_b, mask in dl:
            X_b, mask = X_b.to(device), mask.to(device)
            out = model(X_b, padding_mask=mask)
            if out.dim() > 1 and out.shape[1] == 1:
                out = out.squeeze(1)
            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_prob_all.extend(prob)
            y_pred_all.extend(pred)
            y_true_all.extend(y_b.numpy())
    return np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)


def load_trained_transformers(
    embeddings: Optional[List[str]] = None,
    pct: Optional[int] = None,
) -> Dict[str, object]:
    """Carrega os Transformers já treinados do cache."""
    embeddings = embeddings or ["voyage", "bge", "openai"]
    models = {}
    for emb in embeddings:
        dim = get_embedding_dim(emb)
        try:
            models[emb] = ModelCache.transformer_load(emb, dim, pct=pct)
            print(f"[transformer] Carregado: {emb}")
        except FileNotFoundError as e:
            print(f"[transformer] AVISO: {e}")
    return models


def load_validation_dataloaders(
    embeddings: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict[str, object]:
    """Carrega os DataLoaders do dataset de validação externa (dataset_validation) para os embeddings solicitados."""
    embeddings = embeddings or ["voyage", "bge", "openai"]
    dataloaders = {}
    for emb in embeddings:
        seqs, lbls = load_sequence_embeddings("validation", emb, with_augmented=False)
        dataloaders[emb] = _make_single_dataloader(seqs, lbls, batch_size=batch_size, shuffle=False)
    return dataloaders


def evaluate_ensemble(
    models_dict: Dict[str, object],
    dataloaders_dict: Dict[str, object],
    dataset_name: str = "test",
) -> ExperimentResult:
    """Realiza votação por média de probabilidades entre múltiplos Transformers usando DataLoaders."""
    device = _get_device()
    all_preds = {}
    all_probs = {}
    y_true_global = None

    for emb, model in models_dict.items():
        dl = dataloaders_dict[emb]
        y_true, y_pred, y_prob = _evaluate_dl(model, dl, device)
        all_preds[emb] = y_pred
        all_probs[emb] = y_prob
        if y_true_global is None:
            y_true_global = y_true
        metrics = compute_metrics(y_true, y_pred, y_prob=y_prob,
                                   model_name=f"Transformer_{emb}", dataset_name=dataset_name)
        print(f"  [{emb}] acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}  "
              f"f1_scam={metrics['f1_scam']:.4f}")

    probs_stack = np.stack(list(all_probs.values()), axis=0)
    y_prob_ensemble = probs_stack.mean(axis=0)
    y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)

    metrics_ensemble = compute_metrics(y_true_global, y_pred_ensemble, y_prob=y_prob_ensemble,
                                        model_name="Ensemble", dataset_name=dataset_name)
    print(f"\n  [Ensemble ({dataset_name})] acc={metrics_ensemble['accuracy']:.4f}  "
          f"f1_macro={metrics_ensemble['f1_macro']:.4f}  f1_scam={metrics_ensemble['f1_scam']:.4f}")

    extra = {
        "individual_preds": all_preds,
        "individual_probs": all_probs,
    }

    result = ExperimentResult(
        strategy="transformer",
        config={"ensemble": list(models_dict.keys()), "dataset": dataset_name},
        df_metrics=pd.DataFrame([metrics_ensemble]),
        y_true=y_true_global,
        y_pred=y_pred_ensemble,
        y_prob=y_prob_ensemble,
        extra=extra,
    )
    return result



def main(argv=None) -> int:
    """CLI do pipeline Transformer, usado pelo array SLURM existente."""
    import argparse
    parser = argparse.ArgumentParser(description="Busca e consolidação dos Transformers")
    parser.add_argument("--mode", required=True, choices=["search", "finalist", "finalize"])
    parser.add_argument("--embedding", choices=TRANSFORMER_EMBEDDINGS)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args(argv)
    if args.mode in {"search", "finalist"} and not args.embedding:
        parser.error("--embedding é obrigatório para search/finalist")
    if args.mode == "search":
        run_transformer_search(args.embedding, force_retrain=args.force_retrain)
    elif args.mode == "finalist":
        run_transformer_finalist(args.embedding, force_retrain=args.force_retrain)
    else:
        finalize_transformer_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
