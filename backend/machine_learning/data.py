"""
machine_learning/data.py
------------------------
Carregamento unificado de dados para todas as estratégias.

Modos de carregamento:
- load_single_embedding()   -> Classical ML e Unsupervised (1 embedding/conversa)
- load_concat_embeddings()  -> FCNN (concatena vários embeddings)
- load_sequence_embeddings()-> Transformer (sequência de msgs por conversa)
- load_ham_only()           -> Unsupervised: treino somente com Ham
"""
from __future__ import annotations
import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASETS_ROOT = os.path.join(_BACKEND_DIR, 'datasets')

try:
    from config import EMBEDDERS
except ImportError:
    import importlib.util
    _config_path = os.path.join(_BACKEND_DIR, "config.py")
    _spec = importlib.util.spec_from_file_location("config", _config_path)
    _config_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_config_mod)
    EMBEDDERS = _config_mod.EMBEDDERS


def _dataset_split_name(split: str) -> str:
    """Converte 'train' / 'validation' / 'val' para o nome de pasta correto."""
    if split in ("train",):
        return "dataset_train"
    if split in ("validation", "val"):
        return "dataset_validation"
    raise ValueError(f"split deve ser 'train' ou 'validation', recebido: '{split}'")


def _parquet_path(split: str, path_data: str, embedding: str, transformer: bool = False) -> str:
    """
    Monta o caminho completo para um arquivo parquet.
    
    transformer=True -> usa messages_transformer/ (sequência por mensagem)
    transformer=False -> usa all_data/ ou suspect_turns/ (embedding único)
    """
    folder = "messages_transformer" if transformer else path_data
    return os.path.join(
        _DATASETS_ROOT,
        _dataset_split_name(split),
        "processed",
        folder,
        f"{embedding}_chunks.parquet",
    )


def _load_parquet(path: str, with_augmented: bool = True) -> pd.DataFrame:
    """Carrega parquet e aplica filtro de augmentation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet não encontrado: {path}")
    df = pd.read_parquet(path)
    if not with_augmented and "origin" in df.columns:
        df = df[df["origin"] == "external"].reset_index(drop=True)
        print(f"  [data] Filtro sem augmentação -> {len(df)} amostras.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def load_single_embedding(
    split: str,
    path_data: str,
    embedding: str,
    with_augmented: bool = True,
    label_filter: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega um único embedding por conversa.
    Usado por: Classical ML, Unsupervised (validação com todas as classes).

    Args:
        split: 'train' ou 'validation'
        path_data: 'all_data' ou 'suspect_turns'
        embedding: chave do embedder (ex: 'voyage', 'bge')
        with_augmented: se False, filtra apenas amostras externas (origin='external')
        label_filter: se não None, retorna somente amostras com esse label (ex: 0 para Ham)

    Returns:
        (X, y): arrays numpy
    """
    path = _parquet_path(split, path_data, embedding, transformer=False)
    print(f"[data] Carregando: {os.path.basename(path)}")
    df = _load_parquet(path, with_augmented=with_augmented)

    if label_filter is not None:
        df = df[df["label"] == label_filter].reset_index(drop=True)
        print(f"  [data] Filtro label={label_filter} -> {len(df)} amostras.")

    X = np.array(df["text_embedded"].tolist(), dtype=np.float32)
    y = df["label"].values
    print(f"  [data] Shape: X={X.shape}, y={y.shape}")
    return X, y


def load_ham_only(
    split: str,
    path_data: str,
    embedding: str,
) -> np.ndarray:
    """
    Carrega somente amostras Ham (label=0) para treino não-supervisionado.
    
    Returns:
        X: array numpy com embeddings de Ham
    """
    X, _ = load_single_embedding(
        split=split,
        path_data=path_data,
        embedding=embedding,
        with_augmented=False,   # unsupervised usa somente dados externos (sem augment)
        label_filter=0,
    )
    return X


def load_concat_embeddings(
    split: str,
    path_data: str,
    embedders: Optional[List[str]] = None,
    with_augmented: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega e concatena múltiplos embeddings por conversa.
    Usado por: FCNN (ex: voyage + openai + bge -> 6528 features).
    
    Valida que todos os embedders têm o mesmo número de amostras e labels idênticos.

    Returns:
        (X_concat, y): X concatenado horizontalmente, y do primeiro embedder
    """
    if embedders is None:
        embedders = list(EMBEDDERS.keys())

    print(f"[data] Concatenando {len(embedders)} embeddings: {embedders}")
    X_list = []
    y_ref = None

    for emb in embedders:
        path = _parquet_path(split, path_data, emb, transformer=False)
        print(f"  [data] {emb}: {os.path.basename(path)}")
        df = _load_parquet(path, with_augmented=with_augmented)

        X_emb = np.array(df["text_embedded"].tolist(), dtype=np.float32)
        y_emb = df["label"].values

        if y_ref is None:
            y_ref = y_emb
        else:
            if len(y_emb) != len(y_ref):
                raise ValueError(
                    f"Embedder '{emb}' tem {len(y_emb)} amostras, "
                    f"mas '{embedders[0]}' tem {len(y_ref)}. Os parquets devem ser alinhados."
                )

        X_list.append(X_emb)
        print(f"    Shape: {X_emb.shape}")

    X_concat = np.concatenate(X_list, axis=1)
    print(f"[data] Shape final concatenado: X={X_concat.shape}, y={y_ref.shape}")
    return X_concat, y_ref


def load_sequence_embeddings(
    split: str,
    embedding: str,
    with_augmented: bool = True,
) -> Tuple[list, np.ndarray]:
    """
    Carrega embeddings como SEQUÊNCIA de mensagens por conversa.
    Usado por: Transformer Ensemble (messages_transformer/).
    
    Cada conversa tem N mensagens -> lista de arrays (N_msgs, dim_emb).

    Returns:
        (sequences, labels):
            sequences: lista de arrays numpy, cada um com shape (n_turnos, dim_emb)
            labels: array numpy de labels por conversa
    """
    path = _parquet_path(split, path_data=None, embedding=embedding, transformer=True)
    print(f"[data] Carregando sequências: {os.path.basename(path)}")
    df = _load_parquet(path, with_augmented=with_augmented)

    sequences = []
    import json
    for emb_list in df["text_embedded"]:
        if isinstance(emb_list, str):
            try:
                emb_list = json.loads(emb_list)
            except Exception:
                import ast
                emb_list = ast.literal_eval(emb_list)
        if isinstance(emb_list, (list, np.ndarray, pd.Series)) and len(emb_list) > 0:
            if isinstance(emb_list[0], (list, np.ndarray, pd.Series)):
                arr = np.vstack([np.array(x, dtype=np.float32) for x in emb_list])
            else:
                arr = np.array(emb_list, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
        else:
            arr = np.array([emb_list], dtype=np.float32)
        sequences.append(arr)

    labels = df["label"].values
    dims = [s.shape for s in sequences[:3]]
    print(f"  [data] {len(sequences)} conversas | Primeiras dims: {dims}")
    return sequences, labels


def get_embedding_dim(embedding: str) -> int:
    """Retorna a dimensão do embedding (para instanciar modelos)."""
    _KNOWN_DIMS = {
        "voyage":  1024,
        "openai":  3072,
        "bge":     1024,
        "e5":      1024,
        "minilm":  384,
    }
    if embedding in _KNOWN_DIMS:
        return _KNOWN_DIMS[embedding]
    # Tenta ler do parquet para descobrir
    try:
        path = _parquet_path("train", "all_data", embedding, transformer=False)
        df = pd.read_parquet(path).head(1)
        emb = df["text_embedded"].iloc[0]
        if isinstance(emb, list):
            return len(emb)
    except Exception:
        pass
    raise ValueError(f"Dimensão desconhecida para embedding '{embedding}'. Adicione em _KNOWN_DIMS.")
