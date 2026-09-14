"""
machine_learning/unsupervised/text_data.py
---------------------------------------------
Carregamento de texto bruto para os modelos CVDD e DATE (Manolache et al., 2021),
que operam diretamente sobre o texto (via BGE) em vez de embeddings pré-computados.

Somente o CSV 'all_data.csv' possui a coluna de texto bruta necessária tanto para
treino quanto para validação (o recorte 'suspect_turns' só existe pré-processado
como embeddings, sem raw CSV equivalente em dataset_validation/).
"""
from __future__ import annotations
import os
from typing import Optional, Tuple, List
import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASETS_ROOT = os.path.join(_BACKEND_DIR, "datasets")


def _raw_csv_path(split: str) -> str:
    folder = "dataset_train" if split == "train" else "dataset_validation"
    return os.path.join(_DATASETS_ROOT, folder, "raw", "all_data.csv")


def load_raw_text(
    split: str,
    with_augmented: bool = False,
    label_filter: Optional[int] = None,
) -> Tuple[List[str], "pd.Series"]:
    """
    Carrega (texto, label) a partir do CSV bruto ('all_data.csv').

    Args:
        split: 'train' ou 'validation'.
        with_augmented: se False (padrão para unsupervised), mantém somente origin='external'.
        label_filter: se não None, filtra apenas amostras com esse label (0=Ham, 1=Scam).

    Returns:
        (texts, labels)
    """
    path = _raw_csv_path(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV bruto não encontrado: {path}")
    df = pd.read_csv(path)

    text_col = "text_last_n_turns" if "text_last_n_turns" in df.columns else "text"
    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    if not with_augmented and "origin" in df.columns:
        df = df[df["origin"] == "external"].reset_index(drop=True)

    if label_filter is not None:
        df = df[df["label"] == label_filter].reset_index(drop=True)

    print(f"  [text_data] {split}: {len(df)} amostras carregadas de {os.path.relpath(path, _BACKEND_DIR)}")
    return df[text_col].tolist(), df["label"].values


def load_ham_only_text(split: str) -> List[str]:
    """Carrega somente textos Ham (label=0), usados para treino não-supervisionado."""
    texts, _ = load_raw_text(split, with_augmented=False, label_filter=0)
    return texts
