"""
machine_learning/cache.py
--------------------------
Cache de experimentos centralizado em experiment_results/.

Estrutura de pastas em experiment_results/:
  classical/
    {origin}__{path_data}__{embedding}__{model_class_name}/
      model.joblib
      config.json
      metrics.json
      history.csv
  fcnn/
    {'augmented'|'no_augmented'}__{path_data}/
      model.pth
      config.json
      metrics.json
      history.csv
  transformer/
    {embedding}__pct{'none'|pct}/
      model.pth
      config.json
      metrics.json
      history.csv
  unsupervised/
    {embedding}__{model_type}__{dim}d__{reducer}/
      model.pkl
      reducer.pkl
      config.json
      metrics.json
      history.csv
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPERIMENT_RESULTS_DIR = os.path.join(_BACKEND_DIR, "experiment_results")


def _exp_dir(strategy: str, run_id: str) -> str:
    """Caminho para o diretório de um experimento específico."""
    return os.path.join(_EXPERIMENT_RESULTS_DIR, strategy, run_id)


def _run_id(config: Dict[str, Any]) -> str:
    """Gera um ID único legível para uma configuração de experimento."""
    import hashlib
    parts = []
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, list):
            v = "_".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    res = "__".join(parts).replace(" ", "").replace("/", "_").replace("[", "").replace("]", "")
    if len(res) > 80:
        h = hashlib.md5(res.encode("utf-8")).hexdigest()[:8]
        res = f"{res[:70]}_{h}"
    return res



class ModelCache:
    """
    Gerencia a persistência e carregamento de modelos e experimentos exclusivamente
    a partir de experiment_results/.
    """

    # ── Classical ML ─────────────────────────────────────────────

    @staticmethod
    def classical_folder(origin: str, path_data: str, embedding: str, model_class_name: str) -> str:
        return f"{origin}__{path_data}__{embedding}__{model_class_name}"

    @staticmethod
    def classical_path(origin: str, path_data: str, embedding: str, model_class_name: str) -> str:
        folder = ModelCache.classical_folder(origin, path_data, embedding, model_class_name)
        return os.path.join(_EXPERIMENT_RESULTS_DIR, "classical", folder, "model.joblib")

    @staticmethod
    def classical_exists(origin: str, path_data: str, embedding: str, model_class_name: str) -> bool:
        return os.path.exists(ModelCache.classical_path(origin, path_data, embedding, model_class_name))

    @staticmethod
    def classical_load(origin: str, path_data: str, embedding: str, model_class_name: str):
        import joblib
        path = ModelCache.classical_path(origin, path_data, embedding, model_class_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo classical não encontrado: {path}")
        print(f"[cache] Carregando modelo existente: {os.path.relpath(path, _BACKEND_DIR)}")
        return joblib.load(path)

    @staticmethod
    def classical_save(model, origin: str, path_data: str, embedding: str):
        import joblib
        model_class_name = type(model).__name__
        folder = ModelCache.classical_folder(origin, path_data, embedding, model_class_name)
        d = os.path.join(_EXPERIMENT_RESULTS_DIR, "classical", folder)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "model.joblib")
        joblib.dump(model, path)
        print(f"[cache] Modelo salvo: {os.path.relpath(path, _BACKEND_DIR)}")
        return path

    # ── FCNN ─────────────────────────────────────────────────────

    @staticmethod
    def fcnn_folder(with_augmented: bool, path_data: str) -> str:
        prefix = "augmented" if with_augmented else "no_augmented"
        return f"{prefix}__{path_data}"

    @staticmethod
    def fcnn_path(with_augmented: bool, path_data: str) -> str:
        folder = ModelCache.fcnn_folder(with_augmented, path_data)
        return os.path.join(_EXPERIMENT_RESULTS_DIR, "fcnn", folder, "model.pth")

    @staticmethod
    def fcnn_exists(with_augmented: bool, path_data: str) -> bool:
        return os.path.exists(ModelCache.fcnn_path(with_augmented, path_data))

    @staticmethod
    def fcnn_load(with_augmented: bool, path_data: str, input_size: Optional[int] = None):
        import torch
        from .fcnn.fcnn_classifier import ScamClassifierFCNN
        path = ModelCache.fcnn_path(with_augmented, path_data)
        if not os.path.exists(path):
            raise FileNotFoundError(f"FCNN não encontrada: {path}")
        print(f"[cache] Carregando FCNN: {os.path.relpath(path, _BACKEND_DIR)}")
        state = torch.load(path, map_location="cpu")
        if "network.0.weight" in state:
            checkpoint_input_size = state["network.0.weight"].shape[1]
        else:
            checkpoint_input_size = input_size or 4096
        model = ScamClassifierFCNN(input_size=checkpoint_input_size)
        model.load_state_dict(state)
        return model

    @staticmethod
    def fcnn_save(model, with_augmented: bool, path_data: str):
        import torch
        folder = ModelCache.fcnn_folder(with_augmented, path_data)
        d = os.path.join(_EXPERIMENT_RESULTS_DIR, "fcnn", folder)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "model.pth")
        torch.save(model.to("cpu").state_dict(), path)
        print(f"[cache] FCNN salva: {os.path.relpath(path, _BACKEND_DIR)}")
        return path

    # ── Transformer ──────────────────────────────────────────────

    @staticmethod
    def transformer_folder(embedding: str, pct: Optional[int] = None) -> str:
        pct_str = f"{pct}" if pct is not None else "none"
        return f"{embedding}__pct{pct_str}"

    @staticmethod
    def transformer_path(embedding: str, pct: Optional[int] = None) -> str:
        folder = ModelCache.transformer_folder(embedding, pct)
        return os.path.join(_EXPERIMENT_RESULTS_DIR, "transformer", folder, "model.pth")

    @staticmethod
    def transformer_exists(embedding: str, pct: Optional[int] = None) -> bool:
        return os.path.exists(ModelCache.transformer_path(embedding, pct))

    @staticmethod
    def transformer_load(embedding: str, embedding_dim: int, pct: Optional[int] = None):
        import torch
        from .transformer.runner import _get_transformer_class
        TransformerClass = _get_transformer_class()

        path = ModelCache.transformer_path(embedding, pct)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Transformer não encontrado: {path}")
        print(f"[cache] Carregando Transformer: {os.path.relpath(path, _BACKEND_DIR)}")
        model = TransformerClass(embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model

    @staticmethod
    def transformer_save(model, embedding: str, pct: Optional[int] = None):
        import torch
        folder = ModelCache.transformer_folder(embedding, pct)
        d = os.path.join(_EXPERIMENT_RESULTS_DIR, "transformer", folder)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "model.pth")
        torch.save(model.state_dict(), path)
        print(f"[cache] Transformer salvo: {os.path.relpath(path, _BACKEND_DIR)}")
        return path

    # ── Unsupervised ─────────────────────────────────────────────

    @staticmethod
    def params_hash(model_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Hash curto e determinístico dos hiperparâmetros do modelo, usado para não
        colidir o cache entre runs com o mesmo embedding/model_type/dim/reducer mas
        hiperparâmetros diferentes (ex: default vs. vencedor do grid search)."""
        import hashlib
        if not model_kwargs:
            return "default"
        items = sorted((k, str(v)) for k, v in model_kwargs.items())
        return hashlib.md5(str(items).encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def unsupervised_folder(embedding: str, model_type: str, dim: int, reducer: str = "pca", params_hash: str = "default") -> str:
        return f"{embedding}__{model_type}__{dim}d__{reducer}__{params_hash}"

    @staticmethod
    def unsupervised_path(embedding: str, model_type: str, dim: int, reducer: str = "pca", params_hash: str = "default") -> str:
        folder = ModelCache.unsupervised_folder(embedding, model_type, dim, reducer, params_hash)
        return os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", folder, "model.pkl")

    @staticmethod
    def reducer_path(embedding: str, model_type: str, dim: int, reducer: str = "pca", params_hash: str = "default") -> str:
        folder = ModelCache.unsupervised_folder(embedding, model_type, dim, reducer, params_hash)
        return os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", folder, "reducer.pkl")

    @staticmethod
    def unsupervised_exists(embedding: str, model_type: str, dim: int, reducer: str = "pca", params_hash: str = "default") -> bool:
        model_ok   = os.path.exists(ModelCache.unsupervised_path(embedding, model_type, dim, reducer, params_hash))
        reducer_ok = os.path.exists(ModelCache.reducer_path(embedding, model_type, dim, reducer, params_hash))
        return model_ok and reducer_ok

    @staticmethod
    def unsupervised_load(embedding: str, model_type: str, dim: int, reducer: str = "pca", params_hash: str = "default"):
        import joblib
        model_path   = ModelCache.unsupervised_path(embedding, model_type, dim, reducer, params_hash)
        reducer_path = ModelCache.reducer_path(embedding, model_type, dim, reducer, params_hash)
        if not os.path.exists(model_path) or not os.path.exists(reducer_path):
            raise FileNotFoundError(
                f"Modelo ou redutor não encontrado:\n  {model_path}\n  {reducer_path}"
            )
        print(f"[cache] Carregando {model_type.upper()} + {reducer.upper()} ({dim}d): {embedding}")
        return joblib.load(model_path), joblib.load(reducer_path)

    @staticmethod
    def unsupervised_save(model, reducer, embedding: str, model_type: str, dim: int, reducer_type: str = "pca", params_hash: str = "default"):
        import joblib
        folder = ModelCache.unsupervised_folder(embedding, model_type, dim, reducer_type, params_hash)
        d = os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", folder)
        os.makedirs(d, exist_ok=True)
        model_path   = os.path.join(d, "model.pkl")
        reducer_path = os.path.join(d, "reducer.pkl")
        joblib.dump(model, model_path)
        joblib.dump(reducer, reducer_path)
        print(f"[cache] {model_type.upper()} salvo: {os.path.relpath(model_path, _BACKEND_DIR)}")
        print(f"[cache] {reducer_type.upper()} salvo: {os.path.relpath(reducer_path, _BACKEND_DIR)}")

    # ── Unsupervised (modelos de texto bruto: CVDD / DATE) ───────

    @staticmethod
    def text_model_folder(model_type: str, embedding: str = "bge", params_hash: str = "default") -> str:
        return f"{embedding}__{model_type}__text__{params_hash}"

    @staticmethod
    def text_model_path(model_type: str, embedding: str = "bge", params_hash: str = "default") -> str:
        folder = ModelCache.text_model_folder(model_type, embedding, params_hash)
        return os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", folder, "model.pt")

    @staticmethod
    def text_model_exists(model_type: str, embedding: str = "bge", params_hash: str = "default") -> bool:
        p_hash = ModelCache.text_model_path(model_type, embedding, params_hash)
        p_legacy = os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", f"{embedding}__{model_type}__text", "model.pt")
        return os.path.exists(p_hash) or os.path.exists(p_legacy)

    @staticmethod
    def text_model_load(model_type: str, embedding: str = "bge", map_location: str = "cpu", params_hash: str = "default") -> dict:
        import torch
        p_hash = ModelCache.text_model_path(model_type, embedding, params_hash)
        if os.path.exists(p_hash):
            path = p_hash
        else:
            path = os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", f"{embedding}__{model_type}__text", "model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo de texto não encontrado: {path}")
        print(f"[cache] Carregando {model_type.upper()} (texto): {embedding}")
        return torch.load(path, map_location=map_location)

    @staticmethod
    def text_model_save(state: dict, model_type: str, embedding: str = "bge", params_hash: str = "default"):
        """Salva state_dict + config de um modelo de texto (CVDD/DATE) via torch.save."""
        import torch
        folder = ModelCache.text_model_folder(model_type, embedding, params_hash)
        d = os.path.join(_EXPERIMENT_RESULTS_DIR, "unsupervised", folder)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "model.pt")
        torch.save(state, path)
        print(f"[cache] {model_type.upper()} (texto) salvo: {os.path.relpath(path, _BACKEND_DIR)}")
        return path

    # ── Métricas e CSVs de experimento ───────────────────────────

    @staticmethod
    def save_metrics(strategy: str, run_id: str, metrics: dict):
        """Salva métricas em JSON no diretório de experimentos."""
        d = _exp_dir(strategy, run_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"[cache] Métricas salvas: {os.path.relpath(path, _BACKEND_DIR)}")

    @staticmethod
    def load_metrics(strategy: str, run_id: str) -> Optional[dict]:
        path = os.path.join(_exp_dir(strategy, run_id), "metrics.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_history(strategy: str, run_id: str, df_history: pd.DataFrame):
        """Salva histórico de treino em CSV."""
        d = _exp_dir(strategy, run_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "history.csv")
        df_history.to_csv(path, index=False)
        print(f"[cache] Histórico salvo: {os.path.relpath(path, _BACKEND_DIR)}")

    @staticmethod
    def load_history(strategy: str, run_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(_exp_dir(strategy, run_id), "history.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    @staticmethod
    def save_predictions(strategy: str, run_id: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """Salva arrays de predição em formato .npz no diretório do experimento."""
        import numpy as np
        d = _exp_dir(strategy, run_id)
        os.makedirs(d, exist_ok=True)
        filename = f"predictions_{dataset_name}.npz"
        path = os.path.join(d, filename)
        np.savez_compressed(path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        print(f"[cache] Predições ({dataset_name}) salvas: {os.path.relpath(path, _BACKEND_DIR)}")

    @staticmethod
    def load_predictions(strategy: str, run_id: str, dataset_name: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Carrega arrays de predição salvos em .npz."""
        import numpy as np
        path = os.path.join(_exp_dir(strategy, run_id), f"predictions_{dataset_name}.npz")
        if not os.path.exists(path):
            return None, None, None
        data = np.load(path)
        return data["y_true"], data["y_pred"], data["y_prob"]

    @staticmethod
    def artifacts_dir(strategy: str, run_id: str) -> str:
        d = _exp_dir(strategy, run_id)
        os.makedirs(d, exist_ok=True)
        return d

