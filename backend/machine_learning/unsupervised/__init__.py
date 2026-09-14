"""
machine_learning/unsupervised — Detecção de anomalias para scam detection.

Modelos sobre embeddings pré-computados (runner.py):
    train_anomaly(embedding, model_type=...)  model_type in {ocsvm, iforest, lof, lunar, svdd}
    learning_curve_anomaly, grid_search_anomaly, train_all_anomaly, compare_dimensions_anomaly

Modelos sobre texto bruto + BGE (text_runner.py):
    train_cvdd(), train_date()
"""
from .runner import (
    train_anomaly,
    learning_curve_anomaly,
    grid_search_anomaly,
    train_all_anomaly,
    compare_dimensions_anomaly,
)
from .text_runner import train_cvdd, train_date
