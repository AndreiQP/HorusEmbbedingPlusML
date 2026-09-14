"""
machine_learning — Biblioteca de ML padronizada para detecção de scam.

Importações rápidas:
    from machine_learning import train_classical, train_fcnn, train_transformer, train_anomaly
    from machine_learning import ExperimentResult, ModelCache, ThresholdCalibrator
"""

# Camada comum
from .result import ExperimentResult
from .cache import ModelCache
from .thresholds import ThresholdCalibrator
from .data import (
    load_single_embedding,
    load_concat_embeddings,
    load_sequence_embeddings,
    load_ham_only,
    get_embedding_dim,
)
from .evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_epoch_history,
    plot_dataset_size_curve,
    plot_loss_evolution_all_embeddings,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_pca_variance,
    plot_umap_2d,
    plot_dimension_study,
)
from .classical.classifier import get_all_classifiers
from .fcnn.fcnn_classifier import ScamClassifierFCNN

# Runners por estratégia
from .classical.runner import (
    train_classical,
    train_all_classical,
    learning_curve_classical,
)
from .fcnn.runner import (
    train_fcnn,
    learning_curve_fcnn,
    evaluate_fcnn_cross_turns,
    grid_search_fcnn,
)


from .transformer.runner import (
    train_transformer,
    load_trained_transformers,
    load_transformer_result,
    load_all_transformer_results,
    evaluate_ensemble,
    evaluate_ensemble_from_results,
    plot_ensemble_diagnostics,
    learning_curve_transformer,
)
from .unsupervised.runner import (
    train_anomaly,
    train_all_anomaly,
    compare_dimensions_anomaly,
    learning_curve_anomaly,
    grid_search_anomaly,
)
from .unsupervised.text_runner import train_cvdd, train_date

__all__ = [
    # Core
    "ExperimentResult", "ModelCache", "ThresholdCalibrator",
    "get_all_classifiers", "ScamClassifierFCNN",
    # Data
    "load_single_embedding", "load_concat_embeddings", "load_sequence_embeddings",
    "load_ham_only", "get_embedding_dim",
    # Evaluation
    "compute_metrics", "plot_confusion_matrix", "plot_epoch_history",
    "plot_dataset_size_curve", "plot_loss_evolution_all_embeddings", "plot_precision_recall_curve", "plot_roc_curve",
    "plot_pca_variance", "plot_umap_2d",

    # Runners
    "train_classical", "train_all_classical", "learning_curve_classical",
    "train_fcnn", "learning_curve_fcnn", "evaluate_fcnn_cross_turns", "grid_search_fcnn",
    "train_transformer", "load_trained_transformers", "load_transformer_result",


    "load_all_transformer_results", "evaluate_ensemble", "evaluate_ensemble_from_results",
    "plot_ensemble_diagnostics", "learning_curve_transformer",
    "train_anomaly", "learning_curve_anomaly", "grid_search_anomaly", "train_all_anomaly", "compare_dimensions_anomaly",
    "train_cvdd", "train_date",
]


