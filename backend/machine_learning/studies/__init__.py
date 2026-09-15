"""Cache-first studies for the Transformer and unsupervised finalists."""

from .manifest import StudyManifest, load_study_manifest, select_study_finalists
from .dataset_size import (
    run_transformer_dataset_size_study,
    run_unsupervised_dataset_size_study,
    aggregate_dataset_size_results,
)
from .transformer_analysis import (
    analyze_transformer_complementarity,
    run_transformer_threshold_study,
)
from .explainability import (
    extract_transformer_head_attention,
    analyze_transformer_head_ablation,
    build_transformer_explainability_report,
)

__all__ = [
    "StudyManifest", "load_study_manifest", "select_study_finalists",
    "run_transformer_dataset_size_study", "run_unsupervised_dataset_size_study",
    "aggregate_dataset_size_results", "analyze_transformer_complementarity",
    "run_transformer_threshold_study", "extract_transformer_head_attention",
    "analyze_transformer_head_ablation", "build_transformer_explainability_report",
]
