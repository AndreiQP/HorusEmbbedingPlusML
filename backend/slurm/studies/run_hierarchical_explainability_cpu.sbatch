#!/bin/bash
#SBATCH --output=slurm_logs/model_studies/%x_%j.out
#SBATCH --error=slurm_logs/model_studies/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PROJECT_ROOT="${HORUS_PROJECT_ROOT:?HORUS_PROJECT_ROOT ausente}"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
OUTPUT_DIR="${HIER_STUDY_OUTPUT_DIR:-$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability}"
ACTION="${HIER_STUDY_ACTION:?HIER_STUDY_ACTION ausente}"

cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$ACTION" == "inventory" ]]; then
    python -m machine_learning.studies hierarchical-study \
        --action inventory --split all \
        --primary-seed "${HIER_STUDY_PRIMARY_SEED:-42}" \
        --checkpoint-root "$CHECKPOINT_ROOT" --output-dir "$OUTPUT_DIR"
elif [[ "$ACTION" == "aggregate" ]]; then
    python -m machine_learning.studies hierarchical-study \
        --action aggregate --split all \
        --shard-count "${HIER_STUDY_SHARD_COUNT:-32}" \
        --output-dir "$OUTPUT_DIR"
elif [[ "$ACTION" == "saturation-aggregate" ]]; then
    python -m machine_learning.studies hierarchical-study \
        --action saturation-aggregate --split all \
        --shard-count "${HIER_STUDY_SHARD_COUNT:-32}" \
        --output-dir "$OUTPUT_DIR"
else
    echo "Ação CPU inválida: $ACTION" >&2
    exit 2
fi
