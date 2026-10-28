#!/bin/bash
#SBATCH --output=slurm_logs/model_studies/%x_%j.out
#SBATCH --error=slurm_logs/model_studies/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PROJECT_ROOT="${HORUS_PROJECT_ROOT:?HORUS_PROJECT_ROOT ausente}"
OUTPUT_DIR="${CONTEXT_BLOCK_OUTPUT_DIR:-$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability/context_blocks}"
SHARDS="${CONTEXT_BLOCK_SHARDS:-32}"
cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

python -m machine_learning.studies context-block-study \
    --action aggregate --shard-count "$SHARDS" --output-dir "$OUTPUT_DIR" --device cpu
