#!/bin/bash
#SBATCH --output=slurm_logs/model_studies/%x_%A_%a.out
#SBATCH --error=slurm_logs/model_studies/%x_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd -P)"
GRID_FILE="${1:?informe a grade}"
STRATEGY="${2:?informe a estratégia}"
cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
IFS=$'\t' read -r candidate fraction seed < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID_FILE")
force_args=(); [[ "${STUDY_FORCE:-0}" == "1" ]] && force_args=(--force)
python -m machine_learning.studies dataset-size --strategy "$STRATEGY" --candidate "$candidate" --fraction "$fraction" --seed "$seed" "${force_args[@]}"
