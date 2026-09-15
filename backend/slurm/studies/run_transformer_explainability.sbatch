#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/model_studies/%x_%A_%a.out
#SBATCH --error=slurm_logs/model_studies/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PROJECT_ROOT="${HORUS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"
GRID_FILE="${1:?informe a grade}"
[[ "$GRID_FILE" == /* ]] || GRID_FILE="$PROJECT_ROOT/$GRID_FILE"
[[ -r "$GRID_FILE" ]] || { echo "Grade ausente ou ilegível: $GRID_FILE" >&2; exit 2; }
cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_gpu_final
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA indisponível"'
IFS=$'\t' read -r embedding seed < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID_FILE")
force_args=(); [[ "${STUDY_FORCE:-0}" == "1" ]] && force_args=(--force)
python -m machine_learning.studies explainability --embedding "$embedding" --seed "$seed" --split all "${force_args[@]}"
