#!/bin/bash
#SBATCH --job-name=scriptmind_annotation
#SBATCH --output=slurm_logs/scriptmind/%x_%A_%a.out
#SBATCH --error=slurm_logs/scriptmind/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s
set -euo pipefail

MODE="${1:?mode required: gemini|local|adjudicate}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SCRIPTMIND_NUM_SHARDS:-32}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
mkdir -p "$PROJECT_ROOT/slurm_logs/scriptmind"

CONDA_SH="${HORUS_CONDA_SH:-/home/andrei.pinto/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
if [[ "$MODE" == "local" ]]; then
    conda activate "${HORUS_GPU_ENV:-env_gpu_final}"
    python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "CUDA unavailable")'
else
    conda activate "${HORUS_CPU_ENV:-env_py_3_12}"
    : "${GEMINI_API_KEY:?GEMINI_API_KEY is required for Gemini jobs}"
fi
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT/backend"

FORCE_ARGS=()
[[ "${SCRIPTMIND_FORCE:-0}" == "1" ]] && FORCE_ARGS+=(--force)
if [[ "$MODE" == "adjudicate" ]]; then
    python -m scriptmind adjudicate --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" "${FORCE_ARGS[@]}"
else
    python -m scriptmind annotate --provider "$MODE" --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" "${FORCE_ARGS[@]}"
fi
