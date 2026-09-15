#!/bin/bash
#SBATCH --job-name=scriptmind_train
#SBATCH --output=slurm_logs/scriptmind/%x_%A_%a.out
#SBATCH --error=slurm_logs/scriptmind/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
EXPERIMENT_SET="${SCRIPTMIND_EXPERIMENT_SET:-primary}"
SEEDS=(42 52 62)
MODELS=(llama1b llama8b)

if [[ "$EXPERIMENT_SET" == "primary" ]]; then
    [[ "$TASK_ID" -lt 6 ]] || { echo "Primary task id must be 0..5" >&2; exit 2; }
    MODEL_KEY="${MODELS[$((TASK_ID / 3))]}"
    SEED="${SEEDS[$((TASK_ID % 3))]}"
    ABLATION="multitask"
    ORIGIN_MODE="all"
elif [[ "$EXPERIMENT_SET" == "ablations" ]]; then
    [[ "$TASK_ID" -lt 9 ]] || { echo "Ablation task id must be 0..8" >&2; exit 2; }
    VARIANT=$((TASK_ID / 3))
    MODEL_KEY="llama8b"
    SEED="${SEEDS[$((TASK_ID % 3))]}"
    case "$VARIANT" in
        0) ABLATION="detection_only"; ORIGIN_MODE="all" ;;
        1) ABLATION="no_next_utterance"; ORIGIN_MODE="all" ;;
        2) ABLATION="multitask"; ORIGIN_MODE="external" ;;
    esac
else
    echo "Unknown SCRIPTMIND_EXPERIMENT_SET: $EXPERIMENT_SET" >&2
    exit 2
fi

PROJECT_ROOT="${HORUS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"
mkdir -p "$PROJECT_ROOT/slurm_logs/scriptmind"
source "${HORUS_CONDA_SH:-/home/andrei.pinto/miniconda3/etc/profile.d/conda.sh}"
conda activate "${HORUS_GPU_ENV:-env_gpu_final}"
python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "CUDA unavailable")'
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT/backend"

FORCE_ARGS=()
[[ "${SCRIPTMIND_FORCE:-0}" == "1" ]] && FORCE_ARGS+=(--force)
python -m scriptmind train \
    --model-key "$MODEL_KEY" --seed "$SEED" --ablation "$ABLATION" --origin-mode "$ORIGIN_MODE" \
    "${FORCE_ARGS[@]}"
