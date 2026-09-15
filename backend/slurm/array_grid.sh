#!/bin/bash
#SBATCH --job-name=unsup_grid
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --mail-user=a169194@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Array job que varre a grade (embedding x dimensão) de UM único modelo.
# Não é chamado diretamente — use submit_all_models.sh, que dispara um job
# destes por modelo (todos em paralelo, com --job-name e grade próprios).
#
# Uso manual (para um modelo específico):
#   python3 backend/slurm/make_grid.py
#   sbatch --job-name=unsup_lof \
#          --array=0-$(($(wc -l < backend/experiment_results/slurm_grids/lof_tuning.txt)-1)) \
#          backend/slurm/array_grid.sh backend/experiment_results/slurm_grids/lof_tuning.txt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p backend/experiment_results/unsupervised slurm_logs

GRID_FILE="${1:?informe o arquivo de grade em backend/experiment_results/slurm_grids}"
FAMILY="${2:-unsupervised}"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID_FILE")"
if [[ "${FORCE_RETRAIN:-0}" == "1" && "$LINE" != *"--mode leaderboard"* ]]; then
    LINE="${LINE} --force-retrain"
fi
echo "[array_grid] grid=${GRID_FILE} task=${SLURM_ARRAY_TASK_ID} -> ${LINE}"

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
USE_GPU_ENV=0
REQUIRE_CUDA=0
if [[ "$FAMILY" == "transformer" ]]; then
    USE_GPU_ENV=1
    [[ "$LINE" != *"--mode finalize"* ]] && REQUIRE_CUDA=1
elif [[ "$LINE" == *"--model lunar"* || "$LINE" == *"--model svdd"* ]]; then
    USE_GPU_ENV=1
    REQUIRE_CUDA=1
fi

if [[ "$USE_GPU_ENV" == "1" ]]; then
    conda activate env_gpu_final
    if [[ "$REQUIRE_CUDA" == "1" ]]; then
        python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "ERRO: GPU CUDA indisponível no env_gpu_final")'
    fi
else
    conda activate env_py_3_12
fi
export PYTHONUNBUFFERED=1
cd backend

# Cada linha do grid já vem no formato: --mode ... --model ... --embedding ... --dim ...
if [[ "$FAMILY" == "transformer" ]]; then
    python -m machine_learning.transformer.runner ${LINE}
else
    python -m machine_learning.unsupervised.run_experiment ${LINE} \
        --output-json "experiment_results/unsupervised/slurm_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.json"
fi
