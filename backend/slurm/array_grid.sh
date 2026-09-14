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
#          --array=0-$(($(wc -l < backend/slurm/grids/lof.txt)-1)) \
#          backend/slurm/array_grid.sh backend/slurm/grids/lof.txt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p backend/experiment_results/unsupervised slurm_logs

GRID_FILE="${1:?informe o arquivo de grade, ex: backend/slurm/grids/lof.txt}"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID_FILE")"
echo "[array_grid] grid=${GRID_FILE} task=${SLURM_ARRAY_TASK_ID} -> ${LINE}"

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh && conda activate env_py_3_12
cd backend

# Cada linha do grid já vem no formato: --mode ... --model ... --embedding ... --dim ...
python -m machine_learning.unsupervised.run_experiment ${LINE} \
    --output-json "experiment_results/unsupervised/slurm_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.json"
