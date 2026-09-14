#!/bin/bash
#SBATCH --job-name=unsup_%x
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --mail-user=a169194@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Treina/avalia UM experimento unsupervised. Pensado para ser disparado via
# `sbatch` diretamente ou por um array job (ver array_grid.sh).
#
# Uso:
#   sbatch run_unsupervised.sbatch <mode> <model> <embedding> <dim> [param=value ...]
#
# Exemplos:
#   sbatch run_unsupervised.sbatch train lof bge 300
#   sbatch run_unsupervised.sbatch train svdd voyage 200 nu=0.1 latent_dim=32
#   sbatch run_unsupervised.sbatch train cvdd bge 0        # dim ignorado p/ cvdd/date
#   sbatch run_unsupervised.sbatch grid_search iforest e5 500
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:?informe o modo: train|learning_curve|grid_search|train_all|compare_dimensions}"
MODEL="${2:?informe o modelo: ocsvm|iforest|lof|lunar|svdd|cvdd|date}"
EMBEDDING="${3:-bge}"
DIM="${4:-300}"
shift 4 || shift $#   # os args restantes (se houver) são hiperparâmetros key=value

mkdir -p slurm_logs

# ── Ambiente ──────────────────────────────────────────────────────────────
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh && conda activate env_py_3_12
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/backend"

PARAMS=()
for kv in "$@"; do
    PARAMS+=(--param "$kv")
done

python -m machine_learning.unsupervised.run_experiment \
    --mode "$MODE" \
    --model "$MODEL" \
    --embedding "$EMBEDDING" \
    --dim "$DIM" \
    --output-json "experiment_results/unsupervised/slurm_${SLURM_JOB_ID:-local}_${MODEL}_${EMBEDDING}.json" \
    "${PARAMS[@]}"
