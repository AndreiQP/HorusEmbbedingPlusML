#!/bin/bash
#SBATCH --job-name=unsup_submit_all
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=l40s
#SBATCH --mail-user=a169194@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Submete UM job SLURM (array) por modelo implementado, todos em paralelo:
# ocsvm, iforest, lof, lunar, svdd (embeddings x dimensões) e cvdd, date (BGE).
# Cada modelo tem seu próprio --job-name, logs e grade — filas/nós diferentes
# não bloqueiam uns aos outros. Este script só dispara os `sbatch` dos jobs
# reais (recursos mínimos), então também pode ser submetido via sbatch.
#
# Uso:
#   sbatch backend/slurm/submit_all_models.sh
#   sbatch backend/slurm/submit_all_models.sh lof svdd     # só os modelos informados
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # raiz do repo
mkdir -p slurm_logs
SLURM_DIR="backend/slurm"
GRIDS_DIR="$SLURM_DIR/grids"

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh && conda activate env_py_3_12
python3 "$SLURM_DIR/make_grid.py"

ALL_MODELS=(ocsvm iforest lof lunar svdd cvdd date)
MODELS=("${@:-${ALL_MODELS[@]}}")

for model in "${MODELS[@]}"; do
    grid_file="$GRIDS_DIR/${model}.txt"
    if [[ ! -f "$grid_file" ]]; then
        echo "[submit_all_models] Grade não encontrada para '$model' ($grid_file), pulando." >&2
        continue
    fi
    n_lines=$(wc -l < "$grid_file")
    echo "[submit_all_models] ${model}: ${n_lines} combinações -> --array=0-$((n_lines - 1))"
    sbatch --job-name="unsup_${model}" \
           --array="0-$((n_lines - 1))" \
           "$SLURM_DIR/array_grid.sh" "$grid_file"
done
