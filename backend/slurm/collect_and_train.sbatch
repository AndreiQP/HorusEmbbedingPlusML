#!/bin/bash
#SBATCH --job-name=unsup_collect
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=l40s
#SBATCH --mail-user=a169194@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2 do pipeline com tuning automático: só roda depois que o grid search
# (etapa 1, ver submit_full_pipeline.sh) de TODOS os embeddings do modelo
# tiver terminado (--dependency=afterok:<jobid_etapa1>). Lê os resultados
# (cache) do grid search, monta a grade final com os hiperparâmetros
# vencedores por embedding e dispara o job de treino final com retrain forçado.
#
# Não é chamado manualmente — use submit_full_pipeline.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:?informe o modelo}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
SLURM_DIR="backend/slurm"

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh && conda activate env_py_3_12

python3 "$SLURM_DIR/collect_best_and_make_train_grid.py" "$MODEL"

grid_file="$SLURM_DIR/grids/${MODEL}_tuned.txt"
n_lines=$(wc -l < "$grid_file")
echo "[collect_and_train] ${MODEL}: ${n_lines} combinações finais -> --array=0-$((n_lines - 1))"
sbatch --job-name="unsup_${MODEL}" \
       --array="0-$((n_lines - 1))" \
       "$SLURM_DIR/array_grid.sh" "$grid_file"
