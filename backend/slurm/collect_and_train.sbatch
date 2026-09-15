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
# vencedores por embedding e dispara o job de treino final. FORCE_RETRAIN=1 é
# propagado somente quando o usuário chama submit_full_pipeline.sh --force.
#
# Não é chamado manualmente — use submit_full_pipeline.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:?informe o modelo}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
SLURM_DIR="backend/slurm"
GRIDS_DIR="backend/experiment_results/slurm_grids"

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh && conda activate env_py_3_12

python3 "$SLURM_DIR/collect_best_and_make_train_grid.py" "$MODEL"

grid_file="$GRIDS_DIR/${MODEL}_tuned.txt"
n_lines=$(wc -l < "$grid_file")
echo "[collect_and_train] ${MODEL}: ${n_lines} combinações finais -> --array=0-$((n_lines - 1))"
resource_args=()
if [[ "$MODEL" == "lunar" || "$MODEL" == "svdd" ]]; then
    resource_args+=(--gres=gpu:1)
fi
train_jobid=$(sbatch --parsable --export=ALL,FORCE_RETRAIN --job-name="unsup_${MODEL}" \
       --array="0-$((n_lines - 1))" \
       "${resource_args[@]}" \
       "$SLURM_DIR/array_grid.sh" "$grid_file")

finalize_file="$GRIDS_DIR/${MODEL}_finalize.txt"
sbatch --export=ALL --job-name="unsup_finalize_${MODEL}" \
       --dependency="afterok:${train_jobid}" \
       --cpus-per-task=2 --mem=4G \
       --array="0-0" \
       "$SLURM_DIR/array_grid.sh" "$finalize_file"
