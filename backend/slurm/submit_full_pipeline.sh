#!/bin/bash
#SBATCH --job-name=unsup_pipeline
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=l40s
#SBATCH --mail-user=a169194@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline completo POR MODELO, com tuning de hiperparâmetros automático:
#
#   1) grid_search (--mode grid_search, um array job por embedding, dim fixo
#      em TUNING_DIM=300) -> escolhe os melhores hiperparâmetros pela
#      validação INTERNA (holdout de dataset_train, ver README seção 5.0).
#   2) (dependência: só roda se 1 terminar com sucesso) treino final
#      (--mode train) em CADA embedding x dimensão, já usando os
#      hiperparâmetros vencedores daquele embedding — resultado salvo em
#      experiment_results/unsupervised/ (modelo, métricas train/val/test/
#      validation_external, histórico), pronto para comparar depois.
#
# CVDD/DATE não têm tuning automatizado (ver make_grid.py) — para eles este
# script apenas dispara o treino direto com hiperparâmetros padrão.
#
# Uso:
#   sbatch backend/slurm/submit_full_pipeline.sh                # todos os modelos
#   sbatch backend/slurm/submit_full_pipeline.sh lof svdd       # só os informados
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # raiz do repo
mkdir -p slurm_logs
SLURM_DIR="backend/slurm"
GRIDS_DIR="$SLURM_DIR/grids"

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh && conda activate env_py_3_12
python3 "$SLURM_DIR/make_grid.py"

TUNABLE_MODELS=(ocsvm iforest lof lunar svdd)
TEXT_MODELS=(cvdd date)
ALL_MODELS=("${TUNABLE_MODELS[@]}" "${TEXT_MODELS[@]}")
MODELS=("${@:-${ALL_MODELS[@]}}")

for model in "${MODELS[@]}"; do
    if [[ " ${TEXT_MODELS[*]} " == *" ${model} "* ]]; then
        # cvdd/date: sem tuning automatizado, treina direto com hiperparâmetros padrão
        grid_file="$GRIDS_DIR/${model}.txt"
        n_lines=$(wc -l < "$grid_file")
        echo "[pipeline] ${model} (sem tuning): --array=0-$((n_lines - 1))"
        sbatch --job-name="unsup_${model}" \
               --array="0-$((n_lines - 1))" \
               "$SLURM_DIR/array_grid.sh" "$grid_file"
        continue
    fi

    tuning_file="$GRIDS_DIR/${model}_tuning.txt"
    n_lines=$(wc -l < "$tuning_file")
    echo "[pipeline] ${model} etapa 1/2 (grid_search): --array=0-$((n_lines - 1))"
    tune_jobid=$(sbatch --parsable \
        --job-name="unsup_tune_${model}" \
        --array="0-$((n_lines - 1))" \
        "$SLURM_DIR/array_grid.sh" "$tuning_file")

    echo "[pipeline] ${model} etapa 2/2 (treino final) agendada após job ${tune_jobid}"
    sbatch --job-name="unsup_collect_${model}" \
           --dependency="afterok:${tune_jobid}" \
           "$SLURM_DIR/collect_and_train.sbatch" "$model"
done
