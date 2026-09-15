#!/bin/bash
# Entrada única para o retrain completo. Os notebooks nunca treinam modelos.
set -euo pipefail

EXPECTED_ROOT="/home/andrei.pinto/HorusEmbbedingPlusML"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
if [[ "$PROJECT_ROOT" != "$EXPECTED_ROOT" ]]; then
    echo "ERRO: execute somente em $EXPECTED_ROOT (atual: $PROJECT_ROOT)" >&2
    exit 2
fi

RUN_UNSUPERVISED=0
RUN_TRANSFORMER=0
DRY_RUN=0
export FORCE_RETRAIN=0

if [[ $# -eq 0 ]]; then
    RUN_UNSUPERVISED=1
    RUN_TRANSFORMER=1
fi
for arg in "$@"; do
    case "$arg" in
        --all) RUN_UNSUPERVISED=1; RUN_TRANSFORMER=1 ;;
        --unsupervised) RUN_UNSUPERVISED=1 ;;
        --transformer) RUN_TRANSFORMER=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE_RETRAIN=1 ;;
        *) echo "Opção inválida: $arg" >&2; exit 2 ;;
    esac
done

cd "$PROJECT_ROOT"
mkdir -p slurm_logs backend/experiment_results/slurm_grids
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
python3 backend/slurm/make_grid.py

SLURM_DIR="backend/slurm"
GRIDS_DIR="backend/experiment_results/slurm_grids"
UNSUPERVISED_MODELS=(ocsvm iforest lof lunar svdd)
declare -A GRID_COUNTS=([ocsvm]=280 [iforest]=120 [lof]=160 [lunar]=270 [svdd]=240)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] force=$FORCE_RETRAIN"
    if [[ "$RUN_UNSUPERVISED" == "1" ]]; then
        for model in "${UNSUPERVISED_MODELS[@]}"; do
            gpu="CPU"
            [[ "$model" == "lunar" || "$model" == "svdd" ]] && gpu="1 GPU"
            echo "[dry-run] $model: 5 embeddings x ${GRID_COUNTS[$model]} combinações ($gpu)"
            echo "          grid --afterok--> 15 finalistas (3 seeds) --afterok--> resumo"
        done
    fi
    if [[ "$RUN_TRANSFORMER" == "1" ]]; then
        echo "[dry-run] transformer: 3 embeddings x (12 arquitetura + 18 capacidade + 3 otimizador), 1 GPU cada"
        echo "          busca sequencial -> 9 finalistas (3 seeds) --afterok--> ensembles"
    fi
    exit 0
fi

if [[ "$RUN_UNSUPERVISED" == "1" ]]; then
    for model in "${UNSUPERVISED_MODELS[@]}"; do
        tuning_file="$GRIDS_DIR/${model}_tuning.txt"
        n_lines=$(wc -l < "$tuning_file")
        resource_args=()
        if [[ "$model" == "lunar" || "$model" == "svdd" ]]; then
            resource_args+=(--gres=gpu:1)
        fi
        tune_jobid=$(sbatch --parsable --export=ALL,FORCE_RETRAIN \
            --job-name="unsup_tune_${model}" --array="0-$((n_lines - 1))" \
            "${resource_args[@]}" "$SLURM_DIR/array_grid.sh" "$tuning_file")
        sbatch --export=ALL,FORCE_RETRAIN --job-name="unsup_collect_${model}" \
            --dependency="afterok:${tune_jobid}" \
            "$SLURM_DIR/collect_and_train.sbatch" "$model"
    done
fi

if [[ "$RUN_TRANSFORMER" == "1" ]]; then
    transformer_file="$GRIDS_DIR/transformer.txt"
    n_lines=$(wc -l < "$transformer_file")
    finalist_jobid=$(sbatch --parsable --export=ALL,FORCE_RETRAIN \
        --job-name="transformer_search" --array="0-$((n_lines - 1))" --gres=gpu:1 \
        "$SLURM_DIR/array_grid.sh" "$transformer_file" transformer)
    sbatch --export=ALL --job-name="transformer_finalize" \
        --dependency="afterok:${finalist_jobid}" --array="0-0" --cpus-per-task=2 --mem=4G \
        "$SLURM_DIR/array_grid.sh" "$GRIDS_DIR/transformer_finalize.txt" transformer
fi
