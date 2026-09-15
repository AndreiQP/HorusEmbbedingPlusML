#!/bin/bash
#SBATCH --job-name=all_models
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=l40s
#SBATCH --mail-user=a169194@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL
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
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] force=$FORCE_RETRAIN"
    if [[ "$RUN_UNSUPERVISED" == "1" ]]; then
        echo "[dry-run] unsupervised: 3 jobs CPU + 2 jobs GPU + 1 consolidação = 6 jobs"
        echo "          cada job de modelo executa internamente 5 grids, 15 finalistas e seu resumo"
    fi
    if [[ "$RUN_TRANSFORMER" == "1" ]]; then
        echo "[dry-run] transformer: 3 jobs GPU + 1 consolidação = 4 jobs"
        echo "          cada job executa internamente busca sequencial e 3 seeds"
    fi
    [[ "$RUN_UNSUPERVISED" == "1" && "$RUN_TRANSFORMER" == "1" ]] && echo "[dry-run] TOTAL: 10 jobs na fila"
    exit 0
fi

if [[ "$RUN_UNSUPERVISED" == "1" ]]; then
    cpu_file="$GRIDS_DIR/unsupervised_cpu.txt"
    gpu_file="$GRIDS_DIR/unsupervised_gpu.txt"
    cpu_count=$(wc -l < "$cpu_file")
    gpu_count=$(wc -l < "$gpu_file")
    cpu_jobid=$(sbatch --parsable --export=ALL,FORCE_RETRAIN \
        --job-name="unsup_cpu" --array="0-$((cpu_count - 1))" \
        "$SLURM_DIR/array_grid.sh" "$cpu_file")
    gpu_jobid=$(sbatch --parsable --export=ALL,FORCE_RETRAIN \
        --job-name="unsup_gpu" --array="0-$((gpu_count - 1))" --gres=gpu:1 \
        "$SLURM_DIR/array_grid.sh" "$gpu_file")
    sbatch --export=ALL --job-name="unsup_finalize" \
        --dependency="afterok:${cpu_jobid}:${gpu_jobid}" --array="0-0" \
        "$SLURM_DIR/array_grid.sh" "$GRIDS_DIR/unsupervised_finalize.txt"
fi

if [[ "$RUN_TRANSFORMER" == "1" ]]; then
    transformer_file="$GRIDS_DIR/transformer.txt"
    n_lines=$(wc -l < "$transformer_file")
    finalist_jobid=$(sbatch --parsable --export=ALL,FORCE_RETRAIN \
        --job-name="transformer_search" --array="0-$((n_lines - 1))" --gres=gpu:1 \
        "$SLURM_DIR/array_grid.sh" "$transformer_file" transformer)
    sbatch --export=ALL --job-name="transformer_finalize" \
        --dependency="afterok:${finalist_jobid}" --array="0-0" \
        "$SLURM_DIR/array_grid.sh" "$GRIDS_DIR/transformer_finalize.txt" transformer
fi
