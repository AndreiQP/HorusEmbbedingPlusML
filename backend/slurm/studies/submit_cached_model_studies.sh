#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd -P)"
cd "$PROJECT_ROOT"

DRY_RUN=0
FORCE=0
ONLY="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --resume) ;;
        --force) FORCE=1 ;;
        --only) shift; ONLY="${1:?--only exige um estágio}" ;;
        *) echo "Opção inválida: $1" >&2; exit 2 ;;
    esac
    shift
done

case "$ONLY" in
    all|cached-analysis|dataset-size|transformer-size|unsupervised-size|complementarity|threshold|explainability|report) ;;
    *) echo "Estágio inválido: $ONLY" >&2; exit 2 ;;
esac

mkdir -p slurm_logs/model_studies
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] pipeline principal: NÃO será submetida"
    echo "[dry-run] modo=$ONLY force=$FORCE resume=1"
    case "$ONLY" in
        all|dataset-size|transformer-size) echo "[dry-run] Transformer dataset-size: até 54 tarefas GPU (inclui 100% cache-only), concorrência 2" ;;
    esac
    case "$ONLY" in
        all|dataset-size|unsupervised-size) echo "[dry-run] Unsupervised dataset-size: até 54 tarefas, separadas entre CPU e GPU" ;;
    esac
    case "$ONLY" in
        all|cached-analysis|complementarity|threshold) echo "[dry-run] Análises cacheadas: complementaridade/threshold, 4 CPUs, sem treino" ;;
    esac
    case "$ONLY" in
        all|cached-analysis|explainability) echo "[dry-run] Explicabilidade: 9 tarefas GPU (3 embeddings x 3 seeds), concorrência 2" ;;
    esac
    echo "[dry-run] Relatório depende apenas dos estágios solicitados; nenhum sbatch foi chamado"
    exit 0
fi

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$ONLY" != "report" ]]; then
    PREPARE_STRATEGY="all"
    case "$ONLY" in
        transformer-size|cached-analysis|complementarity|threshold|explainability) PREPARE_STRATEGY="transformer" ;;
        unsupervised-size) PREPARE_STRATEGY="unsupervised" ;;
    esac
    PREPARE_ARGS=()
    [[ "$ONLY" == "complementarity" || "$ONLY" == "threshold" ]] && PREPARE_ARGS=(--no-strict)
    python -m machine_learning.studies prepare --strategy "$PREPARE_STRATEGY" "${PREPARE_ARGS[@]}"
    python -m machine_learning.studies make-slurm-grids
fi

SLURM_DIR="backend/slurm/studies"
GRID_DIR="backend/experiment_results/transformer/studies/slurm_grids"
EXPORTS="ALL,STUDY_FORCE=$FORCE"
MAIL_ARGS=()
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
    MAIL_ARGS=(--mail-user="$SLURM_MAIL_USER" --mail-type=END,FAIL)
fi
JOB_IDS=()

submit_array() {
    local name="$1" grid="$2" runner="$3" concurrency="$4" strategy="${5:-}"
    local count
    count=$(wc -l < "$grid")
    [[ "$count" -eq 0 ]] && return 0
    local array="0-$((count - 1))"
    [[ -n "$concurrency" ]] && array="${array}%${concurrency}"
    local jobid
    jobid=$(sbatch --parsable --export="$EXPORTS" "${MAIL_ARGS[@]}" --job-name="$name" --array="$array" "$runner" "$grid" "$strategy")
    JOB_IDS+=("$jobid")
    echo "[submit] $name -> $jobid ($count tarefas)"
}

case "$ONLY" in
    all|dataset-size|transformer-size)
        submit_array "study_tr_size" "$GRID_DIR/transformer_size.tsv" "$SLURM_DIR/run_dataset_size_gpu.sbatch" 2 transformer ;;
esac
case "$ONLY" in
    all|dataset-size|unsupervised-size)
        submit_array "study_un_cpu" "$GRID_DIR/unsupervised_size_cpu.tsv" "$SLURM_DIR/run_dataset_size_cpu.sbatch" "" unsupervised
        submit_array "study_un_gpu" "$GRID_DIR/unsupervised_size_gpu.tsv" "$SLURM_DIR/run_dataset_size_gpu.sbatch" 2 unsupervised ;;
esac
case "$ONLY" in
    all|cached-analysis)
        jobid=$(sbatch --parsable --export="$EXPORTS" "${MAIL_ARGS[@]}" --job-name="study_cached" "$SLURM_DIR/run_cached_transformer_analysis.sbatch" all)
        JOB_IDS+=("$jobid") ;;
    complementarity|threshold)
        jobid=$(sbatch --parsable --export="$EXPORTS" "${MAIL_ARGS[@]}" --job-name="study_${ONLY}" "$SLURM_DIR/run_cached_transformer_analysis.sbatch" "$ONLY")
        JOB_IDS+=("$jobid") ;;
esac
case "$ONLY" in
    all|cached-analysis|explainability)
        submit_array "study_heads" "$GRID_DIR/explainability.tsv" "$SLURM_DIR/run_transformer_explainability.sbatch" 2 ;;
esac

if [[ "$ONLY" == "report" ]]; then
    sbatch --export="$EXPORTS" "${MAIL_ARGS[@]}" --job-name="study_report" "$SLURM_DIR/run_studies_report.sbatch"
elif [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    dependency=$(IFS=:; echo "${JOB_IDS[*]}")
    sbatch --export="$EXPORTS" "${MAIL_ARGS[@]}" --job-name="study_report" --dependency="afterok:${dependency}" "$SLURM_DIR/run_studies_report.sbatch"
fi
