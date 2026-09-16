#!/bin/bash
# Submete somente a geração cache-first de gráficos de CVDD/DATE.
set -euo pipefail

MODEL="all"
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        cvdd|date|all) MODEL="$arg" ;;
        *) echo "Uso: bash backend/slurm/submit_text_model_reports.sh [cvdd|date|all] [--dry-run]" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$DRY_RUN" == true ]]; then
    echo "sbatch --job-name=text_model_reports $SCRIPT_DIR/run_text_model_reports.sbatch $MODEL"
    exit 0
fi
mkdir -p "$SCRIPT_DIR/../../slurm_logs/text_models"
sbatch --job-name=text_model_reports "$SCRIPT_DIR/run_text_model_reports.sbatch" "$MODEL"
