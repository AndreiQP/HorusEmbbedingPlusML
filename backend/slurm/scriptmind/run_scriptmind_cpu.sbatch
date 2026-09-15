#!/bin/bash
#SBATCH --job-name=scriptmind_cpu
#SBATCH --output=slurm_logs/scriptmind/%x_%A_%a.out
#SBATCH --error=slurm_logs/scriptmind/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=l40s
set -euo pipefail

STAGE="${1:?stage required: audit|build|report|quality|judge}"
STAGE_ARGUMENT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
mkdir -p "$PROJECT_ROOT/slurm_logs/scriptmind"

CONDA_SH="${HORUS_CONDA_SH:-/home/andrei.pinto/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${HORUS_CPU_ENV:-env_py_3_12}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT/backend"

FORCE_ARGS=()
[[ "${SCRIPTMIND_FORCE:-0}" == "1" ]] && FORCE_ARGS+=(--force)

case "$STAGE" in
    audit) python -m scriptmind audit-data "${FORCE_ARGS[@]}" ;;
    quality) python -m scriptmind annotation-quality --pilot-size 200 --enforce ;;
    build) python -m scriptmind build-csid "${FORCE_ARGS[@]}" ;;
    report) python -m scriptmind report ;;
    judge)
        : "${GEMINI_API_KEY:?GEMINI_API_KEY is required for judge jobs}"
        [[ -n "$STAGE_ARGUMENT" ]] || { echo "judge requires an evaluation directory" >&2; exit 2; }
        python -m scriptmind judge-evaluation --evaluation-dir "$STAGE_ARGUMENT" "${FORCE_ARGS[@]}"
        ;;
    *) echo "Unknown CPU stage: $STAGE" >&2; exit 2 ;;
esac
