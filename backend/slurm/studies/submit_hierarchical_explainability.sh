#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
OUTPUT_DIR="$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability"
SHARDS=32
EXCLUDE_NODE=""
DRY_RUN=0
SATURATION_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-root) shift; CHECKPOINT_ROOT="${1:?caminho ausente}" ;;
        --output-dir) shift; OUTPUT_DIR="${1:?caminho ausente}" ;;
        --shards) shift; SHARDS="${1:?valor ausente}" ;;
        --exclude-node) shift; EXCLUDE_NODE="${1:?nó ausente}" ;;
        --dry-run) DRY_RUN=1 ;;
        --saturation-only) SATURATION_ONLY=1 ;;
        *) echo "Opção inválida: $1" >&2; exit 2 ;;
    esac
    shift
done
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "--shards deve ser positivo" >&2; exit 2; }

mkdir -p "$PROJECT_ROOT/slurm_logs/model_studies" "$OUTPUT_DIR"
BASE_EXPORTS="ALL,HORUS_PROJECT_ROOT=$PROJECT_ROOT,HORUS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT,HIER_STUDY_OUTPUT_DIR=$OUTPUT_DIR,HIER_STUDY_SHARD_COUNT=$SHARDS,HIER_STUDY_PRIMARY_SEED=42,HIER_STUDY_STABILITY_SEEDS=42:52:62,HIER_STUDY_TOP_MESSAGES=6,HIER_STUDY_MAX_NGRAM=5,HIER_STUDY_TOP_SPANS=20,HIER_STUDY_CONTROLS=5"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] output=$OUTPUT_DIR checkpoints=$CHECKPOINT_ROOT shards=$SHARDS"
    if [[ "$SATURATION_ONLY" == "1" ]]; then
        echo "[dry-run] 1. arrays GPU de saturação test_internal e validation (0-$((SHARDS - 1)))"
        echo "[dry-run] 2. agregação CPU das curvas estendidas"
    else
        echo "[dry-run] 1. inventory CPU"
        echo "[dry-run] 2. arrays GPU test_internal e validation (0-$((SHARDS - 1)))"
        echo "[dry-run] 3. aggregate CPU após os dois arrays"
        echo "[dry-run] 4. arrays GPU de saturação e agregação final"
    fi
    [[ -n "$EXCLUDE_NODE" ]] && echo "[dry-run] exclude_node=$EXCLUDE_NODE"
    exit 0
fi

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export HORUS_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
python - <<'PY'
import os
from machine_learning.studies import resolve_bge_finalist_checkpoint
for seed in (42, 52, 62):
    print(f"[preflight] seed={seed}: {resolve_bge_finalist_checkpoint(seed, os.environ['HORUS_CHECKPOINT_ROOT'])}")
PY

if [[ "$SATURATION_ONLY" == "1" ]]; then
    gpu_args=()
    [[ -n "$EXCLUDE_NODE" ]] && gpu_args+=(--exclude="$EXCLUDE_NODE")
    saturation_test_id=$(sbatch --parsable "${gpu_args[@]}" --array="0-$((SHARDS - 1))" \
        --export="$BASE_EXPORTS,HIER_STUDY_SPLIT=test_internal,HIER_STUDY_GPU_ACTION=saturation-shard" \
        --job-name=bge_hier_sat_test "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
    saturation_validation_id=$(sbatch --parsable "${gpu_args[@]}" --array="0-$((SHARDS - 1))" \
        --export="$BASE_EXPORTS,HIER_STUDY_SPLIT=validation,HIER_STUDY_GPU_ACTION=saturation-shard" \
        --job-name=bge_hier_sat_val "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
    saturation_aggregate_id=$(sbatch --parsable \
        --dependency="afterok:$saturation_test_id:$saturation_validation_id" \
        --export="$BASE_EXPORTS,HIER_STUDY_ACTION=saturation-aggregate" \
        --job-name=bge_hier_sat_agg "$SCRIPT_DIR/run_hierarchical_explainability_cpu.sbatch")
    echo "[submit] saturation_test=$saturation_test_id saturation_validation=$saturation_validation_id saturation_aggregate=$saturation_aggregate_id"
    echo "[monitor] squeue -j $saturation_test_id,$saturation_validation_id,$saturation_aggregate_id"
    echo "[results] $OUTPUT_DIR/aggregates/saturation_curves.parquet"
    exit 0
fi

inventory_id=$(sbatch --parsable --export="$BASE_EXPORTS,HIER_STUDY_ACTION=inventory" \
    --job-name=bge_hier_inventory "$SCRIPT_DIR/run_hierarchical_explainability_cpu.sbatch")

gpu_args=()
[[ -n "$EXCLUDE_NODE" ]] && gpu_args+=(--exclude="$EXCLUDE_NODE")
test_id=$(sbatch --parsable "${gpu_args[@]}" --dependency="afterok:$inventory_id" \
    --array="0-$((SHARDS - 1))" --export="$BASE_EXPORTS,HIER_STUDY_SPLIT=test_internal" \
    --job-name=bge_hier_test "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
validation_id=$(sbatch --parsable "${gpu_args[@]}" --dependency="afterok:$inventory_id" \
    --array="0-$((SHARDS - 1))" --export="$BASE_EXPORTS,HIER_STUDY_SPLIT=validation" \
    --job-name=bge_hier_validation "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
aggregate_id=$(sbatch --parsable --dependency="afterok:$test_id:$validation_id" \
    --export="$BASE_EXPORTS,HIER_STUDY_ACTION=aggregate" --job-name=bge_hier_aggregate \
    "$SCRIPT_DIR/run_hierarchical_explainability_cpu.sbatch")

saturation_test_id=$(sbatch --parsable "${gpu_args[@]}" --dependency="afterok:$inventory_id" \
    --array="0-$((SHARDS - 1))" \
    --export="$BASE_EXPORTS,HIER_STUDY_SPLIT=test_internal,HIER_STUDY_GPU_ACTION=saturation-shard" \
    --job-name=bge_hier_sat_test "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
saturation_validation_id=$(sbatch --parsable "${gpu_args[@]}" --dependency="afterok:$inventory_id" \
    --array="0-$((SHARDS - 1))" \
    --export="$BASE_EXPORTS,HIER_STUDY_SPLIT=validation,HIER_STUDY_GPU_ACTION=saturation-shard" \
    --job-name=bge_hier_sat_val "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
saturation_aggregate_id=$(sbatch --parsable \
    --dependency="afterok:$saturation_test_id:$saturation_validation_id" \
    --export="$BASE_EXPORTS,HIER_STUDY_ACTION=saturation-aggregate" \
    --job-name=bge_hier_sat_agg "$SCRIPT_DIR/run_hierarchical_explainability_cpu.sbatch")

echo "[submit] inventory=$inventory_id test_internal=$test_id validation=$validation_id aggregate=$aggregate_id saturation_test=$saturation_test_id saturation_validation=$saturation_validation_id saturation_aggregate=$saturation_aggregate_id"
echo "[monitor] squeue -j $inventory_id,$test_id,$validation_id,$aggregate_id,$saturation_test_id,$saturation_validation_id,$saturation_aggregate_id"
echo "[results] $OUTPUT_DIR"
