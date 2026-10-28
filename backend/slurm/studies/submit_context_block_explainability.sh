#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
OUTPUT_DIR="$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability/context_blocks"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
SHARDS=32
MAX_PARALLEL=8
EXCLUDE_NODE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) shift; OUTPUT_DIR="${1:?caminho ausente}" ;;
        --checkpoint-root) shift; CHECKPOINT_ROOT="${1:?caminho ausente}" ;;
        --shards) shift; SHARDS="${1:?valor ausente}" ;;
        --max-parallel) shift; MAX_PARALLEL="${1:?valor ausente}" ;;
        --exclude-node) shift; EXCLUDE_NODE="${1:?nó ausente}" ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Opção inválida: $1" >&2; exit 2 ;;
    esac
    shift
done

[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "--shards inválido" >&2; exit 2; }
mkdir -p "$PROJECT_ROOT/slurm_logs/model_studies" "$OUTPUT_DIR"
BASE="ALL,HORUS_PROJECT_ROOT=$PROJECT_ROOT,HORUS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT,CONTEXT_BLOCK_OUTPUT_DIR=$OUTPUT_DIR,CONTEXT_BLOCK_SHARDS=$SHARDS"
EXPLAIN_TASKS=$((7 * 2 * 2 * SHARDS))

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] cache: 14 tarefas GPU (7 representações × train/validation)"
    echo "[dry-run] train: 18 tarefas GPU (6 representações × 3 seeds)"
    echo "[dry-run] explain: $EXPLAIN_TASKS tarefas GPU (7 × 2 modos × 2 splits × $SHARDS shards)"
    echo "[dry-run] aggregate: 1 tarefa CPU"
    echo "[dry-run] output=$OUTPUT_DIR max_parallel=$MAX_PARALLEL"
    exit 0
fi

gpu_args=()
[[ -n "$EXCLUDE_NODE" ]] && gpu_args+=(--exclude="$EXCLUDE_NODE")

cache_id=$(sbatch --parsable "${gpu_args[@]}" --array="0-13%$MAX_PARALLEL" \
    --export="$BASE,CONTEXT_BLOCK_ACTION=cache" --job-name=bge_ctx_cache \
    "$SCRIPT_DIR/run_context_block_explainability.sbatch")
train_id=$(sbatch --parsable "${gpu_args[@]}" --dependency="afterok:$cache_id" \
    --array="0-17%$MAX_PARALLEL" --export="$BASE,CONTEXT_BLOCK_ACTION=train" \
    --job-name=bge_ctx_train "$SCRIPT_DIR/run_context_block_explainability.sbatch")
explain_id=$(sbatch --parsable "${gpu_args[@]}" --dependency="afterok:$train_id" \
    --array="0-$((EXPLAIN_TASKS - 1))%$MAX_PARALLEL" \
    --export="$BASE,CONTEXT_BLOCK_ACTION=explain" --job-name=bge_ctx_explain \
    "$SCRIPT_DIR/run_context_block_explainability.sbatch")
aggregate_id=$(sbatch --parsable --dependency="afterok:$explain_id" \
    --export="$BASE" --job-name=bge_ctx_aggregate \
    "$SCRIPT_DIR/run_context_block_explainability_cpu.sbatch")

echo "[submit] cache=$cache_id train=$train_id explain=$explain_id aggregate=$aggregate_id"
echo "[monitor] squeue -j $cache_id,$train_id,$explain_id,$aggregate_id"
echo "[results] $OUTPUT_DIR/aggregates"
