#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
OUTPUT_DIR=""
SPLIT="test_internal"
SAMPLES_PER_CATEGORY=5
MIN_MESSAGES=7
STABILITY_SAMPLES_PER_CATEGORY=2
PRIMARY_SEED=42
STABILITY_SEEDS=(42 52 62)
STABILITY_SEEDS_EXPLICIT=0
TOP_MESSAGES=6
MAX_NGRAM=5
TOP_SPANS=20
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-root) shift; CHECKPOINT_ROOT="${1:?--checkpoint-root exige um caminho}" ;;
        --output-dir) shift; OUTPUT_DIR="${1:?--output-dir exige um caminho}" ;;
        --split) shift; SPLIT="${1:?--split exige test_internal ou validation}" ;;
        --samples-per-category) shift; SAMPLES_PER_CATEGORY="${1:?valor ausente}" ;;
        --min-messages) shift; MIN_MESSAGES="${1:?valor ausente}" ;;
        --stability-samples-per-category) shift; STABILITY_SAMPLES_PER_CATEGORY="${1:?valor ausente}" ;;
        --primary-seed) shift; PRIMARY_SEED="${1:?valor ausente}" ;;
        --stability-seed)
            shift
            if [[ "$STABILITY_SEEDS_EXPLICIT" == "0" ]]; then
                STABILITY_SEEDS=()
                STABILITY_SEEDS_EXPLICIT=1
            fi
            STABILITY_SEEDS+=("${1:?valor ausente}")
            ;;
        --skip-stability) STABILITY_SAMPLES_PER_CATEGORY=0 ;;
        --top-messages) shift; TOP_MESSAGES="${1:?valor ausente}" ;;
        --max-ngram) shift; MAX_NGRAM="${1:?valor ausente}" ;;
        --top-spans-per-message) shift; TOP_SPANS="${1:?valor ausente}" ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Opção inválida: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability/$SPLIT"
fi

[[ "$SPLIT" == "test_internal" || "$SPLIT" == "validation" ]] || {
    echo "--split deve ser test_internal ou validation" >&2
    exit 2
}
for value in "$SAMPLES_PER_CATEGORY" "$MIN_MESSAGES" "$TOP_MESSAGES" "$MAX_NGRAM" "$TOP_SPANS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "Parâmetros de contagem devem ser inteiros positivos" >&2
        exit 2
    }
done
[[ "$STABILITY_SAMPLES_PER_CATEGORY" =~ ^[0-9]+$ ]] || {
    echo "--stability-samples-per-category deve ser inteiro não negativo" >&2
    exit 2
}
[[ "$PRIMARY_SEED" =~ ^[0-9]+$ ]] || { echo "--primary-seed inválida" >&2; exit 2; }
if (( STABILITY_SAMPLES_PER_CATEGORY > SAMPLES_PER_CATEGORY )); then
    echo "A amostra de estabilidade não pode exceder a amostra principal" >&2
    exit 2
fi
STABILITY_SEEDS_SERIALIZED=$(IFS=:; echo "${STABILITY_SEEDS[*]}")

mkdir -p "$PROJECT_ROOT/slurm_logs/model_studies" "$OUTPUT_DIR"
EXPORTS="ALL,HORUS_PROJECT_ROOT=$PROJECT_ROOT,HORUS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT,HIER_SPLIT=$SPLIT,HIER_SAMPLES_PER_CATEGORY=$SAMPLES_PER_CATEGORY,HIER_MIN_MESSAGES=$MIN_MESSAGES,HIER_STABILITY_SAMPLES_PER_CATEGORY=$STABILITY_SAMPLES_PER_CATEGORY,HIER_PRIMARY_SEED=$PRIMARY_SEED,HIER_STABILITY_SEEDS=$STABILITY_SEEDS_SERIALIZED,HIER_TOP_MESSAGES=$TOP_MESSAGES,HIER_MAX_NGRAM=$MAX_NGRAM,HIER_TOP_SPANS_PER_MESSAGE=$TOP_SPANS,HIER_OUTPUT_DIR=$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] checkpoints=$CHECKPOINT_ROOT"
    echo "[dry-run] output=$OUTPUT_DIR"
    echo "[dry-run] split=$SPLIT seed_principal=$PRIMARY_SEED casos_por_categoria=$SAMPLES_PER_CATEGORY min_messages=$MIN_MESSAGES"
    echo "[dry-run] estabilidade=$STABILITY_SAMPLES_PER_CATEGORY por categoria seeds=$STABILITY_SEEDS_SERIALIZED"
    echo "[dry-run] top_messages=$TOP_MESSAGES max_ngram=$MAX_NGRAM top_spans=$TOP_SPANS"
    echo "[dry-run] sbatch backend/slurm/studies/run_hierarchical_explainability.sbatch"
    exit 0
fi

source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export HORUS_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export HIER_PRIMARY_SEED="$PRIMARY_SEED"
export HIER_MIN_MESSAGES="$MIN_MESSAGES"
export HIER_STABILITY_SEEDS="$STABILITY_SEEDS_SERIALIZED"
export HIER_STABILITY_SAMPLES_PER_CATEGORY="$STABILITY_SAMPLES_PER_CATEGORY"
python - <<'PY'
import os
from machine_learning.studies import resolve_bge_finalist_checkpoint

root = os.environ["HORUS_CHECKPOINT_ROOT"]
primary = int(os.environ["HIER_PRIMARY_SEED"])
stability_count = int(os.environ["HIER_STABILITY_SAMPLES_PER_CATEGORY"])
stability = [int(value) for value in os.environ["HIER_STABILITY_SEEDS"].split(":") if value]
seeds = [primary] + ([seed for seed in stability if seed != primary] if stability_count else [])
for seed in dict.fromkeys(seeds):
    print(f"[preflight] seed={seed}: {resolve_bge_finalist_checkpoint(seed, root)}")
PY

job_id=$(sbatch --parsable \
    --export="$EXPORTS" \
    --job-name="bge_hier_exp" \
    "$SCRIPT_DIR/run_hierarchical_explainability.sbatch")
echo "[submit] explicabilidade hierárquica BGE -> $job_id"
echo "[monitor] squeue -j $job_id"
echo "[logs] tail -f $PROJECT_ROOT/slurm_logs/model_studies/bge_hier_exp_${job_id}.out"
