#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
DRY_RUN=0
EVALUATE_FINALISTS=1
FORCE_REENCODE=0
EXCLUDE_NODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-root) shift; CHECKPOINT_ROOT="${1:?caminho ausente}" ;;
        --skip-evaluation) EVALUATE_FINALISTS=0 ;;
        --force) FORCE_REENCODE=1 ;;
        --exclude-node) shift; EXCLUDE_NODE="${1:?nó ausente}" ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Opção inválida: $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "$PROJECT_ROOT/slurm_logs/model_studies"
echo "[config] project=$PROJECT_ROOT"
echo "[config] checkpoints=$CHECKPOINT_ROOT"
echo "[config] evaluate_finalists=$EVALUATE_FINALISTS force=$FORCE_REENCODE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] 1. reconstruir raw validation no headnode e preencher o cache LoveFraud02"
    echo "[dry-run] 2. recodificar 1.083 conversas / 67.666 mensagens com BAAI/bge-m3"
    echo "[dry-run] 3. validar e substituir messages_transformer/bge_chunks.parquet"
    [[ "$EVALUATE_FINALISTS" == "1" ]] && \
      echo "[dry-run] 4. reavaliar checkpoints congelados seeds 42, 52 e 62"
    [[ -n "$EXCLUDE_NODE" ]] && echo "[dry-run] exclude_node=$EXCLUDE_NODE"
    exit 0
fi

# Os nós GPU não possuem acesso confiável à internet. O download público dos
# 83 arquivos e a reconstrução do raw acontecem no headnode, antes do sbatch,
# e ficam disponíveis ao job pelo filesystem compartilhado.
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
python -m machine_learning.studies.rebuild_validation_split \
  --source-dir "$PROJECT_ROOT/backend/datasets/dataset_validation/source_cache/lovefraud02_v1"
if [[ "$EVALUATE_FINALISTS" == "1" ]]; then
    export HORUS_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
    python - <<'PY'
import os
from machine_learning.studies import resolve_bge_finalist_checkpoint
for seed in (42, 52, 62):
    print(f"[preflight] seed={seed}: {resolve_bge_finalist_checkpoint(seed, os.environ['HORUS_CHECKPOINT_ROOT'])}")
PY
fi

sbatch_args=()
[[ -n "$EXCLUDE_NODE" ]] && sbatch_args+=(--exclude="$EXCLUDE_NODE")
exports="ALL,HORUS_PROJECT_ROOT=$PROJECT_ROOT,HORUS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT,BGE_REBUILD_EVALUATE_FINALISTS=$EVALUATE_FINALISTS,BGE_REBUILD_FORCE=$FORCE_REENCODE"
job_id=$(sbatch --parsable "${sbatch_args[@]}" \
    --export="$exports" \
    --job-name=bge_validation_rebuild \
    "$SCRIPT_DIR/run_rebuild_validation_bge_messages.sbatch")

echo "[submit] reconstrução validation BGE -> $job_id"
echo "[monitor] squeue -j $job_id"
echo "[logs] tail -f $PROJECT_ROOT/slurm_logs/model_studies/bge_validation_rebuild_${job_id}.out"
echo "[output] $PROJECT_ROOT/backend/datasets/dataset_validation/processed/messages_transformer/bge_chunks.parquet"

