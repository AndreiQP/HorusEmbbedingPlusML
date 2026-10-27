#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/model_studies/%x_%A_%a.out
#SBATCH --error=slurm_logs/model_studies/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PROJECT_ROOT="${HORUS_PROJECT_ROOT:?HORUS_PROJECT_ROOT ausente}"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
OUTPUT_DIR="${HIER_STUDY_OUTPUT_DIR:-$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability}"
SHARD_COUNT="${HIER_STUDY_SHARD_COUNT:-32}"
SPLIT="${HIER_STUDY_SPLIT:?HIER_STUDY_SPLIT ausente}"
GPU_ACTION="${HIER_STUDY_GPU_ACTION:-shard}"

cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_gpu_final
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA indisponível no shard global"
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"[gpu-preflight] device={torch.cuda.get_device_name()} probe={probe.item()}")
PY

seed_args=()
IFS=':' read -r -a stability_seeds <<< "${HIER_STUDY_STABILITY_SEEDS:-42:52:62}"
for seed in "${stability_seeds[@]}"; do
    [[ -n "$seed" ]] && seed_args+=(--stability-seed "$seed")
done

python -m machine_learning.studies hierarchical-study \
    --action "$GPU_ACTION" \
    --split "$SPLIT" \
    --shard-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID ausente}" \
    --shard-count "$SHARD_COUNT" \
    --primary-seed "${HIER_STUDY_PRIMARY_SEED:-42}" \
    "${seed_args[@]}" \
    --top-messages "${HIER_STUDY_TOP_MESSAGES:-6}" \
    --max-ngram "${HIER_STUDY_MAX_NGRAM:-5}" \
    --top-spans-per-message "${HIER_STUDY_TOP_SPANS:-20}" \
    --controls-per-span "${HIER_STUDY_CONTROLS:-5}" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda
