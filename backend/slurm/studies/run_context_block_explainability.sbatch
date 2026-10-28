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
OUTPUT_DIR="${CONTEXT_BLOCK_OUTPUT_DIR:-$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability/context_blocks}"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
ACTION="${CONTEXT_BLOCK_ACTION:?CONTEXT_BLOCK_ACTION ausente}"
SHARDS="${CONTEXT_BLOCK_SHARDS:-32}"

REPRESENTATIONS=(canonical_message speaker_block_4 chronological_window_4 chronological_window_5 chronological_window_6 chronological_window_7 chronological_window_8)
MODES=(frozen_probe retrained)
SPLITS=(test_internal validation)
SEEDS=(42 52 62)

cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_gpu_final
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA indisponível"
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"[gpu-preflight] device={torch.cuda.get_device_name()} probe={probe.item()}")
PY

task="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$ACTION" == "cache" ]]; then
    representation_index=$((task / 2))
    split_index=$((task % 2))
    source_splits=(train validation)
    python -m machine_learning.studies context-block-study \
        --action cache --representation "${REPRESENTATIONS[$representation_index]}" \
        --source-split "${source_splits[$split_index]}" \
        --output-dir "$OUTPUT_DIR" --device cuda
elif [[ "$ACTION" == "train" ]]; then
    # canonical_message reutiliza os finalistas; são 6 representações × 3 seeds.
    representation_index=$((task / 3 + 1))
    seed_index=$((task % 3))
    python -m machine_learning.studies context-block-study \
        --action train --representation "${REPRESENTATIONS[$representation_index]}" \
        --seed "${SEEDS[$seed_index]}" --output-dir "$OUTPUT_DIR" --device cuda
elif [[ "$ACTION" == "explain" ]]; then
    shard_index=$((task % SHARDS))
    combination=$((task / SHARDS))
    split_index=$((combination % 2)); combination=$((combination / 2))
    mode_index=$((combination % 2)); representation_index=$((combination / 2))
    python -m machine_learning.studies context-block-study \
        --action explain --representation "${REPRESENTATIONS[$representation_index]}" \
        --model-mode "${MODES[$mode_index]}" --split "${SPLITS[$split_index]}" \
        --shard-index "$shard_index" --shard-count "$SHARDS" \
        --checkpoint-root "$CHECKPOINT_ROOT" --output-dir "$OUTPUT_DIR" --device cuda
else
    echo "Ação GPU inválida: $ACTION" >&2
    exit 2
fi
