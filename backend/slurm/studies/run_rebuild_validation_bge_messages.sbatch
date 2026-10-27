#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/model_studies/%x_%j.out
#SBATCH --error=slurm_logs/model_studies/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PROJECT_ROOT="${HORUS_PROJECT_ROOT:?HORUS_PROJECT_ROOT ausente}"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
EVALUATE_FINALISTS="${BGE_REBUILD_EVALUATE_FINALISTS:-1}"
FORCE_REENCODE="${BGE_REBUILD_FORCE:-0}"

cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_gpu_final
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA indisponível para reconstrução BGE"
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"[gpu-preflight] device={torch.cuda.get_device_name(0)} probe={probe.item()}")
PY

python -m machine_learning.studies.rebuild_validation_split \
  --source-dir "$PROJECT_ROOT/backend/datasets/dataset_validation/source_cache/lovefraud02_v1"

args=(
  --device cuda
  --encode-batch-size "${BGE_REBUILD_ENCODE_BATCH_SIZE:-32}"
  --conversations-per-shard "${BGE_REBUILD_CONVERSATIONS_PER_SHARD:-20}"
  --checkpoint-root "$CHECKPOINT_ROOT"
)
[[ "$EVALUATE_FINALISTS" == "1" ]] && args+=(--evaluate-finalists)
[[ "$FORCE_REENCODE" == "1" ]] && args+=(--force)

python -m machine_learning.studies.rebuild_validation_bge_messages "${args[@]}"

