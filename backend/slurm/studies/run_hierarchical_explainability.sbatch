#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/model_studies/%x_%j.out
#SBATCH --error=slurm_logs/model_studies/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PROJECT_ROOT="${HORUS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"
CHECKPOINT_ROOT="${HORUS_CHECKPOINT_ROOT:-$PROJECT_ROOT/backend/experiment_results/transformer}"
SPLIT="${HIER_SPLIT:-test_internal}"
SAMPLES_PER_CATEGORY="${HIER_SAMPLES_PER_CATEGORY:-5}"
MIN_MESSAGES="${HIER_MIN_MESSAGES:-7}"
STABILITY_SAMPLES_PER_CATEGORY="${HIER_STABILITY_SAMPLES_PER_CATEGORY:-2}"
PRIMARY_SEED="${HIER_PRIMARY_SEED:-42}"
STABILITY_SEEDS="${HIER_STABILITY_SEEDS:-42:52:62}"
TOP_MESSAGES="${HIER_TOP_MESSAGES:-6}"
MAX_NGRAM="${HIER_MAX_NGRAM:-5}"
TOP_SPANS="${HIER_TOP_SPANS_PER_MESSAGE:-20}"
OUTPUT_DIR="${HIER_OUTPUT_DIR:-$PROJECT_ROOT/backend/experiment_results/transformer/studies/hierarchical_explainability/$SPLIT}"

cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_gpu_final
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
# O BGE-M3 já foi baixado no cache compartilhado do cluster. Os nós de compute
# não têm DNS para o Hugging Face, portanto não tente resolver o Hub durante o job.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA indisponível"
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"[gpu-preflight] device={torch.cuda.get_device_name()} probe={probe.item()}")
PY
python - <<'PY'
import os
from machine_learning.studies import resolve_bge_finalist_checkpoint

root = os.environ.get("HORUS_CHECKPOINT_ROOT")
primary = int(os.environ.get("HIER_PRIMARY_SEED", "42"))
stability_count = int(os.environ.get("HIER_STABILITY_SAMPLES_PER_CATEGORY", "2"))
stability = [int(value) for value in os.environ.get("HIER_STABILITY_SEEDS", "42:52:62").split(":") if value]
seeds = [primary] + ([seed for seed in stability if seed != primary] if stability_count else [])
for seed in dict.fromkeys(seeds):
    print(f"[preflight] seed={seed}: {resolve_bge_finalist_checkpoint(seed, root)}")
PY

stability_args=()
IFS=':' read -r -a stability_seeds <<< "$STABILITY_SEEDS"
for seed in "${stability_seeds[@]}"; do
    [[ -n "$seed" ]] && stability_args+=(--stability-seed "$seed")
done

python -m machine_learning.studies hierarchical-explain \
    --study \
    --split "$SPLIT" \
    --primary-seed "$PRIMARY_SEED" \
    "${stability_args[@]}" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --min-messages "$MIN_MESSAGES" \
    --stability-samples-per-category "$STABILITY_SAMPLES_PER_CATEGORY" \
    --top-messages "$TOP_MESSAGES" \
    --max-ngram "$MAX_NGRAM" \
    --top-spans-per-message "$TOP_SPANS" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$OUTPUT_DIR"
