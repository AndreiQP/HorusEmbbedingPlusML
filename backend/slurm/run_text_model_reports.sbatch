#!/bin/bash
#SBATCH --job-name=text_model_reports
#SBATCH --output=slurm_logs/text_models/%x_%j.out
#SBATCH --error=slurm_logs/text_models/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s

# Gera gráficos de CVDD e DATE somente a partir dos checkpoints e métricas
# existentes. Não usa --force-retrain e não modifica os outros modelos.
set -euo pipefail

MODEL="${1:-all}"
mkdir -p slurm_logs/text_models
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_DIR/backend"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA não está disponível para a inferência cache-first de CVDD/DATE.")
print(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
PY

python -m machine_learning.unsupervised.text_reports --model "$MODEL" --device cuda
