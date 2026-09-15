#!/bin/bash
#SBATCH --output=slurm_logs/model_studies/%x_%j.out
#SBATCH --error=slurm_logs/model_studies/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd -P)"
cd "$PROJECT_ROOT"
source /home/andrei.pinto/miniconda3/etc/profile.d/conda.sh
conda activate env_py_3_12
export PYTHONPATH="$PROJECT_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
python -m machine_learning.studies aggregate
python -m machine_learning.studies report
