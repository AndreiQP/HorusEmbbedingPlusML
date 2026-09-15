# Horus ScriptMind

This package implements the isolated computational replication of the ScriptMind
paper. It never writes to the existing train/validation datasets, embedding files,
split manifests, or experiment caches. All generated files live under:

- `backend/datasets/dataset_scriptmind/`
- `backend/experiment_results/scriptmind/`

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the decisions and acceptance
criteria that define this implementation.

## Setup

Install the base and optional ScriptMind dependencies in the appropriate environments:

```bash
pip install -r requirements.txt
pip install -r requirements-scriptmind.txt
```

Set `GEMINI_API_KEY` only in the execution environment. Llama access may require
`HF_TOKEN`. No command enables provider billing.

## Local smoke workflow

Run from `backend/`:

```bash
python -m scriptmind audit-data --limit 20 --force
python -m scriptmind annotate --provider gemini --shard-index 0 --num-shards 1 --limit 20
python -m scriptmind annotate --provider local --shard-index 0 --num-shards 1 --limit 20
python -m scriptmind adjudicate --shard-index 0 --num-shards 1 --limit 20
python -m scriptmind build-csid --allow-partial --force
```

For full experiments, use `backend/slurm/scriptmind/submit_scriptmind_pipeline.sh`.
Its default is resumable; `--dry-run` submits nothing.

```bash
bash backend/slurm/scriptmind/submit_scriptmind_pipeline.sh --dry-run
bash backend/slurm/scriptmind/submit_scriptmind_pipeline.sh
bash backend/slurm/scriptmind/submit_scriptmind_pipeline.sh --with-ablations
bash backend/slurm/scriptmind/submit_scriptmind_pipeline.sh --from train
```

The CPU environment needs the Gemini/scientific dependencies and the GPU environment
needs the complete `requirements-scriptmind.txt` stack. Override the configured conda
locations with `HORUS_CONDA_SH`, `HORUS_CPU_ENV`, and `HORUS_GPU_ENV` when necessary.

## Public Python API

```python
from scriptmind import (
    ScriptMindPrediction,
    annotate_dialogues,
    build_csid,
    train_scriptmind,
    evaluate_scriptmind,
)
```

The generated metrics can be explored in `backend/notebooks/ScriptMind_metrics.ipynb`.

After an evaluation, its optional 200-example automatic judge can be submitted with:

```bash
sbatch --export=ALL,GEMINI_API_KEY \
  backend/slurm/scriptmind/run_scriptmind_cpu.sbatch judge \
  backend/experiment_results/scriptmind/evaluation/<run-directory>
```

Fill the generated `human_judge_audit.csv`, then run `score-judge-audit`. Judge scores
are not treated as primary metrics unless every available correlation reaches 0.70.
