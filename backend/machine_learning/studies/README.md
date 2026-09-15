# Estudos cache-first

Esta camada analisa somente os finalistas das pipelines `transformer` e
`unsupervised`. Ela não chama busca de hiperparâmetros, não altera os manifests
de split e grava tudo em `experiment_results/<strategy>/studies/`.

## Protocolo

- Transformer: finalistas `voyage`, `bge` e `openai`.
- Unsupervised: melhor embedding de cada algoritmo pela validação interna e,
  depois, os três algoritmos com maior `internal_val_f1_macro`.
- Frações: 5%, 10%, 25%, 50%, 75% e 100%; seeds 42, 52 e 62.
- O ponto de 100% referencia os resultados atuais. Frações menores congelam os
  hiperparâmetros e treinam apenas modelos derivados.
- Threshold `min_fn`: calibrado exclusivamente na validação interna e aplicado
  sem novo ajuste no teste interno e no `dataset_validation`.
- Atenção é tratada como evidência descritiva; ablação de heads e occlusão de
  turnos estimam impacto na decisão.

## Execução local

```bash
cd backend
python -m machine_learning.studies prepare
python -m machine_learning.studies complementarity
python -m machine_learning.studies threshold
python -m machine_learning.studies explainability --embedding voyage --seed 42
python -m machine_learning.studies aggregate
python -m machine_learning.studies report
```

## SLURM

```bash
bash backend/slurm/studies/submit_cached_model_studies.sh --dry-run
bash backend/slurm/studies/submit_cached_model_studies.sh --resume
bash backend/slurm/studies/submit_cached_model_studies.sh --only cached-analysis
bash backend/slurm/studies/submit_cached_model_studies.sh --only explainability
```

O submit falha no preflight quando um checkpoint ou bundle de predição está
ausente. Ele nunca submete `submit_full_pipeline.sh` automaticamente.

## Artefatos

- `dataset_size/metrics.csv` e `aggregate.csv`;
- `complementarity/summary.csv`, `cases.csv` e `ensemble_metrics.csv`;
- `threshold/metrics.csv`, `threshold_curve.csv` e `thresholds.json`;
- `explainability/head_summary.csv`, `head_ablation.csv`,
  `turn_occlusion.csv`, `head_top_connections.csv` e mapas `.npz` dos casos
  representativos.

Os notebooks `judge_decision.ipynb` e `unsupervised.ipynb` apenas leem esses
artefatos em suas novas seções de estudo.
