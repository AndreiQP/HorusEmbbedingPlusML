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
- `hierarchical_explainability/<split>/message_occlusion.csv`,
  `span_occlusion.csv`, `fidelity.csv`, `stability.csv` e explicações JSON por
  amostra. `test_internal` e `validation` são persistidos separadamente.

## Explicabilidade hierárquica BGE

O estudo canônico processa todas as conversas de 	est_internal e
alidation.
Não existe mais o runner amostral. A seed 42 define as explicações; as seeds 52
e 62 medem estabilidade sobre as mesmas perturbações BGE.

Há duas políticas: bsolute_influence, usada para diagnóstico no notebook, e
decision_supporting, usada pela API e pelas métricas de fidelidade. Conversas
curtas são mantidas e cada Top-k informa N_k e metric_N_k.

Para explicar manualmente um texto ou uma amostra cacheada:

`ash
python -m machine_learning.studies hierarchical-explain --conversation-file conversa.txt --seed 42
python -m machine_learning.studies hierarchical-explain --sample-id SAMPLE_ID --split test_internal --seed 42
`

### Execução completa no cluster

`ash
cd /home/andrei.pinto/HorusEmbbedingPlusML
bash backend/slurm/studies/submit_hierarchical_explainability.sh --dry-run
bash backend/slurm/studies/submit_hierarchical_explainability.sh --shards 32
`

O submit valida os três checkpoints, cria o inventário em CPU, executa arrays
GPU para os dois splits e agrega em CPU. Para excluir um nó com falha de ECC:

`ash
bash backend/slurm/studies/submit_hierarchical_explainability.sh --shards 32 --exclude-node dl-01
`

Todos os resultados ficam no único diretório
ackend/experiment_results/transformer/studies/hierarchical_explainability/.
O notebook
otebooks/judge_decision.ipynb lê somente os Parquets dessa
estrutura. O protocolo completo está em
HIERARCHICAL_EXPLAINABILITY_PLAN.md.
