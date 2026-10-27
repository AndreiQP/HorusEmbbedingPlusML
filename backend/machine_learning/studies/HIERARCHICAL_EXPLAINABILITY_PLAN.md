# Estudo global de explicabilidade BGE

Este é o único runner do estudo. Ele processa todas as
conversas de `test_internal` e `validation`, grava Parquets retomáveis por shard
e só consolida resultados quando a cobertura está completa.

## Políticas de ranking

- `absolute_influence`: as seis mensagens com maiores valores de `abs(delta_logit)`, nas duas
  direções. É a visão diagnóstica do notebook.
- `decision_supporting`: somente efeitos que sustentam a classe prevista e têm
  `class_support_delta > 1e-4`. É a política da API e das métricas de fidelidade.

Para Scam, `class_support_delta = delta_logit`; para Ham,
`class_support_delta = -delta_logit`. Os n-grams são gerados para a união das
duas Top-6 da seed 42, codificados uma vez e reutilizados nas seeds 52 e 62.

## Cluster

Valide o fluxo sem submeter jobs:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh --dry-run
```

Execute o estudo completo:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh --shards 32
```

O submit cria um inventário CPU, dois arrays GPU e uma agregação CPU com
dependências `afterok`. Para evitar um nó com falha de ECC:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh \
  --shards 32 --exclude-node dl-01
```

Quando os artefatos globais principais já existem, execute apenas o estudo de
saturação das curvas de fidelity:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh \
  --saturation-only --shards 32
```

Esse caminho usa somente a seed 42, não recalcula n-grams e avalia a remoção
cumulativa desde Top-1 até todas as mensagens disponíveis. `k95` é o primeiro
`k` que alcança 95% do maior comprehensiveness alinhado no logit observado na
conversa. Os resultados registram tanto `k` quanto `k / total_de_mensagens` e
as curvas correspondentes de sufficiency.

Cada shard tem manifesto, hashes dos Parquets e `complete.json`. Uma nova
submissão reutiliza shards íntegros; arquivos incompletos ou com checksum
divergente são recalculados.

## Saídas

Os artefatos ficam em
`backend/experiment_results/transformer/studies/hierarchical_explainability`.
As pastas dos splits contêm resultados por conversa; `aggregates/` contém
curvas, distribuições, cobertura `N_k`, controles e estabilidade. O notebook
`backend/notebooks/judge_decision.ipynb` lê somente esses artefatos na seção de explicabilidade.

As curvas adicionais de saturação ficam em
`aggregates/saturation_curves.parquet`,
`saturation_fraction_curves.parquet`, `saturation_points.parquet` e
`saturation_summary.parquet`.

`N_k` é a quantidade de conversas com ao menos `k` turnos. `metric_N_k` é a
quantidade que também possui ao menos `k` evidências válidas alinhadas à
decisão. A distinção é intencional e aparece no primeiro gráfico.
