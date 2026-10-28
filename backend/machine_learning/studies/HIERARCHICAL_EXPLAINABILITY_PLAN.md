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

## Comparação por unidade contextual

O estudo comparativo fica dentro do mesmo estudo hierárquico, em
`hierarchical_explainability/context_blocks/`. Ele não altera a API nem a
extensão. As sete representações avaliadas são:

- `canonical_message`;
- `speaker_block_4`;
- `chronological_window_4` até `chronological_window_8`.

O corte das últimas 100 mensagens acontece antes da construção dos blocos.
Blocos por interlocutor preservam sobras; janelas cronológicas usam stride 1 e
um bloco parcial em conversas menores que a janela. Cada bloco guarda os IDs,
índices, interlocutores, textos e offsets das mensagens originais.

Cada representação é avaliada em `frozen_probe` e `retrained`. O primeiro usa
os três Transformers finalistas canônicos sem retreino. O segundo usa um
Transformer treinado especificamente para a representação, com a configuração
finalista BGE (batch 16, LR 1e-6, 2 heads, 4 layers, hidden 256, dropout 0,1,
até 100 épocas e early stopping). O canônico reutiliza os finalistas existentes
nos dois modos e funciona como controle de identidade.

Valide a submissão:

```bash
bash backend/slurm/studies/submit_context_block_explainability.sh --dry-run
```

Execute o fluxo completo:

```bash
bash backend/slurm/studies/submit_context_block_explainability.sh \
  --shards 32 --max-parallel 8
```

Por padrão, cada fase envia notificações SLURM `BEGIN`, `END` e `FAIL` para
`a169194@dac.unicamp.br`. O endereço pode ser substituído com
`--mail-user endereco@dominio` ou pela variável `SLURM_MAIL_USER`. Use
`--no-mail` para desabilitar as notificações.

Para evitar um nó conhecido com erro de GPU:

```bash
bash backend/slurm/studies/submit_context_block_explainability.sh \
  --shards 32 --max-parallel 8 --exclude-node dl-01
```

A submissão encadeia quatro fases com `afterok`: cache BGE, treinamento,
explicabilidade e agregação. Os resultados consolidados ficam em
`context_blocks/aggregates/`. O notebook
`backend/notebooks/bge_explainability.ipynb` lê esses Parquets e mantém sempre
separados split, representação, modo, seed, política de ranking e população
(`all_available` ou `paired_common`).

As métricas de fidelidade são calculadas até todas as unidades disponíveis e
registram simultaneamente unidades selecionadas, mensagens originais únicas,
fração da conversa e redundância. `k95` e `k_joint` são comparados em mensagens
originais únicas, evitando favorecer artificialmente janelas sobrepostas.
