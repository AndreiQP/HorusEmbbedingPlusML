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

O estudo local por perturbação seleciona as seis mensagens com maior impacto
absoluto no logit e recodifica oclusões de n-grams apenas nessas mensagens. Os
demais vetores da conversa permanecem congelados. Efeitos positivos apoiam Scam;
efeitos negativos apoiam Ham.

O classificador usado é o finalista BGE escolhido por
`internal_val_f1_macro`: 2 heads, 4 camadas, dimensão oculta 256, dropout 0,1,
batch 16, Adam sem weight decay e learning rate `1e-6`. A seed 42 é a execução
canônica nas 20 conversas do estudo principal. As seeds 52 e 62 são executadas
somente em duas conversas de cada categoria para medir estabilidade da explicação
canônica. Seeds não são escolhidas posteriormente por desempenho no teste ou na
validação externa.

No caminho cacheado, `text_embedded` é a fonte oficial dos vetores originais. O
texto concatenado é separado apenas para recuperar mensagens e offsets. O BGE
faz uma auditoria numérica em uma conversa por job e depois codifica somente os
dois baselines, as perturbações e os textos de sufficiency. Os mesmos embeddings
perturbados são reutilizados nas três seeds.

O estudo cacheado exclui conversas com seis ou menos mensagens: o default
`--min-messages 7` exige pelo menos sete turnos após o corte dos últimos 100.
Assim, cada explicação selecionada tem ao menos seis mensagens reais para a
etapa macro. Cada categoria usa os cinco casos elegíveis solicitados ou, quando
isso não for possível, todos os casos disponíveis; o job registra a redução no
log e no manifesto, sem descartar as demais categorias.

Para uma conversa nova:

```bash
cd backend
python -m machine_learning.studies hierarchical-explain \
  --conversation-file conversa.txt --seed 42 \
  --checkpoint-root /caminho/para/experiment_results/transformer \
  --output-dir experiment_results/transformer/studies/hierarchical_explainability
```

Sem `--checkpoint-root`, a busca ocorre em
`backend/experiment_results/transformer`. O loader exige o `run_id` finalista
registrado em `finalists/bge/summary.json`, mesmo que existam outros
`model.pth` na árvore.

Para uma amostra cacheada, usando os embeddings persistidos e pontuando as
mesmas perturbações nos Transformers solicitados:

```bash
python -m machine_learning.studies hierarchical-explain \
  --sample-id SAMPLE_ID --split test_internal --seed 42 --seed 52 --seed 62
```

Para selecionar cinco casos TP/TN/FP/FN na seed 42 e duas conversas de cada
categoria para estabilidade nas seeds 42/52/62:

```bash
python -m machine_learning.studies hierarchical-explain \
  --study --split test_internal --primary-seed 42 \
  --stability-seed 42 --stability-seed 52 --stability-seed 62 \
  --samples-per-category 5 --stability-samples-per-category 2 --min-messages 7
```

Sem `--output-dir`, o split é acrescentado automaticamente ao destino. Assim,
as duas execuções abaixo produzem, respectivamente,
`hierarchical_explainability/test_internal/` e
`hierarchical_explainability/validation/`, sem sobrescrever artefatos:

```bash
python -m machine_learning.studies hierarchical-explain --study --split test_internal
python -m machine_learning.studies hierarchical-explain --study --split validation
```

Para executar somente a explicação principal da seed 42:

```bash
python -m machine_learning.studies hierarchical-explain \
  --study --split test_internal --primary-seed 42 \
  --samples-per-category 5 --skip-stability
```

### Execução no cluster

O estudo hierárquico tem um job GPU próprio porque recodifica as perturbações
textuais no BGE-M3. No cluster, a execução completa recomendada é:

```bash
cd /home/andrei.pinto/HorusEmbbedingPlusML
bash backend/slurm/studies/submit_hierarchical_explainability.sh --dry-run
bash backend/slurm/studies/submit_hierarchical_explainability.sh \
  --split test_internal --samples-per-category 5
```

Os defaults usam `--primary-seed 42` e
`--stability-samples-per-category 2`. Para eliminar completamente a etapa de
estabilidade e exigir apenas o checkpoint 42:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh \
  --split test_internal --skip-stability
```

Se um job falhar com `uncorrectable ECC error`, identifique o nó pelo `sacct` e
reenvie excluindo-o; por exemplo, para o nó `dl-01`:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh \
  --split test_internal --exclude-node dl-01
```

O job usa o cache local do BGE-M3 em modo offline, pois os nós de compute não
precisam — e normalmente não conseguem — acessar o Hugging Face Hub.

O job verifica antes da análise o checkpoint da seed 42 e, quando a estabilidade
está habilitada, também os checkpoints 52 e 62. Ele solicita uma GPU L40S e grava os resultados em
`backend/experiment_results/transformer/studies/hierarchical_explainability/<split>`.
Se os pesos estiverem fora da árvore do projeto:

```bash
bash backend/slurm/studies/submit_hierarchical_explainability.sh \
  --checkpoint-root /caminho/para/experiment_results/transformer
```

Monitoramento:

```bash
squeue -u "$USER"
tail -f slurm_logs/model_studies/bge_hier_exp_JOBID.out
```

O notebook `notebooks/judge_decision.ipynb` apenas lê os artefatos produzidos.
Defina `HIER_SPLIT = 'test_internal'` ou `HIER_SPLIT = 'validation'` para
visualizar cada distribuição separadamente;
ele não recodifica textos nem carrega o Transformer.

O protocolo completo está em `HIERARCHICAL_EXPLAINABILITY_PLAN.md`.

Os notebooks `judge_decision.ipynb` e `unsupervised.ipynb` apenas leem esses
artefatos em suas novas seções de estudo.
