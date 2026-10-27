# Reconstrução do split externo de validação

Fonte autoritativa das 83 conversas Scam: LoveFraud02 v1, DOI
`10.17632/kmhvb4x5d8.1`.

## Estado atual

| Artefato | Estado | Observação |
|---|---|---|
| `raw/all_data.csv` | reconstruído localmente | 1.000 Ham preservadas e 83 Scam reconstruídas; `PFB* = Innocent` |
| Manifesto das 83 conversas | concluído | `data/validation_lovefraud_manifest.json` |
| `processed/messages_transformer/bge_chunks.parquet` | aguardando job do cluster | será totalmente substituído após validação |
| Predições BGE de validação, seeds 42/52/62 | aguardando job do cluster | checkpoints permanecem congelados |
| Métricas BGE de validação, seeds 42/52/62 | aguardando job do cluster | serão recalculadas sobre o novo Parquet |
| Estudos globais de explicabilidade | obsoletos | executar novamente somente após reconstruir predições BGE |
| `messages_transformer` OpenAI/Voyage/E5/MiniLM | obsoletos | reconstrução futura |
| `all_data` e `suspect_turns` processados | obsoletos | reconstrução futura |
| Recortes de 1 a 20 turnos | obsoletos | reconstrução futura |

## Contrato preservado

O Parquet BGE reconstruído mantém exatamente as colunas consumidas pelo código
existente:

```text
text
label
text_embedded
```

`text_embedded` continua sendo uma sequência por conversa, com um vetor BGE-M3
de 1.024 dimensões para cada mensagem. O corte dos últimos 100 turnos continua
sendo responsabilidade do `TransformerDataset`, como no treinamento original.

Não há pooling manual e não há normalização adicional. A codificação usa
`SentenceTransformer("BAAI/bge-m3").encode(..., normalize_embeddings=False)`.

## Execução no cluster

Dry-run:

```bash
bash backend/slurm/studies/submit_rebuild_validation_bge_messages.sh --dry-run
```

Reconstrução completa e reavaliação das três seeds finalistas:

```bash
bash backend/slurm/studies/submit_rebuild_validation_bge_messages.sh
```

O submit baixa os 83 arquivos públicos e reconstrói o `raw` no headnode antes
de enfileirar o job. Assim, o nó GPU não depende de rede: ele usa o
LoveFraud02 e o BGE presentes no cache do filesystem compartilhado.

Somente reconstrução do Parquet, sem alterar métricas/predições:

```bash
bash backend/slurm/studies/submit_rebuild_validation_bge_messages.sh --skip-evaluation
```

Use `--force` somente para ignorar os shards retomáveis e recodificar todas as
mensagens novamente.

## Artefatos e backups

O job produz:

```text
backend/datasets/dataset_validation/processed/messages_transformer/
├── bge_chunks.parquet
├── bge_chunks.manifest.json
├── bge_chunks.pre_speaker_reconstruction.parquet
└── bge_validation_metrics_rebuilt.json
```

Antes de atualizar cada finalista, o job preserva uma cópia de
`metrics.json` e `predictions_validation.npz` com o sufixo
`pre_speaker_reconstruction`.

## Critérios de conclusão do BGE

- 1.083 conversas e distribuição `{Ham: 1000, Scam: 83}`;
- texto e ordem idênticos ao `raw` reconstruído;
- exatamente um embedding de 1.024 dimensões por mensagem;
- nenhum `NaN` ou infinito;
- reprodução de uma conversa Ham inalterada com `atol=1e-5` e `rtol=1e-4`;
- 67.666 mensagens codificadas com a versão atual do raw;
- IDs de predição regenerados a partir do texto corrigido;
- métricas externas recalculadas sem retreinar os Transformers.
