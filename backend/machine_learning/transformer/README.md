# Pipeline Transformer

A pipeline compara os Transformers dos embeddings `voyage`, `bge` e `openai`. Todos têm limite de 100 épocas, early stopping em `internal_val_f1_macro` (`patience=8`, `min_delta=1e-4`), restauração do melhor checkpoint, `ReduceLROnPlateau`, mixed precision e gradient clipping em `1.0`.

A busca ocorre em três etapas, sempre usando apenas treino e validação interna:

1. `num_heads=[2,4,8]` e `num_layers=[1,2,3,4]`;
2. `hidden_dim=[128,256,512]`, `dropout=[0.1,0.2,0.3]` e `batch_size=[16,32]`;
3. Adam sem weight decay e AdamW com `weight_decay=[1e-4,1e-2]`.

São preservados os LRs já definidos: `voyage=2e-6`, `bge=1e-6` e `openai=1e-6`. A configuração vencedora de cada embedding é repetida com seeds `42`, `52` e `62`, mantendo o split fixo em 42.

No `dataset_validation`, são comparados os três modelos, majority voting, média das probabilidades e soma de logits calibrados por temperatura. A temperatura é ajustada somente na validação interna. IDs e rótulos precisam estar perfeitamente alinhados antes do ensemble.

No cluster:

```bash
bash backend/slurm/submit_full_pipeline.sh --transformer --force
bash backend/slurm/submit_full_pipeline.sh --transformer --dry-run
```

O notebook `backend/notebooks/judge_decision.ipynb` chama a mesma pipeline com `force_retrain=False`. Após o processamento no cluster, as buscas e os finalistas são recuperados do cache; se algo estiver faltando, o notebook pode produzir a etapa ausente.
