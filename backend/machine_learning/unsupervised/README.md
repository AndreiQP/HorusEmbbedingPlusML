# Pipeline unsupervised

A pipeline compara OCSVM, Isolation Forest, LOF, LUNAR e Deep SVDD nos embeddings `voyage`, `openai`, `e5`, `bge` e `minilm`.

Para cada embedding, `grid_search_anomaly()` cruza as dimensões PCA `25, 60, 100, 150, 200, 250, 300, 500, 750, 1000` com a grade completa do algoritmo. Essa etapa carrega somente o treino e usa `internal_val_f1_macro` para selecionar dimensão e hiperparâmetros. Os desempates são PR-AUC, recall de Scam e menor complexidade.

O finalista de cada embedding é treinado novamente em treino + validação interna com as seeds `42`, `52` e `62`. O mesmo artefato é avaliado no teste interno e no `dataset_validation`. O embedding campeão de cada algoritmo e o campeão geral são escolhidos pela média de `val_f1_macro`.

Os IDs do split ficam em `backend/experiment_results/protocol/unsupervised_split_manifest.json`. Resultados finais, grids, métricas e referências às predições ficam nos resumos em `backend/experiment_results/unsupervised/pipeline/`.

No cluster:

```bash
bash backend/slurm/submit_full_pipeline.sh --unsupervised --force
bash backend/slurm/submit_full_pipeline.sh --unsupervised --dry-run
```

O notebook `backend/notebooks/unsupervised.ipynb` chama a mesma pipeline com `force_retrain=False`. Depois da execução no cluster, todas as chamadas batem no cache; se houver artefatos ausentes, o notebook pode calcular somente as etapas faltantes.
