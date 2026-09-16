# Pipeline unsupervised

A pipeline compara OCSVM, Isolation Forest, LOF, LUNAR e Deep SVDD nos embeddings `voyage`, `openai`, `e5`, `bge` e `minilm`.

Para cada embedding, `grid_search_anomaly()` cruza as dimensões PCA `25, 60, 100, 150, 200, 250, 300, 500, 750, 1000` com a grade completa do algoritmo. Essa etapa carrega somente o treino e usa `internal_val_f1_macro` para selecionar dimensão e hiperparâmetros. Os desempates são PR-AUC, recall de Scam e menor complexidade.

O finalista de cada embedding é treinado novamente em treino + validação interna com as seeds `42`, `52` e `62`. O mesmo artefato é avaliado no teste interno e no `dataset_validation`. O embedding campeão de cada algoritmo e o campeão geral são escolhidos pela média de `val_f1_macro`.

## Estudos adicionais

O estudo cache-first em `machine_learning/studies/` seleciona separadamente os
três campeões pela validação interna e produz curvas em 5%, 10%, 25%, 50%, 75%
e 100% do Ham de ajuste. Não repete o grid search e não altera este ranking ou
seus caches. As visualizações estão em `notebooks/unsupervised.ipynb`.

Os IDs do split ficam em `backend/experiment_results/protocol/unsupervised_split_manifest.json`. Resultados finais, grids, métricas e referências às predições ficam nos resumos em `backend/experiment_results/unsupervised/pipeline/`.

No cluster:

```bash
bash backend/slurm/submit_full_pipeline.sh --unsupervised --force
bash backend/slurm/submit_full_pipeline.sh --unsupervised --dry-run
```

## CVDD e DATE (texto bruto)

CVDD e DATE usam diretamente o texto das conversas com BGE, em vez dos
embeddings de conversa pré-computados usados pelos demais detectores. Depois de
treinados, seus resultados podem ser visualizados sem novo treino nos notebooks
`notebooks/CVDD_results.ipynb` e `notebooks/DATE_results.ipynb`.

Para reconstruir as curvas ROC/PR, histogramas de score e tabelas no cluster a
partir dos checkpoints já persistidos:

```bash
bash backend/slurm/submit_text_model_reports.sh all
# ou: bash backend/slurm/submit_text_model_reports.sh cvdd
# ou: bash backend/slurm/submit_text_model_reports.sh date
```

Os artefatos ficam em `experiment_results/unsupervised/text_reports/`. Esse job
somente carrega checkpoints; não usa `--force-retrain` e não altera pesos nem
os resultados dos outros detectores.

O notebook `backend/notebooks/unsupervised.ipynb` chama a mesma pipeline com `force_retrain=False`. Depois da execução no cluster, todas as chamadas batem no cache; se houver artefatos ausentes, o notebook pode calcular somente as etapas faltantes.
