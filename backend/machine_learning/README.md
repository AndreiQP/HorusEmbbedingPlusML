# `backend/machine_learning/` — Biblioteca de Machine Learning para Detecção de Scam

Biblioteca Python que reúne **4 abordagens de modelagem** diferentes para o mesmo
problema — detectar conversas de golpe (`Scam`, label=1) contra conversas legítimas
(`Ham`, label=0) — todas compartilhando a mesma infraestrutura de **carregamento de
dados, cache de experimentos, métricas e objeto de resultado** (`ExperimentResult`).

Este documento explica **o que cada arquivo faz**. Para entender **como cada abordagem
funciona por dentro** (arquitetura do modelo, fluxo de treino, estrutura do cache),
veja o README dedicado de cada subpasta:

| Abordagem | Pasta | README |
|---|---|---|
| Classificadores clássicos supervisionados (SVM, RF, KNN, NaiveBayes, XGBoost) | `classical/` | [`classical/README.md`](classical/README.md) |
| Rede neural Fully Connected multi-embedding | `fcnn/` | [`fcnn/README.md`](fcnn/README.md) |
| Transformer Encoder sequencial + Ensemble | `transformer/` | [`transformer/README.md`](transformer/README.md) |
| Detecção de anomalias (7 modelos: OCSVM, IForest, LOF, LUNAR, Deep SVDD, CVDD, DATE) | `unsupervised/` | [`unsupervised/README.md`](unsupervised/README.md) |

---

## 1. Visão geral: o que é comum a todas as abordagens

```
backend/machine_learning/
├── __init__.py         # Ponto único de importação pública da biblioteca
├── result.py            # ExperimentResult — objeto de retorno unificado de todos os runners
├── cache.py              # ModelCache — persistência/carregamento em experiment_results/
├── data.py                # Carregamento unificado dos parquets/CSVs de embeddings
├── evaluation.py           # compute_metrics() + todas as funções de plot
├── thresholds.py            # ThresholdCalibrator — calibração de threshold de decisão
├── classical/                # Abordagem 1 (ver classical/README.md)
├── fcnn/                      # Abordagem 2 (ver fcnn/README.md)
├── transformer/                 # Abordagem 3 (ver transformer/README.md)
├── unsupervised/                  # Abordagem 4 (ver unsupervised/README.md)
└── (embedder.py, predict.py,        # Scripts legados/utilitários — ver seção 3
    train.py, visualize_results.py)
```

Todas as 4 abordagens seguem o mesmo contrato:
1. Uma função `train_*()` que recebe uma config (embedding, hiperparâmetros, etc.),
   treina (ou carrega do cache) e retorna um `ExperimentResult`.
2. `ExperimentResult` expõe o modelo treinado, as métricas (`df_metrics`), o histórico
   de treino (`df_history`) e métodos de plot prontos (`.plot_confusion_matrix()`,
   `.plot_roc_curve()`, etc.).
3. `ModelCache` decide, de forma transparente, se re-treina do zero ou carrega de
   `experiment_results/<estrategia>/<run_id>/` — cada runner define sua própria
   convenção de nome de pasta (documentado no README de cada abordagem).

### 1.1 Datasets e split (convenção geral)

Duas fontes de dados físicas, usadas por todas as abordagens:

- **`backend/datasets/dataset_train/`** — usado para treinar **e** para a avaliação
  interna (holdout/cross-validation/curva de aprendizado). A maioria dos runners separa
  um percentual fixo (geralmente 20%) como "teste interno"; o `unsupervised/` vai além e
  faz um split 3-way explícito (`train`/`val`/`test`).
- **`backend/datasets/dataset_validation/`** — dataset **externo**, nunca usado no
  treino. Representa o comportamento esperado em produção/"vida real". Nem todo runner
  avalia automaticamente aqui dentro da função de treino (ver README de cada abordagem).

### 1.2 Embeddings suportados (`config.EMBEDDERS`)

| Chave | Modelo por trás | Uso |
|---|---|---|
| `voyage` | `voyage-3-large` (Voyage AI) | Todas as abordagens |
| `openai` | `text-embedding-3-large` (OpenAI) | Todas as abordagens |
| `bge` | `BAAI/bge-m3` (local, sentence-transformers) | Todas as abordagens — único usado por CVDD/DATE |
| `e5` | `intfloat/multilingual-e5-large` (local) | Todas as abordagens |
| `minilm` | `all-MiniLM-L6-v2` (local) | Todas as abordagens |

---

## 2. Arquivos comuns — o que cada um faz

### `__init__.py`
Ponto único de importação pública (`from machine_learning import train_anomaly, ...`).
Reexporta:
- Núcleo: `ExperimentResult`, `ModelCache`, `ThresholdCalibrator`.
- Dados: `load_single_embedding`, `load_concat_embeddings`, `load_sequence_embeddings`,
  `load_ham_only`, `get_embedding_dim`.
- Avaliação/plots: `compute_metrics`, `plot_confusion_matrix`, `plot_epoch_history`,
  `plot_dataset_size_curve`, `plot_precision_recall_curve`, `plot_roc_curve`,
  `plot_pca_variance`, `plot_umap_2d`, `plot_dimension_study`, etc.
- Runners de cada abordagem: `train_classical`/`train_all_classical`/
  `learning_curve_classical`, `train_fcnn`/`learning_curve_fcnn`/
  `evaluate_fcnn_cross_turns`/`grid_search_fcnn`, `train_transformer`/
  `load_trained_transformers`/`evaluate_ensemble`/`plot_ensemble_diagnostics`/
  `learning_curve_transformer`, `train_anomaly`/`learning_curve_anomaly`/
  `grid_search_anomaly`/`train_all_anomaly`/`compare_dimensions_anomaly`/
  `train_cvdd`/`train_date`.
- Auxiliares: `get_all_classifiers` (classical), `ScamClassifierFCNN` (fcnn).

Sempre que uma função nova for adicionada a um runner e precisar ser usada fora do
pacote, ela deve ser importada e adicionada em `__all__` aqui.

### `result.py` — `ExperimentResult`
Classe contêiner **retornada por todos os runners** (`train_*`, `learning_curve_*`,
`grid_search_*`). Armazena:
- `strategy` (`"classical"`, `"fcnn"`, `"transformer"`, `"unsupervised"`), `config`
  (dict com os parâmetros que geraram o experimento — usado para calcular o `run_id`
  do cache), `model` (objeto sklearn/PyTorch treinado).
- `df_metrics` (métricas finais, uma linha por partição avaliada — `train`/`val`/
  `test`/`validation_external`, dependendo da abordagem), `df_history` (por
  época/fold/fração, conforme o modo), `y_true`/`y_pred`/`y_prob`.
- `artifacts_dir` — pasta onde os plots (`.png`) são salvos automaticamente.

Métodos prontos para uso direto em notebook (cada um salva o PNG em `artifacts_dir` se
`save=True`, além de exibir inline):
```python
result.print_metrics()                       # tabela formatada
result.print_classification_report()         # sklearn.classification_report
result.plot_confusion_matrix()
result.plot_learning_curves(metrics=["loss","f1"])   # FCNN/Transformer
result.plot_dataset_size_curve(metric=...)           # curvas de aprendizado (%dataset)
result.plot_confusion_matrix_evolution()
result.plot_confusion_matrices_by_fraction()
result.plot_roc_curve()
result.plot_precision_recall_curve()
result.plot_pca_variance()                           # unsupervised (PCA)
result.plot_umap()                                   # unsupervised (UMAP)
result.calibrate_threshold(strategy="target_recall", target_recall=0.95)
result.evaluate_on_validation()                       # reavalia no dataset_validation
result.compare_with(other_result, metric="f1_macro")
```

### `cache.py` — `ModelCache`
Sistema de persistência central. Tudo é salvo dentro de
`backend/experiment_results/<estrategia>/`. Principais peças:
- `_run_id(config)` — gera um identificador determinístico a partir do dict `config`
  (concatena `chave=valor` ordenado por chave; se ficar maior que 80 caracteres, trunca
  e acrescenta um hash MD5 de 8 caracteres do texto completo, para nunca colidir e
  nunca estourar limites de tamanho de path).
- `params_hash(model_kwargs)` — hash curto e determinístico de um dict de
  hiperparâmetros; usado pelo `unsupervised/` para não confundir, no cache, um modelo
  treinado com hiperparâmetros padrão com um treinado com hiperparâmetros vencedores de
  grid search (mesmo embedding/dimensão).
- Um par `<estrategia>_save()`/`<estrategia>_load()`/`<estrategia>_exists()` por
  abordagem (`classical_*`, `fcnn_*`, `transformer_*`, `unsupervised_*`,
  `text_model_*` para CVDD/DATE) — cada um sabe montar o nome de pasta específico
  daquela abordagem (documentado no README de cada uma).
- Genéricos, usados por todas as abordagens: `save_metrics`/`load_metrics` (JSON),
  `save_history`/`load_history` (CSV), `save_predictions`/`load_predictions`
  (`y_true`/`y_pred`/`y_prob` em `.npz`), `artifacts_dir(strategy, run_id)`.

### `data.py` — carregamento unificado de dados
Abstrai a leitura dos parquets processados (`datasets/dataset_train/processed/...` e
equivalente em `dataset_validation/`), sempre a partir de `config.EMBEDDERS`:
- `load_single_embedding(split, path_data, embedding, with_augmented, label_filter)` —
  1 vetor por conversa. Usado por `classical/` e `unsupervised/`.
- `load_concat_embeddings(split, path_data, embedders, with_augmented)` — concatena
  horizontalmente vários embeddings da mesma conversa (valida que todos têm o mesmo
  número de amostras/labels). Usado por `fcnn/`.
- `load_sequence_embeddings(split, embedding, with_augmented)` — carrega a sequência de
  embeddings por turno de mensagem (não agregado). Usado por `transformer/`.
- `load_ham_only(split, path_data, embedding)` — atalho de `load_single_embedding` com
  `label_filter=0`; usado pelo treino one-class do `unsupervised/`.
- `path_data` ∈ `{"all_data", "suspect_turns"}` — dois recortes diferentes do dataset
  (todas as conversas vs. apenas os turnos suspeitos).
- `with_augmented=False` filtra para `origin == "external"` (remove dados de
  augmentação sintética) — convenção usada pelo `unsupervised/`, já que ali o objetivo é
  aprender o padrão "real" de Ham.

### `evaluation.py` — métricas e plots
- `compute_metrics(y_true, y_pred, y_score=None, model_name="", dataset_name="")` —
  função central de métricas, usada por **todas** as abordagens. Retorna `accuracy`,
  `f1_macro`, `f1_scam`, `precision_macro`, `recall_macro`, `precision_scam`,
  `recall_scam`, `roc_auc`, `pr_auc` (calculados a partir de `y_score`/`y_prob` quando
  informado, senão `None`), `TP`, `TN`, `FP`, `FN`.
- Funções de plot com assinatura padronizada `(..., ax=None, save_path=None)` — se
  `ax=None`, cria a própria figura e mostra inline; se recebe `ax`, desenha nele (para
  compor grades de subplots); se `save_path` for informado, também salva PNG:
  `plot_confusion_matrix`, `plot_epoch_history`, `plot_dataset_size_curve`,
  `plot_confusion_matrix_evolution`, `plot_confusion_matrices_by_fraction`,
  `plot_precision_recall_curve`, `plot_roc_curve`, `plot_pca_variance`, `plot_umap_2d`,
  `plot_dimension_study`, `plot_loss_evolution_all_embeddings`.
- `print_metrics_table(df_metrics)` — impressão tabular simples.

### `thresholds.py` — `ThresholdCalibrator`
Ajusta o ponto de corte de probabilidade (padrão 0.5) para otimizar um objetivo de
negócio, tipicamente **minimizar Falsos Negativos** (deixar passar um Scam custa caro):
- `strategy="target_recall"` — menor threshold que garante `recall_scam >= target`.
- `strategy="min_fn"` — minimiza FN, com o menor FP possível.
- `strategy="f_beta"` — maximiza F-beta (dá mais peso ao recall conforme `beta`).
- `strategy="youden"` — maximiza `TPR - FPR` (ponto de Youden da curva ROC).
```python
calibrator = result.calibrate_threshold(strategy="target_recall", target_recall=0.98)
calibrator.plot_fp_fn_tradeoff()
calibrator.plot_precision_recall_curve()
y_pred_calibrado = calibrator.apply(y_prob_validacao)
```

---

## 3. Arquivos legados/utilitários (fora do fluxo principal)

Estes arquivos ficam soltos em `machine_learning/` mas **não fazem parte da API atual**
usada pelos notebooks/runners (`classical/`, `fcnn/`, `transformer/`, `unsupervised/`).
São scripts de uma versão anterior do projeto (antes da refatoração em subpacotes) ou
utilitários pontuais — importam módulos com caminhos antigos
(`machine_learning.classifier`, `machine_learning.fcnn_classifier`) que não existem mais
na raiz do pacote:
- `train.py` — script original de treino em lote dos classificadores clássicos.
- `predict.py` — função `predict()` para carregar um `.pth` da FCNN e gerar previsões.
- `visualize_results.py` — utilitário para parsear nomes de arquivo de modelos salvos
  (`{origin}_{path_data}_{embedding}_{ModelName}.joblib`) e plotar comparações.
- `embedder.py` — script de **geração** de embeddings (chama as APIs Voyage/OpenAI e os
  modelos locais via `sentence-transformers`) a partir dos CSVs brutos — usado offline
  para produzir os parquets consumidos por `data.py`, não durante o treino/avaliação.

Se for reaproveitar algo daqui, verifique os imports antes — provavelmente precisam ser
ajustados para os caminhos atuais (`from .classical.classifier import ...`, etc.).

---

## 4. Notebooks de experimentos (`backend/notebooks/`)

| Notebook | Abordagem | Conteúdo |
|---|---|---|
| `ML_models.ipynb` | `classical/` | Comparação dos classificadores × 5 embeddings, grid search de SVM, calibração de threshold, curvas de aprendizado. |
| `FCNN_test.ipynb` | `fcnn/` | Treino da FCNN com múltiplos embeddings concatenados, histórico de loss/f1 por época. |
| `unsupervised.ipynb` | `unsupervised/` | Os 7 modelos de detecção de anomalias: UMAP/PCA exploratórios, estudo de dimensionalidade, grid search de hiperparâmetros, treino final por embedding, curvas de aprendizado e leaderboard comparativo final. Ver `unsupervised/README.md` para o detalhe completo de cada seção. |
| `judge_decision.ipynb` | — | Notebook auxiliar de análise/decisão (fora do escopo dos 4 runners principais). |

---

## 5. Onde experimentos ficam salvos

Todo `run_id`/pasta de cache vive sob `backend/experiment_results/<estrategia>/`, uma
subpasta por abordagem (`classical/`, `fcnn/`, `transformer/`, `unsupervised/`). A
convenção exata do nome de pasta e do conteúdo salvo está documentada no README de cada
abordagem — veja a tabela no topo deste arquivo.
