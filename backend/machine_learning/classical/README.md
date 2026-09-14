# `machine_learning/classical/` — Classificadores Supervisionados Clássicos

Abordagem **totalmente supervisionada**: cada conversa é representada por **um único vetor
de embedding** e um classificador tradicional (scikit-learn / XGBoost) aprende a separar
`Ham` (0) de `Scam` (1) a partir de exemplos rotulados dos dois lados — diferente do
`unsupervised/`, aqui o modelo **vê exemplos de Scam durante o treino**.

---

## 1. Arquivos

| Arquivo | Conteúdo |
|---|---|
| `classifier.py` | `get_classifier(name)` / `get_all_classifiers()` — fábrica dos modelos (RF, KNN, SVM, NaiveBayes, XGBoost) com hiperparâmetros padrão já definidos. |
| `runner.py` | `train_classical()`, `train_all_classical()`, `learning_curve_classical()` — todo o ciclo de treino/avaliação/cache. |

## 2. Modelos disponíveis (`classifier.py`)

| Chave | Modelo | Observações |
|---|---|---|
| `RF` | `RandomForestClassifier` | `n_estimators=100` (ou `sqrt(n_amostras)` se `length_data` for passado). |
| `KNN` | `KNeighborsClassifier` | `n_neighbors=5`. |
| `SVM` | `SVC(kernel="linear", probability=True)` | `probability=True` habilita `predict_proba` para ROC/PR e calibração de threshold. |
| `NAIVEBAYES` | `Pipeline(MinMaxScaler → MultinomialNB)` | precisa de features não-negativas, por isso o `MinMaxScaler`. |
| `XGBOOST` | `XGBClassifier` | opcional — só aparece em `get_all_classifiers()` se o pacote `xgboost` estiver instalado. |

## 3. Fluxo de treino (`train_classical()`)

```
load_single_embedding("train", path_data, embedding, with_augmented)
        │
        ├── train_test_split (80% pool / 20% teste interno, estratificado)
        │
        ├── StratifiedKFold (n_cv_folds, padrão 5) sobre o pool de treino
        │     → mede estabilidade (f1_macro por fold) ANTES do treino final
        │
        ├── fit() no pool de treino inteiro (modelo final)
        │
        └── avaliação no holdout de teste interno (20%)
              → predict_proba/decision_function quando disponível → roc_auc/pr_auc
```

- `with_augmented=True` usa o dataset com augmentação (`origin="augmented"`); `False` usa
  somente `origin="external"` dos dados brutos (`origin="external_only"` no cache).
- Diferente do `unsupervised/`, **não há avaliação automática em `dataset_validation`**
  dentro de `train_classical()` — o holdout de 20% é o único conjunto de teste. Para
  avaliar no dataset externo, use `evaluate_model_on_external()` (`evaluation.py`) ou
  chame `train_classical()` e rode `model.predict()` manualmente sobre
  `load_single_embedding("validation", ...)`.

### Funções auxiliares

- `train_all_classical(path_data, embedders=None, model_names=None)` — treina o produto
  cartesiano modelo × embedding (por padrão, todos os 5 modelos × todos os embeddings) e
  retorna um dict `{"{model}_{embedding}": ExperimentResult}`.
- `learning_curve_classical(embedding, model_name, fractions=[0.05..1.00], n_repeats=3)` —
  treina o modelo com frações crescentes do pool de treino (holdout de 20% **fixo**) e
  avalia tanto no holdout interno quanto em `dataset_validation` (vida real), retornando
  `df_history` com `frac`, `f1_macro_mean/std`, `val_f1_macro_mean/std`, contagens de
  `TP/TN/FP/FN` (absolutas e percentuais) para os dois conjuntos.

## 4. Cache (`ModelCache`, ver `cache.py`)

Modelo persistido em:
```
experiment_results/classical/{origin}__{path_data}__{embedding}__{ModelClassName}/
    model.joblib      # joblib.dump do estimador treinado
    metrics.json       # saída de compute_metrics() no holdout de teste interno
    history.csv         # métricas por fold da cross-validation
    confusion_matrix_test.png
```
- `origin` ∈ `{augmented, external_only}`, conforme `with_augmented`.
- `ModelClassName` é `type(model).__name__` (ex.: `RandomForestClassifier`, `SVC`).
- Cache-hit: se `force_retrain=False` e o `.joblib` já existir, o modelo é carregado
  direto do disco (sem CV, sem re-treino); métricas/histórico são recarregados dos
  respectivos `.json`/`.csv` quando existirem.
- Curvas de aprendizado (`learning_curve_classical`) são cacheadas separadamente em
  `experiment_results/classical/lc_{run_id}/history.csv`.

## 5. Como rodar

```python
from machine_learning import train_classical, train_all_classical, learning_curve_classical

# Um modelo específico
result = train_classical(embedding="bge", path_data="all_data", model_name="SVM")
result.print_metrics()
result.plot_confusion_matrix()
result.plot_roc_curve()

# Todos os modelos x todos os embeddings
results = train_all_classical(path_data="all_data")

# Curva de aprendizado
lc = learning_curve_classical(embedding="voyage", path_data="all_data", model_name="RF")
lc.plot_dataset_size_curve(metric="f1_macro_mean")
```

## 6. Métricas

Todas calculadas por `evaluation.compute_metrics()`: `accuracy`, `f1_macro`, `f1_scam`,
`precision_macro`, `recall_macro`, `precision_scam`, `recall_scam`, `roc_auc`, `pr_auc`,
`TP`/`TN`/`FP`/`FN`. `roc_auc`/`pr_auc` só são calculados quando o modelo expõe
`predict_proba` (a maioria) ou `decision_function` (fallback usado por alguns modelos).
