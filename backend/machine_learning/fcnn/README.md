# `machine_learning/fcnn/` — Rede Neural Fully Connected (Multi-Embedding)

Abordagem **totalmente supervisionada**: uma rede neural *feed-forward* (MLP) em PyTorch
recebe a **concatenação de vários embeddings** da mesma conversa (ex.: `voyage + openai +
bge + e5 + minilm`) e aprende a classificar `Ham` (0) vs `Scam` (1). A ideia é que
combinar múltiplas representações vetoriais capture sinais complementares que um único
embedding não capturaria sozinho.

---

## 1. Arquivos

| Arquivo | Conteúdo |
|---|---|
| `fcnn_classifier.py` | `ScamClassifierFCNN` (arquitetura da rede) e `FraudDataset` (Dataset PyTorch auxiliar, não usado pelo `runner.py` atual — ele monta `TensorDataset` diretamente). |
| `runner.py` | `train_fcnn()`, `learning_curve_fcnn()`, `evaluate_fcnn_cross_turns()`, `grid_search_fcnn()` — ciclo completo de treino/avaliação/cache. |

## 2. Arquitetura (`ScamClassifierFCNN`)

```
input_size (soma das dimensões dos embeddings concatenados, ex: 1024+1536+1024+1024+384)
   → Linear(input_size, 2048) → BatchNorm1d → ReLU → Dropout(0.3)
   → Linear(2048, 1024)       → BatchNorm1d → ReLU → Dropout(0.3)
   → Linear(1024, 516)        → BatchNorm1d → ReLU → Dropout(0.2)
   → Linear(516, 2)                                            # 2 logits: [Ham, Scam]
```
- Saída de 2 logits (não 1 + sigmoide): o treino usa `nn.CrossEntropyLoss` e a
  probabilidade de Scam é obtida via `softmax(out, dim=1)[:, 1]`.
- `BatchNorm1d` exige batches com mais de 1 amostra (cuidado com `batch_size=1` ou o
  último batch incompleto de tamanho 1 — o `DataLoader` padrão do runner evita isso na
  prática pois usa `batch_size=64`).

## 3. Fluxo de treino (`train_fcnn()`)

```
load_concat_embeddings("train", path_data, embedders, with_augmented)
        │  (concatena horizontalmente cada embedding: X.shape[1] = soma das dims)
        ├── train_test_split (test_size, padrão 20%, estratificado)
        ├── DataLoader(train) / DataLoader(test)
        │
        ├── loop de épocas (padrão 50, Adam, CrossEntropyLoss):
        │     train_one_epoch() → val (mesmo holdout de teste interno) a cada época
        │     Early Stopping: guarda o melhor state_dict por val_f1 (macro);
        │     interrompe após `patience` épocas sem melhora
        │
        └── restaura o melhor checkpoint (por val_f1) e avalia no holdout de teste
```
- `embedders=None` usa **todos** os embeddings definidos em `config.EMBEDDERS` (5 no
  total); pode-se passar uma lista menor (ex.: `["voyage", "bge"]`) para reduzir a
  dimensão de entrada.
- `with_augmented=True` inclui as conversas aumentadas sinteticamente; o pipeline valida
  que todos os embedders têm o mesmo número de amostras e mesmos labels antes de
  concatenar (ver `load_concat_embeddings` em `data.py`).
- **Não há avaliação automática em `dataset_validation`** dentro de `train_fcnn()` — só
  o holdout interno de teste. Use `evaluate_fcnn_cross_turns()` ou carregue
  `load_concat_embeddings("validation", ...)` manualmente para a checagem externa.

### Funções auxiliares

- `learning_curve_fcnn(...)` — treina com frações crescentes do pool de treino (mesmo
  padrão dos outros runners: holdout de teste interno fixo + validação externa).
- `evaluate_fcnn_cross_turns(...)` — avalia um modelo já treinado no dataset externo
  (`dataset_validation`), útil para checar generalização "vida real".
- `grid_search_fcnn(...)` — varre hiperparâmetros (ex.: `lr`, `weight_decay`,
  `batch_size`) chamando `train_fcnn()` repetidamente.

## 4. Cache (`ModelCache`)

```
experiment_results/fcnn/{augmented|no_augmented}__{path_data}/
    model.pth      # torch.save(model.state_dict())
    metrics.json
    history.csv    # loss/f1 por época (treino e validação)
```
- Pasta determinada só por `with_augmented` + `path_data` (não pelos embedders
  escolhidos) — **cuidado**: treinar com um subconjunto diferente de `embedders` para a
  mesma combinação `with_augmented`/`path_data` sobrescreve o cache anterior, pois a
  dimensão de entrada (`input_size`) muda mas a chave de cache não reflete isso.
  Se for comparar diferentes conjuntos de embedders, use `force_retrain=True` ou trate
  manualmente os resultados antes de re-treinar.
- Cache-hit: recarrega o modelo (`ScamClassifierFCNN(input_size=X.shape[1])` reconstruído
  a partir da 1ª camada salva, ver `ModelCache.fcnn_load`), reavalia no holdout de teste
  interno (recalculado a partir dos dados, não do cache) para gerar `y_true/y_pred/y_prob`
  atualizados, e recarrega `history.csv`/`metrics.json` do disco.

## 5. Como rodar

```python
from machine_learning import train_fcnn, learning_curve_fcnn

result = train_fcnn(path_data="all_data", with_augmented=True, epochs=50, lr=1e-3)
result.print_metrics()
result.plot_learning_curves(metrics=["loss", "f1"])
result.plot_confusion_matrix()

lc = learning_curve_fcnn(path_data="all_data")
lc.plot_dataset_size_curve(metric="f1_macro_mean")
```

## 6. Métricas

`compute_metrics()` padrão (`accuracy`, `f1_macro`, `f1_scam`, `precision_scam`,
`recall_scam`, `roc_auc`, `pr_auc`, `TP/TN/FP/FN`), calculadas a partir de
`softmax(...)[:, 1]` como `y_prob`.
