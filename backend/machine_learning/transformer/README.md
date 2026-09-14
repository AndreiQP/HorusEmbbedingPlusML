# `machine_learning/transformer/` — Transformer Encoder Sequencial + Ensemble

Abordagem **totalmente supervisionada**, mas diferente de `classical/`/`fcnn/`: em vez de
um único vetor por conversa, cada conversa é representada como uma **sequência de
embeddings por turno de mensagem** (uma mensagem = um vetor). Um Transformer Encoder
processa essa sequência (com *positional encoding* e *padding mask*) e faz *pooling* para
gerar uma única predição Ham/Scam por conversa. Os 3 melhores embeddings (`voyage`, `bge`,
`openai`) são treinados individualmente e depois combinados em um **Ensemble por votação**.

---

## 1. Arquivos

| Arquivo | Conteúdo |
|---|---|
| `runner.py` | Arquitetura (`TransformerScamClassifier`, `PositionalEncoding`), treino (`train_transformer`), carregamento (`load_transformer_result`, `load_all_transformer_results`), ensemble (`evaluate_ensemble`, `evaluate_ensemble_from_results`, `plot_ensemble_diagnostics`) e `learning_curve_transformer`. |

## 2. Arquitetura (`TransformerScamClassifier`)

```
sequência de embeddings (B, T, emb_dim), T = nº de turnos da conversa
   → PositionalEncoding (soma senoidal, suporta T > max_len dinamicamente)
   → nn.TransformerEncoder (num_layers=2, nhead=4, dim_feedforward=128, dropout=0.1)
   → mean-pooling MASCARADO sobre T (ignora posições de padding via padding_mask)
   → Linear(emb_dim, 128) → ReLU → Dropout → Linear(128, 1)   # 1 logit (BCEWithLogitsLoss)
```
- `padding_mask`: sequências de conversas têm números diferentes de turnos; o
  `collate_fn` faz *padding* com zeros e gera uma máscara booleana para que o
  Transformer e o *pooling* ignorem as posições preenchidas.
- Sequências muito longas são truncadas nos **últimos 100 turnos** (`s[-100:]`) em
  `_make_single_dataloader`.

## 3. Fluxo de treino (`train_transformer(embedding, ...)`)

```
load_sequence_embeddings("train", embedding, with_augmented)
        │ (opcional: sub-amostra pct% das conversas, para curva de aprendizado)
        ├── split 3-way: treino / val (interno, usado p/ early stopping) / teste interno
        │     (_make_dataloaders: test_size=0.2, val_size=0.1)
        │
        ├── loop de épocas (Adam, BCEWithLogitsLoss):
        │     treina no dl_train, mede train/val loss+f1+acc a cada época
        │     guarda o melhor state_dict por val_f1
        │
        ├── restaura melhor checkpoint
        │
        ├── avalia no TESTE INTERNO (dl_test)                    → dataset_name="test_internal"
        └── avalia na VALIDAÇÃO EXTERNA (dataset_validation)      → dataset_name="validation"
```
- Único runner (além de `unsupervised/`) que já avalia tanto no holdout interno quanto no
  dataset externo dentro da própria função de treino.
- `pct`: percentual do dataset de treino usado (para estudos de curva de aprendizado);
  `pct=None`/`100` usa tudo.

### Ensemble

- `load_all_transformer_results(embeddings=["voyage","bge","openai"], dataset_name=...)`
  — carrega (ou recalcula, se faltar cache de predições) os 3 modelos já treinados.
- `evaluate_ensemble_from_results(results_list, voting="majority"|"soft")` — combina as
  predições dos 3 modelos:
  - `"majority"`: pelo menos 2 dos 3 modelos votam Scam.
  - `"soft"`: média das probabilidades ≥ 0.5.
- `plot_ensemble_diagnostics(results_list, ensemble_result)` — 3 diagnósticos: F1 por
  modelo vs ensemble, quantas vezes o ensemble "salvou" um erro individual, e mapa de
  concordância de acertos/erros entre os 3 modelos.

## 4. Cache (`ModelCache`)

```
experiment_results/transformer/{embedding}__pct{none|N}/
    model.pth
    metrics.json          # {"test": {...}, "val": {...}}
    history.csv            # loss/f1/acc por época (treino e validação interna)
    predictions_test_internal.npz   # y_true, y_pred, y_prob (via save_predictions)
    predictions_validation.npz
```
- Diferente de `classical/`/`fcnn/`, o transformer **salva as predições brutas** em
  `.npz` (`ModelCache.save_predictions`/`load_predictions`), permitindo recompor o
  ensemble sem reprocessar os dados toda vez.
- `train_transformer()` **sempre sobrescreve** o cache (comentário no código: "Salvar
  (Sobrescreve no treino mais recente)") — não há checagem de `force_retrain` antes do
  treino em si, só antes de iniciar (`ModelCache.transformer_exists`) para decidir se
  carrega via `load_transformer_result()` em vez de re-treinar.
- `load_transformer_result()` é resiliente: se não achar `history.csv` no cache novo,
  tenta um *fallback* de histórico legado (`_try_load_legacy_history`); se não achar
  predições salvas, recalcula e grava.

## 5. Como rodar

```python
from machine_learning import (
    train_transformer, load_all_transformer_results,
    evaluate_ensemble_from_results, plot_ensemble_diagnostics,
)

# Treina os 3 embeddings do ensemble
for emb in ["voyage", "bge", "openai"]:
    res = train_transformer(embedding=emb, num_epochs=50, lr=1e-5)
    res.print_metrics()

# Monta o ensemble a partir do cache
results = load_all_transformer_results(dataset_name="validation")
ensemble = evaluate_ensemble_from_results(results, dataset_name="validation", voting="majority")
plot_ensemble_diagnostics(results, ensemble)
```

## 6. Métricas

`compute_metrics()` padrão, calculado separadamente para `test_internal` (holdout de
teste) e `validation` (dataset externo), tanto por modelo individual quanto para o
ensemble combinado.
