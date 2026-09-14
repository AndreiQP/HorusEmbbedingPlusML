# `machine_learning/unsupervised/` — Benchmark de Detecção de Anomalias (Scam)

Este módulo reúne todos os modelos **não-supervisionados** usados para detectar
scams: o modelo aprende apenas com conversas **Ham** (legítimas, label=0) e,
em produção, qualquer coisa que não se pareça com "normal" é sinalizada como
possível **Scam** (label=1). Nenhum modelo aqui vê exemplos de Scam durante o
treino — eles só aparecem na avaliação.

## 0. Objetivo

O objetivo deste benchmark é encontrar a **melhor combinação modelo + embedding**
para detectar scams em conversas de aplicativos de mensagens. Por isso, cada
modelo é (em geral) testado com os **5 embeddings** disponíveis no projeto
(`config.EMBEDDERS`): `voyage`, `openai`, `e5`, `bge`, `minilm` — exceto CVDD e
DATE, que por decisão do projeto usam exclusivamente `bge` (ver seção 3). As
funções `train_all_anomaly()` e `compare_dimensions_anomaly()` (seção 5.1)
existem justamente para automatizar essa comparação entre embeddings/dimensões
para um mesmo modelo.

Existem dois grupos de modelos:

| Grupo | Modelos | Entrada | Arquivo |
|---|---|---|---|
| Embeddings pré-computados + PCA/UMAP | OCSVM, Isolation Forest, LOF, LUNAR, Deep SVDD | vetor de embedding (voyage/openai/e5/bge/minilm — os 5 embeddings do projeto) | `runner.py`, `svdd_torch.py` |
| Texto bruto + BGE | CVDD, DATE | texto da conversa (string) | `text_runner.py`, `cvdd.py`, `date_model.py` |

`FATE` foi cogitado no planejamento inicial mas fica **adiado** (não implementado).

---

## 1. Divisão dos dados (train / val / test / validation externa)

O projeto tem dois datasets físicos:

- `backend/datasets/dataset_train/` — usado para treinar E para a avaliação
  interna. É dividido de forma estratificada em **3 partes**:
  1. `test_size` (padrão 20%) é separado primeiro → **test** (holdout final).
  2. Do restante, `val_size / (1 - test_size)` (padrão 10% do total) vira
     **val** (usado para tuning/monitoramento).
  3. O que sobra é o **train** de fato.
  - **Somente as amostras Ham do split de treino** são usadas para ajustar
    scaler/PCA/UMAP/modelo (treino "one-class").
  - `train`, `val` e `test` contêm Ham + Scam, para medir o quão bem o modelo
    separa as duas classes mesmo nunca tendo visto Scam no fit.
- `backend/datasets/dataset_validation/` — dataset **externo**, nunca usado no
  treino ou tuning. Simula "vida real" e aparece nas métricas como
  `validation_external`.

Essa mesma convenção (3-way split + validação externa) é usada pelos runners
`classical`, `fcnn`, `transformer` e agora também por todo o `unsupervised`.

---

## 2. Modelos sobre embeddings (`runner.py`)

Pipeline comum para todos os modelos deste grupo (`train_anomaly()`):

```
X_train (Ham) -> StandardScaler -> PCA ou UMAP (pca_dim) -> modelo.fit(X_ham_reduzido)
```

O mesmo `scaler` + `reducer` (ajustados só no Ham de treino) são reaplicados a
`train`, `val`, `test` e `validation_external` antes de cada avaliação.

### 2.1 One-Class SVM (`ocsvm`)
Fronteira de decisão via kernel (RBF/linear/poly) que engloba a região "normal"
no espaço reduzido; `nu` controla a fração esperada de outliers no treino.
- **Hiperparâmetros**: `kernel` (linear/rbf/poly), `gamma`, `nu`.
- Implementação: `sklearn.svm.OneClassSVM` (wrapper direto).

### 2.2 Isolation Forest (`iforest`)
Constrói árvores de isolamento aleatórias; anomalias são isoladas com poucos
cortes (caminho médio menor).
- **Hiperparâmetros**: `n_estimators`, `contamination`.
- Implementação: `sklearn.ensemble.IsolationForest`.

### 2.3 Local Outlier Factor (`lof`)
Mede a densidade local de cada ponto relativa à densidade dos seus
`n_neighbors` vizinhos; pontos em regiões muito menos densas que suas
vizinhanças são outliers. Rodado em modo *novelty* (fit no Ham, predict em
dados novos).
- **Hiperparâmetros**: `n_neighbors`, `metric`, `contamination`.
- Implementação: `pyod.models.lof.LOF`.

### 2.4 LUNAR (`lunar`)
Unifica métodos clássicos de detecção local de outliers (kNN, LOF, etc.) usando
uma rede neural (GNN-like) que aprende a combinar as distâncias aos vizinhos
mais próximos de forma otimizada, em vez de uma fórmula fixa.
- **Hiperparâmetros**: `n_neighbors`, `lunar_model_type` (`WEIGHT`/`SCORE`),
  `epsilon`, `lr`, `weight_decay`, `n_epochs`.
- Implementação: `pyod.models.lunar.LUNAR`.

### 2.5 Deep SVDD (`svdd`)
Baseado em Ruff et al. (2018) — *"Deep One-Class Classification"*. Uma rede
encoder (MLP) aprende a mapear os embeddings Ham para uma **hiperesfera**
compacta no espaço latente; o raio `R` é otimizado junto ao encoder (variante
*soft-boundary*), permitindo que uma fração `nu` das amostras de treino fique
fora da esfera — isso evita o colapso trivial (todos os pesos → 0).
- **Por que não usamos o Deep SVDD do PyOD?** A implementação do PyOD depende
  de TensorFlow, o que conflita com o stack do projeto (PyTorch). Por isso
  implementamos uma versão própria em PyTorch (`svdd_torch.py`).
- **Hiperparâmetros**: `latent_dim`, `hidden_dims`, `nu`, `lr`, `weight_decay`,
  `n_epochs`, `batch_size`, `warmup_epochs`.
- Implementação: `svdd_torch.DeepSVDD` (classe própria, com API compatível
  com sklearn: `fit`/`predict`/`decision_function`).

### 2.6 Convenções de score/predição (`_anomaly_predict`, `_anomaly_score`)
Os dois grupos de bibliotecas usam convenções opostas, unificadas internamente:
- **sklearn** (OCSVM/IForest/SVDD): `predict` → `+1` normal / `-1` anomalia;
  `decision_function` → **maior = mais normal**.
- **PyOD** (LOF/LUNAR): `predict` → `0` normal / `1` anomalia;
  `decision_function` → **maior = mais anômalo**.

Essas funções convertem tudo para o padrão do projeto: `0=Ham`, `1=Scam`, e um
score contínuo onde **maior = mais provável ser Scam** (usado para AUROC/AUPRC).

---

## 3. Modelos sobre texto bruto + BGE (`text_runner.py`)

Diferente da seção anterior, CVDD e DATE **não usam embeddings pré-computados
nem PCA/UMAP** — eles processam o texto da conversa diretamente com o modelo
`BAAI/bge-m3` (via `transformers`), conforme decidido no planejamento (BGE foi
escolhido como único embedding para esses dois modelos, dado o custo
computacional de fine-tuning/atenção sobre tokens).

O texto bruto vem de `backend/datasets/{dataset_train,dataset_validation}/raw/all_data.csv`
(carregado por `text_data.py`, coluna `text`, filtrando `origin='external'` —
mesma regra de "sem augmentação" usada pelos demais modelos unsupervised).

### 3.1 CVDD — Context Vector Data Description (`cvdd.py`)
Baseado em Ruff et al. (2019) — *"Self-Attentive, Multi-Context One-Class
Classification for Unsupervised Anomaly Detection on Text"*.
- Cada token da conversa é representado pelo embedding contextual do BGE
  (**congelado**, sem fine-tuning).
- Um mecanismo de **self-attention com `n_contexts` vetores de contexto
  treináveis** aprende `K` "perspectivas"/tópicos distintos do texto normal:
  para cada contexto `k`, calcula uma atenção sobre os tokens e produz uma
  representação ponderada da frase.
- **Treino**: minimiza a distância cosseno entre cada representação ponderada
  e seu vetor de contexto, mais uma regularização de ortogonalidade
  (`lambda_ortho`) que impede os `K` contextos de colapsarem no mesmo vetor.
- **Score de anomalia**: distância cosseno até o contexto mais próximo — textos
  fora dos "tópicos normais" aprendidos ficam distantes de todos os contextos.
- **Hiperparâmetros**: `n_contexts`, `attention_hidden`, `lambda_ortho`, `lr`,
  `n_epochs`, `batch_size`, `max_tokens`, `nu` (define o threshold de corte).

### 3.2 DATE — Detecting Anomalies via self-supervision of Transformers (`date_model.py`)
Baseado em Manolache et al. (2021) — *"DATE: Detecting Anomalies in Text via
Self-Supervision of Transformers"*.
- Tarefa **pretexto auto-supervisionada**: aplica uma transformação aleatória
  ao texto (identidade, embaralhar palavras, remover palavras, duplicar
  palavra, reverter turnos) e treina um classificador (BGE + cabeça linear,
  com as últimas `unfreeze_last_n_layers` camadas do BGE fine-tunáveis) para
  prever **qual** transformação foi aplicada.
- Depois de treinado, ajusta uma **Gaussiana por classe pretexto** (média +
  covariância compartilhada via Ledoit-Wolf) sobre as representações `[CLS]`
  dos textos Ham de treino.
- **Score de anomalia**: distância de **Mahalanobis mínima** entre a
  representação do texto original (sem transformação) e as `K` Gaussianas —
  textos fora da distribuição de treino tendem a ficar longe de todas as
  classes aprendidas.
- **Simplificações em relação ao paper original** (documentadas em código):
  transformações pretexto offline (sem tradução automática/paráfrase via API),
  e uso do Mahalanobis apenas com covariância compartilhada (não por classe).
- **Hiperparâmetros**: `n_pretext_transforms` (K, máx. 5 transformações
  disponíveis), `unfreeze_last_n_layers`, `lr`, `n_epochs`, `batch_size`,
  `max_tokens`.

---

## 4. Métricas avaliadas

Todas as avaliações passam por `compute_metrics()` (`machine_learning/evaluation.py`),
que agora inclui, para **todos** os modelos deste módulo:

- `accuracy`, `f1_macro`, `f1_scam`, `precision_macro`, `recall_macro`,
  `precision_scam`, `recall_scam`, `TP`/`TN`/`FP`/`FN`
- **`roc_auc`** — AUROC calculado a partir do score contínuo de anomalia
  (`y_score`), independente do threshold escolhido.
- **`pr_auc`** — AUPRC (average precision), mais informativo que AUROC quando
  a classe Scam é minoritária (nosso caso).

Cada `train_anomaly()`/`train_cvdd()`/`train_date()` retorna um `df_metrics`
com uma linha por partição: `train`, `val`, `test`, `validation_external`.

---

## 5. Como rodar

### 5.0 Fluxo recomendado (tuning → treino final)

**Nenhuma das funções abaixo é um pré-requisito obrigatório da outra** — são
ferramentas independentes que você combina conforme o objetivo:

1. **(Opcional, recomendado) Tuning de hiperparâmetros** — `grid_search_anomaly()`.
   Treina cada combinação do `param_grid` só com Ham (split de treino) e
   **seleciona o melhor combo pela validação INTERNA** (`internal_val_*`,
   holdout de `dataset_train` — nunca pela validação externa, para não
   "vazar" o conjunto de vida real no tuning). `test_*` e `external_val_*`
   aparecem no resultado só como conferência, não são usados para escolher.
2. **Treino final + avaliação completa** — `train_anomaly()` / `train_cvdd()` /
   `train_date()`, já usando os hiperparâmetros escolhidos no passo 1 (via
   `**model_kwargs`). Essa é a função que de fato salva o modelo em
   `experiment_results/unsupervised/` e reporta `train` / `val` / `test` /
   `validation_external` (ver seção 1).

Se você já sabe os hiperparâmetros (ou está satisfeito com os padrões
documentados na seção 2/3), pode pular direto para o passo 2 — o grid search
é uma etapa de tuning independente, não uma dependência de execução.

### 5.1 Em notebook / Python
```python
from machine_learning import train_anomaly, train_cvdd, train_date
from machine_learning.unsupervised.runner import grid_search_anomaly

# 1) (opcional) tuning — escolhe o melhor combo pela validação interna
gs = grid_search_anomaly(embedding="bge", model_type="lof", pca_dim=300)
gs.print_metrics()  # melhor combo, com internal_val_f1 usado na seleção

# 2) treino final com os hiperparâmetros escolhidos
result = train_anomaly(embedding="bge", model_type="lof", pca_dim=300,
                        n_neighbors=20, contamination=0.1)  # valores vindos do passo 1
result.print_metrics()
result.plot_roc_curve()
result.plot_precision_recall_curve()

# Deep SVDD com hiperparâmetros customizados (sem passar por grid search)
result = train_anomaly(embedding="voyage", model_type="svdd", pca_dim=200,
                        nu=0.1, latent_dim=32, n_epochs=150)

# Modelos de texto bruto (sempre BGE)
result = train_cvdd(n_contexts=10, n_epochs=20)
result = train_date(n_pretext_transforms=5, n_epochs=5)
```

Outras funções disponíveis em `runner.py`:
- `learning_curve_anomaly(...)` — curva de aprendizado por % de dados de treino
  (independente do grid search; usa `test` interno + `validation_external`).
- `grid_search_anomaly(embedding, model_type, param_grid=None)` — tuning de
  hiperparâmetros (ver 5.0); sem `param_grid`, usa uma grade padrão por modelo
  (`_default_param_grid`).
- `train_all_anomaly(model_type, embeddings=None, ...)` — treina o mesmo
  modelo (com hiperparâmetros fixos) em todos os embedders, para comparar
  qual embedding funciona melhor.
- `compare_dimensions_anomaly(model_type, dimensions=None, ...)` — estudo de
  dimensionalidade do PCA/UMAP.

### 5.2 Via terminal (CLI única, pensada para SLURM)
Script: `machine_learning/unsupervised/run_experiment.py`.

```bash
cd backend

# Treino simples (embeddings + PCA/UMAP)
python -m machine_learning.unsupervised.run_experiment \
    --mode train --model lof --embedding bge --dim 300 --reducer pca

# Deep SVDD com hiperparâmetros custom (--param chave=valor, repetível)
python -m machine_learning.unsupervised.run_experiment \
    --mode train --model svdd --embedding voyage --dim 200 \
    --param nu=0.1 --param latent_dim=32

# Curva de aprendizado / grid search / todos os embeddings / estudo de dimensão
python -m machine_learning.unsupervised.run_experiment --mode learning_curve --model iforest --embedding e5 --dim 500
python -m machine_learning.unsupervised.run_experiment --mode grid_search --model lunar --embedding bge --dim 300
python -m machine_learning.unsupervised.run_experiment --mode train_all --model lof --dim 300
python -m machine_learning.unsupervised.run_experiment --mode compare_dimensions --model ocsvm --embedding bge

# Modelos de texto bruto (BGE fixo; --embedding/--dim são ignorados)
python -m machine_learning.unsupervised.run_experiment --mode train --model cvdd --param n_contexts=10
python -m machine_learning.unsupervised.run_experiment --mode train --model date --param n_pretext_transforms=5
```

Todos os modos salvam automaticamente em `experiment_results/unsupervised/`
(modelo, config, métricas, histórico) via `ModelCache` — reexecuções com a
mesma config usam o cache, a menos que `--force-retrain` seja passado.

### 5.3 Via SLURM
Templates em `backend/slurm/`:

> ⚠️ **"grid" aqui ≠ grid search de hiperparâmetros.** `make_grid.py`/`array_grid.sh`
> geram uma varredura de **`--mode train` por embedding x dimensão** (para
> comparar modelo+embedding, seção 0), sempre com os hiperparâmetros
> **padrão** de `_build_anomaly_model`. Eles **não** chamam `--mode grid_search`
> (tuning de hiperparâmetros, seção 5.0). Se quiser rodar tuning via SLURM,
> use `run_unsupervised.sbatch` com `<mode>=grid_search` (exemplo abaixo).

- `run_unsupervised.sbatch` — roda **um** experimento por job (qualquer `--mode`,
  incluindo `grid_search`):
  ```bash
  sbatch backend/slurm/run_unsupervised.sbatch train lof bge 300
  sbatch backend/slurm/run_unsupervised.sbatch train svdd voyage 200 nu=0.1 latent_dim=32
  sbatch backend/slurm/run_unsupervised.sbatch train cvdd bge 0
  sbatch backend/slurm/run_unsupervised.sbatch grid_search lof bge 300   # tuning de hiperparâmetros
  ```
- `submit_all_models.sh` — submete **um job (array) por modelo implementado**,
  todos em paralelo (cada um com seu próprio `--job-name`, grade e logs, sem
  bloquear os demais), rodando `--mode train` com hiperparâmetros padrão em
  todos os embeddings x dimensões:
  ```bash
  sbatch backend/slurm/submit_all_models.sh                # todos os 7 modelos
  sbatch backend/slurm/submit_all_models.sh lof svdd cvdd  # só os informados
  ```
  Internamente, ele chama `make_grid.py` (gera uma grade por modelo em
  `backend/slurm/grids/<model>.txt`: embedding x dimensão para os modelos de
  embedding, uma única linha para cvdd/date) e dispara `array_grid.sh` uma vez
  por modelo com `--array=0-N` cobrindo sua grade.
- `submit_full_pipeline.sh` — **pipeline completo com tuning automático**, em
  2 etapas encadeadas por modelo (via `--dependency=afterok`, sem precisar
  acompanhar manualmente):
  1. `--mode grid_search` em cada embedding (dim fixo em `TUNING_DIM=300`) —
      sempre executado com `--force-retrain`, escolhe os melhores hiperparâmetros
      pela validação **interna** (seção 5.0), sem reutilizar grid antigo.
  2. Só depois que TODOS os embeddings do modelo terminarem o tuning com
     sucesso, dispara `--mode train` em cada embedding x dimensão, já com os
     hiperparâmetros vencedores daquele embedding aplicados
      (`collect_best_and_make_train_grid.py` monta essa grade lendo o resultado
      recém-gerado; cada linha final também recebe `--force-retrain`).
  ```bash
  sbatch backend/slurm/submit_full_pipeline.sh                # todos os modelos
  sbatch backend/slurm/submit_full_pipeline.sh lof svdd cvdd  # só os informados
  ```
  CVDD/DATE não têm tuning automatizado (grid_search não suportado para
  modelos de texto no CLI) — para eles o pipeline dispara o treino direto com
  `--force-retrain` e hiperparâmetros padrão.

  **Diferença resumida:** `submit_all_models.sh` = treino rápido com
  hiperparâmetros padrão (bom para comparar modelo+embedding); `submit_full_pipeline.sh`
  = tuning + treino final com os melhores hiperparâmetros por embedding (mais
  lento, mas metodologicamente mais correto — é o fluxo recomendado antes de
  reportar resultados finais).
Ajuste `#SBATCH` (partição, `--gres=gpu`, tempo, memória) conforme o cluster
antes de submeter — os cabeçalhos nos scripts têm comentários indicando onde.

---

## 6. Onde os resultados ficam salvos (para comparar depois)

Todo experimento (`train`, `grid_search`, `learning_curve`, `train_all`,
`compare_dimensions`, CVDD/DATE) é persistido automaticamente por `ModelCache`
em `backend/experiment_results/unsupervised/<run_id>/`, sem nenhuma ação extra
sua — inclusive quando disparado via SLURM (`array_grid.sh`/`run_unsupervised.sbatch`).
Cada pasta contém:
- `model.pkl` + `reducer.pkl` (ou `model.pt` para CVDD/DATE/SVDD) — o modelo treinado.
- `config.json` — os parâmetros exatos que geraram aquele experimento (embedding,
  model_type, dim, reducer, test_size, val_size, hiperparâmetros).
- `metrics.json` — métricas de `train`/`val`/`test`/`validation_external`
  (accuracy, f1, recall_scam, roc_auc, pr_auc, TP/TN/FP/FN).
- `history.csv` — histórico completo (todas as combinações do grid search,
  curva de aprendizado por fração, ou estudo de dimensionalidade, conforme o modo).
- PNGs de plots (matriz de confusão, PR/ROC, etc.), quando aplicável.

Como o `run_id` é determinístico (hash da config), reexecutar o mesmo
experimento faz cache-hit em vez de re-treinar — isso vale tanto localmente
quanto entre jobs SLURM diferentes.

**Para comparar modelos/embeddings depois**, use (em notebook ou script):
```python
from machine_learning import train_all_anomaly, compare_dimensions_anomaly

df = train_all_anomaly(model_type="lof", pca_dim=300)     # 1 linha por embedding
df = compare_dimensions_anomaly(model_type="svdd")        # 1 linha por embedding x dim
```
Essas funções chamam `train_anomaly()` internamente (cache-hit se já rodado
via SLURM) e retornam um `DataFrame` comparativo pronto para ordenar por
`val_roc_auc`/`val_pr_auc`/`val_f1_macro`. Para juntar tudo (todos os modelos),
basta chamar `train_all_anomaly()` uma vez por `model_type` e concatenar os
DataFrames — não há (ainda) um agregador único para os 7 modelos de uma vez.

---

## 7. Dependências

- `pyod` — LOF e LUNAR.
- `transformers` — backbone BGE (`BAAI/bge-m3`) para CVDD/DATE (também usado
  via `sentence-transformers`, já presente no projeto).
- `umap-learn` — opcional, apenas se `reducer="umap"`.
- PyTorch (já usado no projeto) — Deep SVDD, CVDD, DATE.

Instale via `pip install -r requirements.txt` no ambiente usado pelos jobs SLURM.
