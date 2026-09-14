# 🛡️ Horus — Detecção de Golpes (Scam) em Conversas via Machine Learning

Projeto de pesquisa que detecta tentativas de golpe (*scam*) em conversas de texto
(ex.: WhatsApp) usando **embeddings de linguagem natural + Machine Learning**. O
repositório reúne três partes:

1. **Biblioteca de ML** (`backend/machine_learning/`) — o núcleo do projeto: 4 abordagens
   de modelagem diferentes, todas comparáveis entre si, com cache de experimentos e
   notebooks de análise.
2. **API + extensão de navegador** (`backend/api/`, `horus_extension/`, `server.js`) —
   uma prova de conceito que roda um dos modelos treinados em produção, monitorando o
   WhatsApp Web.
3. **Datasets e resultados de experimentos** (`backend/datasets/`,
   `backend/experiment_results/`) — dados brutos/processados e todo o histórico de
   modelos/métricas já treinados.

Se você está chegando agora no projeto, **comece por aqui**, depois vá para
[`backend/machine_learning/README.md`](backend/machine_learning/README.md) (explica cada
arquivo da biblioteca) e, por fim, para o README da abordagem específica que te
interessa (tabela abaixo).

---

## 1. Mapa do repositório

```
HorusEmbbedingPlusML/
├── backend/
│   ├── config.py                 # chaves de API e config.EMBEDDERS (5 embeddings suportados)
│   ├── api/                       # API Flask que serve o modelo treinado (produção/demo)
│   ├── datasets/                  # dados brutos e processados (dataset_train / dataset_validation)
│   ├── machine_learning/          # 👉 biblioteca de ML — ver README dedicado
│   ├── notebooks/                 # notebooks Jupyter de treino/análise por abordagem
│   ├── experiment_results/        # cache de modelos/métricas/históricos treinados
│   └── slurm/                     # scripts para rodar treinos/tuning em cluster SLURM
├── horus_extension/               # extensão de navegador (Chrome/Edge) para WhatsApp Web
├── server.js / package.json       # proxy Node.js opcional entre a extensão e a API Flask
├── backup/                        # snapshot histórico de uma versão anterior do código
├── requirements.txt               # dependências Python (backend + ML)
└── slurm_logs/                    # saída (stdout/stderr) dos jobs SLURM já rodados
```

---

## 2. A biblioteca de Machine Learning (`backend/machine_learning/`)

O problema é sempre o mesmo — classificar uma conversa como `Ham` (0, legítima) ou
`Scam` (1, golpe) — mas resolvido por **4 abordagens independentes**, cada uma na sua
subpasta, todas com a mesma interface (`train_*()` → `ExperimentResult`) e o mesmo
sistema de cache (`ModelCache`, ver `backend/machine_learning/cache.py`):

| Abordagem | Tipo | Modelos | Pasta | Documentação |
|---|---|---|---|---|
| **Clássica supervisionada** | 1 embedding por conversa | SVM, KNN, Random Forest, Naive Bayes, XGBoost | `classical/` | [`classical/README.md`](backend/machine_learning/classical/README.md) |
| **Rede Neural FCNN** | Multi-embedding concatenado | MLP em PyTorch (`ScamClassifierFCNN`) | `fcnn/` | [`fcnn/README.md`](backend/machine_learning/fcnn/README.md) |
| **Transformer sequencial** | Sequência de mensagens por conversa | Transformer Encoder + Ensemble por votação (voyage/bge/openai) | `transformer/` | [`transformer/README.md`](backend/machine_learning/transformer/README.md) |
| **Detecção de anomalias** | Semi-supervisionado (treina só com Ham) | OCSVM, Isolation Forest, LOF, LUNAR, Deep SVDD, CVDD, DATE | `unsupervised/` | [`unsupervised/README.md`](backend/machine_learning/unsupervised/README.md) |

A visão completa de arquitetura (o que cada arquivo comum — `result.py`, `cache.py`,
`data.py`, `evaluation.py`, `thresholds.py` — faz) está em
[`backend/machine_learning/README.md`](backend/machine_learning/README.md).

### 2.1 Embeddings suportados

Definidos em `backend/config.py` (`EMBEDDERS`):

| Chave | Modelo | Origem |
|---|---|---|
| `voyage` | `voyage-3-large` | API Voyage AI |
| `openai` | `text-embedding-3-large` | API OpenAI |
| `bge` | `BAAI/bge-m3` | Local (`sentence-transformers`) |
| `e5` | `intfloat/multilingual-e5-large` | Local (`sentence-transformers`) |
| `minilm` | `all-MiniLM-L6-v2` | Local (`sentence-transformers`) |

### 2.2 Datasets

- `backend/datasets/dataset_train/` — usado para treinar e para a avaliação interna
  (holdout / cross-validation / curva de aprendizado); em `raw/` tem o texto original,
  em `processed/` tem os parquets já com embeddings pré-computados.
- `backend/datasets/dataset_validation/` — dataset externo, nunca usado no treino;
  simula o comportamento em produção ("vida real").
- `path_data` ∈ `{"all_data", "suspect_turns"}` — dois recortes: todas as conversas ou
  só os turnos sinalizados como suspeitos.

### 2.3 Notebooks

Em `backend/notebooks/`:
- `ML_models.ipynb` — abordagem clássica.
- `FCNN_test.ipynb` — rede neural FCNN.
- `unsupervised.ipynb` — os 7 modelos de detecção de anomalias, com estudo de
  dimensionalidade, grid search de hiperparâmetros e leaderboard comparativo final.
- `judge_decision.ipynb` — análise auxiliar.

### 2.4 Rodando experimentos em SLURM

`backend/slurm/` contém os scripts para rodar o benchmark completo de detecção de
anomalias em um cluster SLURM (tuning de hiperparâmetros + treino final, para todos os
modelos e embeddings, em paralelo). Ver a seção "Como rodar" de
[`unsupervised/README.md`](backend/machine_learning/unsupervised/README.md) para o
passo a passo completo (`submit_full_pipeline.sh`, `submit_all_models.sh`, etc.).

---

## 3. API + Extensão de navegador (demonstração / produção)

- **`backend/api/main.py`** — API Flask que recebe o texto de uma conversa, gera os
  embeddings (via `machine_learning/embedder.py`) e roda a predição (via
  `machine_learning/predict.py`, usando o modelo FCNN salvo em `fcnn_trained/`),
  retornando se a conversa é `Ham` ou `Scam`.
- **`horus_extension/`** — extensão de navegador (Manifest V3) que injeta um *content
  script* no WhatsApp Web (`content.js`), captura as mensagens e as envia para a API via
  `background.js`.
- **`server.js`** (Node.js/Express) — proxy CORS opcional entre a extensão e a API
  Flask (porta 3000); dependências em `package.json`.

Esses componentes são uma prova de conceito de como um dos modelos treinados poderia
rodar em produção — o foco de desenvolvimento e experimentação está na biblioteca de ML.

---

## 4. Como instalar

```bash
# Dependências Python (backend + ML)
pip install -r requirements.txt

# Dependências do proxy Node.js (opcional, só se for rodar a extensão)
npm install
```

Configure as chaves de API em `backend/config.py` (`OPENAI_API_KEY`, `VOYAGE_API_KEY`)
antes de gerar novos embeddings ou rodar a API.

---

## 5. Por onde continuar

1. [`backend/machine_learning/README.md`](backend/machine_learning/README.md) — visão
   geral da biblioteca e explicação arquivo por arquivo.
2. Escolha a abordagem que te interessa e leia o README dela (tabela da seção 2).
3. Abra o notebook correspondente em `backend/notebooks/` para ver os experimentos
   rodando de ponta a ponta.
