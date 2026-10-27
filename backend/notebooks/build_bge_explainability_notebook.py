"""Gera o notebook autônomo de visualização da explicabilidade BGE.

A seção analítica continua tendo uma única fonte no ``judge_decision.ipynb``.
Este gerador copia essa seção e acrescenta apenas a inicialização necessária para
que o notebook de explicabilidade possa ser executado isoladamente.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = HERE / "judge_decision.ipynb"
TARGET_NOTEBOOK = HERE / "bge_explainability.ipynb"
SECTION_TITLE = "## Explicabilidade hierárquica BGE"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


source = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
section_start = next(
    index
    for index, cell in enumerate(source["cells"])
    if SECTION_TITLE in "".join(cell.get("source", []))
)

# A primeira célula de código da seção original pressupõe imports e STUDY_ROOT
# do notebook principal. Ela é substituída pela inicialização autônoma abaixo.
analysis_cells = source["cells"][section_start + 2 :]
for cell in analysis_cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        source_text = "".join(cell.get("source", []))
        source_text = source_text.replace(
            "def plot_hierarchical_curve(metric, title, ylabel):\n"
            "    view = hierarchical_curves[(hierarchical_curves.metric.eq(metric))",
            "def plot_hierarchical_curve(metric, title, ylabel):\n"
            "    if hierarchical_curves.empty:\n"
            "        return\n"
            "    view = hierarchical_curves[(hierarchical_curves.metric.eq(metric))",
        )
        cell["source"] = source_text.splitlines(keepends=True)


cells = [
    markdown(
        """# Explicabilidade BGE — resultados globais e auditoria por conversa

Notebook dedicado **somente** à explicabilidade hierárquica do Transformer com embeddings BGE.

Ele não executa BGE nem o classificador: lê os artefatos consolidados produzidos pelo estudo e reproduz todas as análises já validadas:

- cobertura e denominadores `N_k`;
- comprehensiveness e sufficiency distance, com média, dispersão e IC95%;
- saturação por número e por proporção de mensagens;
- `k95` e massa acumulada de evidência;
- distribuições globais e por TP, TN, FP e FN;
- n-grams selecionados versus controles aleatórios;
- estabilidade entre seeds 42, 52 e 62;
- auditoria completa de conversas de validação;
- explorador de mensagens, trechos, fidelidade, controles e estabilidade.

As métricas globais usam `decision_supporting`; o explorador também mostra `absolute_influence`. Todos os valores globais são mantidos separados dos valores de uma conversa específica.
"""
    ),
    markdown(
        """## Leitura rápida das métricas

- **Comprehensiveness:** quanto o suporte à classe prevista cai ao remover as evidências escolhidas. Maior é melhor para uma explicação causal; valores assinados negativos indicam interação ou evidência contrária.
- **Sufficiency distance:** distância entre a saída original e a saída usando somente as evidências escolhidas. Menor é melhor e a versão absoluta nunca é negativa.
- **IC95%:** intervalo de confiança bootstrap para a média entre conversas; não descreve a variação individual, que aparece nos boxplots.
- **`N_k`:** conversas com pelo menos `k` mensagens. **`metric_N_k`:** conversas com pelo menos `k` evidências válidas para a política avaliada.
- **`k95`:** primeiro `k` que alcança 95% do maior comprehensiveness observado naquela conversa.
- **Massa de evidência:** soma dos impactos individuais. Ela é monotônica dentro de cada conversa, mas não substitui a reavaliação contrafactual conjunta.

Nas curvas por fração da conversa, cada ponto agrega uma faixa percentual distinta e pode ter outro conjunto de conversas. Por isso a curva proporcional não precisa ter o mesmo formato da curva por `k` absoluto.
"""
    ),
    code(
        """from pathlib import Path
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 120)
pd.set_option('display.max_colwidth', 180)

def find_repository_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'backend').is_dir() and (candidate / 'backend' / 'notebooks').is_dir():
            return candidate
    # Útil quando o kernel é iniciado em outro diretório, mas o caminho padrão existe.
    fallback = Path(r'C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML')
    if fallback.exists():
        return fallback
    raise FileNotFoundError('Não foi possível localizar a raiz do repositório HorusEmbbedingPlusML.')

REPO_ROOT = find_repository_root()
default_root = REPO_ROOT / 'backend' / 'experiment_results' / 'transformer' / 'studies' / 'hierarchical_explainability'
HIER_STUDY_ROOT = Path(os.environ.get('HORUS_BGE_EXPLAINABILITY_ROOT', default_root)).resolve()

def read_hierarchical_parquet(relative, required=False):
    path = HIER_STUDY_ROOT / relative
    if not path.exists():
        message = f'Artefato ausente: {path}'
        if required:
            raise FileNotFoundError(message)
        print(message)
        return pd.DataFrame()
    return pd.read_parquet(path)

def concat_available(frames):
    available = [frame for frame in frames if not frame.empty]
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()

manifest_path = HIER_STUDY_ROOT / 'manifest.json'
HIER_MANIFEST = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}

hierarchical_coverage = read_hierarchical_parquet('aggregates/coverage.parquet')
hierarchical_curves = read_hierarchical_parquet('aggregates/fidelity_curves.parquet')
hierarchical_metrics = read_hierarchical_parquet('aggregates/metric_distributions.parquet')
hierarchical_saturation_curves = read_hierarchical_parquet('aggregates/saturation_curves.parquet')
hierarchical_saturation_fraction_curves = read_hierarchical_parquet('aggregates/saturation_fraction_curves.parquet')
hierarchical_saturation_points = read_hierarchical_parquet('aggregates/saturation_points.parquet')
hierarchical_saturation_conversations = read_hierarchical_parquet('aggregates/saturation_conversations.parquet')
hierarchical_validation_saturation = read_hierarchical_parquet('validation/fidelity_saturation.parquet')
hierarchical_control_summary = read_hierarchical_parquet('aggregates/random_control_summary.parquet')
hierarchical_controls = concat_available([
    read_hierarchical_parquet(f'{split}/random_controls.parquet')
    for split in ('test_internal', 'validation')
])
hierarchical_stability = concat_available([
    read_hierarchical_parquet(f'{split}/stability.parquet')
    for split in ('test_internal', 'validation')
])
hierarchical_messages = concat_available([
    read_hierarchical_parquet(f'{split}/message_effects.parquet')
    for split in ('test_internal', 'validation')
])
hierarchical_spans = concat_available([
    read_hierarchical_parquet(f'{split}/span_effects.parquet')
    for split in ('test_internal', 'validation')
])

HIER_SPLIT_COLORS = {'test_internal': '#2563eb', 'validation': '#f97316'}
HIER_CATEGORY_COLORS = {'TP': '#16a34a', 'TN': '#2563eb', 'FP': '#dc2626', 'FN': '#9333ea'}

print(f'Repositório: {REPO_ROOT}')
print(f'Artefatos:   {HIER_STUDY_ROOT}')
print(f'Manifesto:   {"carregado" if HIER_MANIFEST else "ausente"}')
"""
    ),
    markdown("## Inventário e proveniência dos artefatos"),
    code(
        """artifact_inventory = pd.DataFrame([
    {'artefato': 'coverage', 'linhas': len(hierarchical_coverage), 'colunas': len(hierarchical_coverage.columns)},
    {'artefato': 'fidelity_curves', 'linhas': len(hierarchical_curves), 'colunas': len(hierarchical_curves.columns)},
    {'artefato': 'metric_distributions', 'linhas': len(hierarchical_metrics), 'colunas': len(hierarchical_metrics.columns)},
    {'artefato': 'saturation_curves', 'linhas': len(hierarchical_saturation_curves), 'colunas': len(hierarchical_saturation_curves.columns)},
    {'artefato': 'saturation_fraction_curves', 'linhas': len(hierarchical_saturation_fraction_curves), 'colunas': len(hierarchical_saturation_fraction_curves.columns)},
    {'artefato': 'saturation_points', 'linhas': len(hierarchical_saturation_points), 'colunas': len(hierarchical_saturation_points.columns)},
    {'artefato': 'message_effects', 'linhas': len(hierarchical_messages), 'colunas': len(hierarchical_messages.columns)},
    {'artefato': 'span_effects', 'linhas': len(hierarchical_spans), 'colunas': len(hierarchical_spans.columns)},
    {'artefato': 'random_controls', 'linhas': len(hierarchical_controls), 'colunas': len(hierarchical_controls.columns)},
    {'artefato': 'stability', 'linhas': len(hierarchical_stability), 'colunas': len(hierarchical_stability.columns)},
])
display(artifact_inventory)
if HIER_MANIFEST:
    display(pd.json_normalize(HIER_MANIFEST, sep='.').T.rename(columns={0: 'valor'}))
"""
    ),
    *analysis_cells,
    markdown(
        """## Checklist de interpretação

1. Leia `N`/cobertura junto com toda curva global; médias em `k` diferentes podem conter populações diferentes.
2. Use comprehensiveness e sufficiency em conjunto: uma explicação boa tende a remover suporte quando retirada e a preservar a decisão quando isolada.
3. Use os boxplots para verificar heterogeneidade; a média e o IC95% não mostram casos individuais extremos.
4. Controles aleatórios avaliam especificidade, não veracidade factual do trecho.
5. Estabilidade entre seeds mede robustez do modelo, não causalidade no mundo real.
6. No explorador, `absolute_influence` é diagnóstico; `decision_supporting` é a política usada para justificar a classe prevista.
"""
    ),
]

for index, cell in enumerate(cells):
    identity = f"bge-explainability-{index}-{cell['cell_type']}".encode("utf-8")
    cell["id"] = hashlib.sha1(identity).hexdigest()[:12]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET_NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Notebook criado: {TARGET_NOTEBOOK}")
