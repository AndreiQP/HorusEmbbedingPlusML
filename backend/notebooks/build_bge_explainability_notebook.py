"""Gera o notebook autônomo da explicabilidade BGE canônica e contextual."""

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

# A comparação contextual substitui as visualizações legadas neste notebook.
# O método antigo continua presente como ``canonical_message`` e passa pelo
# mesmo protocolo das novas representações.
analysis_cells = []


CONTEXT_CELLS = [
    markdown(
        """## Comparação uniforme das sete representações

Todos os gráficos abaixo aplicam o mesmo protocolo a `canonical_message`,
`speaker_block_4` e às janelas cronológicas 4–8. `frozen_probe` mede apenas
compatibilidade com o checkpoint antigo; `retrained` mede modelos treinados
para a nova unidade contextual.
"""
    ),
    code(
        """CONTEXT_ROOT = HIER_STUDY_ROOT / 'context_blocks'
CONTEXT_AGG = CONTEXT_ROOT / 'aggregates'

def read_context(name):
    path = CONTEXT_AGG / f'{name}.parquet'
    if not path.exists():
        print(f'Artefato contextual ausente: {path}')
        return pd.DataFrame()
    return pd.read_parquet(path)

ctx_conversations = read_context('conversation_metrics')
ctx_units = read_context('unit_effects')
ctx_spans = read_context('span_effects')
ctx_fidelity = read_context('fidelity')
ctx_controls = read_context('random_controls')
ctx_stability = read_context('stability')
ctx_summary = read_context('metric_summary')
ctx_categories = read_context('category_summary')
ctx_points = read_context('quantity_points')
ctx_classification = read_context('classification_metrics')
ctx_agreement = read_context('canonical_agreement')
ctx_paired = read_context('paired_common_fidelity')

CTX_REPRESENTATIONS = [
    'canonical_message', 'speaker_block_4',
    'chronological_window_4', 'chronological_window_5',
    'chronological_window_6', 'chronological_window_7',
    'chronological_window_8',
]
CTX_COLORS = {
    'canonical_message': '#111827', 'speaker_block_4': '#dc2626',
    'chronological_window_4': '#2563eb', 'chronological_window_5': '#0891b2',
    'chronological_window_6': '#16a34a', 'chronological_window_7': '#ca8a04',
    'chronological_window_8': '#7c3aed',
}
CTX_MODE_STYLES = {'frozen_probe': '--', 'retrained': '-'}
CTX_MODEL_MODE = 'retrained'
CTX_POPULATION = 'paired_common'
CTX_SEED = 42
CTX_RANKING_POLICY = 'decision_supporting'
CTX_TOP_K_BOXPLOT = 6

display(pd.DataFrame({
    'configuração': ['modo', 'população', 'seed', 'ranking'],
    'valor': [CTX_MODEL_MODE, CTX_POPULATION, CTX_SEED, CTX_RANKING_POLICY],
}))
"""
    ),
    markdown("### 1. Auditoria, cobertura, blocos parciais e truncamento"),
    code(
        """cache_rows = []
for representation in CTX_REPRESENTATIONS:
    for source_split in ('train', 'validation'):
        path = CONTEXT_ROOT / 'cache' / representation / f'{source_split}_blocks.parquet'
        if not path.exists():
            continue
        columns = [
            'sample_id', 'is_partial', 'was_truncated', 'token_count',
            'max_token_length', 'overlap_fraction_with_previous',
        ]
        view = pd.read_parquet(path, columns=columns)
        cache_rows.append({
            'representation': representation,
            'source_split': source_split,
            'conversations': view.sample_id.nunique(),
            'units': len(view),
            'partial_rate': view.is_partial.mean(),
            'truncation_rate': view.was_truncated.mean(),
            'tokens_median': view.token_count.median(),
            'tokens_max': view.token_count.max(),
            'encoder_max_tokens': view.max_token_length.max(),
            'overlap_medio_anterior': view.overlap_fraction_with_previous.mean(),
        })
ctx_cache_audit = pd.DataFrame(cache_rows)
if not ctx_cache_audit.empty:
    display(ctx_cache_audit.style.format({
        'partial_rate': '{:.2%}', 'truncation_rate': '{:.2%}',
        'tokens_median': '{:.0f}', 'tokens_max': '{:.0f}',
        'overlap_medio_anterior': '{:.2%}',
    }))
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    sns.barplot(data=ctx_cache_audit, x='representation', y='units', hue='source_split', ax=axes[0])
    sns.barplot(data=ctx_cache_audit, x='representation', y='truncation_rate', hue='source_split', ax=axes[1])
    axes[0].set_title('Unidades geradas por representação'); axes[1].set_title('Taxa de truncamento BGE')
    for ax in axes: ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()
"""
    ),
    markdown("### 2. Métricas preditivas e matrizes TP/TN/FP/FN"),
    code(
        """if not ctx_classification.empty:
    all_primary = ctx_classification[ctx_classification.seed.eq(CTX_SEED)]
    primary = all_primary[all_primary.model_mode.eq(CTX_MODEL_MODE)]
    display(primary.sort_values(['model_mode', 'split', 'representation']))
    metrics = ['f1_macro', 'pr_auc', 'recall_scam', 'precision_scam']
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for ax, metric in zip(axes.flat, metrics):
        sns.barplot(
            data=primary, x='representation', y=metric,
            hue='split', palette=HIER_SPLIT_COLORS, ax=ax,
        )
        ax.set_title(f'{metric} | {CTX_MODEL_MODE} | seed {CTX_SEED}')
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()

    confusion = all_primary.melt(
        id_vars=['split', 'representation', 'model_mode'],
        value_vars=[name for name in ('TP', 'TN', 'FP', 'FN') if name in all_primary],
        var_name='categoria', value_name='quantidade',
    )
    g = sns.catplot(
        data=confusion, x='representation', y='quantidade', hue='categoria',
        col='split', row='model_mode', kind='bar', height=4, aspect=1.7,
        palette=HIER_CATEGORY_COLORS,
    )
    g.set_xticklabels(rotation=30); g.fig.suptitle('Matrizes de decisão por representação', y=1.02)
"""
    ),
    markdown("### 3. Concordância dos probes congelados com o canônico"),
    code(
        """if not ctx_agreement.empty:
    frozen = ctx_agreement[
        ctx_agreement.model_mode.eq('frozen_probe') & ctx_agreement.seed.eq(CTX_SEED)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    sns.barplot(data=frozen, x='representation', y='prediction_agreement', hue='split', palette=HIER_SPLIT_COLORS, ax=axes[0])
    sns.barplot(data=frozen, x='representation', y='mean_abs_logit_shift', hue='split', palette=HIER_SPLIT_COLORS, ax=axes[1])
    axes[0].set_ylim(0, 1); axes[0].set_title('Concordância da classe com canonical_message')
    axes[1].set_title('Mudança absoluta média do logit')
    for ax in axes: ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()
"""
    ),
    markdown("### 4–5. Comprehensiveness e sufficiency por Top-k"),
    code(
        """def plot_context_summary(metric, title, ylabel):
    if ctx_summary.empty:
        return
    view = ctx_summary[
        ctx_summary.metric.eq(metric)
        & ctx_summary.population.eq(CTX_POPULATION)
        & ctx_summary.model_mode.eq(CTX_MODEL_MODE)
        & ctx_summary.seed.eq(CTX_SEED)
        & ctx_summary.ranking_policy.eq(CTX_RANKING_POLICY)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(19, 6), sharey=True)
    for ax, split in zip(axes, ('test_internal', 'validation')):
        for representation, group in view[view.split.eq(split)].groupby('representation'):
            group = group.sort_values('top_k')
            ax.plot(group.top_k, group['mean'], color=CTX_COLORS[representation], label=representation)
            ax.fill_between(group.top_k, group.ci95_low, group.ci95_high, color=CTX_COLORS[representation], alpha=.10)
        ax.set(title=f'{title} | {split}\\n{CTX_MODEL_MODE} | {CTX_POPULATION} | seed {CTX_SEED}', xlabel='Top-k unidades', ylabel=ylabel)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

plot_context_summary('comprehensiveness_logit_aligned', 'Comprehensiveness', 'Queda alinhada do logit')
plot_context_summary('sufficiency_distance_logit', 'Sufficiency distance', '|logit original − logit suficiente|')
"""
    ),
    markdown("### 6–8. Curvas por unidades, mensagens únicas e fração da conversa"),
    code(
        """def plot_context_xaxis(x, xlabel):
    if ctx_fidelity.empty:
        return
    view = ctx_fidelity[
        ctx_fidelity.model_mode.eq(CTX_MODEL_MODE)
        & ctx_fidelity.seed.eq(CTX_SEED)
        & ctx_fidelity.ranking_policy.eq(CTX_RANKING_POLICY)
    ].copy()
    if x == 'coverage_fraction':
        view['x_bin'] = (view[x] * 20).round() / 20
        x_plot = 'x_bin'
    else:
        x_plot = x
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey='row')
    for column, split in enumerate(('test_internal', 'validation')):
        data = view[view.split.eq(split)]
        split_n = data.sample_id.nunique()
        for representation, group in data.groupby('representation'):
            aggregate = group.groupby(x_plot, as_index=False).agg(
                comprehensiveness=('comprehensiveness_logit_aligned', 'mean'),
                sufficiency=('sufficiency_distance_logit', 'mean'),
                n=('sample_id', 'nunique'),
            ).sort_values(x_plot)
            axes[0, column].plot(aggregate[x_plot], aggregate.comprehensiveness, color=CTX_COLORS[representation], label=representation)
            axes[1, column].plot(aggregate[x_plot], aggregate.sufficiency, color=CTX_COLORS[representation], label=representation)
        axes[0, column].set_title(f'Comprehensiveness | {split} | {CTX_MODEL_MODE} | seed {CTX_SEED} | N={split_n}')
        axes[1, column].set_title(f'Sufficiency distance | {split} | {CTX_MODEL_MODE} | seed {CTX_SEED} | N={split_n}')
        for ax in axes[:, column]: ax.set_xlabel(xlabel); ax.grid(alpha=.25); ax.legend(fontsize=7)
    axes[0, 0].set_ylabel('Queda alinhada do logit'); axes[1, 0].set_ylabel('Distância no logit')
    plt.tight_layout(); plt.show()

plot_context_xaxis('top_k', 'Unidades acumuladas')
plot_context_xaxis('unique_messages_covered', 'Mensagens originais únicas')
plot_context_xaxis('coverage_fraction', 'Fração da conversa (faixas de 5%)')
"""
    ),
    markdown("### 9–11. Boxplots globais e por TP/TN/FP/FN"),
    code(
        """if not ctx_fidelity.empty:
    view = ctx_fidelity[
        ctx_fidelity.model_mode.eq(CTX_MODEL_MODE)
        & ctx_fidelity.seed.eq(CTX_SEED)
        & ctx_fidelity.ranking_policy.eq(CTX_RANKING_POLICY)
        & ctx_fidelity.top_k.eq(CTX_TOP_K_BOXPLOT)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey='row')
    for column, split in enumerate(('test_internal', 'validation')):
        data = view[view.split.eq(split)]
        sns.boxplot(data=data, x='representation', y='comprehensiveness_logit_aligned', palette=CTX_COLORS, showfliers=False, ax=axes[0, column])
        sns.boxplot(data=data, x='representation', y='sufficiency_distance_logit', palette=CTX_COLORS, showfliers=False, ax=axes[1, column])
        axes[0, column].set_title(f'Comprehensiveness | {split} | Top-{CTX_TOP_K_BOXPLOT}')
        axes[1, column].set_title(f'Sufficiency | {split} | Top-{CTX_TOP_K_BOXPLOT}')
        for ax in axes[:, column]: ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()

    g = sns.catplot(
        data=view, x='representation', y='comprehensiveness_logit_aligned',
        hue='prediction_category', col='split', kind='box', showfliers=False,
        palette=HIER_CATEGORY_COLORS, height=5, aspect=1.7,
    )
    g.set_xticklabels(rotation=30); g.fig.suptitle('Comprehensiveness por TP/TN/FP/FN', y=1.03)
"""
    ),
    markdown("### 12. Quantidade ideal: k95 e k_joint"),
    code(
        """if not ctx_points.empty:
    reached = ctx_points[
        ctx_points.status.eq('reached')
        & ctx_points.model_mode.eq(CTX_MODEL_MODE)
        & ctx_points.seed.eq(CTX_SEED)
        & ctx_points.ranking_policy.eq(CTX_RANKING_POLICY)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    for column, point_type in enumerate(('k95', 'k_joint')):
        data = reached[reached.point_type.eq(point_type)]
        sns.boxplot(data=data, x='representation', y='unique_messages', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[0, column])
        sns.boxplot(data=data, x='representation', y='fraction', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[1, column])
        axes[0, column].set_title(f'{point_type}: mensagens originais únicas')
        axes[1, column].set_title(f'{point_type}: fração da conversa')
        axes[1, column].yaxis.set_major_formatter(lambda value, _: f'{100*value:.0f}%')
        for ax in axes[:, column]: ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()
    display(reached.groupby(['split', 'representation', 'point_type']).agg(
        N=('sample_id', 'nunique'), units_mediana=('units', 'median'),
        mensagens_mediana=('unique_messages', 'median'), fracao_mediana=('fraction', 'median'),
    ))
"""
    ),
    markdown("### 13. Massa aditiva, comprehensiveness e redundancy gap"),
    code(
        """if not ctx_fidelity.empty:
    mass = ctx_fidelity[
        ctx_fidelity.model_mode.eq(CTX_MODEL_MODE)
        & ctx_fidelity.seed.eq(CTX_SEED)
        & ctx_fidelity.ranking_policy.eq(CTX_RANKING_POLICY)
    ]
    fig, axes = plt.subplots(3, 2, figsize=(20, 16), sharex='col')
    metric_specs = [
        ('massa', 'Massa aditiva'),
        ('conjunto', 'Comprehensiveness conjunto'),
        ('redundancia', 'Redundancy gap'),
    ]
    for column, split in enumerate(('test_internal', 'validation')):
        data = mass[mass.split.eq(split)]
        for representation, group in data.groupby('representation'):
            aggregate = group.groupby('unique_messages_covered', as_index=False).agg(
                massa=('cumulative_evidence_mass_logit', 'mean'),
                conjunto=('comprehensiveness_logit_aligned', 'mean'),
                redundancia=('redundancy_gap_logit', 'mean'),
            )
            for row, (metric, _) in enumerate(metric_specs):
                axes[row, column].plot(
                    aggregate.unique_messages_covered, aggregate[metric],
                    color=CTX_COLORS[representation], label=representation,
                )
        for row, (_, title) in enumerate(metric_specs):
            axes[row, column].set_title(
                f'{title} | {split} | {CTX_MODEL_MODE} | seed {CTX_SEED} | N={data.sample_id.nunique()}'
            )
            axes[row, column].set_xlabel('Mensagens originais únicas cobertas')
            axes[row, column].set_ylabel('Logit')
            axes[row, column].grid(alpha=.25); axes[row, column].legend(fontsize=7)
    plt.tight_layout(); plt.show()
"""
    ),
    markdown("### 14. Trechos versus controles aleatórios"),
    code(
        """if not ctx_controls.empty:
    controls = ctx_controls[
        ctx_controls.model_mode.eq(CTX_MODEL_MODE) & ctx_controls.seed.eq(CTX_SEED)
    ].copy()
    controls['specificity_ratio'] = controls.selected_abs_delta_logit / controls.control_abs_delta_logit.replace(0, np.nan)
    impact = pd.concat([
        controls[['split', 'representation', 'selected_abs_delta_logit']].rename(columns={'selected_abs_delta_logit': 'impact'}).assign(tipo='Selecionado'),
        controls[['split', 'representation', 'control_abs_delta_logit']].rename(columns={'control_abs_delta_logit': 'impact'}).assign(tipo='Controle'),
    ])
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    sns.boxplot(data=impact, x='representation', y='impact', hue='tipo', showfliers=False, ax=axes[0])
    sns.boxplot(data=controls, x='representation', y='specificity_ratio', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[1])
    axes[1].axhline(1, color='black', ls='--')
    for ax in axes: ax.tick_params(axis='x', rotation=30)
    axes[0].set_title('Impacto dos trechos versus controles'); axes[1].set_title('Razão de especificidade')
    plt.tight_layout(); plt.show()
"""
    ),
    markdown("### 15. Estabilidade entre seeds"),
    code(
        """if not ctx_stability.empty:
    view = ctx_stability[ctx_stability.model_mode.eq(CTX_MODEL_MODE)]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    stability_metrics = [
        ('top_k_jaccard', 'Jaccard Top-6'),
        ('direction_agreement', 'Concordância de direção'),
        ('rank_spearman', 'Spearman dos rankings'),
        ('rank_kendall', 'Kendall dos rankings'),
        ('impact_cv', 'Coeficiente de variação do impacto'),
    ]
    for ax, (metric, title) in zip(axes.flat, stability_metrics):
        sns.boxplot(
            data=view, x='representation', y=metric, hue='split',
            palette=HIER_SPLIT_COLORS, showfliers=False, ax=ax,
        )
        ax.set_title(title); ax.tick_params(axis='x', rotation=30)
    agreement = view.groupby(['split', 'representation'], as_index=False).prediction_agreement.mean()
    sns.barplot(data=agreement, x='representation', y='prediction_agreement', hue='split', palette=HIER_SPLIT_COLORS, ax=axes.flat[-1])
    axes.flat[-1].set_ylim(0, 1); axes.flat[-1].tick_params(axis='x', rotation=30)
    axes.flat[-1].set_title('Concordância da classe')
    plt.tight_layout(); plt.show()
"""
    ),
    markdown("### 16. Comparação pareada entre representações"),
    code(
        """if not ctx_paired.empty:
    paired_view = ctx_paired[
        ctx_paired.model_mode.eq(CTX_MODEL_MODE)
        & ctx_paired.seed.eq(CTX_SEED)
        & ctx_paired.ranking_policy.eq(CTX_RANKING_POLICY)
        & ctx_paired.top_k.eq(CTX_TOP_K_BOXPLOT)
    ]
    canonical = paired_view[paired_view.representation.eq('canonical_message')][
        ['split', 'sample_id', 'comprehensiveness_logit_aligned', 'sufficiency_distance_logit']
    ].rename(columns={
        'comprehensiveness_logit_aligned': 'canonical_comprehensiveness',
        'sufficiency_distance_logit': 'canonical_sufficiency',
    })
    comparison = paired_view.merge(canonical, on=['split', 'sample_id'])
    comparison['delta_comprehensiveness'] = comparison.comprehensiveness_logit_aligned - comparison.canonical_comprehensiveness
    comparison['delta_sufficiency'] = comparison.sufficiency_distance_logit - comparison.canonical_sufficiency
    fig, axes = plt.subplots(1, 2, figsize=(19, 6))
    sns.boxplot(data=comparison, x='representation', y='delta_comprehensiveness', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[0])
    sns.boxplot(data=comparison, x='representation', y='delta_sufficiency', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[1])
    for ax in axes: ax.axhline(0, color='black', ls='--'); ax.tick_params(axis='x', rotation=30)
    axes[0].set_title('Δ comprehensiveness versus canônico (maior é melhor)')
    axes[1].set_title('Δ sufficiency versus canônico (menor é melhor)')
    plt.tight_layout(); plt.show()
"""
    ),
    markdown("### 16.1. Resumo específico das janelas cronológicas 4–8"),
    code(
        """chronological = [f'chronological_window_{size}' for size in range(4, 9)]
if not ctx_classification.empty:
    predictive = ctx_classification[
        ctx_classification.representation.isin(chronological)
        & ctx_classification.seed.eq(CTX_SEED)
    ].copy()
    predictive['window_size'] = predictive.representation.str.rsplit('_', n=1).str[-1].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for mode in ('frozen_probe', 'retrained'):
        for split in ('test_internal', 'validation'):
            group = predictive[predictive.model_mode.eq(mode) & predictive.split.eq(split)].sort_values('window_size')
            label = f'{split} | {mode}'
            axes[0].plot(group.window_size, group.f1_macro, color=HIER_SPLIT_COLORS[split], ls=CTX_MODE_STYLES[mode], marker='o', label=label)
            axes[1].plot(group.window_size, group.pr_auc, color=HIER_SPLIT_COLORS[split], ls=CTX_MODE_STYLES[mode], marker='o', label=label)
    axes[0].set_title('F1 macro por tamanho da janela'); axes[1].set_title('PR-AUC por tamanho da janela')
    for ax in axes: ax.set_xlabel('n mensagens por janela'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

if not ctx_points.empty:
    chronological_points = ctx_points[
        ctx_points.representation.isin(chronological)
        & ctx_points.model_mode.eq(CTX_MODEL_MODE)
        & ctx_points.seed.eq(CTX_SEED)
        & ctx_points.ranking_policy.eq(CTX_RANKING_POLICY)
        & ctx_points.status.eq('reached')
    ].copy()
    chronological_points['window_size'] = chronological_points.representation.str.rsplit('_', n=1).str[-1].astype(int)
    display(chronological_points.groupby(['split', 'window_size', 'point_type']).agg(
        N=('sample_id', 'nunique'), mensagens_mediana=('unique_messages', 'median'),
        fracao_mediana=('fraction', 'median'), redundancia_mediana=('redundancy_ratio', 'median'),
    ))
"""
    ),
    markdown("### 17–18. Explorador: a mesma conversa nas sete representações"),
    code(
        """CTX_EXPLORER_SPLIT = 'validation'
CTX_EXPLORER_SAMPLE_ID = None

if not ctx_conversations.empty:
    available = ctx_conversations[
        ctx_conversations.split.eq(CTX_EXPLORER_SPLIT)
        & ctx_conversations.model_mode.eq(CTX_MODEL_MODE)
        & ctx_conversations.seed.eq(CTX_SEED)
    ]
    if CTX_EXPLORER_SAMPLE_ID is None and not available.empty:
        counts = available.groupby('sample_id').representation.nunique()
        complete = counts[counts.eq(len(CTX_REPRESENTATIONS))]
        CTX_EXPLORER_SAMPLE_ID = str(complete.index[0]) if len(complete) else str(available.sample_id.iloc[0])
    print('sample_id:', CTX_EXPLORER_SAMPLE_ID)
    display(available[available.sample_id.astype(str).eq(str(CTX_EXPLORER_SAMPLE_ID))].sort_values('representation'))

    case_units = ctx_units[
        ctx_units.split.eq(CTX_EXPLORER_SPLIT)
        & ctx_units.model_mode.eq(CTX_MODEL_MODE)
        & ctx_units.seed.eq(CTX_SEED)
        & ctx_units.sample_id.astype(str).eq(str(CTX_EXPLORER_SAMPLE_ID))
    ]
    for representation in CTX_REPRESENTATIONS:
        view = case_units[case_units.representation.eq(representation)].nlargest(6, 'abs_delta_logit').sort_values('delta_logit')
        if view.empty: continue
        display(Markdown(f'#### {representation}'))
        display(view[['block_index', 'message_model_indices', 'text', 'delta_logit', 'class_support_delta']])
        spans = ctx_spans[
            ctx_spans.split.eq(CTX_EXPLORER_SPLIT)
            & ctx_spans.model_mode.eq(CTX_MODEL_MODE)
            & ctx_spans.seed.eq(CTX_SEED)
            & ctx_spans.sample_id.astype(str).eq(str(CTX_EXPLORER_SAMPLE_ID))
            & ctx_spans.representation.eq(representation)
        ].nlargest(12, 'class_support_delta')
        display(spans[['block_index', 'message_model_index', 'span_text', 'class_support_delta']])

    case_fidelity = ctx_fidelity[
        ctx_fidelity.split.eq(CTX_EXPLORER_SPLIT)
        & ctx_fidelity.model_mode.eq(CTX_MODEL_MODE)
        & ctx_fidelity.seed.eq(CTX_SEED)
        & ctx_fidelity.ranking_policy.eq(CTX_RANKING_POLICY)
        & ctx_fidelity.sample_id.astype(str).eq(str(CTX_EXPLORER_SAMPLE_ID))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for representation, group in case_fidelity.groupby('representation'):
        group = group.sort_values('unique_messages_covered')
        axes[0].plot(group.unique_messages_covered, group.comprehensiveness_logit_aligned, color=CTX_COLORS[representation], label=representation)
        axes[1].plot(group.unique_messages_covered, group.sufficiency_distance_logit, color=CTX_COLORS[representation], label=representation)
    axes[0].set_title('Conversa: comprehensiveness'); axes[1].set_title('Conversa: sufficiency')
    for ax in axes: ax.set_xlabel('Mensagens originais únicas'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()
"""
    ),
]


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
import hashlib
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
hierarchical_inventory = read_hierarchical_parquet('inventory.parquet')
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
hierarchical_conversations = concat_available([
    read_hierarchical_parquet(f'{split}/conversation_metrics.parquet')
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
    {'artefato': 'inventory', 'linhas': len(hierarchical_inventory), 'colunas': len(hierarchical_inventory.columns)},
    {'artefato': 'conversation_metrics', 'linhas': len(hierarchical_conversations), 'colunas': len(hierarchical_conversations.columns)},
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
    markdown(
        """## 0. Auditoria do dataset dividido por mensagens

O estudo não usa um embedding único por conversa. Sua fonte é
`processed/messages_transformer/bge_chunks.parquet`: cada linha representa uma conversa e
`text_embedded` contém uma sequência `(n_mensagens, 1024)`.

Antes de gerar as explicações, o inventário compara, para **todas** as conversas:

- mensagens reconhecidas no texto versus vetores BGE antes do corte;
- mensagens versus vetores depois do corte das últimas 100;
- IDs e rótulos do dataset versus bundles de predição.

A auditoria abaixo verifica novamente nos resultados que cada mensagem chegou uma única vez,
na ordem correta, com `model_index` contíguo e interlocutor coerente com o prefixo textual.
Ela não recalcula embeddings e não executa o modelo.
"""
    ),
    code(
        """STRICT_MESSAGE_AUDIT = False

def audit_message_dataset(inventory, messages, primary_seed=42):
    if inventory.empty or messages.empty:
        print('Auditoria indisponível: execute/agregue o estudo global para produzir inventory.parquet e message_effects.parquet.')
        return pd.DataFrame(), pd.DataFrame()

    primary = messages[messages.seed.eq(primary_seed)].copy()
    inventory = inventory.copy()
    inventory['sample_id'] = inventory.sample_id.astype(str)
    primary['sample_id'] = primary.sample_id.astype(str)

    duplicate_mask = primary.duplicated(['split', 'sample_id', 'model_index'], keep=False)
    prefix_ok = primary.text.astype(str).str.startswith(primary.speaker.astype(str) + ':')
    per_conversation = []
    for (split, sample_id), group in primary.groupby(['split', 'sample_id'], sort=False):
        ordered = group.sort_values('model_index')
        observed_indices = ordered.model_index.astype(int).tolist()
        expected_indices = list(range(len(ordered)))
        original_indices = ordered.original_index.astype(int).tolist()
        row = inventory[(inventory.split.eq(split)) & (inventory.sample_id.eq(sample_id))]
        expected_turns = int(row.iloc[0].n_turns) if len(row) == 1 else None
        source_turns = int(row.iloc[0].source_n_turns) if len(row) == 1 else None
        expected_original = (
            list(range(source_turns - expected_turns, source_turns))
            if expected_turns is not None and source_turns is not None else []
        )
        per_conversation.append({
            'split': split,
            'sample_id': sample_id,
            'mensagens_no_resultado': len(ordered),
            'mensagens_esperadas': expected_turns,
            'fonte_antes_do_corte': source_turns,
            'contagem_ok': expected_turns == len(ordered),
            'model_index_contiguo': observed_indices == expected_indices,
            'janela_original_correta': original_indices == expected_original,
            'prefixos_ok': bool(prefix_ok.loc[ordered.index].all()),
            'sem_duplicatas': not bool(duplicate_mask.loc[ordered.index].any()),
        })

    detail = pd.DataFrame(per_conversation)
    inventory_keys = set(zip(inventory.split, inventory.sample_id))
    result_keys = set(zip(detail.split, detail.sample_id))
    rows = []
    for split in sorted(set(inventory.split) | set(detail.split)):
        inv = inventory[inventory.split.eq(split)]
        det = detail[detail.split.eq(split)]
        rows.append({
            'split': split,
            'conversas_inventario': inv.sample_id.nunique(),
            'conversas_message_effects_seed42': det.sample_id.nunique(),
            'texto_embedding_integral_ok': bool((inv.source_n_turns == inv.source_embedding_turns).all()),
            'texto_embedding_apos_corte_ok': bool((inv.n_turns == inv.embedding_turns).all()),
            'contagem_resultado_ok': bool(det.contagem_ok.all()),
            'model_index_contiguo': bool(det.model_index_contiguo.all()),
            'janela_ultimas_100_ok': bool(det.janela_original_correta.all()),
            'prefixo_interlocutor_ok': bool(det.prefixos_ok.all()),
            'sem_mensagens_duplicadas': bool(det.sem_duplicatas.all()),
            'ids_ausentes': sum(1 for key in inventory_keys - result_keys if key[0] == split),
            'ids_extras': sum(1 for key in result_keys - inventory_keys if key[0] == split),
        })
    summary = pd.DataFrame(rows)
    boolean_columns = [
        'texto_embedding_integral_ok', 'texto_embedding_apos_corte_ok',
        'contagem_resultado_ok', 'model_index_contiguo', 'janela_ultimas_100_ok',
        'prefixo_interlocutor_ok', 'sem_mensagens_duplicadas',
    ]
    failures = detail[~detail[[
        'contagem_ok', 'model_index_contiguo', 'janela_original_correta',
        'prefixos_ok', 'sem_duplicatas',
    ]].all(axis=1)]
    valid = (
        summary[boolean_columns].all(axis=None)
        and summary[['ids_ausentes', 'ids_extras']].eq(0).all(axis=None)
        and summary.conversas_inventario.eq(summary.conversas_message_effects_seed42).all()
    )
    return summary, failures if not valid else detail.iloc[0:0]

message_audit_summary, message_audit_failures = audit_message_dataset(
    hierarchical_inventory, hierarchical_messages,
)
if not message_audit_summary.empty:
    display(message_audit_summary)
    if message_audit_failures.empty:
        display(Markdown('**Resultado: dataset por mensagens corretamente alinhado em todas as conversas.**'))
    else:
        display(Markdown(f'**Falha: {len(message_audit_failures)} conversas apresentam desalinhamento.**'))
        display(message_audit_failures.head(50))
        if STRICT_MESSAGE_AUDIT:
            raise AssertionError('A auditoria do dataset dividido por mensagens falhou.')
"""
    ),
    markdown(
        """### 0.1. Proveniência: resultados versus Parquet BGE atual

A auditoria estrutural anterior prova que `inventory` e `message_effects` concordam entre si.
Esta segunda auditoria impede um erro mais sutil: visualizar resultados antigos depois que o
dataset foi reconstruído. Ela recalcula somente os IDs derivados do texto — sem carregar a
coluna de embeddings — e os compara com o inventário usado no estudo.
"""
    ),
    code(
        """def stable_text_ids(frame):
    text_hash = frame['text'].astype(str).map(
        lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    )
    occurrence = text_hash.groupby(text_hash).cumcount().astype(str)
    return (text_hash + ':' + occurrence).astype(str)

def current_dataset_id_audit(inventory):
    dataset_root = REPO_ROOT / 'backend' / 'datasets'
    rows = []
    for split, dataset_name in [('test_internal', 'dataset_train'), ('validation', 'dataset_validation')]:
        path = dataset_root / dataset_name / 'processed' / 'messages_transformer' / 'bge_chunks.parquet'
        if not path.exists():
            rows.append({'split': split, 'status': 'dataset ausente', 'path': str(path)})
            continue
        try:
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(path).schema.names)
        except Exception:
            names = {'text'}
        columns = ['text'] + (['origin'] if 'origin' in names else [])
        frame = pd.read_parquet(path, columns=columns)
        if 'origin' in frame and split == 'test_internal':
            frame = frame[frame.origin.eq('external')].reset_index(drop=True)
        current_ids = set(stable_text_ids(frame))
        result_ids = set(inventory[inventory.split.eq(split)].sample_id.astype(str))
        missing = result_ids - current_ids
        new = current_ids - result_ids
        # validation representa o dataset inteiro; test_internal é subconjunto do treino.
        valid = (not missing) and (not new if split == 'validation' else True)
        rows.append({
            'split': split,
            'status': 'atual' if valid else 'DESATUALIZADO',
            'ids_no_inventario': len(result_ids),
            'ids_no_dataset_fonte': len(current_ids),
            'ids_do_resultado_ausentes_no_dataset_atual': len(missing),
            'ids_novos_no_dataset_atual': len(new),
            'path': str(path),
        })
    return pd.DataFrame(rows)

dataset_id_audit = current_dataset_id_audit(hierarchical_inventory) if not hierarchical_inventory.empty else pd.DataFrame()
if not dataset_id_audit.empty:
    display(dataset_id_audit)
    stale = dataset_id_audit[dataset_id_audit.status.eq('DESATUALIZADO')]
    if not stale.empty:
        display(Markdown(
            '**Os resultados não correspondem ao dataset BGE atual. Reexecute o estudo antes de interpretar as métricas.**'
        ))
        if STRICT_MESSAGE_AUDIT:
            raise AssertionError('Artefatos de explicabilidade desatualizados em relação ao dataset BGE atual.')
"""
    ),
    *CONTEXT_CELLS,
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
