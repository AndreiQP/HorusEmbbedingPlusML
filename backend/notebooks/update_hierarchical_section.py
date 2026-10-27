"""Reconstrói somente a seção global do judge_decision.ipynb."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("judge_decision.ipynb")
SECTION_TITLE = "## Explicabilidade hierárquica BGE"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


NEW_CELLS = [
    markdown("""## Explicabilidade hierárquica BGE — estudo global

Esta seção lê **somente** os Parquets de `hierarchical_explainability`. As métricas globais usam a política `decision_supporting` e seed 42; o explorador também preserva `absolute_influence`. Os gráficos sempre distinguem split, política, seeds e número de observações. Nenhum BGE ou Transformer é executado no notebook.
"""),
    code("""from IPython.display import display
HIER_STUDY_ROOT = STUDY_ROOT / 'hierarchical_explainability'

def read_hierarchical_parquet(relative):
    path = HIER_STUDY_ROOT / relative
    if not path.exists():
        print(f'Artefato ausente: {path}')
        print('Execute: bash backend/slurm/studies/submit_hierarchical_explainability.sh')
        return pd.DataFrame()
    return pd.read_parquet(path)

hierarchical_coverage = read_hierarchical_parquet('aggregates/coverage.parquet')
hierarchical_curves = read_hierarchical_parquet('aggregates/fidelity_curves.parquet')
hierarchical_metrics = read_hierarchical_parquet('aggregates/metric_distributions.parquet')
hierarchical_saturation_curves = read_hierarchical_parquet('aggregates/saturation_curves.parquet')
hierarchical_saturation_fraction_curves = read_hierarchical_parquet('aggregates/saturation_fraction_curves.parquet')
hierarchical_saturation_points = read_hierarchical_parquet('aggregates/saturation_points.parquet')
hierarchical_control_summary = read_hierarchical_parquet('aggregates/random_control_summary.parquet')
hierarchical_controls = pd.concat([
    read_hierarchical_parquet(f'{split}/random_controls.parquet')
    for split in ('test_internal', 'validation')
], ignore_index=True)
hierarchical_stability = pd.concat([
    read_hierarchical_parquet(f'{split}/stability.parquet')
    for split in ('test_internal', 'validation')
], ignore_index=True)
hierarchical_messages = pd.concat([
    read_hierarchical_parquet(f'{split}/message_effects.parquet')
    for split in ('test_internal', 'validation')
], ignore_index=True)
hierarchical_spans = pd.concat([
    read_hierarchical_parquet(f'{split}/span_effects.parquet')
    for split in ('test_internal', 'validation')
], ignore_index=True)
HIER_SPLIT_COLORS = {'test_internal': '#2563eb', 'validation': '#f97316'}
HIER_CATEGORY_COLORS = {'TP': '#16a34a', 'TN': '#2563eb', 'FP': '#dc2626', 'FN': '#9333ea'}
"""),
    markdown("### 1. Cobertura global e denominadores por Top-k"),
    code("""if not hierarchical_coverage.empty:
    display(hierarchical_coverage.sort_values(['split', 'top_k']).style.format({'N_k': '{:,.0f}', 'metric_N_k': '{:,.0f}'}))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, split in zip(axes, ('test_internal', 'validation')):
        view = hierarchical_coverage[hierarchical_coverage.split.eq(split)]
        ax.bar(view.top_k - .16, view.N_k, width=.32, color=HIER_SPLIT_COLORS[split], label='N_k: conversas com ≥ k turnos')
        ax.bar(view.top_k + .16, view.metric_N_k, width=.32, color='#64748b', label='N métrico: ≥ k evidências válidas')
        ax.set(title=f'Cobertura global — {split} — seed 42', xlabel='Top-k', ylabel='Conversas')
        ax.legend(); ax.grid(axis='y', alpha=.2)
    plt.tight_layout(); plt.show()
"""),
    markdown("### 2. Comprehensiveness global — média e IC95% bootstrap"),
    code("""def plot_hierarchical_curve(metric, title, ylabel):
    view = hierarchical_curves[(hierarchical_curves.metric.eq(metric)) & (hierarchical_curves.aggregation.eq('population'))]
    if view.empty:
        return
    plt.figure(figsize=(9, 5))
    for split, group in view.groupby('split'):
        group = group.sort_values('top_k')
        plt.plot(group.top_k, group['mean'], marker='o', color=HIER_SPLIT_COLORS[split], label=f'{split} (N por ponto em tabela)')
        plt.fill_between(group.top_k, group.ci95_low, group.ci95_high, color=HIER_SPLIT_COLORS[split], alpha=.18)
    plt.axhline(0, color='black', lw=1)
    plt.title(f'{title}\\nGLOBAL | decision_supporting | seed 42')
    plt.xlabel('Quantidade acumulada de mensagens'); plt.ylabel(ylabel)
    plt.legend(title='Split / IC95%'); plt.grid(alpha=.25); plt.show()
    display(view[['split', 'top_k', 'n', 'mean', 'std', 'sem', 'ci95_low', 'ci95_high', 'negative_rate']].sort_values(['split', 'top_k']))

plot_hierarchical_curve('comprehensiveness', 'Comprehensiveness por Top-k', 'Queda na probabilidade da classe prevista')
"""),
    markdown("### 3. Sufficiency distance global — média e IC95% bootstrap"),
    code("""plot_hierarchical_curve(
    'sufficiency_distance',
    'Sufficiency distance por Top-k',
    '|P(classe prevista | conversa) − P(classe prevista | evidências)|',
)
"""),
    markdown("""### 3A. Saturação da explicação além das Top-6

O **logit** é o valor bruto produzido pelo classificador antes da sigmoid. Em probabilidades próximas de 0 ou 1, uma mudança relevante no logit pode parecer quase nula após a sigmoid. Por isso, a curva abaixo usa `comprehensiveness_logit_aligned`: positivo sempre significa queda do suporte à classe prevista após remover mensagens.

`k95` é o primeiro número de mensagens que alcança 95% do maior comprehensiveness em logit observado para aquela conversa. Ele também é apresentado como fração do total de mensagens. A sufficiency distance mede a distância entre a decisão original e a decisão usando somente as evidências; **menor é melhor**.
"""),
    code("""if not hierarchical_saturation_curves.empty:
    decision = hierarchical_saturation_curves[
        (hierarchical_saturation_curves.ranking_policy.eq('decision_supporting')) &
        (hierarchical_saturation_curves.split.eq('validation'))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric, title, ylabel in [
        (axes[0], 'comprehensiveness_logit_aligned', 'Comprehensiveness conjunto em logit', 'Queda alinhada do logit'),
        (axes[1], 'sufficiency_distance_logit', 'Sufficiency distance em logit', '|logit original − logit somente evidências|'),
    ]:
        view = decision[decision.metric.eq(metric)].sort_values('top_k')
        ax.plot(view.top_k, view['mean'], color=HIER_SPLIT_COLORS['validation'], marker='o', ms=3, label='validation — média')
        ax.fill_between(view.top_k, view.ci95_low, view.ci95_high, color=HIER_SPLIT_COLORS['validation'], alpha=.18, label='IC95%')
        ax.set(title=f'{title}\\nGLOBAL | validation | decision_supporting | seed 42', xlabel='Mensagens removidas/acumuladas (k)', ylabel=ylabel)
        ax.grid(alpha=.25); ax.legend()
    plt.tight_layout(); plt.show()
    display(decision[decision.metric.isin(['comprehensiveness_logit_aligned', 'sufficiency_distance_logit'])][['metric', 'top_k', 'n', 'mean', 'std', 'ci95_low', 'ci95_high']])
"""),
    markdown("### 3B. Mesmas curvas pela fração da conversa utilizada"),
    code("""if not hierarchical_saturation_fraction_curves.empty:
    decision_fraction = hierarchical_saturation_fraction_curves[
        hierarchical_saturation_fraction_curves.ranking_policy.eq('decision_supporting')
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric, title, ylabel in [
        (axes[0], 'comprehensiveness_logit_aligned', 'Comprehensiveness versus tamanho da conversa', 'Queda alinhada do logit'),
        (axes[1], 'sufficiency_distance_logit', 'Sufficiency versus tamanho da conversa', 'Distância de sufficiency no logit'),
    ]:
        for split, group in decision_fraction[decision_fraction.metric.eq(metric)].groupby('split'):
            group = group.sort_values('removed_fraction_bin')
            ax.plot(100 * group.removed_fraction_bin, group['mean'], marker='o', color=HIER_SPLIT_COLORS[split], label=split)
            ax.fill_between(100 * group.removed_fraction_bin, group.ci95_low, group.ci95_high, color=HIER_SPLIT_COLORS[split], alpha=.14)
        ax.set(title=f'{title}\\nGLOBAL | decision_supporting | seed 42', xlabel='% do total de mensagens acumuladas', ylabel=ylabel)
        ax.grid(alpha=.25); ax.legend(title='Split / IC95%')
    plt.tight_layout(); plt.show()
"""),
    markdown("### 3C. Quantas mensagens são necessárias para atingir k95?"),
    code("""if not hierarchical_saturation_points.empty:
    reached = hierarchical_saturation_points[hierarchical_saturation_points.status.eq('reached')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(data=reached, x='split', y='k95', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[0])
    axes[0].set_title('Mensagens para alcançar 95% do máximo observado'); axes[0].set_ylabel('k95 (mensagens)')
    sns.boxplot(data=reached, x='split', y='k95_fraction_total', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[1])
    axes[1].set_title('k95 relativo ao tamanho da conversa'); axes[1].set_ylabel('k95 / total de mensagens')
    axes[1].yaxis.set_major_formatter(lambda value, _: f'{100*value:.0f}%')
    plt.tight_layout(); plt.show()
    display(reached.groupby('split').agg(
        conversas=('sample_id', 'nunique'),
        k95_mediana=('k95', 'median'),
        k95_media=('k95', 'mean'),
        fracao_mediana=('k95_fraction_total', 'median'),
        fracao_media=('k95_fraction_total', 'mean'),
        sufficiency_logit_k95=('sufficiency_distance_logit_at_k95', 'mean'),
    ))
"""),
    markdown("""### 3D. Massa aditiva de evidência

Estas curvas são somas e, portanto, não diminuem. `absolute_influence` soma os módulos dos impactos individuais; `decision_supporting` soma somente impactos alinhados à decisão. Elas **não são comprehensiveness**, pois não reavaliam conjuntamente as interações do Transformer.
"""),
    code("""if not hierarchical_saturation_curves.empty:
    mass = hierarchical_saturation_curves[
        (hierarchical_saturation_curves.split.eq('validation')) &
        (hierarchical_saturation_curves.metric.eq('cumulative_evidence_mass_logit'))
    ]
    plt.figure(figsize=(10, 5))
    colors = {'decision_supporting': '#16a34a', 'absolute_influence': '#7c3aed'}
    for policy, group in mass.groupby('ranking_policy'):
        group = group.sort_values('top_k')
        plt.plot(group.top_k, group['mean'], marker='o', ms=3, color=colors[policy], label=f'{policy} (N varia por k)')
        plt.fill_between(group.top_k, group.ci95_low, group.ci95_high, color=colors[policy], alpha=.14)
    plt.title('Massa acumulada de evidência — GLOBAL | validation | seed 42')
    plt.xlabel('Quantidade acumulada de mensagens'); plt.ylabel('Soma acumulada de impactos no logit')
    plt.grid(alpha=.25); plt.legend(title='Política / IC95%'); plt.show()
"""),
    markdown("### 4–5. Distribuições globais por split e Top-k"),
    code("""if not hierarchical_metrics.empty:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    sns.boxplot(data=hierarchical_metrics, x='top_k', y='comprehensiveness', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[0])
    axes[0].axhline(0, color='black', lw=1); axes[0].set_title('Comprehensiveness — GLOBAL | decision_supporting | seed 42')
    sns.boxplot(data=hierarchical_metrics, x='top_k', y='sufficiency_distance', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[1])
    axes[1].set_title('Sufficiency distance — GLOBAL | decision_supporting | seed 42 (sempre ≥ 0)')
    for ax in axes: ax.set_xlabel('Top-k'); ax.legend(title='Split')
    plt.tight_layout(); plt.show()
"""),
    markdown("### 6. Variação por TP, TN, FP e FN"),
    code("""if not hierarchical_metrics.empty:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True)
    for column, split in enumerate(('test_internal', 'validation')):
        view = hierarchical_metrics[hierarchical_metrics.split.eq(split)]
        sns.boxplot(data=view, x='top_k', y='comprehensiveness', hue='prediction_category', palette=HIER_CATEGORY_COLORS, showfliers=False, ax=axes[0, column])
        axes[0, column].axhline(0, color='black', lw=1); axes[0, column].set_title(f'{split} — comprehensiveness')
        sns.boxplot(data=view, x='top_k', y='sufficiency_distance', hue='prediction_category', palette=HIER_CATEGORY_COLORS, showfliers=False, ax=axes[1, column])
        axes[1, column].set_title(f'{split} — sufficiency distance')
    fig.suptitle('GLOBAL | decision_supporting | seed 42 | cores = TP/TN/FP/FN')
    plt.tight_layout(); plt.show()
"""),
    markdown("### 7–8. Trechos selecionados versus controles aleatórios"),
    code("""if not hierarchical_controls.empty:
    unique_controls = hierarchical_controls.drop_duplicates(['split', 'sample_id', 'seed', 'model_index', 'start_char', 'end_char'])
    impact = pd.concat([
        unique_controls[['split', 'prediction_category', 'selected_abs_delta_logit']].rename(columns={'selected_abs_delta_logit': 'impact'}).assign(tipo='Trecho selecionado'),
        hierarchical_controls[['split', 'prediction_category', 'control_abs_delta_logit']].rename(columns={'control_abs_delta_logit': 'impact'}).assign(tipo='Controle aleatório'),
    ], ignore_index=True).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.boxplot(data=impact, x='split', y='impact', hue='tipo', showfliers=False, ax=axes[0])
    axes[0].set_title(f'Impacto |Δ logit| — GLOBAL | seed 42 | N={len(unique_controls):,} trechos')
    sns.boxplot(data=unique_controls, x='split', y='specificity_ratio', hue='prediction_category', palette=HIER_CATEGORY_COLORS, showfliers=False, ax=axes[1])
    axes[1].axhline(1, color='black', ls='--', label='Mesmo impacto dos controles')
    axes[1].set_title('Razão de especificidade — GLOBAL | decision_supporting | seed 42')
    axes[1].legend(title='Categoria / referência')
    plt.tight_layout(); plt.show()
    global_control_ci = hierarchical_control_summary[hierarchical_control_summary.aggregation.eq('population')]
    display(global_control_ci[['split', 'metric', 'n', 'mean', 'std', 'ci95_low', 'ci95_high']])
"""),
    markdown("### 9–10. Estabilidade global entre seeds 42, 52 e 62"),
    code("""if not hierarchical_stability.empty:
    summary_stability = hierarchical_stability[hierarchical_stability.item_type.eq('summary')].copy()
    metrics = ['message_top_k_jaccard', 'message_direction_agreement', 'message_rank_spearman', 'message_rank_kendall', 'span_top_k_jaccard', 'span_direction_agreement', 'span_rank_kendall']
    long = summary_stability.melt(id_vars=['split', 'sample_id', 'compared_seed', 'prediction_agreement'], value_vars=[m for m in metrics if m in summary_stability], var_name='métrica', value_name='valor').dropna()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    sns.boxplot(data=long, x='métrica', y='valor', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False, ax=axes[0])
    axes[0].tick_params(axis='x', rotation=20); axes[0].set_title(f'Estabilidade de rankings — GLOBAL | seeds 42×52/62 | N={summary_stability.sample_id.nunique():,} conversas')
    agreement = summary_stability.groupby(['split', 'compared_seed'], as_index=False).prediction_agreement.mean()
    sns.barplot(data=agreement, x='compared_seed', y='prediction_agreement', hue='split', palette=HIER_SPLIT_COLORS, ax=axes[1])
    axes[1].set_ylim(0, 1); axes[1].set_title('Concordância da classe prevista (discordâncias excluídas das métricas de justificativa)')
    plt.tight_layout(); plt.show()
    item_stability = hierarchical_stability[hierarchical_stability.item_type.isin(['message', 'span'])].dropna(subset=['impact_coefficient_variation'])
    if not item_stability.empty:
        sns.boxplot(data=item_stability, x='item_type', y='impact_coefficient_variation', hue='split', palette=HIER_SPLIT_COLORS, showfliers=False)
        plt.title('Coeficiente de variação do impacto — GLOBAL | seeds 42/52/62')
        plt.xlabel('Elemento'); plt.ylabel('desvio-padrão / média |Δ logit|'); plt.legend(title='Split'); plt.show()
"""),
    markdown("""### 11. Explorador por conversa

Defina `HIER_STUDY_SAMPLE_ID` e `HIER_STUDY_SPLIT`. A esquerda ficam as seis mensagens por influência absoluta; a direita, apenas justificativas alinhadas à classe prevista. Os painéis inferiores são **desta conversa**, não médias globais.
"""),
    code("""HIER_STUDY_SPLIT = 'test_internal'
HIER_STUDY_SAMPLE_ID = None
HIER_STUDY_SEED = 42

available = (
    hierarchical_messages[(hierarchical_messages.split.eq(HIER_STUDY_SPLIT)) & (hierarchical_messages.seed.eq(HIER_STUDY_SEED))]
    if not hierarchical_messages.empty else pd.DataFrame()
)
if not available.empty:
    if HIER_STUDY_SAMPLE_ID is None:
        HIER_STUDY_SAMPLE_ID = str(available.sample_id.iloc[0])
    case_messages = available[available.sample_id.astype(str).eq(str(HIER_STUDY_SAMPLE_ID))]
    case_spans = hierarchical_spans[(hierarchical_spans.split.eq(HIER_STUDY_SPLIT)) & (hierarchical_spans.seed.eq(HIER_STUDY_SEED)) & (hierarchical_spans.sample_id.astype(str).eq(str(HIER_STUDY_SAMPLE_ID)))]
    case_fidelity = hierarchical_metrics[(hierarchical_metrics.split.eq(HIER_STUDY_SPLIT)) & (hierarchical_metrics.sample_id.astype(str).eq(str(HIER_STUDY_SAMPLE_ID)))]
    absolute = case_messages[case_messages.selected_absolute_influence].nlargest(6, 'abs_delta_logit').sort_values('delta_logit')
    supporting = case_messages[case_messages.selected_decision_supporting].sort_values('class_support_delta').tail(6)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].barh(absolute.text.str.slice(0, 75), absolute.delta_logit, color=np.where(absolute.delta_logit.ge(0), '#dc2626', '#2563eb'))
    axes[0].set_title(f'CONVERSA {HIER_STUDY_SAMPLE_ID}: Top-6 absolute_influence\\nvermelho=pró-Scam; azul=pró-Ham')
    axes[1].barh(supporting.text.str.slice(0, 75), supporting.class_support_delta, color='#16a34a')
    supported = supporting.supported_class.iloc[0] if len(supporting) else 'sem evidência válida'
    axes[1].set_title(f'Justificativas decision_supporting para {supported}\\nseed 42 | N={len(supporting)}')
    plt.tight_layout(); plt.show()

    selected_indices = set(supporting.model_index.astype(int))
    span_view = case_spans[case_spans.model_index.isin(selected_indices) & case_spans.class_support_delta.gt(1e-4)]
    span_view = span_view.sort_values('class_support_delta').groupby('model_index', group_keys=False).tail(3)
    if not span_view.empty:
        plt.figure(figsize=(12, max(4, len(span_view) * .35)))
        plt.barh(span_view.span_text.str.slice(0, 90), span_view.class_support_delta, color='#dc2626' if supported == 'scam' else '#2563eb')
        plt.title(f'N-grams alinhados à decisão — CONVERSA {HIER_STUDY_SAMPLE_ID} | {HIER_STUDY_SPLIT} | seed 42 | N={len(span_view)}')
        plt.xlabel('class_support_delta'); plt.tight_layout(); plt.show()

    if not case_fidelity.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(case_fidelity.top_k, case_fidelity.comprehensiveness, marker='o', label='Comprehensiveness')
        axes[0].axhline(0, color='black', lw=1); axes[0].legend(); axes[0].set_title('CONVERSA: remoção cumulativa')
        axes[1].plot(case_fidelity.top_k, case_fidelity.sufficiency_distance, marker='s', color='#f97316', label='Sufficiency distance')
        axes[1].legend(); axes[1].set_title('CONVERSA: somente evidências')
        for ax in axes: ax.set_xlabel('Top-k'); ax.grid(alpha=.25)
        plt.tight_layout(); plt.show()

    case_controls = hierarchical_controls[(hierarchical_controls.split.eq(HIER_STUDY_SPLIT)) & (hierarchical_controls.sample_id.astype(str).eq(str(HIER_STUDY_SAMPLE_ID)))]
    case_stability = hierarchical_stability[(hierarchical_stability.split.eq(HIER_STUDY_SPLIT)) & (hierarchical_stability.sample_id.astype(str).eq(str(HIER_STUDY_SAMPLE_ID))) & hierarchical_stability.item_type.eq('summary')]
    display(case_controls.drop_duplicates(['model_index', 'start_char', 'end_char'])[['span_text', 'selected_abs_delta_logit', 'control_mean', 'specificity_ratio']])
    display(case_stability)
"""),
]


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
start = next(
    index for index, cell in enumerate(notebook["cells"])
    if SECTION_TITLE in "".join(cell.get("source", []))
)
notebook["cells"] = notebook["cells"][:start] + NEW_CELLS
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
