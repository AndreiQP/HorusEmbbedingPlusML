# Plano de explicabilidade hierárquica do Transformer BGE

## Objetivo

Implementar explicabilidade local por perturbação em duas etapas:

1. triagem macro pela substituição de cada mensagem por um baseline neutro;
2. análise micro por oclusão de n-grams nas seis mensagens de maior impacto absoluto.

Cada efeito é classificado como evidência pró-Scam, pró-Ham ou neutra. Evidência
pró-Ham representa um sinal de legitimidade aprendido pelo classificador e não uma
afirmação de veracidade factual.

## Decisões metodológicas

- Usar a configuração BGE vencedora por `internal_val_f1_macro` (2 heads, 4
  camadas, hidden 256, dropout 0,1, batch 16, Adam, `lr=1e-6`). Carregar o
  checkpoint finalista pelo `run_id` registrado, com seed 42 como execução
  canônica e seeds 42/52/62 no estudo de estabilidade. Nunca escolher uma seed
  por teste interno ou validação externa.
- Tratar `text_embedded` como fonte oficial dos embeddings originais. Separar o
  texto bruto apenas para recuperar mensagens, palavras e offsets.
- Auditar a reprodução BGE uma única vez por job em uma conversa determinística,
  com `atol=1e-5` e `rtol=1e-4`; interromper o estudo em caso de divergência.
- Manter somente os últimos 100 turnos e preservar os índices originais.
- Selecionar para o estudo somente conversas com mais de seis mensagens, isto é,
  pelo menos sete turnos após esse corte. Isso garante seis mensagens reais na
  visualização macro; conversas curtas não entram na amostra TP/TN/FP/FN.
- Na etapa macro, codificar uma vez os baselines `Innocent:` e `Suspect:` e
  reutilizar esses vetores.
- Definir `delta_logit = logit_original - logit_perturbado`.
- Classificar `delta_logit > 1e-4` como pró-Scam, `< -1e-4` como pró-Ham e os
  demais efeitos como neutros.
- Selecionar as seis mensagens com maior `abs(delta_logit)`, sem impor divisão
  entre as direções.
- Na etapa micro, remover n-grams contíguos de uma a cinco palavras, recodificar
  apenas a mensagem alterada e manter os demais vetores congelados.
- Usar `abs(delta_logit)` no ranking, preservando o sinal e preferindo o menor
  trecho em caso de empate.

## Fidelidade e estabilidade

- Medir comprehensiveness e sufficiency para os principais trechos.
- Comparar os trechos selecionados com n-grams aleatórios do mesmo tamanho.
- Medir o efeito cumulativo das top 1 a top 6 mensagens.
- Executar as cinco conversas de cada categoria somente na seed canônica 42.
- Executar seeds 52 e 62 somente nas duas primeiras conversas determinísticas de
  cada categoria e reutilizar os n-grams definidos pela seed 42.
- Codificar cada perturbação e texto de sufficiency uma única vez por conversa e
  reutilizar os vetores em todos os Transformers.
- Avaliar TP, TN, FP e FN separadamente quando rótulos estiverem disponíveis.
- Usar validação externa somente para avaliação final, sem seleção de parâmetros.

## Interface prevista

```python
explain_conversation(
    conversation: str,
    seed: int = 42,
    top_messages: int = 6,
    max_ngram: int = 5,
    top_spans_per_message: int = 20,
) -> HierarchicalExplanation
```

O estudo cacheado usa:

```python
run_cached_dataset_study(
    split="test_internal",
    primary_seed=42,
    stability_seeds=(42, 52, 62),
    samples_per_category=5,
    stability_samples_per_category=2,
    min_messages=7,
)
```

Os artefatos ficam em
`experiment_results/transformer/studies/hierarchical_explainability/<split>/` e
incluem manifesto, oclusão de mensagens, oclusão de trechos, fidelidade,
estabilidade e explicações JSON por amostra. `test_internal` e `validation`
usam diretórios independentes para impedir sobrescrita e permitir visualização
separada das duas distribuições.

## Execução e visualização

- Executar o estudo completo no cluster pelo job GPU
  `backend/slurm/studies/submit_hierarchical_explainability.sh`.
- Fazer preflight da seed 42 e exigir 52/62 somente quando a estabilidade estiver
  habilitada.
- Usar `backend/notebooks/judge_decision.ipynb` somente para leitura dos
  artefatos e visualização de impacto macro, impacto micro, fidelidade top 1–6
  e estabilidade entre seeds.
- Selecionar a distribuição no notebook por `HIER_SPLIT`, preservando tabelas e
  gráficos independentes para `test_internal` e `validation`.
- Não carregar o BGE nem refazer inferência dentro do notebook.

## Critérios de aceite

- Usar embeddings cacheados sem recodificação no fluxo normal e reproduzir os
  bundles de probabilidade/logit para cada checkpoint.
- Executar a auditoria BGE uma vez por job.
- Reutilizar embeddings neutros na etapa macro.
- Alterar somente o vetor alvo na etapa micro.
- Retornar índices, textos, offsets e direção dos efeitos.
- Produzir rankings determinísticos e resultados separados por seed.
- Produzir até 20 explicações principais e até oito casos de estabilidade com os
  defaults: cada categoria usa cinco exemplos elegíveis quando disponíveis ou
  todos os disponíveis quando houver menos.
- Entregar os trechos mais fortes pró-Scam e pró-Ham quando existirem.
