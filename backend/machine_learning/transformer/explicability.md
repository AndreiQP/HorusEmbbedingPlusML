A implementação é um estudo de explicabilidade local por perturbação. O BGE-M3 e os Transformers ficam completamente congelados: não há treinamento, ajuste de pesos ou uso de gradientes. A explicação é obtida observando como a saída muda quando removemos informações específicas.

O fluxo completo é:

```text
Dataset cacheado
    │
    ├── texto original ──► mensagens, palavras e offsets
    └── text_embedded ───► embeddings BGE originais
                              │
                              ▼
                    Transformer da seed 42
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Macro: mensagens               Micro: expressões
       todas as mensagens             apenas top 6 mensagens
               │                             │
               └──────────────┬──────────────┘
                              ▼
                 Fidelidade e estabilidade
                              │
                              ▼
                  CSVs, JSONs e notebook
```

## 1. O que a explicação mede

O efeito de uma perturbação é:

\[
\Delta z = z_{\text{original}} - z_{\text{perturbado}}
\]

Onde \(z\) é o logit de Scam.

A interpretação é:

- `Δz > 1e-4`: ao remover o conteúdo, o logit de Scam diminuiu. Portanto, o conteúdo era evidência pró-Scam.
- `Δz < -1e-4`: ao remover o conteúdo, o logit de Scam aumentou. Portanto, o conteúdo estava funcionando como evidência pró-Ham.
- `|Δz| ≤ 1e-4`: efeito considerado neutro.

Isso identifica causalidade em relação ao comportamento do modelo, não causalidade no mundo real. Da mesma forma, “evidência pró-Ham” não prova que uma mensagem é verdadeira; significa apenas que o modelo aprendeu a associá-la à classe Ham.

A implementação central está em [hierarchical_perturbation.py](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/machine_learning/studies/hierarchical_perturbation.py:267).

## 2. Seleção dos checkpoints

O código não procura novamente “o melhor `model.pth`” olhando métricas de teste.

Ele lê:

```text
backend/experiment_results/transformer/finalists/bge/summary.json
```

Esse arquivo contém o `run_id` previamente selecionado para cada seed. O código procura exatamente o `model.pth` desse `run_id`.

Isso evita uma seleção pós-hoc com base no conjunto de teste.

Para cada checkpoint:

1. Recupera a configuração do Transformer.
2. Instancia `TransformerScamClassifier`.
3. Carrega o `state_dict`.
4. Coloca o modelo em modo `eval`.
5. Desabilita gradientes de todos os parâmetros.

Essa resolução acontece em [resolve_bge_finalist_checkpoint()](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/machine_learning/studies/hierarchical_perturbation.py:977).

## 3. Carregamento do dataset

O estudo usa `_dataset("bge", split)`, que devolve:

- `sequences`: embeddings BGE já calculados;
- `labels`: rótulos reais;
- `ids`: identificadores das conversas;
- `text_by_id`: texto concatenado de cada conversa.

O ponto importante é:

> Os embeddings originais usados pelo Transformer vêm de `text_embedded`. O BGE não recodifica todas as mensagens originais durante o estudo.

O texto concatenado é usado somente para recuperar:

- limites das mensagens;
- interlocutor;
- palavras;
- offsets;
- texto dos n-grams.

## 4. Separação das mensagens

A função `parse_conversation()` identifica os marcadores:

```text
Innocent:
Suspect:
```

Ela produz um `ConversationTurn` com:

- `original_index`: posição original na conversa;
- `model_index`: posição dentro da janela do Transformer;
- `speaker`;
- texto completo da mensagem;
- posição onde começa o conteúdo depois do prefixo;
- offsets dentro da conversa completa.

São mantidos somente os últimos 100 turnos, reproduzindo a entrada utilizada pelo Transformer.

Depois disso, há uma validação obrigatória:

```text
quantidade de turnos textuais == quantidade de embeddings
```

Também é verificado que cada embedding possui a dimensão esperada do BGE, 1024.

Se houver qualquer divergência, o estudo é interrompido com `ReproductionError`. O código não tenta corrigir silenciosamente o alinhamento.

## 5. Seleção das conversas

A seed 42 é usada para separar as conversas em:

- TP: Scam real, classificado como Scam;
- TN: Ham real, classificado como Ham;
- FP: Ham real, classificado como Scam;
- FN: Scam real, classificado como Ham.

Dentro de cada categoria, os `sample_id` são ordenados pelo SHA-256 do próprio ID. Isso deixa a seleção determinística e evita escolher exemplos pela confiança do modelo.

Com os defaults, são selecionados:

- 5 TP;
- 5 TN;
- 5 FP;
- 5 FN.

Total esperado: 20 conversas, desde que existam pelo menos cinco exemplos em cada categoria. Atualmente, se uma categoria tiver menos de cinco exemplos, o código usa os disponíveis; ele não falha apenas por causa disso.

Para estabilidade, são escolhidas as duas primeiras conversas da ordenação determinística de cada categoria:

- 2 TP;
- 2 TN;
- 2 FP;
- 2 FN.

Total: 8 conversas.

A seleção está em [_select_samples_by_prediction_category()](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/machine_learning/studies/hierarchical_perturbation.py:1217).

## 6. Auditoria única do BGE

Embora os embeddings cacheados sejam a fonte oficial, precisamos garantir que o BGE disponível no cluster é compatível com aquele que criou o dataset.

Por isso, o job escolhe deterministicamente a primeira conversa do estudo principal e:

1. Recupera seus embeddings cacheados.
2. Recodifica todos os turnos dessa única conversa usando:

```python
SentenceTransformer("BAAI/bge-m3").encode(...)
```

3. Compara os novos embeddings com os salvos usando:

```text
atol = 1e-5
rtol = 1e-4
```

São registrados:

- maior erro absoluto;
- erro absoluto médio;
- menor similaridade cosseno;
- número de turnos auditados.

Se os embeddings não forem compatíveis, o job para antes de gerar as explicações.

Essa auditoria acontece uma única vez por execução do estudo, não uma vez por conversa.

## 7. Reprodução das predições

Para cada checkpoint necessário, o código carrega o bundle de predições original.

Ele valida que:

- os `sample_ids` estão na mesma ordem do dataset;
- os rótulos estão alinhados;
- a probabilidade recalculada pelo Transformer usando `text_embedded` reproduz `y_prob`;
- o logit derivado também é compatível.

Portanto, antes de aceitar uma explicação, o sistema prova que o conjunto:

```text
embedding cacheado + checkpoint carregado
```

reproduz a predição original.

Isso é importante porque uma explicação sobre um modelo incorretamente reconstruído não seria confiável.

## 8. Baselines neutros

O BGE codifica somente duas mensagens neutras no início do job:

```text
Innocent:
Suspect:
```

São os textos contendo apenas o prefixo do interlocutor.

O vetor usado depende de quem escreveu a mensagem:

- uma mensagem `Innocent:` é substituída pelo embedding de `Innocent:`;
- uma mensagem `Suspect:` é substituída pelo embedding de `Suspect:`.

Esses dois embeddings são reutilizados para todas as conversas, mensagens e seeds.

Isso preserva:

- quantidade de turnos;
- posição do turno;
- positional encoding;
- identidade do interlocutor.

O que é retirado é o conteúdo semântico da mensagem.

## 9. Etapa macro: importância das mensagens

Para uma conversa com \(M\) mensagens, o sistema cria \(M\) intervenções.

Em cada intervenção:

1. Copia a sequência original de embeddings.
2. Escolhe uma posição.
3. Substitui somente aquele embedding pelo baseline neutro correspondente.
4. Mantém os outros embeddings intactos.
5. Executa novamente o Transformer.

Exemplo:

```text
Original:
[E0, E1, E2, E3, E4]

Teste da mensagem 2:
[E0, E1, NEUTRO, E3, E4]
```

Não há chamada ao BGE dentro desse loop. São apenas inferências do Transformer sobre matrizes de embeddings.

Para cada mensagem são registrados:

- logit perturbado;
- probabilidade perturbada;
- `delta_logit`;
- `delta_probability`;
- direção `scam`, `ham` ou `neutral`;
- se a classe prevista mudou;
- índices e texto original.

Depois, todas as mensagens são ordenadas por:

1. maior `abs(delta_logit)`;
2. maior `abs(delta_probability)`;
3. menor índice em caso de empate.

As seis primeiras formam a seleção macro.

A implementação está em [_score_macro()](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/machine_learning/studies/hierarchical_perturbation.py:434).

## 10. Etapa micro: expressões e frases

A etapa micro é executada apenas nas seis mensagens selecionadas pela seed 42.

As “palavras” são identificadas por separação textual com `\S+`. Portanto, são palavras delimitadas por espaços, não os sub-tokens WordPiece internos do XLM-R/BGE.

Para cada mensagem, são criados todos os n-grams contíguos de 1 a 5 palavras.

Para uma mensagem com \(W \ge 5\) palavras, a quantidade de perturbações é:

\[
W + (W-1) + (W-2) + (W-3) + (W-4) = 5W - 10
\]

Por exemplo:

```text
Suspect: clique no link de pagamento agora
```

Algumas perturbações seriam:

```text
Suspect: no link de pagamento agora
Suspect: clique link de pagamento agora
Suspect: clique no de pagamento agora
Suspect: clique no link agora
Suspect: clique no link de agora
Suspect: clique no link de pagamento
Suspect: clique no link agora
Suspect: clique agora
```

O prefixo `Suspect:` nunca é removido.

Para cada n-gram são preservados:

- texto exato;
- posição da primeira e última palavra;
- offsets dentro da mensagem;
- offsets dentro da conversa;
- texto perturbado;
- tamanho do n-gram.

Todas as versões perturbadas daquela conversa são enviadas em lote para o BGE.

Depois, para cada candidato:

1. Usa-se o embedding BGE da mensagem perturbada.
2. Substitui-se somente o vetor da mensagem alvo.
3. Os outros 99 ou menos embeddings permanecem congelados.
4. Executa-se novamente o Transformer.
5. Calcula-se `delta_logit` e `delta_probability`.

Também é calculado:

\[
\text{delta\_logit\_per\_word}
=
\frac{\Delta z}{\text{tamanho do n-gram}}
\]

A geração dos trechos está em [generate_ngram_perturbations()](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/machine_learning/studies/hierarchical_perturbation.py:149).

## 11. Como o BGE é reutilizado entre seeds

A seed 42 é a fonte canônica das perturbações.

Para cada conversa:

1. A seed 42 calcula o ranking macro.
2. Ela seleciona as top 6 mensagens.
3. São gerados os n-grams dessas mensagens.
4. O BGE codifica a coleção de textos perturbados.
5. Os embeddings resultantes ficam em `PreparedPerturbations`.

Esse objeto contém:

- embeddings originais;
- embeddings neutros;
- candidatos textuais;
- embeddings BGE perturbados;
- ranking macro da seed 42;
- top 6 da seed 42;
- identificação da seed que originou os candidatos.

Nas seeds 52 e 62, o sistema não chama novamente o BGE para esses n-grams. Os mesmos vetores perturbados são inseridos nos outros Transformers.

Assim, estamos respondendo:

> Os Transformers treinados com seeds diferentes reagem de forma semelhante às mesmas perturbações definidas pela seed operacional 42?

Não estamos produzindo três universos independentes de explicações.

## 12. Escopo das seeds

Com os defaults:

### Seed 42

É executada nas 20 conversas:

```text
5 TP + 5 TN + 5 FP + 5 FN
```

Produz:

- macro completo;
- top 6 mensagens;
- micro completo;
- fidelidade;
- controles aleatórios.

### Seeds 52 e 62

São executadas somente nas oito conversas de estabilidade:

```text
2 TP + 2 TN + 2 FP + 2 FN
```

Para essas seeds:

- o ranking macro é recalculado;
- as top 6 mensagens próprias são recuperadas para comparação;
- os n-grams avaliados continuam sendo os definidos pela seed 42;
- os embeddings BGE desses n-grams são reutilizados.

O total padrão é:

```text
20 explicações da seed 42
 8 explicações da seed 52
 8 explicações da seed 62
─────────────────────────
36 objetos de explicação
```

Usando `--skip-stability`, são carregados somente a seed 42 e seus 20 casos.

## 13. Fidelidade das mensagens

Além de medir perturbações individuais, são avaliadas cumulativamente as top 1 até top 6 mensagens.

### Comprehensiveness

As mensagens selecionadas são progressivamente neutralizadas:

```text
top 1 removida
top 1–2 removidas
top 1–3 removidas
...
top 1–6 removidas
```

A métrica verifica quanto a confiança na classe originalmente prevista cai quando essas evidências são removidas.

\[
\text{comprehensiveness}
=
s_{\text{original}} - s_{\text{removido}}
\]

Quanto maior e mais positiva, mais importantes eram as mensagens removidas.

### Sufficiency

O processo inverso também é executado:

1. Começa-se com todas as mensagens neutralizadas.
2. Restauram-se somente as top 1, depois top 2, até top 6.

\[
\text{sufficiency gap}
=
s_{\text{original}} - s_{\text{suficiente}}
\]

Um gap pequeno significa que somente aquelas mensagens já conseguem recuperar boa parte da decisão original.

Essas métricas são calculadas para a classe originalmente prevista. Se o modelo previu Ham, o score analisado é \(1-p(\text{Scam})\).

## 14. Fidelidade dos trechos

Para o melhor trecho de cada uma das seis mensagens, são feitos dois testes.

### Remoção

Usa o resultado já obtido ao apagar aquele n-gram da mensagem.

### Trecho isolado

Cria uma mensagem contendo apenas:

```text
Speaker: trecho selecionado
```

Esse texto é codificado pelo BGE e colocado na posição da mensagem original. O restante da conversa continua presente.

Isso mede se o trecho isolado é suficiente para preservar a evidência encontrada.

Os textos de sufficiency necessários pelas seeds ativas são:

1. reunidos;
2. deduplicados;
3. codificados uma única vez;
4. reutilizados entre os Transformers.

## 15. Controles aleatórios

Para cada trecho principal, o sistema procura outros n-grams:

- da mesma mensagem;
- com o mesmo número de palavras;
- com offsets diferentes.

Até cinco controles são escolhidos usando uma seed metodológica fixa 42.

Depois compara:

```text
abs(delta_logit do trecho escolhido)
─────────────────────────────────────
média de abs(delta_logit dos controles)
```

O resultado é `selected_to_random_ratio`.

Interpretação:

- valor próximo de 1: o trecho escolhido não é muito melhor do que trechos aleatórios do mesmo tamanho;
- valor bem acima de 1: o trecho possui impacto mais específico;
- valor abaixo de 1: os controles aleatórios tiveram impacto médio maior.

Nenhuma nova inferência BGE é necessária para os controles, pois seus efeitos já estão na coleção completa de n-grams.

## 16. Estabilidade entre seeds

O arquivo `stability.csv` contém três tipos de linha.

### Mensagens

Para cada mensagem:

- número de seeds avaliadas;
- frequência com que entrou nas top 6;
- média de `abs(delta_logit)`;
- desvio padrão do impacto;
- direção majoritária;
- concordância de direção.

### Trechos

Para cada n-gram canônico da seed 42:

- frequência com que apareceu no ranking de trechos;
- impacto absoluto médio;
- variação do impacto;
- direção majoritária;
- concordância de direção.

### Resumo por seed

Compara a seed 42 com 52 e 62:

- quantidade de mensagens top 6 em comum;
- Jaccard das top 6;
- concordância da direção das mensagens;
- Spearman do ranking macro;
- concordância da direção dos spans;
- Spearman do ranking dos spans;
- quantidade de spans compartilhados.

A seed 42 não é recalculada nessa fase. Seus resultados da etapa principal são reutilizados.

## 17. Artefatos produzidos

A pasta final é:

```text
backend/experiment_results/transformer/studies/hierarchical_explainability/
```

### `message_occlusion.csv`

Contém todas as perturbações macro.

- Seed 42: todas as 20 conversas.
- Seeds 52/62: somente as oito conversas de estabilidade.

### `span_occlusion.csv`

Contém todos os n-grams avaliados, inclusive aqueles que não ficaram entre os 20 melhores de cada mensagem.

### `fidelity.csv`

Contém:

- fidelidade cumulativa top 1–6;
- fidelidade dos melhores trechos;
- controles aleatórios;
- comprehensiveness;
- sufficiency.

### `stability.csv`

Contém os resultados agregados de estabilidade. Se estabilidade estiver desabilitada, o arquivo continua sendo criado, mas somente com cabeçalhos.

### `explanations/<sample_id>/seed_<seed>.json`

Resumo navegável de cada explicação:

- baseline;
- top mensagens;
- trecho Scam mais forte;
- trecho Ham mais forte;
- ranking resumido dos spans;
- fidelity;
- metadados de reprodução.

### `manifest.json`

Registra:

- seeds utilizadas;
- amostras principais;
- subconjunto de estabilidade;
- contagens por categoria;
- configuração;
- proveniência e revisão do BGE;
- resultado da auditoria;
- hashes da análise;
- tolerâncias;
- política de seleção.

A persistência está em [save_hierarchical_explanations()](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/machine_learning/studies/hierarchical_perturbation.py:1609).

## 18. Campos `analysis_scope` e `candidate_source_seed`

Todos os CSVs recebem esses campos.

`analysis_scope`:

- `primary`: resultado principal da seed 42;
- `stability`: execução adicional da seed 52 ou 62.

`candidate_source_seed`:

```text
42
```

Esse campo indica que os n-grams avaliados foram definidos pelo ranking canônico da seed 42, mesmo quando o Transformer avaliado é o da seed 52 ou 62.

## 19. Execução no cluster

O submit:

1. Interpreta os argumentos.
2. Valida contagens e seeds.
3. Cria as pastas de log e resultado.
4. Ativa um ambiente leve para o preflight.
5. Confirma que os checkpoints necessários existem.
6. Submete o job via `sbatch`.

Se estabilidade estiver ligada, o preflight exige 42, 52 e 62. Com `--skip-stability`, exige somente 42.

O job solicita:

```text
1 GPU L40S
8 CPUs
64 GB de RAM
até 2 dias
```

Dentro do nó:

1. Ativa `env_gpu_final`.
2. Confirma que CUDA está disponível.
3. Repete o preflight dos checkpoints.
4. Executa:

```bash
python -m machine_learning.studies hierarchical-explain --study ...
```

Os scripts são:

- [submit_hierarchical_explainability.sh](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/slurm/studies/submit_hierarchical_explainability.sh)
- [run_hierarchical_explainability.sbatch](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/slurm/studies/run_hierarchical_explainability.sbatch)

## 20. Notebook

O notebook não executa BGE nem Transformer. Ele lê diretamente:

```text
message_occlusion.csv
span_occlusion.csv
fidelity.csv
stability.csv
```

Por padrão:

```python
HIER_SAMPLE_ID = None
HIER_SEED = 42
```

Quando o ID é `None`, ele escolhe automaticamente a conversa da seed 42 com o maior impacto macro individual.

A visualização possui quatro painéis:

1. Top 6 mensagens por impacto absoluto.
2. Quinze expressões mais influentes.
3. Comprehensiveness e sufficiency para top 1–6.
4. Estabilidade entre seeds.

As cores significam:

- vermelho: pró-Scam;
- azul: pró-Ham;
- cinza: neutro.

Também são exibidas tabelas com:

- texto integral das mensagens;
- trecho textual exato;
- offsets;
- tamanho do n-gram;
- direção;
- impacto;
- mudança de classe;
- comparação com controles aleatórios;
- métricas de estabilidade.

Se a conversa não estiver entre as oito selecionadas, o notebook informa:

```text
Caso não selecionado para estabilidade
```

A explicação principal da seed 42 permanece disponível normalmente.

A seção está em [judge_decision.ipynb](C:/Users/Master/Desktop/Recod_ia/HorusEmbbedingPlusML/backend/notebooks/judge_decision.ipynb).

## 21. Quando o BGE é chamado

No estudo cacheado, o BGE é usado somente nestes pontos:

1. Auditoria dos turnos de uma única conversa.
2. Codificação dos dois baselines neutros.
3. Codificação dos textos perturbados das top 6 mensagens de cada conversa.
4. Codificação deduplicada dos textos de sufficiency.

Ele não é usado para:

- recodificar todas as mensagens originais;
- executar a etapa macro mensagem a mensagem;
- repetir as perturbações para 52 e 62;
- gerar controles aleatórios;
- alimentar o notebook.

Para uma conversa nova, sem `text_embedded`, `explain_conversation()` codifica as mensagens originais uma vez e depois segue o mesmo fluxo.

Em resumo, o estudo responde duas perguntas complementares:

1. **Quais mensagens alteram mais a decisão quando neutralizadas?**
2. **Dentro dessas mensagens, quais trechos alteram mais a decisão quando removidos?**

E faz isso preservando o contexto completo da conversa, usando o Transformer finalista real, medindo evidências tanto pró-Scam quanto pró-Ham e controlando o custo ao reutilizar os mesmos embeddings perturbados entre as seeds.