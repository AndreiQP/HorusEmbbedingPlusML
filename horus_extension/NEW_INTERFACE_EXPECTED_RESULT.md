# Resultado esperado para a interface do Horus

## Situação atual

A branch `main` permanece como a versão operacional de referência do fluxo de captura.
A nova implementação visual está isolada em `feature/new_interface_v2` e não deve ser
incorporada à `main` enquanto os critérios deste documento não forem comprovados no
WhatsApp real.

## Objetivo visual

A futura interface deverá substituir o olho móvel e o painel semelhante a um terminal
por uma aba lateral fixa e uma gaveta clara, sem alterar o comportamento funcional da
extensão.

O símbolo do Horus será exclusivamente `util/logo_recod.png` dentro de um escudo. A
gaveta deverá conter:

- cabeçalho com nome Horus, escudo, ajuda `?` e botão de fechar;
- estado explícito **Aguardando gatilho**;
- botão **Analisar agora** para ignorar a pré-triagem local;
- carregamento por etapas, sem porcentagem artificial;
- probabilidade de Scam em destaque;
- cartões das mensagens e dos trechos pró-Scam;
- ação **Ver na conversa** para localizar o balão correspondente;
- logs, dispositivo, quantidade de balões e exportação TXT recolhidos em
  **Detalhes técnicos**.

## Estados esperados

### Minimizado

- aba fixa na lateral direita, não arrastável;
- escudo com o logo Recod;
- indicador cinza enquanto aguarda, azul durante processamento, vermelho para Scam,
  verde para Ham e laranja para erro;
- resultado Ham atualiza apenas o indicador, sem abrir a gaveta automaticamente.

### Aguardando gatilho

- informar que somente a conversa aberta está sendo observada;
- explicar que a IA ainda não foi executada;
- mostrar exemplos de sinais locais, como pagamento, senha, urgência e links;
- esclarecer que o gatilho não é uma confirmação de fraude;
- permitir início manual por **Analisar agora**.

### Analisando

Exibir as etapas reais:

1. conversa preparada;
2. classificação de risco com BGE e Transformer;
3. localização das mensagens e dos trechos influentes, somente quando necessário.

### Scam

- abrir a gaveta automaticamente;
- mostrar probabilidade de Scam;
- contornar em vermelho somente os balões pró-Scam;
- realçar somente trechos pró-Scam com correspondência exata;
- listar evidências com ação **Ver na conversa**;
- avisar que os destaques representam sinais aprendidos pelo modelo e não prova factual
  de fraude.

### Ham

- mostrar baixo risco estimado quando a gaveta for aberta manualmente;
- não contornar balões e não realçar trechos;
- não abrir a gaveta automaticamente.

### Erro

- informar falha de conexão ou processamento em linguagem simples;
- oferecer **Tentar novamente**;
- manter os detalhes técnicos recolhidos.

### Resultado desatualizado

- quando novas mensagens chegarem, manter marcações anteriores esmaecidas;
- informar que a explicação pertence a uma versão anterior;
- disponibilizar **Atualizar análise**.

## Ajuda `?`

O botão deverá explicar de onde veio o resultado:

1. **Gatilho local:** identifica sinais iniciais sem GPU.
2. **Classificação por IA:** analisa a conversa completa formatada em turnos.
3. **Explicabilidade:** encontra mensagens e trechos que influenciaram a decisão.
4. **Resultado:** apresenta risco estimado e evidências ao usuário.

A ajuda deverá repetir que influência no modelo não equivale a comprovação factual de
fraude.

## Regras funcionais que não podem mudar

A nova interface será apenas uma consumidora do estado produzido pela implementação da
`main`. As rotinas abaixo não poderão ser reescritas durante a troca visual:

- identificação da conversa após clique na lista lateral;
- seleção e leitura dos balões do WhatsApp;
- distinção entre `Suspect` e `Innocent`;
- geração e preservação dos IDs dos balões;
- estabilização do DOM após troca de conversa;
- ordenação cronológica e deduplicação;
- agrupamento dos balões em turnos;
- memória temporária por conversa;
- gatilhos lexicais;
- formatação do histórico enviado à API;
- chamadas de classificação e explicabilidade;
- aplicação e restauração das marcações nos balões.

O código visual deverá ficar separado da captura. Uma exceção na interface nunca poderá
interromper o `MutationObserver`, a memória ou a comunicação com a API.

## Memória e privacidade

- nenhuma conversa será armazenada apenas por carregar o WhatsApp Web;
- somente conversas explicitamente abertas pelo usuário poderão entrar na memória;
- não haverá histórico persistente;
- trocar de conversa restaurará apenas o estado temporário daquela conversa;
- recarregar a aba apagará toda a memória da extensão.

## Critérios obrigatórios antes de substituir a interface antiga

1. Executar a `main` e a nova interface sobre as mesmas conversas reais.
2. Confirmar igualdade dos IDs, textos, interlocutores, ordem e quantidade dos balões.
3. Confirmar igualdade dos turnos enviados para a API.
4. Confirmar igualdade da classificação, probabilidade e referências da explicação.
5. Testar conversa já aberta, troca de conversa, reload, rolagem e chegada de mensagens.
6. Testar balões consecutivos do mesmo interlocutor e mensagens antigas carregadas por
   virtualização.
7. Confirmar que abrir, fechar ou usar a ajuda da interface não altera a memória.
8. Validar manualmente no WhatsApp Web; um DOM simulado não será aceito como evidência
   suficiente.
9. Manter a interface antiga disponível como rollback até a validação completa.

Somente após todos esses itens passarem a nova interface poderá ser integrada à
`main`.
