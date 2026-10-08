# Extensão Horus para WhatsApp Web

Extensão Chrome/Edge Manifest V3 que observa a conversa aberta no WhatsApp Web,
executa uma pré-análise lexical local e consulta a API BGE quando uma análise profunda
é necessária. Para classificações Scam, a extensão projeta a explicabilidade sobre os
próprios balões da conversa.

## Fluxo

1. O usuário clica em uma conversa na lista lateral do WhatsApp.
2. A extensão passa a acompanhar somente essa conversa e os balões renderizados nela.
3. Balões consecutivos do mesmo interlocutor são agrupados em turnos `Suspect` ou
   `Innocent`, preservando seus IDs individuais.
4. Uma regra lexical em `triggers.js` pode acionar a IA automaticamente. O botão
   **Forçar IA** ignora essa pré-triagem e inicia a análise diretamente.
5. O histórico de até 100 turnos é enviado para `POST /api/analyse_history`.
6. O Transformer retorna a probabilidade de Scam. Se o resultado for positivo, a
   extensão consulta o job assíncrono de explicabilidade.
7. As mensagens pró-Scam recebem contorno vermelho e o trecho perigoso é realçado no
   texto do balão.

Nenhuma conversa é enviada apenas porque o WhatsApp foi aberto. A memória começa após
um clique explícito na conversa e existe somente na aba atual.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `manifest.json` | Permissões, content scripts e service worker. |
| `background.js` | Executa as requisições HTTP fora do contexto da página. |
| `content.js` | Captura balões, mantém sessões, chama a API, atualiza o painel e aplica marcações. |
| `triggers.js` | Regras lexicais da pré-análise local. |
| `annotation_utils.js` | Conversão de offsets Unicode/UTF-16 e mapeamento seguro do texto para o DOM. |
| `options.html` / `options.js` | Configuração da URL da API. |
| `tests/annotation_utils.test.js` | Testes unitários dos offsets e da normalização textual. |

## Memória e identidade

Cada conversa clicada recebe uma sessão em memória com:

- balões conhecidos e sua ordem cronológica;
- versão da conversa, incrementada quando o conteúdo muda;
- última classificação, probabilidade de Scam e dispositivo utilizado;
- último resultado de explicabilidade;
- índice dos balões e trechos que devem ser marcados.

O ID principal é o `data-id` exposto pelo WhatsApp. Quando ele não está disponível, a
extensão cria um ID determinístico a partir de interlocutor, metadados e texto. Esse ID
é enviado à API e retorna em `message.balloonIds` e
`dangerousSpan.balloonReferences`, permitindo reencontrar o mesmo balão no front.

A ordem combina a ordem atual do DOM, balões já conhecidos e o timestamp disponível em
`data-pre-plain-text`. Isso permite inserir mensagens antigas carregadas após rolagem
sem colocá-las depois das mensagens recentes.

Somente elementos dentro de `.message-in` ou `.message-out` são aceitos como balões.
Campos `contenteditable`, o rodapé de composição e mensagens citadas são excluídos, de
modo que rascunhos e estados intermediários de digitação nunca entrem no histórico. Na
troca de conversa, a captura aguarda duas observações estáveis e rejeita a assinatura do
DOM anterior antes de associar mensagens ao novo chat.

## Classificação por conversa

O ícone representa a conversa atualmente aberta:

- `👁️`: ainda não analisada;
- `✅`: Ham;
- `⚠️`: Scam.

O painel mostra a probabilidade de Scam. Se novas mensagens chegarem depois da
classificação, o resultado é identificado como anterior à versão atual até que outra
análise seja executada. Trocar de conversa restaura o estado correto de cada sessão e
uma resposta atrasada da API nunca altera o chat errado.

## Marcação no WhatsApp

Somente evidências pró-Scam são projetadas na conversa:

- o turno influente pode abranger vários balões; todos recebem contorno vermelho;
- apenas os caracteres referenciados por `dangerousSpan` recebem realce;
- marcas atuais usam contorno sólido;
- marcas de uma versão anterior ficam tracejadas e esmaecidas;
- conversas Ham e evidências pró-Ham não recebem marcação.

O realce usa a CSS Highlight API e objetos `Range`. O texto gerenciado pelo React do
WhatsApp não é modificado. Como o WhatsApp virtualiza o histórico, as marcações são
removidas e reaplicadas sempre que o DOM muda ou o usuário rola a conversa.

Offsets retornados pelo Python contam pontos de código Unicode; o navegador usa UTF-16.
`annotation_utils.js` realiza a conversão, valida o texto exato e só cria o destaque
quando a correspondência é inequívoca. Em caso de divergência, o balão continua
contornado, mas nenhum trecho incorreto é realçado.

## Instalação no navegador

1. Abra `chrome://extensions` ou `edge://extensions`.
2. Ative o modo de desenvolvedor.
3. Escolha **Carregar sem compactação** e selecione `horus_extension/`.
4. Abra as opções da extensão e configure a API. Para o túnel descrito em
   [`../starter.md`](../starter.md), use `http://127.0.0.1:5500`.
5. Recarregue a extensão sempre que algum arquivo desta pasta mudar.
6. Recarregue o WhatsApp Web e clique na conversa desejada pela lista lateral.

## Testes

```bash
node --test horus_extension/tests/annotation_utils.test.js
node --check horus_extension/annotation_utils.js
node --check horus_extension/content.js
node --check horus_extension/background.js
node --check horus_extension/options.js
```

## Limitações atuais

- sessões, classificações e marcações são perdidas ao recarregar a página;
- somente mensagens textuais renderizadas pelo WhatsApp podem ser capturadas e marcadas;
- nomes iguais no cabeçalho do WhatsApp ainda podem representar uma identidade de chat
  menos forte que um identificador interno oficial;
- o painel e as marcações representam evidências aprendidas pelo modelo, não uma prova
  factual de fraude.
