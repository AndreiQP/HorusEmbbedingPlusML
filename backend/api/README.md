# API online BGE + Transformer

API Flask que serve o modelo operacional Horus: BGE-M3 para codificação de cada turno e
o Transformer finalista da seed 42 para classificação da conversa. Quando a previsão é
Scam, a mesma instância produz explicabilidade hierárquica por perturbação.

## Arquitetura

O processo carrega uma única vez:

1. `SentenceTransformer("BAAI/bge-m3")`;
2. o checkpoint finalista BGE da seed 42;
3. `HierarchicalPerturbationExplainer`.

Cada requisição codifica os turnos originais, executa o Transformer e retorna a
probabilidade de Scam. Explicações usam:

- top 6 turnos por impacto absoluto;
- n-grams de 1 a 5 palavras;
- até 100 turnos da conversa;
- `dangerousSpan` como o maior trecho com `deltaLogit > 1e-4` em direção pró-Scam.

O campo `span` original é mantido para compatibilidade e pode representar a maior
magnitude em qualquer direção. `dangerousSpan` é o campo correto para uma marcação
visual de perigo.

## Dispositivo e checkpoint

Variáveis aceitas:

| Variável | Padrão | Uso |
|---|---|---|
| `HORUS_DEVICE` | `auto` | `auto`, `cuda` ou `cpu`. CUDA é preferida e CPU é fallback. |
| `HORUS_BGE_CHECKPOINT_ROOT` | busca padrão do projeto | Raiz dos resultados do Transformer. |
| `HORUS_BGE_MODEL` | `BAAI/bge-m3` | Nome ou caminho local do encoder. |
| `HORUS_EXPLANATION_TTL_SECONDS` | `1800` | Tempo de retenção dos jobs concluídos em memória. |
| `HF_HUB_OFFLINE` | não definido | Use `1` no cluster sem acesso ao Hugging Face Hub. |
| `TRANSFORMERS_OFFLINE` | não definido | Use `1` junto com o cache local do checkpoint. |

Se CUDA não estiver disponível na inicialização, a API usa CPU. Se ocorrer uma falha
CUDA durante inferência, o runtime é reconstruído em CPU e a operação é repetida.

## Endpoints

### `GET /api/health`

```json
{
  "status": "ok",
  "modelLoaded": true,
  "device": "cuda"
}
```

### `POST /api/analyse_history`

Payload recomendado:

```json
{
  "apiVersion": "v1",
  "analysisId": "uuid-da-analise",
  "chatId": "identidade-da-conversa",
  "conversationVersion": 12,
  "trigger": {
    "category": "request_payment",
    "matchedText": "pix"
  },
  "turns": [
    {
      "turnId": "turn:balloon-1",
      "speaker": "Suspect",
      "balloons": [
        {"balloonId": "balloon-1", "text": "Faça um pix agora"}
      ]
    },
    {
      "turnId": "turn:balloon-2",
      "speaker": "Innocent",
      "balloons": [
        {"balloonId": "balloon-2", "text": "Por quê?"}
      ]
    }
  ]
}
```

`turnId` e `balloonId` devem ser únicos. `speaker` aceita somente `Suspect` ou
`Innocent`. Balões do mesmo turno são unidos com um espaço e os últimos 100 turnos são
enviados ao modelo.

Resposta imediata:

```json
{
  "apiVersion": "v1",
  "analysisId": "uuid-da-analise",
  "chatId": "identidade-da-conversa",
  "conversationVersion": 12,
  "probabilidade": 0.94,
  "isScam": true,
  "model": {
    "name": "bge-transformer",
    "seed": 42,
    "device": "cuda"
  },
  "explanation": {
    "status": "queued",
    "jobId": "uuid-do-job",
    "pollUrl": "/api/explanations/uuid-do-job",
    "reused": false
  }
}
```

Para Ham, `explanation.status` será `not_requested` e nenhum job será criado.

### `GET /api/explanations/<job_id>`

Enquanto processa, responde HTTP 202 com `queued` ou `running`. Ao concluir, responde
HTTP 200 e inclui `baseline` e `items`. Cada item contém:

- `message.balloonIds`: balões pertencentes ao turno macro;
- `message.direction`: `scam`, `ham` ou `neutral`;
- `span`: trecho de maior magnitude, mantido para compatibilidade;
- `dangerousSpan`: maior trecho estritamente pró-Scam;
- `dangerousSpan.balloonReferences`: ID do balão, offsets locais e texto exato.

Exemplo reduzido:

```json
{
  "status": "completed",
  "items": [
    {
      "rank": 1,
      "message": {
        "balloonIds": ["balloon-1"],
        "direction": "scam",
        "deltaLogit": 1.21
      },
      "dangerousSpan": {
        "text": "pix agora",
        "direction": "scam",
        "deltaLogit": 0.73,
        "balloonReferences": [
          {
            "balloonId": "balloon-1",
            "startChar": 9,
            "endChar": 18,
            "text": "pix agora"
          }
        ]
      }
    }
  ]
}
```

Jobs inexistentes ou expirados retornam 404. Falhas do worker retornam HTTP 500.

`POST /api/analyse_message` está descontinuado e retorna HTTP 410, pois a pré-análise
lexical é executada pela extensão.

## Fila, memória e reutilização

A explicabilidade usa um worker por processo para evitar várias análises pesadas
competindo pela mesma GPU. Jobs e resultados existem somente na memória da API.

Conversas idênticas são deduplicadas por um hash que inclui conteúdo, IDs, versão,
seed e configuração. Requisições repetidas reutilizam o mesmo trabalho, mas cada
solicitação recebe um `jobId` e `analysisId` próprios. O cache expira pelo TTL.

Reiniciar a API remove fila, aliases e resultados. Não há banco de dados nem histórico
persistente.

## Execução e testes

```bash
export HORUS_DEVICE=auto
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u -m backend.api.main
```

A API escuta em `0.0.0.0:5000`. O procedimento completo para GPU e túnel está em
[`../../starter.md`](../../starter.md).

```bash
python -m pytest backend/tests/test_online_explainability.py -q
```
