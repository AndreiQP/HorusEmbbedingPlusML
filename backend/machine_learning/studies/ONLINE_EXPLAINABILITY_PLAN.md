# Plano de explicabilidade online BGE

## Objetivo

Integrar a extensão do WhatsApp Web ao Transformer BGE finalista seed 42 em três
níveis: pré-análise lexical local, classificação profunda e explicabilidade
hierárquica assíncrona apenas para decisões Scam.

## Fluxo

1. A extensão mantém, somente durante a vida da aba, os `data-id` e textos
   originais dos balões observados em cada chat.
2. Uma cópia temporária em minúsculas do novo turno `Suspect` é comparada por
   substring com gatilhos em português e inglês. O texto original nunca é
   alterado.
3. Quando há gatilho, ou quando o usuário força a análise, a extensão envia os
   turnos estruturados. Cada turno contém seus balões com `balloonId` e texto.
4. A API concatena os balões deterministicamente, calcula a posição de cada um
   dentro do turno e mantém o mapeamento `modelIndex -> turnId -> balloonIds`.
5. O BGE-M3 e o Transformer seed 42 classificam os últimos 100 turnos. CUDA é
   preferida, com fallback automático para CPU.
6. Se a decisão for Scam, um job em memória neutraliza as mensagens, seleciona
   até seis turnos e encontra o n-gram mais influente de cada um.
7. Os offsets do n-gram são convertidos para offsets locais dos balões. A API
   devolve `balloonReferences`, sem tentar reencontrar mensagens por texto.

## Contrato essencial

Requisição:

```json
{
  "apiVersion": "v1",
  "analysisId": "uuid",
  "chatId": "chat",
  "conversationVersion": 18,
  "turns": [
    {
      "turnId": "turn:balloon-10",
      "speaker": "Suspect",
      "balloons": [
        {"balloonId": "balloon-10", "text": "Your account was blocked."},
        {"balloonId": "balloon-11", "text": "Click this link now."}
      ]
    }
  ]
}
```

Resposta de explicabilidade:

```json
{
  "analysisId": "uuid",
  "conversationVersion": 18,
  "status": "completed",
  "items": [
    {
      "message": {
        "modelIndex": 4,
        "turnId": "turn:balloon-10",
        "balloonIds": ["balloon-10", "balloon-11"]
      },
      "span": {
        "text": "Click this link",
        "balloonReferences": [
          {
            "balloonId": "balloon-11",
            "startChar": 0,
            "endChar": 15,
            "text": "Click this link"
          }
        ]
      }
    }
  ]
}
```

## Estado e execução

- A extensão usa memória JavaScript por aba; não usa `chrome.storage` para
  conversas e perde tudo ao recarregar ou fechar a aba.
- A API usa uma fila FIFO com uma explicação por vez para evitar concorrência de
  BGE na GPU ou CPU.
- Jobs idênticos podem reutilizar o mesmo trabalho por 30 minutos, mas cada
  requisição conserva seu próprio `analysisId`.
- O cache e a fila existem apenas em RAM e desaparecem ao reiniciar a API.
- O front valida `chatId`, `analysisId` e `conversationVersion` antes de exibir
  um resultado.

## Critérios de aceite

- IDs nunca são derivados do texto.
- Textos idênticos em balões distintos continuam distinguíveis.
- Um turno retorna todos os balões que o compõem.
- Um n-gram que cruza balões retorna uma referência para cada balão afetado.
- A ausência de gatilho não é apresentada como decisão Ham.
- Ham não gera explicação; Scam retorna no máximo seis turnos rastreáveis.
- CUDA saudável é usada e a indisponibilidade ou falha de CUDA aciona CPU.
