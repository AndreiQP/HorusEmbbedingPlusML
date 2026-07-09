# Guia de Debug - IC Horus

## ✅ Passos para Resolver o Problema

### 1. **Reiniciar o Servidor Node.js** (PORTA ALTERADA PARA 5000)
```bash
cd backend
node ../server.js
# ou
npm start
```
**Aguarde a mensagem**: `🛡️ Servidor Backend IC Horus rodando na porta 5000...`

### 2. **Recarregar a Extensão no Chrome**
- Abra `chrome://extensions/`
- Encontre "IC Horus"
- Clique no botão **Recarregar** (ícone circular)

### 3. **Abrir WhatsApp Web**
- Vá para `https://web.whatsapp.com`
- Abra um chat
- Procure o painel do **IC Horus** (ícone 👁️ verde na esquerda)

---

## 🔍 Verificações de Debug

### Verificar se as mensagens estão sendo carregadas:

**Abra o Console (F12) e procure por:**
- ✅ `[IC Horus]` - Logs da extensão
- ✅ `Conectado! Aguardando mensagens...` - Extensão iniciada
- ✅ `Memória: X msgs` - Mensagens sendo capturadas

### Se NÃO ver mensagens sendo carregadas:

1. **Verificar seletores CSS no Console:**
```javascript
// Digite no console do WhatsApp Web (F12):
document.querySelectorAll('.message-in, .message-out').length
document.querySelectorAll('div[data-testid*="message"]').length
document.querySelectorAll('.copyable-text').length
```

Se algum retornar um número > 0, o seletor funcionou!

2. **Ver o que foi encontrado:**
```javascript
document.querySelectorAll('.message-in, .message-out')[0]?.innerText
```

---

## 🚨 Problemas Comuns

| Problema | Solução |
|----------|---------|
| Painel não aparece | Recarregar extensão em `chrome://extensions` |
| "Nenhuma mensagem encontrada" | Verificar seletores CSS acima |
| Conexão recusada (localhost:5000) | Servidor não está rodando |
| Servidor retorna erro HTML | Python/Flask não está iniciado |

---

## 📝 Checklist

- [ ] Servidor Node.js rodando na porta 5000
- [ ] Chrome extensão recarregada
- [ ] WhatsApp Web aberto
- [ ] Painel IC Horus visível
- [ ] Console mostrando "Conectado!"
- [ ] Enviar uma mensagem no chat

Se tudo estiver correto, a extensão deve capturar a mensagem em tempo real!

---

## 🔗 URLs Importantes

- **WhatsApp Web**: https://web.whatsapp.com
- **Extensões**: chrome://extensions/
- **DevTools**: F12 (quando em WhatsApp Web)
