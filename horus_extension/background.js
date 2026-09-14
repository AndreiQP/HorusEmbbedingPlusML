console.log("🛡️ IC Horus: Background Service Worker iniciado com sucesso!");

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    
    if (request.action === "chamarAPI") {
        console.log(`[Background] 📡 Enviando para: http://localhost:5000${request.endpoint}`);
        
        fetch(`http://localhost:5000${request.endpoint}`, {
            method: request.method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request.dados)
        })
        .then(async (response) => {
            // Verifica o tipo de arquivo que o Python enviou de volta
            const contentType = response.headers.get("content-type");
            
            if (contentType && contentType.includes("application/json")) {
                return response.json(); // Tudo certo, é JSON!
            } else {
                // Opa, não é JSON. Vamos ler como texto/HTML para ver o erro.
                const textoErro = await response.text();
                console.error(`[Background] 🚨 O Servidor devolveu HTML/Texto em vez de JSON! Status: ${response.status}`);
                console.error(`[Background] 🚨 Conteúdo recebido (primeiros 200 caracteres):\n`, textoErro.substring(0, 200));
                throw new Error(`Erro do Servidor (Status ${response.status}). Olhe o terminal do Python!`);
            }
        })
        .then(data => {
            console.log(`[Background] 🧠 Resposta da IA:`, data);
            sendResponse({ sucesso: true, dados: data });
        })
        .catch(erro => {
            console.error("[Background] ❌ Falha na requisição:", erro.message);
            sendResponse({ sucesso: false, erro: erro.message });
        });

        return true; 
    }
});