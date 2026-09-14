// =====================================================================
// ESTADO GLOBAL (Memória Dinâmica Segura)
// =====================================================================
const logOriginal = console.log;

let baseDeDadosMensagens = {};
let ordemCronologicaGlobal = [];
let chatAtivoId = null; 
let ultimoTurnoAnalisadoId = null; 

// =====================================================================
// SISTEMA DE DEBUG VISUAL (Terminal Horus)
// =====================================================================
function horusLog(msg) {
    try { logOriginal("[IC Horus] " + msg); } catch(e){} 
    
    const terminal = document.getElementById('ic-horus-terminal');
    if (terminal) {
        const linha = document.createElement('div');
        linha.innerText = "> " + msg;
        terminal.appendChild(linha);
        terminal.scrollTop = terminal.scrollHeight;
    }
}

// =====================================================================
// EXTRAÇÃO E HIGIENIZAÇÃO DO TEXTO
// =====================================================================
function extrairTextoLimpo(node) {
    const clone = node.cloneNode(true);
    
    // 1. Remove Citações
    const citacoes = clone.querySelectorAll('span[data-testid="quoted-message"]');
    citacoes.forEach(citacao => {
        const blockPai = citacao.closest('div');
        if (blockPai) blockPai.remove();
        else citacao.remove();
    });

    // 2. Remove botões nativos do WPP (ex: botão de contatos, ler mais)
    const botoes = clone.querySelectorAll('[role="button"]');
    botoes.forEach(b => b.remove());

    const selectors = ['span.copyable-text', 'span.selectable-text', 'div[dir="ltr"]', 'span[dir="ltr"]'];
    let textoFinal = "";

    for (const sel of selectors) {
        const spansTexto = clone.querySelectorAll(sel);
        if (spansTexto && spansTexto.length > 0) {
            spansTexto.forEach(s => {
                if (s && s.innerText) textoFinal += s.innerText + " ";
            });
            if (textoFinal.trim().length > 0) {
                textoFinal = textoFinal.trim();
                break;
            }
        }
    }

    if (!textoFinal) {
        const raw = clone.textContent ? clone.textContent.trim() : "";
        if (raw && raw.length > 0) {
            textoFinal = raw.replace(/^\[.*?\]\s*/, '').trim();
        } else {
            const pre = clone.querySelector('[data-pre-plain-text]');
            if (pre) {
                textoFinal = pre.parentElement ? pre.parentElement.textContent.replace(pre.getAttribute('data-pre-plain-text') || '', '').trim() : '';
            }
        }
    }

    // 3. FILTRO DE HIGIENIZAÇÃO (Remove lixos do sistema WhatsApp)
    if (textoFinal) {
        const lixosDoWhatsApp = [
            "clique para mostrar os dados do contato",
            "clique para mostrar os dados do contacto",
            "esta mensagem foi apagada",
            "mensagem apagada",
            "leia mais",
            "read more"
        ];
        
        lixosDoWhatsApp.forEach(lixo => {
            // Remove as frases ignorando letras maiúsculas/minúsculas (gi)
            const regex = new RegExp(lixo, "gi");
            textoFinal = textoFinal.replace(regex, "");
        });
        
        // Remove espaços duplicados que podem ter sobrado e limpa as bordas
        textoFinal = textoFinal.replace(/\s+/g, ' ').trim();
    }

    return textoFinal;
}

// =====================================================================
// GERADOR DE HISTÓRICO (Agrupa Turnos)
// =====================================================================
function obterHistoricoAninhado() {
    let textoCompilado = "";
    let remetenteAtual = null;
    let mensagensDoTurno = [];

    ordemCronologicaGlobal.forEach((id) => {
        const msg = baseDeDadosMensagens[id];
        if (!msg) return;

        if (remetenteAtual === null) {
            remetenteAtual = msg.remetente;
            mensagensDoTurno.push(msg.texto);
        } else if (remetenteAtual === msg.remetente) {
            mensagensDoTurno.push(msg.texto);
        } else {
            textoCompilado += `${remetenteAtual}: ${mensagensDoTurno.join(' ')}. `;
            remetenteAtual = msg.remetente;
            mensagensDoTurno = [msg.texto];
        }
    });
    
    if (remetenteAtual !== null) {
        textoCompilado += `${remetenteAtual}: ${mensagensDoTurno.join(' ')}.`;
    }
    
    return textoCompilado.trim();
}

// =====================================================================
// FLUXO DE INTELIGÊNCIA ARTIFICIAL
// =====================================================================
function enviarHistoricoParaIA() {
    const historico = obterHistoricoAninhado();
    if (!historico) return;

    horusLog(`Enviando contexto profundo para o Background...`);
    
    chrome.runtime.sendMessage({
        action: "chamarAPI",
        endpoint: "/api/analyse_history",
        method: "POST",
        dados: { historico: historico, contato: chatAtivoId }
    }, (resposta) => {
        if (chrome.runtime.lastError) return horusLog(`❌ Erro Crítico: ${chrome.runtime.lastError.message}`);
        if (!resposta || !resposta.sucesso) return horusLog(`❌ Erro IA (Fase 2).`);

        const data = resposta.dados;
        if (data.isScam) {
            horusLog(`🚨 ALERTA: Scam Confirmado! (${(data.probabilidade * 100).toFixed(1)}%)`);
            const toggleBtn = document.getElementById('ic-horus-toggle');
            if (toggleBtn) {
                toggleBtn.style.backgroundColor = '#e74c3c';
                toggleBtn.style.borderColor = '#c0392b';
                toggleBtn.style.color = '#fff';
            }
        } else {
            horusLog(`✅ Seguro: ${(data.probabilidade * 100).toFixed(1)}% Scam`);
        }
    });
}

function analisarNovaMensagem(textoTurno) {
    horusLog(`Analisando Fase 1: "${textoTurno.substring(0, 20)}..."`);
    
    chrome.runtime.sendMessage({
        action: "chamarAPI",
        endpoint: "/api/analyse_message",
        method: "POST",
        dados: { mensagem: textoTurno, contato: chatAtivoId }
    }, (resposta) => {
        if (chrome.runtime.lastError) return;
        if (!resposta || !resposta.sucesso) return;

        const data = resposta.dados;
        const prob = (data.probabilidade * 100).toFixed(1);
        horusLog(`Risco Fase 1: ${prob}%`);

        if (data.probabilidade > 0.40) {
            horusLog(`⚠️ Risco alto! Acionando Fase 2...`);
            enviarHistoricoParaIA();
        }
    });
}

// =====================================================================
// NÚCLEO DO SISTEMA (Radar Dinâmico Contínuo)
// =====================================================================
function atualizarPainel() {
    const contador = document.getElementById('ic-horus-contador');
    const chatStatus = document.getElementById('ic-horus-chatid');
    const toggleBtn = document.getElementById('ic-horus-toggle');
    
    if (contador) contador.innerText = `Memória: ${ordemCronologicaGlobal.length} msgs`;
    if (chatStatus) chatStatus.innerText = chatAtivoId ? `Contato: ${chatAtivoId}` : "Contato: Nenhum";
    
    if (toggleBtn && ordemCronologicaGlobal.length > 0 && toggleBtn.style.backgroundColor !== 'rgb(231, 76, 60)') {
        toggleBtn.style.boxShadow = "0px 0px 10px #25D366";
        setTimeout(() => toggleBtn.style.boxShadow = "0px 4px 6px rgba(0,0,0,0.4)", 500);
    }
}

function sincronizarMemoria() {
    if (!document.getElementById('ic-horus-panel')) injetarPainel();
    
    const mainEl = document.getElementById('main');
    
    if (!mainEl) {
        if (chatAtivoId !== null) { chatAtivoId = null; atualizarPainel(); }
        return; 
    }

    const cabecalhoNome = mainEl.querySelector('header span[dir="auto"][title], header span[dir="auto"]');
    if (!cabecalhoNome) return;

    const chatDetectado = cabecalhoNome.innerText.trim();
    if (chatDetectado && chatDetectado !== chatAtivoId) {
        horusLog(`🔄 Trocando contato para: '${chatDetectado}'`);
        chatAtivoId = chatDetectado;
        ordemCronologicaGlobal = []; 
        baseDeDadosMensagens = {};   
        ultimoTurnoAnalisadoId = null; 
        
        const toggleBtn = document.getElementById('ic-horus-toggle');
        if (toggleBtn) {
            toggleBtn.style.backgroundColor = 'rgba(20, 20, 20, 0.95)';
            toggleBtn.style.borderColor = '#25D366';
        }
        atualizarPainel();
    }

    const spansMensagem = mainEl.querySelectorAll('span[data-testid="selectable-text"], span.copyable-text');
    if (spansMensagem.length === 0) return;

    const mainRect = mainEl.getBoundingClientRect();
    const midX = mainRect.left + (mainRect.width / 2);
    
    let idsVisiveisNaTela = [];

    spansMensagem.forEach((span) => {
        if (span.closest('[data-testid="quoted-message"]')) return;

        const texto = extrairTextoLimpo(span.closest('div[data-id]') || span);
        if (!texto || texto.length === 0) return;

        let remetente = "Unknown";
        const spanRect = span.getBoundingClientRect();
        if (spanRect.width > 0) {
            const centroDoSpan = spanRect.left + (spanRect.width / 2);
            remetente = (centroDoSpan < midX) ? "Suspect" : "Innocent";
        }
        if (remetente === "Unknown") return;

        const divComId = span.closest('div[data-id]');
        const idWpp = divComId ? divComId.getAttribute('data-id') : null;
        const idUnico = idWpp ? `${idWpp}_${texto.substring(0, 15)}` : `HASH_${remetente}_${texto.replace(/\s/g, '').substring(0, 20)}`;

        if (!baseDeDadosMensagens[idUnico]) {
            baseDeDadosMensagens[idUnico] = { remetente, texto };
        }
        
        if (!idsVisiveisNaTela.includes(idUnico)) {
            idsVisiveisNaTela.push(idUnico);
        }
    });

    if (idsVisiveisNaTela.length === 0) return;

    // Sincronização e Scroll Perfeito
    if (ordemCronologicaGlobal.length === 0) {
        ordemCronologicaGlobal = [...idsVisiveisNaTela];
    } else {
        const primeiroMatch = idsVisiveisNaTela.find(id => ordemCronologicaGlobal.includes(id));
        const ultimoMatch = idsVisiveisNaTela.slice().reverse().find(id => ordemCronologicaGlobal.includes(id));

        if (primeiroMatch) {
            const idxVisivelPrimeiro = idsVisiveisNaTela.indexOf(primeiroMatch);
            const mensagensAnterioresNovas = idsVisiveisNaTela.slice(0, idxVisivelPrimeiro);
            ordemCronologicaGlobal = [...mensagensAnterioresNovas, ...ordemCronologicaGlobal];
        } 
        if (ultimoMatch) {
            const idxVisivelUltimo = idsVisiveisNaTela.indexOf(ultimoMatch);
            const mensagensFuturasNovas = idsVisiveisNaTela.slice(idxVisivelUltimo + 1);
            const futurasFiltradas = mensagensFuturasNovas.filter(id => !ordemCronologicaGlobal.includes(id));
            ordemCronologicaGlobal = [...ordemCronologicaGlobal, ...futurasFiltradas];
        }
        if (!primeiroMatch && !ultimoMatch) {
            const novosIds = idsVisiveisNaTela.filter(id => !ordemCronologicaGlobal.includes(id));
            ordemCronologicaGlobal = [...ordemCronologicaGlobal, ...novosIds];
        }
    }

    // Gatilho Inteligente da IA
    if (ordemCronologicaGlobal.length > 0) {
        const ultimoId = ordemCronologicaGlobal[ordemCronologicaGlobal.length - 1];
        
        if (ultimoId !== ultimoTurnoAnalisadoId) {
            ultimoTurnoAnalisadoId = ultimoId; 
            const msg = baseDeDadosMensagens[ultimoId];
            
            if (msg && msg.remetente === "Suspect") {
                let turnoTextos = [];
                for (let i = ordemCronologicaGlobal.length - 1; i >= 0; i--) {
                    const m = baseDeDadosMensagens[ordemCronologicaGlobal[i]];
                    if (m.remetente === "Suspect") turnoTextos.unshift(m.texto);
                    else break; 
                }
                
                const textoTurno = turnoTextos.join(" ");
                analisarNovaMensagem(textoTurno);
            }
        }
    }

    atualizarPainel();
}

setInterval(sincronizarMemoria, 1500);

// =====================================================================
// EXPORTAÇÃO MANUAL
// =====================================================================
function gerarDataset() {
    if (ordemCronologicaGlobal.length === 0) return alert("A memória está vazia.");
    
    const textoCompilado = obterHistoricoAninhado();
    
    const blob = new Blob([textoCompilado], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `Dataset_${chatAtivoId.replace(/\s+/g, '_')}_${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    horusLog("✅ Dataset dinâmico descarregado!");
}


// =====================================================================
// INTERFACE VISUAL WIDGET 
// =====================================================================
function injetarPainel() {
    if (document.getElementById('ic-horus-panel')) return;

    const painel = document.createElement('div');
    painel.id = 'ic-horus-panel';
    painel.style.cssText = `
        position: fixed; top: 150px; left: 15px; z-index: 10000; 
        display: flex; flex-direction: row; align-items: flex-start; gap: 10px;
        font-family: sans-serif;
    `;
    
    const toggleBtn = document.createElement('div');
    toggleBtn.id = 'ic-horus-toggle';
    toggleBtn.style.cssText = `
        width: 44px; height: 44px; background-color: rgba(20, 20, 20, 0.95); 
        border: 2px solid #25D366; border-radius: 50%; 
        display: flex; justify-content: center; align-items: center; 
        cursor: grab; box-shadow: 0px 4px 6px rgba(0,0,0,0.4);
        font-size: 20px; user-select: none; transition: background-color 0.3s, border-color 0.3s;
    `;
    toggleBtn.innerHTML = '👁️';

    const bodyPanel = document.createElement('div');
    bodyPanel.id = 'ic-horus-body';
    bodyPanel.style.cssText = `
        background-color: rgba(20, 20, 20, 0.95); border: 2px solid #25D366; 
        border-radius: 10px; padding: 15px; display: none; flex-direction: column; gap: 10px;
        width: 260px; box-shadow: 0px 5px 15px rgba(0,0,0,0.5); font-family: monospace;
    `;

    const statusDiv = document.createElement('div');
    statusDiv.innerHTML = `
        <strong style="font-family:sans-serif; font-size:15px; color:#fff;">IC Horus - Scan Dinâmico</strong><br>
        <span id="ic-horus-chatid" style="color: #f39c12; font-size: 12px; font-weight: bold; word-break: break-all;">Contato: Aguardando...</span><br>
        <span id="ic-horus-contador" style="color: #25D366; font-size: 13px; font-weight: bold;">Memória: 0 msgs</span>
    `;
    bodyPanel.appendChild(statusDiv);

    const terminal = document.createElement('div');
    terminal.id = 'ic-horus-terminal';
    terminal.style.cssText = `
        height: 100px; overflow-y: auto; background: #000; color: #0f0; 
        font-size: 11px; padding: 5px; border-radius: 5px; border: 1px solid #333;
    `;
    bodyPanel.appendChild(terminal);

    const divBotoes = document.createElement('div');
    divBotoes.style.display = "flex";
    divBotoes.style.gap = "5px";

    const btnAnalisar = document.createElement('button');
    btnAnalisar.innerText = '🔍 Forçar IA';
    btnAnalisar.style.cssText = `flex:1; background: #3498db; color: white; border: none; padding: 8px; border-radius: 5px; cursor: pointer; font-weight: bold;`;
    btnAnalisar.onclick = enviarHistoricoParaIA;

    const btnBaixar = document.createElement('button');
    btnBaixar.innerText = '📥 Baixar TXT';
    btnBaixar.style.cssText = `flex:1; background: #25D366; color: black; border: none; padding: 8px; border-radius: 5px; cursor: pointer; font-weight: bold;`;
    btnBaixar.onclick = gerarDataset;

    divBotoes.appendChild(btnAnalisar);
    divBotoes.appendChild(btnBaixar);
    bodyPanel.appendChild(divBotoes);

    painel.appendChild(toggleBtn);
    painel.appendChild(bodyPanel);
    document.body.appendChild(painel);

    let isDragging = false, hasMoved = false, dragStart = { x: 0, y: 0 }, isAberto = false;

    toggleBtn.onmousedown = (e) => {
        isDragging = true; hasMoved = false;
        const rect = painel.getBoundingClientRect();
        dragStart.x = e.clientX - rect.left; dragStart.y = e.clientY - rect.top;
        toggleBtn.style.cursor = 'grabbing';
    };

    document.onmousemove = (e) => {
        if (!isDragging) return;
        hasMoved = true; 
        painel.style.left = (e.clientX - dragStart.x) + 'px';
        painel.style.top = (e.clientY - dragStart.y) + 'px';
        painel.style.bottom = 'auto'; painel.style.right = 'auto';
    };

    document.onmouseup = () => {
        if (isDragging) { isDragging = false; toggleBtn.style.cursor = 'grab'; }
    };

    toggleBtn.onclick = (e) => {
        if (hasMoved) return; 
        isAberto = !isAberto;
        bodyPanel.style.display = isAberto ? 'flex' : 'none';
        
        if (isAberto && toggleBtn.style.backgroundColor !== 'rgb(231, 76, 60)') {
            toggleBtn.style.backgroundColor = '#25D366';
            toggleBtn.style.color = '#fff';
        } else if (!isAberto && toggleBtn.style.backgroundColor !== 'rgb(231, 76, 60)') {
            toggleBtn.style.backgroundColor = 'rgba(20, 20, 20, 0.95)';
        }
    };
}

injetarPainel();