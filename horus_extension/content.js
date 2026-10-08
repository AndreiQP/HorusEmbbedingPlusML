// IC Horus — transient WhatsApp memory, lexical gate and BGE explanations.

const originalConsoleLog = console.log.bind(console);
const chatSessions = new Map();
const fallbackIds = new WeakMap();
let activeChatId = null;
let observedSequence = 0;
let observedMain = null;
let mainObserver = null;
let scanScheduled = false;

function sessionFor(chatId) {
    if (!chatSessions.has(chatId)) {
        chatSessions.set(chatId, {
            conversationVersion: 0,
            messages: new Map(),
            chronologicalIds: [],
            lastGateBalloonId: null,
            activeAnalysisId: null,
            lastExplanation: null
        });
    }
    return chatSessions.get(chatId);
}

function horusLog(message) {
    originalConsoleLog(`[IC Horus] ${message}`);
    const terminal = document.getElementById("ic-horus-terminal");
    if (!terminal) return;
    const line = document.createElement("div");
    line.textContent = `> ${message}`;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}

function setAnalysisStatus(text, color = "#f39c12") {
    const status = document.getElementById("ic-horus-analysis-status");
    if (status) {
        status.textContent = text;
        status.style.color = color;
    }
}

function extractCleanText(node) {
    const clone = node.cloneNode(true);
    clone.querySelectorAll('span[data-testid="quoted-message"]').forEach((quoted) => {
        const parent = quoted.closest("div");
        (parent || quoted).remove();
    });
    clone.querySelectorAll('[role="button"]').forEach((button) => button.remove());

    let text = "";
    const selectors = [
        "span.copyable-text",
        "span.selectable-text",
        'div[dir="ltr"]',
        'span[dir="ltr"]'
    ];
    for (const selector of selectors) {
        const candidates = clone.querySelectorAll(selector);
        if (!candidates.length) continue;
        text = Array.from(candidates)
            .map((item) => item.innerText || item.textContent || "")
            .join(" ")
            .trim();
        if (text) break;
    }
    if (!text) text = (clone.textContent || "").replace(/^\[.*?\]\s*/, "").trim();

    const ignored = [
        "clique para mostrar os dados do contato",
        "clique para mostrar os dados do contacto",
        "esta mensagem foi apagada",
        "mensagem apagada",
        "leia mais",
        "read more"
    ];
    for (const phrase of ignored) {
        text = text.replace(new RegExp(phrase, "gi"), "");
    }
    return text.replace(/\s+/g, " ").trim();
}

function fallbackBalloonId(element) {
    if (!fallbackIds.has(element)) {
        fallbackIds.set(element, `session:${crypto.randomUUID()}`);
    }
    return fallbackIds.get(element);
}

function speakerFor(element, mainElement) {
    const textNode = element.matches("span")
        ? element
        : element.querySelector('span[data-testid="selectable-text"], span.copyable-text');
    if (!textNode) return null;
    const rect = textNode.getBoundingClientRect();
    const mainRect = mainElement.getBoundingClientRect();
    if (rect.width <= 0) return null;
    return rect.left + rect.width / 2 < mainRect.left + mainRect.width / 2
        ? "Suspect"
        : "Innocent";
}

function visibleBalloons(mainElement) {
    const found = [];
    const seen = new Set();
    const textNodes = mainElement.querySelectorAll(
        'span[data-testid="selectable-text"], span.copyable-text'
    );
    for (const textNode of textNodes) {
        if (textNode.closest('[data-testid="quoted-message"]')) continue;
        const element = textNode.closest("div[data-id]") || textNode;
        const rawId = element.getAttribute ? element.getAttribute("data-id") : null;
        const balloonId = rawId || fallbackBalloonId(element);
        if (seen.has(balloonId)) continue;
        const text = extractCleanText(element);
        const speaker = speakerFor(element, mainElement);
        if (!text || !speaker) continue;
        seen.add(balloonId);
        found.push({ balloonId, speaker, text });
    }
    return found;
}

function reconcileChronology(state, visibleIds) {
    if (!visibleIds.length) return;
    if (!state.chronologicalIds.length) {
        state.chronologicalIds = [...visibleIds];
        return;
    }
    const firstMatch = visibleIds.find((id) => state.chronologicalIds.includes(id));
    const lastMatch = [...visibleIds].reverse().find((id) => state.chronologicalIds.includes(id));
    if (firstMatch) {
        const previous = visibleIds.slice(0, visibleIds.indexOf(firstMatch));
        state.chronologicalIds = [
            ...previous.filter((id) => !state.chronologicalIds.includes(id)),
            ...state.chronologicalIds
        ];
    }
    if (lastMatch) {
        const following = visibleIds.slice(visibleIds.indexOf(lastMatch) + 1);
        state.chronologicalIds.push(
            ...following.filter((id) => !state.chronologicalIds.includes(id))
        );
    }
    if (!firstMatch && !lastMatch) {
        state.chronologicalIds.push(
            ...visibleIds.filter((id) => !state.chronologicalIds.includes(id))
        );
    }
}

function buildTurns(state) {
    const turns = [];
    let current = null;
    for (const balloonId of state.chronologicalIds) {
        const message = state.messages.get(balloonId);
        if (!message) continue;
        if (!current || current.speaker !== message.speaker) {
            if (current) turns.push(current);
            current = {
                turnId: `turn:${balloonId}`,
                speaker: message.speaker,
                balloons: []
            };
        }
        current.balloons.push({ balloonId, text: message.originalText });
    }
    if (current) turns.push(current);
    return turns.slice(-100);
}

function lastSuspectText(state) {
    const parts = [];
    for (let index = state.chronologicalIds.length - 1; index >= 0; index -= 1) {
        const message = state.messages.get(state.chronologicalIds[index]);
        if (!message || message.speaker !== "Suspect") break;
        parts.unshift(message.originalText);
    }
    return parts.join(" ");
}

function callApi(endpoint, method = "GET", data) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({
            action: "chamarAPI",
            endpoint,
            method,
            dados: data
        }, (response) => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
            } else if (!response || !response.sucesso) {
                reject(new Error(response?.erro || "Falha desconhecida na API"));
            } else {
                resolve(response.dados);
            }
        });
    });
}

function directionLabel(direction) {
    if (direction === "scam") return "Aumentou o risco de Scam";
    if (direction === "ham") return "Reduziu o risco de Scam";
    return "Efeito neutro";
}

function directionColor(direction) {
    if (direction === "scam") return "#e74c3c";
    if (direction === "ham") return "#3498db";
    return "#95a5a6";
}

function renderExplanation(result, stale = false) {
    const container = document.getElementById("ic-horus-explanations");
    if (!container) return;
    container.replaceChildren();

    if (stale) {
        const warning = document.createElement("div");
        warning.textContent = "Esta explicação pertence a uma versão anterior da conversa.";
        warning.style.cssText = "color:#f39c12;font-size:11px;margin-bottom:8px;";
        container.appendChild(warning);
    }

    for (const item of result.items || []) {
        const card = document.createElement("div");
        card.dataset.turnId = item.message.turnId;
        card.dataset.balloonIds = (item.message.balloonIds || []).join(",");
        card.style.cssText = `
            background:#151515;border-left:4px solid ${directionColor(item.span?.direction || item.message.direction)};
            border-radius:5px;padding:8px;margin-bottom:8px;color:#eee;font-size:11px;
        `;

        const heading = document.createElement("div");
        heading.style.cssText = "font-weight:bold;margin-bottom:4px;color:#fff;";
        heading.textContent = `#${item.rank} ${item.message.speaker} — ${directionLabel(item.span?.direction || item.message.direction)}`;

        const message = document.createElement("div");
        message.style.cssText = "color:#bbb;margin-bottom:5px;white-space:normal;";
        message.textContent = item.message.text;

        const span = document.createElement("div");
        span.style.cssText = "font-weight:bold;color:#fff;margin-bottom:5px;";
        span.textContent = item.span ? `Trecho: “${item.span.text}”` : "Trecho indisponível";

        const impactTrack = document.createElement("div");
        impactTrack.style.cssText = "height:5px;background:#333;border-radius:3px;overflow:hidden;";
        const impact = document.createElement("div");
        impact.style.cssText = `height:100%;width:${Math.round(100 * (item.span?.relativeImpact || item.message.relativeImpact || 0))}%;background:${directionColor(item.span?.direction || item.message.direction)};`;
        impactTrack.appendChild(impact);

        card.append(heading, message, span, impactTrack);
        container.appendChild(card);
    }
}

async function pollExplanation(pollUrl, context) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < 10 * 60 * 1000) {
        await new Promise((resolve) => setTimeout(resolve, Date.now() - startedAt < 30000 ? 2000 : 5000));
        let result;
        try {
            result = await callApi(pollUrl, "GET");
        } catch (error) {
            horusLog(`Erro consultando explicação: ${error.message}`);
            setAnalysisStatus("Falha ao consultar a explicação", "#e74c3c");
            return;
        }
        if (result.status === "queued" || result.status === "running") continue;
        if (result.status === "failed") {
            horusLog(`Explicação falhou: ${result.error || "erro desconhecido"}`);
            setAnalysisStatus("Não foi possível produzir a explicação", "#e74c3c");
            return;
        }

        const state = chatSessions.get(context.chatId);
        if (!state || result.analysisId !== context.analysisId) return;
        state.lastExplanation = result;
        const stale = state.conversationVersion !== context.conversationVersion;
        renderExplanation(result, stale);
        setAnalysisStatus(
            stale ? "Explicação concluída para uma versão anterior" : "Explicação concluída",
            stale ? "#f39c12" : "#2ecc71"
        );
        return;
    }
    setAnalysisStatus("A explicação excedeu o tempo de espera", "#e74c3c");
}

async function sendDeepAnalysis(trigger = { category: "manual", matchedText: "manual" }) {
    if (!activeChatId) return;
    const state = sessionFor(activeChatId);
    const turns = buildTurns(state);
    if (!turns.length) return;

    const context = {
        analysisId: crypto.randomUUID(),
        chatId: activeChatId,
        conversationVersion: state.conversationVersion
    };
    state.activeAnalysisId = context.analysisId;
    setAnalysisStatus("Executando BGE + Transformer...", "#f39c12");
    horusLog(`Análise profunda acionada (${trigger.category}).`);

    let result;
    try {
        result = await callApi("/api/analyse_history", "POST", {
            apiVersion: "v1",
            ...context,
            trigger,
            turns
        });
    } catch (error) {
        horusLog(`Falha na análise profunda: ${error.message}`);
        setAnalysisStatus("Falha na análise profunda", "#e74c3c");
        return;
    }

    const probability = Number(result.probabilidade || 0);
    const device = result.model?.device || "desconhecido";
    if (!result.isScam) {
        setAnalysisStatus(`Transformer: Ham (${(probability * 100).toFixed(1)}%, ${device})`, "#2ecc71");
        horusLog(`Transformer classificou como Ham (${(probability * 100).toFixed(1)}% Scam).`);
        return;
    }

    const toggle = document.getElementById("ic-horus-toggle");
    if (toggle) {
        toggle.style.backgroundColor = "#e74c3c";
        toggle.style.borderColor = "#c0392b";
    }
    setAnalysisStatus(`Scam ${(probability * 100).toFixed(1)}% — identificando trechos (${device})`, "#e74c3c");
    horusLog(`Scam detectado. Job de explicabilidade iniciado em ${device}.`);
    if (result.explanation?.pollUrl) {
        pollExplanation(result.explanation.pollUrl, context);
    }
}

function maybeTriggerAnalysis(state) {
    if (!state.chronologicalIds.length) return;
    const lastId = state.chronologicalIds[state.chronologicalIds.length - 1];
    if (lastId === state.lastGateBalloonId) return;
    state.lastGateBalloonId = lastId;
    const lastMessage = state.messages.get(lastId);
    if (!lastMessage || lastMessage.speaker !== "Suspect") return;

    const trigger = globalThis.findHorusTrigger(lastSuspectText(state));
    if (!trigger) {
        setAnalysisStatus("Nenhum gatilho — conversa não analisada profundamente", "#95a5a6");
        horusLog("Pré-análise concluída: nenhum gatilho encontrado.");
        return;
    }
    setAnalysisStatus(`Gatilho: ${trigger.category}`, "#f39c12");
    sendDeepAnalysis(trigger);
}

function updatePanel(state) {
    const counter = document.getElementById("ic-horus-contador");
    const chatStatus = document.getElementById("ic-horus-chatid");
    if (counter) counter.textContent = `Memória da aba: ${state?.chronologicalIds.length || 0} balões`;
    if (chatStatus) chatStatus.textContent = activeChatId ? `Contato: ${activeChatId}` : "Contato: nenhum";
}

function scanActiveChat() {
    scanScheduled = false;
    if (!document.getElementById("ic-horus-panel")) injectPanel();
    const mainElement = document.getElementById("main");
    if (!mainElement) return;

    const header = mainElement.querySelector('header span[dir="auto"][title], header span[dir="auto"]');
    const detectedChat = header?.innerText?.trim();
    if (!detectedChat) return;
    if (detectedChat !== activeChatId) {
        activeChatId = detectedChat;
        horusLog(`Chat ativo: ${activeChatId}`);
        const state = sessionFor(activeChatId);
        if (state.lastExplanation) renderExplanation(state.lastExplanation, false);
        else document.getElementById("ic-horus-explanations")?.replaceChildren();
    }

    const state = sessionFor(activeChatId);
    const visible = visibleBalloons(mainElement);
    for (const item of visible) {
        const existing = state.messages.get(item.balloonId);
        if (!existing) {
            state.messages.set(item.balloonId, {
                balloonId: item.balloonId,
                speaker: item.speaker,
                originalText: item.text,
                observedOrder: ++observedSequence,
                observedAt: Date.now()
            });
            state.conversationVersion += 1;
        } else if (existing.originalText !== item.text || existing.speaker !== item.speaker) {
            existing.originalText = item.text;
            existing.speaker = item.speaker;
            state.conversationVersion += 1;
        }
    }
    reconcileChronology(state, visible.map((item) => item.balloonId));
    updatePanel(state);
    maybeTriggerAnalysis(state);
}

function scheduleScan() {
    if (scanScheduled) return;
    scanScheduled = true;
    setTimeout(scanActiveChat, 100);
}

function attachObserver() {
    const mainElement = document.getElementById("main");
    if (!mainElement || mainElement === observedMain) return;
    mainObserver?.disconnect();
    observedMain = mainElement;
    mainObserver = new MutationObserver(scheduleScan);
    mainObserver.observe(mainElement, { childList: true, subtree: true, characterData: true });
    scheduleScan();
}

function exportCurrentChat() {
    if (!activeChatId) return;
    const state = sessionFor(activeChatId);
    const text = buildTurns(state)
        .map((turn) => `${turn.speaker}: ${turn.balloons.map((item) => item.text).join(" ")}`)
        .join("\n");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `Horus_${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

function injectPanel() {
    if (document.getElementById("ic-horus-panel")) return;
    const panel = document.createElement("div");
    panel.id = "ic-horus-panel";
    panel.style.cssText = "position:fixed;top:150px;left:15px;z-index:10000;display:flex;align-items:flex-start;gap:10px;font-family:sans-serif;";

    const toggle = document.createElement("button");
    toggle.id = "ic-horus-toggle";
    toggle.textContent = "👁️";
    toggle.style.cssText = "width:44px;height:44px;background:#141414;color:#fff;border:2px solid #25D366;border-radius:50%;cursor:grab;box-shadow:0 4px 6px rgba(0,0,0,.4);font-size:20px;";

    const body = document.createElement("div");
    body.id = "ic-horus-body";
    body.style.cssText = "display:none;width:360px;max-height:70vh;overflow:auto;background:#141414;border:2px solid #25D366;border-radius:10px;padding:14px;box-shadow:0 5px 15px rgba(0,0,0,.5);font-family:monospace;";

    const title = document.createElement("strong");
    title.textContent = "IC Horus — análise dinâmica";
    title.style.cssText = "display:block;color:#fff;font-family:sans-serif;margin-bottom:6px;";
    const chat = document.createElement("div");
    chat.id = "ic-horus-chatid";
    chat.style.cssText = "color:#f39c12;font-size:12px;font-weight:bold;word-break:break-all;";
    const counter = document.createElement("div");
    counter.id = "ic-horus-contador";
    counter.style.cssText = "color:#25D366;font-size:12px;font-weight:bold;";
    const status = document.createElement("div");
    status.id = "ic-horus-analysis-status";
    status.textContent = "Aguardando mensagens";
    status.style.cssText = "color:#95a5a6;font-size:11px;margin:7px 0;";
    const terminal = document.createElement("div");
    terminal.id = "ic-horus-terminal";
    terminal.style.cssText = "height:70px;overflow:auto;background:#000;color:#0f0;font-size:10px;padding:5px;border-radius:5px;border:1px solid #333;margin-bottom:8px;";
    const explanations = document.createElement("div");
    explanations.id = "ic-horus-explanations";

    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;gap:5px;margin-bottom:8px;";
    const analyse = document.createElement("button");
    analyse.textContent = "Forçar IA";
    analyse.style.cssText = "flex:1;background:#3498db;color:#fff;border:0;padding:8px;border-radius:5px;cursor:pointer;font-weight:bold;";
    analyse.onclick = () => sendDeepAnalysis();
    const download = document.createElement("button");
    download.textContent = "Baixar TXT";
    download.style.cssText = "flex:1;background:#25D366;color:#000;border:0;padding:8px;border-radius:5px;cursor:pointer;font-weight:bold;";
    download.onclick = exportCurrentChat;
    buttons.append(analyse, download);
    body.append(title, chat, counter, status, terminal, buttons, explanations);
    panel.append(toggle, body);
    document.body.appendChild(panel);

    let open = false;
    let dragging = false;
    let moved = false;
    let offsetX = 0;
    let offsetY = 0;
    toggle.addEventListener("mousedown", (event) => {
        dragging = true;
        moved = false;
        const rect = panel.getBoundingClientRect();
        offsetX = event.clientX - rect.left;
        offsetY = event.clientY - rect.top;
    });
    document.addEventListener("mousemove", (event) => {
        if (!dragging) return;
        moved = true;
        panel.style.left = `${event.clientX - offsetX}px`;
        panel.style.top = `${event.clientY - offsetY}px`;
    });
    document.addEventListener("mouseup", () => { dragging = false; });
    toggle.addEventListener("click", () => {
        if (moved) return;
        open = !open;
        body.style.display = open ? "block" : "none";
    });
}

injectPanel();
attachObserver();
setInterval(() => {
    attachObserver();
    scheduleScan();
}, 1500);
