// IC Horus — transient WhatsApp memory, lexical gate and BGE explanations.

const originalConsoleLog = console.log.bind(console);
const chatSessions = new Map();
const trackedChats = new Set();
let activeChatId = null;
let observedSequence = 0;
let observedMain = null;
let mainObserver = null;
let scanScheduled = false;
let pendingChatSelection = null;
const decoratedBalloonElements = new Set();
const CURRENT_HIGHLIGHT_NAME = "horus-scam-danger";
const STALE_HIGHLIGHT_NAME = "horus-scam-danger-stale";

function sessionFor(chatId) {
    if (!chatSessions.has(chatId)) {
        chatSessions.set(chatId, {
            conversationVersion: 0,
            messages: new Map(),
            chronologicalIds: [],
            lastGateBalloonId: null,
            activeAnalysisId: null,
            lastExplanation: null,
            annotations: null,
            classification: null,
            analysisStatus: null
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

function setAnalysisStatus(text, color = "#f39c12", chatId = activeChatId) {
    const state = chatId ? chatSessions.get(chatId) : null;
    if (state) state.analysisStatus = { text, color };
    if (chatId && chatId !== activeChatId) return;
    const status = document.getElementById("ic-horus-analysis-status");
    if (status) {
        status.textContent = text;
        status.style.color = color;
    }
}

function balloonTextRoots(element) {
    const selectors = [
        "span.copyable-text",
        "span.selectable-text",
        'div[dir="ltr"]',
        'span[dir="ltr"]'
    ];
    for (const selector of selectors) {
        const candidates = [
            ...(element.matches?.(selector) ? [element] : []),
            ...element.querySelectorAll(selector)
        ].filter((candidate, index, all) => (
            !all.some((other, otherIndex) => otherIndex < index && other.contains(candidate))
        ));
        if (!candidates.length) continue;
        return candidates;
    }
    return [element];
}

function readBalloonContent(element) {
    const chunks = [];
    for (const [rootIndex, root] of balloonTextRoots(element).entries()) {
        if (rootIndex) chunks.push({ text: " ", node: null });
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        let textNode = walker.nextNode();
        while (textNode) {
            const parent = textNode.parentElement;
            const excluded = parent?.closest(
                '[data-testid="quoted-message"], [role="button"], #ic-horus-panel'
            );
            if (!excluded || !root.contains(excluded)) {
                chunks.push({ text: textNode.nodeValue || "", node: textNode });
            }
            textNode = walker.nextNode();
        }
    }

    const mapped = globalThis.HorusAnnotationUtils.canonicalizeMappedChunks(chunks);
    const prefix = mapped.text.match(/^\[.*?\]\s*/)?.[0] || "";
    let start = prefix.length;
    let end = mapped.text.length;
    while (start < end && /\s/u.test(mapped.text[start])) start += 1;
    while (end > start && /\s/u.test(mapped.text[end - 1])) end -= 1;
    let text = mapped.text.slice(start, end);
    let positions = mapped.positions.slice(start, end);

    const ignored = [
        "clique para mostrar os dados do contato",
        "clique para mostrar os dados do contacto",
        "esta mensagem foi apagada",
        "mensagem apagada",
        "leia mais",
        "read more"
    ];
    for (const phrase of ignored) {
        if (text.toLocaleLowerCase("pt-BR") === phrase) {
            text = "";
            positions = [];
        }
    }
    return { text, positions };
}

function stableHash(value) {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16).padStart(8, "0");
}

function chronologyFromMetadata(metadata) {
    const match = String(metadata || "").match(
        /\[(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)?\s*,\s*(\d{1,2})\/(\d{1,2})\/(\d{2,4})\]/i
    );
    if (!match) return null;
    let hour = Number(match[1]);
    const minute = Number(match[2]);
    const meridiem = String(match[3] || "").toUpperCase();
    if (meridiem === "PM" && hour < 12) hour += 12;
    if (meridiem === "AM" && hour === 12) hour = 0;
    const day = Number(match[4]);
    const month = Number(match[5]);
    let year = Number(match[6]);
    if (year < 100) year += 2000;
    const timestamp = new Date(year, month - 1, day, hour, minute).getTime();
    return Number.isFinite(timestamp) ? timestamp : null;
}

function speakerFor(element, mainElement) {
    if (element.closest(".message-in")) return "Suspect";
    if (element.closest(".message-out")) return "Innocent";
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
    const fingerprintOccurrences = new Map();
    const textNodes = mainElement.querySelectorAll(
        'span[data-testid="selectable-text"], span.copyable-text'
    );
    for (const textNode of textNodes) {
        if (textNode.closest('[data-testid="quoted-message"]')) continue;
        const element = textNode.closest(
            'div[data-id], [data-testid="msg-container"], .message-in, .message-out'
        ) || textNode;
        const content = readBalloonContent(element);
        const text = content.text;
        const speaker = speakerFor(element, mainElement);
        if (!text || !speaker) continue;
        const metadataNode = textNode.closest("[data-pre-plain-text]")
            || element.querySelector?.("[data-pre-plain-text]");
        const metadata = metadataNode?.getAttribute("data-pre-plain-text") || "";
        const rawId = element.getAttribute?.("data-id")
            || element.closest?.("[data-id]")?.getAttribute("data-id")
            || null;
        const fingerprint = `${speaker}\u241f${metadata}\u241f${text}`;
        const occurrence = fingerprintOccurrences.get(fingerprint) || 0;
        fingerprintOccurrences.set(fingerprint, occurrence + 1);
        const balloonId = rawId || `message:${stableHash(fingerprint)}:${occurrence}`;
        if (seen.has(balloonId)) continue;
        seen.add(balloonId);
        found.push({
            balloonId,
            speaker,
            text,
            metadata,
            chronology: chronologyFromMetadata(metadata),
            domElement: element,
            content
        });
    }
    return found;
}

function reconcileChronology(state, visibleItems) {
    const visibleIds = visibleItems.map((item) => item.balloonId);
    if (!visibleIds.length) return;
    if (!state.chronologicalIds.length) {
        state.chronologicalIds = [...visibleIds];
        return;
    }
    const merged = [...new Set(state.chronologicalIds)];
    for (let index = 0; index < visibleIds.length; index += 1) {
        const balloonId = visibleIds[index];
        if (merged.includes(balloonId)) continue;
        const previous = [...visibleIds.slice(0, index)].reverse().find((id) => merged.includes(id));
        const next = visibleIds.slice(index + 1).find((id) => merged.includes(id));
        if (previous) {
            merged.splice(merged.indexOf(previous) + 1, 0, balloonId);
        } else if (next) {
            merged.splice(merged.indexOf(next), 0, balloonId);
        } else {
            const message = state.messages.get(balloonId);
            const chronology = message?.chronology;
            const laterIndex = Number.isFinite(chronology)
                ? merged.findIndex((id) => {
                    const candidate = state.messages.get(id)?.chronology;
                    return Number.isFinite(candidate) && candidate > chronology;
                })
                : -1;
            if (laterIndex >= 0) merged.splice(laterIndex, 0, balloonId);
            else merged.push(balloonId);
        }
    }
    state.chronologicalIds = merged;
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

function ensureAnnotationStyles() {
    if (document.getElementById("ic-horus-annotation-styles")) return;
    const style = document.createElement("style");
    style.id = "ic-horus-annotation-styles";
    style.textContent = `
        [data-horus-scam-message="current"] {
            outline: 3px solid #e53935 !important;
            outline-offset: 2px !important;
            border-radius: 8px !important;
        }
        [data-horus-scam-message="stale"] {
            outline: 2px dashed rgba(229, 57, 53, .48) !important;
            outline-offset: 2px !important;
            border-radius: 8px !important;
        }
        ::highlight(${CURRENT_HIGHLIGHT_NAME}) {
            background-color: rgba(255, 61, 61, .48);
            text-decoration: underline 2px #b71c1c;
            text-decoration-skip-ink: none;
        }
        ::highlight(${STALE_HIGHLIGHT_NAME}) {
            background-color: rgba(255, 99, 99, .20);
            text-decoration: underline 1px rgba(183, 28, 28, .45);
            text-decoration-style: dashed;
        }
    `;
    document.documentElement.appendChild(style);
}

function clearWhatsappAnnotations() {
    for (const element of decoratedBalloonElements) {
        if (element instanceof HTMLElement) {
            delete element.dataset.horusScamMessage;
            delete element.dataset.horusScamRank;
        }
    }
    decoratedBalloonElements.clear();
    if (globalThis.CSS?.highlights) {
        globalThis.CSS.highlights.delete(CURRENT_HIGHLIGHT_NAME);
        globalThis.CSS.highlights.delete(STALE_HIGHLIGHT_NAME);
    }
}

function rangesForBalloonReference(content, reference) {
    const interval = globalThis.HorusAnnotationUtils.resolveReferenceInterval(
        content.text,
        reference
    );
    if (!interval) return [];
    const ranges = [];
    for (const segment of globalThis.HorusAnnotationUtils.mappedSegments(
        content.positions,
        interval
    )) {
        try {
            const range = document.createRange();
            range.setStart(segment.node, segment.start);
            range.setEnd(segment.node, segment.end);
            ranges.push(range);
        } catch (error) {
            horusLog(`Trecho não pôde ser ligado ao DOM: ${error.message}`);
        }
    }
    return ranges;
}

function buildAnnotationIndex(result) {
    return {
        conversationVersion: result.conversationVersion,
        items: (result.items || [])
            .filter((item) => item.message?.direction === "scam")
            .map((item) => ({
                rank: item.rank,
                balloonIds: [...(item.message.balloonIds || [])],
                dangerousSpan: item.dangerousSpan
                    || (item.span?.direction === "scam" ? item.span : null)
            }))
    };
}

function applyWhatsappAnnotations(state, visibleItems) {
    clearWhatsappAnnotations();
    if (!state?.classification?.isScam || !state.annotations) return;

    const stale = state.annotations.conversationVersion !== state.conversationVersion;
    const stateName = stale ? "stale" : "current";
    const visibleById = new Map(visibleItems.map((item) => [item.balloonId, item]));
    const ranges = [];

    for (const item of state.annotations.items) {
        for (const balloonId of item.balloonIds) {
            const visible = visibleById.get(balloonId);
            if (!visible?.domElement) continue;
            visible.domElement.dataset.horusScamMessage = stateName;
            visible.domElement.dataset.horusScamRank = String(item.rank || "");
            decoratedBalloonElements.add(visible.domElement);
        }

        const dangerousSpan = item.dangerousSpan;
        if (!dangerousSpan || dangerousSpan.direction !== "scam") continue;
        for (const reference of dangerousSpan.balloonReferences || []) {
            const visible = visibleById.get(reference.balloonId);
            if (!visible?.content) continue;
            ranges.push(...rangesForBalloonReference(visible.content, reference));
        }
    }

    if (!ranges.length || !globalThis.CSS?.highlights || !globalThis.Highlight) return;
    globalThis.CSS.highlights.set(
        stale ? STALE_HIGHLIGHT_NAME : CURRENT_HIGHLIGHT_NAME,
        new globalThis.Highlight(...ranges)
    );
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
        const displayedSpan = item.dangerousSpan || item.span;
        const card = document.createElement("div");
        card.dataset.turnId = item.message.turnId;
        card.dataset.balloonIds = (item.message.balloonIds || []).join(",");
        card.style.cssText = `
            background:#151515;border-left:4px solid ${directionColor(displayedSpan?.direction || item.message.direction)};
            border-radius:5px;padding:8px;margin-bottom:8px;color:#eee;font-size:11px;
        `;

        const heading = document.createElement("div");
        heading.style.cssText = "font-weight:bold;margin-bottom:4px;color:#fff;";
        heading.textContent = `#${item.rank} ${item.message.speaker} — ${directionLabel(displayedSpan?.direction || item.message.direction)}`;

        const message = document.createElement("div");
        message.style.cssText = "color:#bbb;margin-bottom:5px;white-space:normal;";
        message.textContent = item.message.text;

        const span = document.createElement("div");
        span.style.cssText = "font-weight:bold;color:#fff;margin-bottom:5px;";
        span.textContent = displayedSpan ? `Trecho: “${displayedSpan.text}”` : "Trecho perigoso não encontrado";

        const impactTrack = document.createElement("div");
        impactTrack.style.cssText = "height:5px;background:#333;border-radius:3px;overflow:hidden;";
        const impact = document.createElement("div");
        impact.style.cssText = `height:100%;width:${Math.round(100 * (displayedSpan?.relativeImpact || item.message.relativeImpact || 0))}%;background:${directionColor(displayedSpan?.direction || item.message.direction)};`;
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
            setAnalysisStatus("Falha ao consultar a explicação", "#e74c3c", context.chatId);
            return;
        }
        if (result.status === "queued" || result.status === "running") continue;
        if (result.status === "failed") {
            horusLog(`Explicação falhou: ${result.error || "erro desconhecido"}`);
            setAnalysisStatus("Não foi possível produzir a explicação", "#e74c3c", context.chatId);
            return;
        }

        const state = chatSessions.get(context.chatId);
        if (!state || result.analysisId !== context.analysisId) return;
        result.conversationVersion = context.conversationVersion;
        state.lastExplanation = result;
        state.annotations = buildAnnotationIndex(result);
        state.activeAnalysisId = null;
        const stale = state.conversationVersion !== context.conversationVersion;
        if (activeChatId === context.chatId) renderExplanation(result, stale);
        setAnalysisStatus(
            stale ? "Explicação concluída para uma versão anterior" : "Explicação concluída",
            stale ? "#f39c12" : "#2ecc71",
            context.chatId
        );
        if (activeChatId === context.chatId) scheduleScan();
        return;
    }
    setAnalysisStatus("A explicação excedeu o tempo de espera", "#e74c3c", context.chatId);
}

async function sendDeepAnalysis(trigger = { category: "manual", matchedText: "manual" }) {
    if (!activeChatId) return;
    if (!trackedChats.has(activeChatId)) {
        setAnalysisStatus(
            "Clique no chat na lista do WhatsApp antes de iniciar a análise",
            "#f39c12"
        );
        return;
    }
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
        state.activeAnalysisId = null;
        setAnalysisStatus("Falha na análise profunda", "#e74c3c", context.chatId);
        return;
    }

    const probability = Number(result.probabilidade || 0);
    const device = result.model?.device || "desconhecido";
    state.classification = {
        isScam: Boolean(result.isScam),
        probabilityScam: probability,
        device,
        conversationVersion: context.conversationVersion,
        analysedAt: Date.now()
    };
    if (activeChatId === context.chatId) {
        applyConversationVisual(state);
        scheduleScan();
    }
    if (!result.isScam) {
        state.activeAnalysisId = null;
        state.lastExplanation = null;
        state.annotations = null;
        if (activeChatId === context.chatId) {
            document.getElementById("ic-horus-explanations")?.replaceChildren();
            clearWhatsappAnnotations();
        }
        setAnalysisStatus(
            `Transformer: Ham (${(probability * 100).toFixed(1)}% de Scam, ${device})`,
            "#2ecc71",
            context.chatId
        );
        horusLog(`Transformer classificou como Ham (${(probability * 100).toFixed(1)}% Scam).`);
        return;
    }

    setAnalysisStatus(
        `Scam ${(probability * 100).toFixed(1)}% — identificando trechos (${device})`,
        "#e74c3c",
        context.chatId
    );
    horusLog(`Scam detectado. Job de explicabilidade iniciado em ${device}.`);
    if (result.explanation?.pollUrl) {
        pollExplanation(result.explanation.pollUrl, context);
    } else {
        state.activeAnalysisId = null;
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
    applyConversationVisual(state);
}

function applyConversationVisual(state) {
    const toggle = document.getElementById("ic-horus-toggle");
    const label = document.getElementById("ic-horus-classification");
    const classification = state?.classification;
    if (!classification) {
        if (toggle) {
            toggle.textContent = "👁️";
            toggle.style.backgroundColor = "#141414";
            toggle.style.borderColor = "#25D366";
            toggle.title = "Conversa ainda não analisada";
        }
        if (label) {
            label.textContent = state ? "Classificação: ainda não analisada" : "Classificação: memória não iniciada";
            label.style.color = "#95a5a6";
        }
        return;
    }

    const probability = Math.max(0, Math.min(1, Number(classification.probabilityScam || 0)));
    const stale = classification.conversationVersion !== state.conversationVersion;
    if (toggle) {
        toggle.textContent = classification.isScam ? "⚠️" : "✅";
        toggle.style.backgroundColor = classification.isScam ? "#c0392b" : "#16803a";
        toggle.style.borderColor = classification.isScam ? "#ff6b5f" : "#2ecc71";
        toggle.title = `${classification.isScam ? "Scam" : "Ham"}: ${(probability * 100).toFixed(1)}% de Scam`;
    }
    if (label) {
        label.textContent = `${classification.isScam ? "Scam" : "Ham"} — ${(probability * 100).toFixed(1)}% de Scam${stale ? " (há mensagens novas)" : ""}`;
        label.style.color = classification.isScam ? "#e74c3c" : "#2ecc71";
    }
}

function restoreConversationUi(state) {
    updatePanel(state);
    const status = state.analysisStatus;
    if (status) setAnalysisStatus(status.text, status.color, activeChatId);
    else setAnalysisStatus("Aguardando análise", "#95a5a6", activeChatId);
    if (state.lastExplanation) {
        const stale = state.lastExplanation.conversationVersion !== state.conversationVersion;
        renderExplanation(state.lastExplanation, stale);
    } else {
        document.getElementById("ic-horus-explanations")?.replaceChildren();
    }
}

function rememberExplicitChatClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const pane = target?.closest("#pane-side");
    if (!target || !pane) return;

    let cursor = target;
    let titledName = target.closest("[title]")?.getAttribute("title")?.trim() || null;
    while (!titledName && cursor && cursor !== pane) {
        const titles = cursor.querySelectorAll?.("span[title]") || [];
        if (titles.length === 1) titledName = titles[0].getAttribute("title")?.trim() || null;
        cursor = cursor.parentElement;
    }

    if (titledName && titledName === activeChatId) {
        const wasTracked = trackedChats.has(activeChatId);
        trackedChats.add(activeChatId);
        const state = sessionFor(activeChatId);
        horusLog(`Chat adicionado à memória da aba por clique explícito: ${activeChatId}`);
        if (!wasTracked) restoreConversationUi(state);
    }
    pendingChatSelection = {
        expectedChatId: titledName || null,
        previousChatId: activeChatId,
        notBefore: Date.now() + 150,
        expiresAt: Date.now() + 5000
    };
    setTimeout(scheduleScan, 220);
}

function scanActiveChat() {
    scanScheduled = false;
    if (!document.getElementById("ic-horus-panel")) injectPanel();
    const mainElement = document.getElementById("main");
    if (!mainElement) return;

    const header = mainElement.querySelector('header span[dir="auto"][title], header span[dir="auto"]');
    const detectedChat = header?.innerText?.trim();
    if (!detectedChat) return;
    const chatChanged = detectedChat !== activeChatId;
    if (detectedChat !== activeChatId) {
        clearWhatsappAnnotations();
        activeChatId = detectedChat;
        horusLog(`Chat ativo: ${activeChatId}`);
        document.getElementById("ic-horus-explanations")?.replaceChildren();
    }

    const now = Date.now();
    if (pendingChatSelection && now >= pendingChatSelection.notBefore) {
        const expectedMatches = (
            pendingChatSelection.expectedChatId === detectedChat
            || detectedChat !== pendingChatSelection.previousChatId
        );
        if (now <= pendingChatSelection.expiresAt && expectedMatches) {
            const wasTracked = trackedChats.has(detectedChat);
            trackedChats.add(detectedChat);
            horusLog(`Chat adicionado à memória da aba por clique explícito: ${detectedChat}`);
            pendingChatSelection = null;
            if (!wasTracked) restoreConversationUi(sessionFor(detectedChat));
        } else if (now > pendingChatSelection.expiresAt) {
            pendingChatSelection = null;
        }
    }

    if (!trackedChats.has(detectedChat)) {
        clearWhatsappAnnotations();
        updatePanel(null);
        setAnalysisStatus(
            "Chat restaurado automaticamente — memória não iniciada até um clique na lista",
            "#95a5a6"
        );
        return;
    }

    const state = sessionFor(activeChatId);
    if (chatChanged) restoreConversationUi(state);
    const visible = visibleBalloons(mainElement);
    for (const item of visible) {
        const existing = state.messages.get(item.balloonId);
        if (!existing) {
            state.messages.set(item.balloonId, {
                balloonId: item.balloonId,
                speaker: item.speaker,
                originalText: item.text,
                metadata: item.metadata,
                chronology: item.chronology,
                observedOrder: ++observedSequence,
                observedAt: Date.now()
            });
            state.conversationVersion += 1;
        } else {
            if (existing.originalText !== item.text || existing.speaker !== item.speaker) {
                existing.originalText = item.text;
                existing.speaker = item.speaker;
                state.conversationVersion += 1;
            }
            existing.metadata = item.metadata;
            existing.chronology = item.chronology;
        }
    }
    reconcileChronology(state, visible);
    updatePanel(state);
    applyWhatsappAnnotations(state, visible);
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
    if (!activeChatId || !trackedChats.has(activeChatId)) {
        alert("Clique primeiro neste chat na lista do WhatsApp para iniciar sua memória temporária.");
        return;
    }
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
    ensureAnnotationStyles();
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
    const classification = document.createElement("div");
    classification.id = "ic-horus-classification";
    classification.textContent = "Classificação: memória não iniciada";
    classification.style.cssText = "color:#95a5a6;font-size:12px;font-weight:bold;margin-top:4px;";
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
    body.append(title, chat, counter, classification, status, terminal, buttons, explanations);
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
document.addEventListener("click", rememberExplicitChatClick, true);
attachObserver();
setInterval(() => {
    attachObserver();
    scheduleScan();
}, 1500);
