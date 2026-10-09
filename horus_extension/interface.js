(function exposeHorusInterface(root) {
    const view = {
        mounted: false,
        open: false,
        helpOpen: false,
        chatId: null,
        balloonCount: 0,
        classification: null,
        conversationVersion: 0,
        status: { text: "Aguardando mensagens", color: "#66758a" },
        hasExplanation: false,
        lastAutoOpenedScam: null,
        callbacks: {}
    };

    function byId(id) {
        return document.getElementById(id);
    }

    function shield(size = "normal") {
        const wrapper = document.createElement("span");
        wrapper.className = `horus-shield horus-shield--${size}`;
        const image = document.createElement("img");
        image.src = chrome.runtime.getURL("util/logo_recod.png");
        image.alt = "Recod";
        wrapper.appendChild(image);
        return wrapper;
    }

    function setOpen(open) {
        view.open = Boolean(open);
        const panel = byId("ic-horus-panel");
        if (panel) panel.dataset.open = String(view.open);
        byId("ic-horus-toggle")?.setAttribute("aria-expanded", String(view.open));
        if (!view.open) setHelpOpen(false);
    }

    function setHelpOpen(open) {
        view.helpOpen = Boolean(open);
        const panel = byId("ic-horus-panel");
        if (panel) panel.dataset.help = String(view.helpOpen);
        byId("ic-horus-help-button")?.setAttribute("aria-expanded", String(view.helpOpen));
    }

    function normalizedStatus() {
        return String(view.status?.text || "").toLocaleLowerCase("pt-BR");
    }

    function presentation() {
        const status = normalizedStatus();
        if (!view.chatId) {
            return {
                phase: "inactive",
                tone: "waiting",
                symbol: "○",
                title: "Abra uma conversa",
                description: "Clique em um chat na lista do WhatsApp para iniciar a memória temporária."
            };
        }
        if (/falha|não foi possível|excedeu o tempo|erro/.test(status)) {
            return {
                phase: "error",
                tone: "error",
                symbol: "!",
                title: "Não foi possível concluir",
                description: view.status.text
            };
        }
        if (/executando bge|gatilho:/.test(status)) {
            return {
                phase: "classifying",
                tone: "analysing",
                symbol: "↻",
                title: "Análise em andamento",
                description: "O Horus está classificando o risco da conversa completa."
            };
        }
        if (/identificando trechos/.test(status)) {
            return {
                phase: "explaining",
                tone: "analysing",
                symbol: "↻",
                title: "Localizando evidências",
                description: "A classificação terminou. Agora o Horus identifica mensagens e trechos influentes."
            };
        }
        if (view.classification?.isScam) {
            const stale = view.classification.conversationVersion !== view.conversationVersion;
            return {
                phase: "scam",
                tone: "scam",
                symbol: "!",
                title: stale ? "Risco detectado · análise anterior" : "Risco detectado",
                description: stale
                    ? "Há mensagens novas. Atualize a análise para considerar a conversa atual."
                    : "O modelo encontrou sinais relevantes de fraude nesta conversa."
            };
        }
        if (view.classification) {
            const stale = view.classification.conversationVersion !== view.conversationVersion;
            return {
                phase: "ham",
                tone: "ham",
                symbol: "✓",
                title: stale ? "Baixo risco · análise anterior" : "Baixo risco estimado",
                description: stale
                    ? "Há mensagens novas. Atualize a análise para considerar a conversa atual."
                    : "Nenhum sinal relevante de fraude foi encontrado nesta conversa."
            };
        }
        return {
            phase: "waiting",
            tone: "waiting",
            symbol: "○",
            title: "Aguardando gatilho",
            description: "A pré-análise local acompanha somente esta conversa. A IA ainda não foi executada."
        };
    }

    function updateProgress(phase) {
        const progress = byId("ic-horus-progress");
        if (!progress) return;
        const visible = phase === "classifying" || phase === "explaining";
        progress.hidden = !visible;
        if (!visible) return;
        const states = phase === "classifying"
            ? ["done", "active", "pending"]
            : ["done", "done", "active"];
        progress.querySelectorAll(".horus-step").forEach((step, index) => {
            step.dataset.state = states[index];
            const marker = step.querySelector(".horus-step__marker");
            if (marker) marker.textContent = states[index] === "done" ? "✓" : String(index + 1);
        });
    }

    function render() {
        if (!view.mounted) return;
        const current = presentation();
        const panel = byId("ic-horus-panel");
        if (panel) panel.dataset.tone = current.tone;

        const symbol = byId("ic-horus-state-symbol");
        const title = byId("ic-horus-state-title");
        const description = byId("ic-horus-state-description");
        if (symbol) symbol.textContent = current.symbol;
        if (title) title.textContent = current.title;
        if (description) description.textContent = current.description;

        const trigger = byId("ic-horus-trigger-box");
        if (trigger) trigger.hidden = current.phase !== "waiting";
        updateProgress(current.phase);

        const probability = Math.max(0, Math.min(1, Number(view.classification?.probabilityScam || 0)));
        const risk = byId("ic-horus-risk-card");
        if (risk) {
            risk.hidden = !view.classification
                || current.phase === "classifying"
                || current.phase === "error";
            const value = risk.querySelector("strong");
            if (value) value.textContent = `${Math.round(probability * 100)}% Scam`;
        }

        const stale = Boolean(
            view.classification
            && view.classification.conversationVersion !== view.conversationVersion
        );
        const analyse = byId("ic-horus-analyse");
        if (analyse) {
            const busy = current.phase === "classifying" || current.phase === "explaining";
            analyse.hidden = current.phase === "error"
                || ((current.phase === "scam" || current.phase === "ham") && !stale);
            analyse.disabled = !view.chatId || busy;
            analyse.textContent = busy
                ? "Análise em andamento…"
                : stale
                    ? "Atualizar análise"
                    : "Analisar agora";
        }
        const retry = byId("ic-horus-retry");
        if (retry) retry.hidden = current.phase !== "error";

        const evidenceTitle = byId("ic-horus-evidence-title");
        const explanations = byId("ic-horus-explanations");
        const showEvidence = Boolean(view.classification?.isScam)
            && current.phase !== "classifying"
            && current.phase !== "error";
        if (evidenceTitle) evidenceTitle.hidden = !showEvidence;
        if (explanations) explanations.hidden = !showEvidence;

        const chat = byId("ic-horus-chatid");
        const count = byId("ic-horus-contador");
        const device = byId("ic-horus-device");
        const technicalStatus = byId("ic-horus-analysis-status");
        if (chat) chat.textContent = view.chatId || "Nenhuma conversa selecionada";
        if (count) count.textContent = `${view.balloonCount} balões em memória`;
        if (device) device.textContent = view.classification?.device || "ainda não utilizado";
        if (technicalStatus) {
            technicalStatus.textContent = view.status.text;
            technicalStatus.style.color = view.status.color;
        }

        const toggle = byId("ic-horus-toggle");
        if (toggle) {
            const suffix = view.classification ? ` · ${Math.round(probability * 100)}% Scam` : "";
            toggle.title = `${current.title}${suffix}`;
            toggle.setAttribute("aria-label", `Horus: ${current.title}${suffix}`);
        }

        if (view.classification?.isScam) {
            const scamKey = `${view.chatId}:${view.classification.conversationVersion}`;
            if (scamKey !== view.lastAutoOpenedScam) {
                view.lastAutoOpenedScam = scamKey;
                setOpen(true);
            }
        }
    }

    function staticMarkup() {
        return `
            <button id="ic-horus-toggle" type="button" aria-controls="ic-horus-drawer" aria-expanded="false">
                <span class="horus-toggle__logo"></span><span class="horus-indicator"></span>
            </button>
            <aside id="ic-horus-drawer" aria-label="Painel Horus">
                <header class="horus-header">
                    <span class="horus-header__logo"></span>
                    <div class="horus-brand"><strong>Horus</strong><span>Proteção inteligente contra golpes</span></div>
                    <button id="ic-horus-help-button" class="horus-icon-button" type="button" aria-controls="ic-horus-help" aria-expanded="false" title="Como o Horus funciona">?</button>
                    <button id="ic-horus-close" class="horus-icon-button" type="button" title="Fechar Horus">×</button>
                </header>
                <div class="horus-content">
                    <section class="horus-state">
                        <span id="ic-horus-state-symbol" class="horus-state__symbol">○</span>
                        <div><h2 id="ic-horus-state-title">Abra uma conversa</h2><p id="ic-horus-state-description"></p></div>
                    </section>
                    <section id="ic-horus-trigger-box" class="horus-trigger" hidden>
                        <strong>Exemplos de sinais iniciais</strong>
                        <div class="horus-chips"><span>pagamento</span><span>senha</span><span>urgente</span><span>link</span></div>
                        <p>São gatilhos locais para iniciar a IA; não indicam fraude por si só.</p>
                    </section>
                    <section id="ic-horus-progress" class="horus-progress" hidden>
                        <div class="horus-step"><span class="horus-step__marker">1</span><div><strong>Conversa preparada</strong><small>Mensagens organizadas em ordem</small></div></div>
                        <div class="horus-step"><span class="horus-step__marker">2</span><div><strong>Classificando risco</strong><small>BGE + Transformer analisam a conversa</small></div></div>
                        <div class="horus-step"><span class="horus-step__marker">3</span><div><strong>Localizando evidências</strong><small>Mensagens e trechos influentes</small></div></div>
                    </section>
                    <section id="ic-horus-risk-card" class="horus-risk" hidden><span>Risco estimado</span><strong>0% Scam</strong></section>
                    <button id="ic-horus-analyse" class="horus-primary" type="button">Analisar agora</button>
                    <button id="ic-horus-retry" class="horus-primary" type="button" hidden>Tentar novamente</button>
                    <h3 id="ic-horus-evidence-title" class="horus-section-title" hidden>Evidências principais</h3>
                    <div id="ic-horus-explanations" class="horus-explanations" hidden></div>
                    <details class="horus-technical">
                        <summary>Detalhes técnicos</summary>
                        <div class="horus-technical__body">
                            <div>Conversa: <strong id="ic-horus-chatid">Nenhuma conversa selecionada</strong></div>
                            <div id="ic-horus-contador">0 balões em memória</div>
                            <div>Dispositivo: <strong id="ic-horus-device">ainda não utilizado</strong></div>
                            <div id="ic-horus-analysis-status">Aguardando mensagens</div>
                            <div id="ic-horus-terminal"></div>
                            <button id="ic-horus-export" class="horus-secondary" type="button">Baixar conversa em TXT</button>
                        </div>
                    </details>
                </div>
                <div id="ic-horus-help" class="horus-help" role="dialog" aria-modal="true" aria-labelledby="ic-horus-help-title">
                    <section class="horus-help__card">
                        <div class="horus-help__header"><span class="horus-help__logo"></span><h2 id="ic-horus-help-title">Como o Horus chegou a este resultado?</h2><button id="ic-horus-help-close" class="horus-icon-button" type="button">×</button></div>
                        <div class="horus-help-step"><b>1</b><div><strong>Gatilho local</strong><p>Identifica sinais iniciais sem usar GPU.</p></div></div>
                        <div class="horus-help-step"><b>2</b><div><strong>Classificação por IA</strong><p>Analisa a conversa completa com BGE e Transformer.</p></div></div>
                        <div class="horus-help-step"><b>3</b><div><strong>Explicabilidade</strong><p>Encontra mensagens e trechos que influenciaram a decisão.</p></div></div>
                        <div class="horus-help-step"><b>4</b><div><strong>Resultado</strong><p>Apresenta risco estimado e evidências na conversa.</p></div></div>
                        <div class="horus-disclaimer">Os destaques mostram sinais aprendidos pelo modelo; não são prova factual de fraude.</div>
                    </section>
                </div>
            </aside>`;
    }

    function mount(callbacks = {}) {
        if (view.mounted || byId("ic-horus-panel")) return;
        view.callbacks = callbacks;
        const panel = document.createElement("div");
        panel.id = "ic-horus-panel";
        panel.dataset.open = "false";
        panel.dataset.help = "false";
        panel.dataset.tone = "waiting";
        panel.innerHTML = staticMarkup();
        document.body.appendChild(panel);

        byId("ic-horus-toggle").querySelector(".horus-toggle__logo").appendChild(shield());
        byId("ic-horus-drawer").querySelector(".horus-header__logo").appendChild(shield("small"));
        byId("ic-horus-help").querySelector(".horus-help__logo").appendChild(shield("small"));

        byId("ic-horus-toggle").onclick = () => setOpen(!view.open);
        byId("ic-horus-close").onclick = () => setOpen(false);
        byId("ic-horus-help-button").onclick = () => setHelpOpen(true);
        byId("ic-horus-help-close").onclick = () => setHelpOpen(false);
        byId("ic-horus-help").onclick = (event) => {
            if (event.target === byId("ic-horus-help")) setHelpOpen(false);
        };
        byId("ic-horus-analyse").onclick = () => view.callbacks.onAnalyse?.();
        byId("ic-horus-retry").onclick = () => view.callbacks.onAnalyse?.();
        byId("ic-horus-export").onclick = () => view.callbacks.onExport?.();
        document.addEventListener("keydown", (event) => {
            if (event.key !== "Escape") return;
            if (view.helpOpen) setHelpOpen(false);
            else if (view.open) setOpen(false);
        });
        view.mounted = true;
        render();
    }

    function log(message) {
        const terminal = byId("ic-horus-terminal");
        if (!terminal) return;
        const line = document.createElement("div");
        line.textContent = `> ${message}`;
        terminal.appendChild(line);
        terminal.scrollTop = terminal.scrollHeight;
    }

    function setStatus(text, color) {
        view.status = { text, color };
        render();
    }

    function updateConversation(snapshot = {}) {
        view.chatId = snapshot.chatId || null;
        view.balloonCount = Number(snapshot.balloonCount || 0);
        view.classification = snapshot.classification || null;
        view.conversationVersion = Number(snapshot.conversationVersion || 0);
        render();
    }

    function clearExplanation() {
        view.hasExplanation = false;
        byId("ic-horus-explanations")?.replaceChildren();
        render();
    }

    function renderExplanation(result, stale = false) {
        const container = byId("ic-horus-explanations");
        if (!container) return;
        container.replaceChildren();
        view.hasExplanation = true;

        if (stale) {
            const warning = document.createElement("div");
            warning.className = "horus-stale";
            warning.textContent = "Esta explicação pertence a uma versão anterior da conversa.";
            container.appendChild(warning);
        }

        const items = (result.items || []).filter((item) => (
            item.message?.direction === "scam"
            || item.dangerousSpan?.direction === "scam"
            || item.span?.direction === "scam"
        ));
        for (const item of items) {
            const span = item.dangerousSpan || (item.span?.direction === "scam" ? item.span : null);
            const card = document.createElement("article");
            card.className = "horus-evidence";
            const rank = document.createElement("div");
            rank.className = "horus-evidence__rank";
            rank.textContent = `Evidência #${item.rank} · ${item.message.speaker}`;
            const quote = document.createElement("div");
            quote.className = "horus-evidence__quote";
            quote.textContent = span ? `“${span.text}”` : "Mensagem influente sem trecho isolado";
            const message = document.createElement("p");
            message.textContent = item.message.text;
            const action = document.createElement("button");
            action.type = "button";
            action.textContent = "Ver na conversa ↗";
            action.onclick = () => view.callbacks.onLocate?.(item.message.balloonIds || []);
            card.append(rank, quote, message, action);
            container.appendChild(card);
        }
        if (!items.length) {
            const empty = document.createElement("div");
            empty.className = "horus-notice";
            empty.textContent = "O modelo classificou a conversa como Scam, mas não isolou um trecho pró-Scam confiável.";
            container.appendChild(empty);
        }
        const disclaimer = document.createElement("div");
        disclaimer.className = "horus-disclaimer";
        disclaimer.textContent = "Os destaques mostram sinais aprendidos pelo modelo; não são prova factual de fraude.";
        container.appendChild(disclaimer);
        render();
    }

    root.HorusInterface = Object.freeze({
        clearExplanation,
        log,
        mount,
        renderExplanation,
        setStatus,
        updateConversation
    });
}(globalThis));
