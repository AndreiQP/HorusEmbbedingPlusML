const DEFAULT_API_BASE_URL = "http://localhost:5000";

console.log("IC Horus: background service worker iniciado.");

async function apiBaseUrl() {
    const stored = await chrome.storage.local.get({ horusApiBaseUrl: DEFAULT_API_BASE_URL });
    return String(stored.horusApiBaseUrl || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}

async function callApi(request) {
    const baseUrl = await apiBaseUrl();
    const endpoint = String(request.endpoint || "");
    const url = `${baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const method = String(request.method || "GET").toUpperCase();
    const options = {
        method,
        headers: { "Content-Type": "application/json" }
    };
    if (method !== "GET" && method !== "HEAD" && request.dados !== undefined) {
        options.body = JSON.stringify(request.dados);
    }

    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
        const body = await response.text();
        throw new Error(`API retornou conteúdo não JSON (${response.status}): ${body.slice(0, 160)}`);
    }
    const data = await response.json();
    if (!response.ok && response.status !== 202) {
        throw new Error(data.message || data.error || `Erro HTTP ${response.status}`);
    }
    return { status: response.status, data };
}

chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
    if (request.action !== "chamarAPI") return false;
    callApi(request)
        .then(({ status, data }) => sendResponse({ sucesso: true, status, dados: data }))
        .catch((error) => {
            console.error("[IC Horus] Falha na API:", error);
            sendResponse({ sucesso: false, erro: error.message });
        });
    return true;
});
