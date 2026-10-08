const DEFAULT_API_BASE_URL = "http://localhost:5000";

async function loadOptions() {
    const stored = await chrome.storage.local.get({ horusApiBaseUrl: DEFAULT_API_BASE_URL });
    document.getElementById("api-url").value = stored.horusApiBaseUrl;
}

async function saveOptions() {
    const status = document.getElementById("status");
    try {
        const raw = document.getElementById("api-url").value.trim();
        const parsed = new URL(raw);
        if (!/^https?:$/.test(parsed.protocol)) throw new Error("Use uma URL HTTP ou HTTPS.");
        const baseUrl = `${parsed.protocol}//${parsed.host}${parsed.pathname.replace(/\/+$/, "")}`;
        const originPattern = `${parsed.protocol}//${parsed.host}/*`;
        const granted = await chrome.permissions.request({ origins: [originPattern] });
        if (!granted) throw new Error("A permissão para acessar esse servidor não foi concedida.");
        await chrome.storage.local.set({ horusApiBaseUrl: baseUrl });
        status.textContent = "Configuração salva.";
        status.style.color = "#16803a";
    } catch (error) {
        status.textContent = error.message;
        status.style.color = "#b00020";
    }
}

document.getElementById("save").addEventListener("click", saveOptions);
loadOptions();
