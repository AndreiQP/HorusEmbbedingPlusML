globalThis.HORUS_TRIGGER_RULES = Object.freeze({
    strong: [
        ["request_payment", ["pix", "wire transfer", "bank transfer", "gift card", "processing fee", "taxa de processamento", "taxa antecipada"]],
        ["request_credentials_or_otp", ["otp", "one-time password", "verification code", "codigo de verificacao", "código de verificação", "senha", "password", "pin"]],
        ["link_download_or_installation", ["click this link", "clique neste link", "clique no link", "download now", "baixe agora", "install this app", "instale este aplicativo"]],
        ["urgency_fear_or_threat", ["act now", "immediately", "urgente", "imediatamente", "conta bloqueada", "account blocked", "account suspended", "conta suspensa"]],
        ["secrecy_or_isolation", ["do not tell anyone", "não conte para ninguém", "nao conte para ninguem", "do not contact the bank", "não ligue para o banco", "nao ligue para o banco"]],
        ["problem_or_opportunity", ["you won", "voce ganhou", "você ganhou", "prize money", "premio", "prêmio", "unexpected refund", "reembolso inesperado"]]
    ],
    contextual: [
        ["impersonation", [["bank", "verification"], ["banco", "verificacao"], ["banco", "verificação"], ["police", "payment"], ["policia", "pagamento"], ["polícia", "pagamento"]]],
        ["urgency_fear_or_threat", [["account", "blocked"], ["conta", "bloqueada"], ["card", "cancel"], ["cartao", "cancelar"], ["cartão", "cancelar"]]],
        ["request_payment", [["payment", "fee"], ["pagamento", "taxa"], ["transfer", "urgent"], ["transferencia", "urgente"], ["transferência", "urgente"]]]
    ]
});

globalThis.findHorusTrigger = function findHorusTrigger(originalText) {
    const searchable = String(originalText || "").toLocaleLowerCase("pt-BR");
    for (const [category, phrases] of globalThis.HORUS_TRIGGER_RULES.strong) {
        for (const phrase of phrases) {
            if (searchable.includes(phrase)) {
                return { category, matchedText: phrase };
            }
        }
    }
    for (const [category, combinations] of globalThis.HORUS_TRIGGER_RULES.contextual) {
        for (const terms of combinations) {
            if (terms.every((term) => searchable.includes(term))) {
                return { category, matchedText: terms.join(" + ") };
            }
        }
    }
    return null;
};
