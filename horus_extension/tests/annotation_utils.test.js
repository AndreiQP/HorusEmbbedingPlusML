const test = require("node:test");
const assert = require("node:assert/strict");

const utils = require("../annotation_utils.js");

test("converte offsets Python em offsets UTF-16 com emoji", () => {
    const text = "pague 💳 agora";
    const startChar = Array.from("pague 💳 ").length;
    const endChar = startChar + Array.from("agora").length;
    assert.deepEqual(
        utils.resolveReferenceInterval(text, { startChar, endChar, text: "agora" }),
        { start: text.indexOf("agora"), end: text.length }
    );
});

test("não escolhe silenciosamente entre ocorrências repetidas", () => {
    assert.equal(
        utils.resolveReferenceInterval("pix agora pix", {
            startChar: 1,
            endChar: 4,
            text: "pix"
        }),
        null
    );
});

test("normaliza espaços mantendo o mapa para os nós originais", () => {
    const first = { id: "first" };
    const second = { id: "second" };
    const mapped = utils.canonicalizeMappedChunks([
        { text: "  clique\n", node: first },
        { text: "  agora  ", node: second }
    ]);
    assert.equal(mapped.text, "clique agora");
    const interval = utils.resolveReferenceInterval(mapped.text, {
        startChar: 7,
        endChar: 12,
        text: "agora"
    });
    assert.deepEqual(utils.mappedSegments(mapped.positions, interval), [
        { node: second, start: 2, end: 7 }
    ]);
});

test("rejeita DOM antigo e aguarda estabilização depois da troca de chat", () => {
    let gate = {
        chatId: "chat-b",
        previousSignature: "assinatura-chat-a",
        candidateSignature: null,
        candidateSince: 0,
        observations: 0
    };

    let outcome = utils.advanceCaptureGate(gate, "assinatura-chat-a", 1000, 400);
    assert.equal(outcome.ready, false);
    assert.equal(outcome.reason, "previous_chat_dom");

    outcome = utils.advanceCaptureGate(outcome.gate, "assinatura-chat-b", 1100, 400);
    assert.equal(outcome.ready, false);
    assert.equal(outcome.reason, "new_candidate");

    outcome = utils.advanceCaptureGate(outcome.gate, "assinatura-chat-b", 1300, 400);
    assert.equal(outcome.ready, false);
    assert.equal(outcome.reason, "settling");

    outcome = utils.advanceCaptureGate(outcome.gate, "assinatura-chat-b", 1500, 400);
    assert.equal(outcome.ready, true);
    assert.equal(outcome.reason, "stable");
    assert.equal(outcome.gate, null);
});
