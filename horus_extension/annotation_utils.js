(function exposeHorusAnnotationUtils(root, factory) {
    const api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    root.HorusAnnotationUtils = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildUtils() {
    function codePointOffsetToCodeUnit(text, codePointOffset) {
        const safeOffset = Math.max(0, Number(codePointOffset) || 0);
        let codePoints = 0;
        let codeUnits = 0;
        for (const character of String(text || "")) {
            if (codePoints >= safeOffset) break;
            codeUnits += character.length;
            codePoints += 1;
        }
        return codeUnits;
    }

    function canonicalizeMappedChunks(chunks) {
        const characters = [];
        const positions = [];
        let pendingWhitespace = null;
        let hasPendingWhitespace = false;

        for (const chunk of chunks || []) {
            const value = String(chunk?.text || "");
            for (let offset = 0; offset < value.length; offset += 1) {
                const character = value[offset];
                const position = chunk?.node
                    ? { node: chunk.node, start: offset, end: offset + 1 }
                    : null;
                if (/\s/u.test(character)) {
                    if (characters.length && !hasPendingWhitespace) {
                        pendingWhitespace = position;
                        hasPendingWhitespace = true;
                    }
                    continue;
                }
                if (hasPendingWhitespace && characters.length) {
                    characters.push(" ");
                    positions.push(pendingWhitespace);
                }
                pendingWhitespace = null;
                hasPendingWhitespace = false;
                characters.push(character);
                positions.push(position);
            }
        }
        return { text: characters.join(""), positions };
    }

    function exactOccurrences(text, needle) {
        const indexes = [];
        if (!needle) return indexes;
        let cursor = 0;
        while (cursor <= text.length - needle.length) {
            const index = text.indexOf(needle, cursor);
            if (index < 0) break;
            indexes.push(index);
            cursor = index + Math.max(1, needle.length);
        }
        return indexes;
    }

    function resolveReferenceInterval(canonicalText, reference) {
        const text = String(canonicalText || "");
        const expected = String(reference?.text || "");
        const start = codePointOffsetToCodeUnit(text, reference?.startChar);
        const end = codePointOffsetToCodeUnit(text, reference?.endChar);
        if (start <= end && text.slice(start, end) === expected) return { start, end };

        const occurrences = exactOccurrences(text, expected);
        if (occurrences.length !== 1) return null;
        return { start: occurrences[0], end: occurrences[0] + expected.length };
    }

    function mappedSegments(positions, interval) {
        if (!interval || interval.start >= interval.end) return [];
        const segments = [];
        let current = null;
        for (let index = interval.start; index < interval.end; index += 1) {
            const position = positions[index];
            if (!position?.node) {
                current = null;
                continue;
            }
            if (current && current.node === position.node && current.end === position.start) {
                current.end = position.end;
            } else {
                current = { ...position };
                segments.push(current);
            }
        }
        return segments;
    }

    function advanceCaptureGate(gate, signature, now, settleMs) {
        const next = { ...gate };
        if (next.previousSignature && signature === next.previousSignature) {
            return { ready: false, gate: next, reason: "previous_chat_dom" };
        }
        if (next.candidateSignature !== signature) {
            next.candidateSignature = signature;
            next.candidateSince = now;
            next.observations = 1;
            return { ready: false, gate: next, reason: "new_candidate" };
        }
        next.observations += 1;
        if (next.observations < 2 || now - next.candidateSince < settleMs) {
            return { ready: false, gate: next, reason: "settling" };
        }
        return { ready: true, gate: null, reason: "stable" };
    }

    return {
        advanceCaptureGate,
        canonicalizeMappedChunks,
        codePointOffsetToCodeUnit,
        exactOccurrences,
        mappedSegments,
        resolveReferenceInterval
    };
}));
