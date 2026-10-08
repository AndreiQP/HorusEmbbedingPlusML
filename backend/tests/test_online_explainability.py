from __future__ import annotations

import time
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.explainability_service import (
    ExplanationJobQueue,
    OnlineExplainabilityService,
    PredictionSnapshot,
    canonicalize_payload,
    format_online_explanation,
)
from api.main import create_app


def _payload(analysis_id: str = "analysis-1") -> dict:
    return {
        "apiVersion": "v1",
        "analysisId": analysis_id,
        "chatId": "chat-1",
        "conversationVersion": 3,
        "turns": [
            {
                "turnId": "turn:b1",
                "speaker": "Suspect",
                "balloons": [
                    {"balloonId": "b1", "text": "first part"},
                    {"balloonId": "b2", "text": "second part"},
                ],
            }
        ],
    }


class FakeRuntime:
    seed = 42
    device = "cpu"

    def __init__(self, probability: float):
        self.probability = probability
        self.explain_count = 0

    def predict(self, conversation):
        return PredictionSnapshot(
            logit=1.0,
            probability=self.probability,
            embeddings=np.zeros((len(conversation.turns), 4), dtype=np.float32),
            device="cpu",
        )

    def explain(self, conversation, original_embeddings):
        self.explain_count += 1
        return {"device": "cpu", "items": []}


def _wait(service: OnlineExplainabilityService, job_id: str) -> dict:
    for _ in range(100):
        result = service.explanation_status(job_id)
        if result and result["status"] not in {"queued", "running"}:
            return result
        time.sleep(0.01)
    raise AssertionError("job não concluiu")


def test_canonical_contract_preserves_balloon_ids_and_original_text() -> None:
    conversation = canonicalize_payload(_payload())

    assert conversation.conversation_text == "Suspect: first part second part"
    assert conversation.turns[0].balloon_ids == ["b1", "b2"]
    assert conversation.turns[0].balloons[0].start_char == 0
    assert conversation.turns[0].balloons[1].start_char == len("first part ")


def test_span_crossing_balloons_returns_local_references() -> None:
    conversation = canonicalize_payload(_payload())
    turn = conversation.turns[0]
    content_start = turn.content_start
    explanation = SimpleNamespace(
        baseline={"logit": 2.0, "probability_scam": 0.88, "predicted_label": 1},
        top_messages=[{
            "model_index": 0,
            "direction": "scam",
            "delta_logit": 1.2,
        }],
        span_ranking=[{
            "model_index": 0,
            "rank_within_message": 1,
            "span_text": "part second",
            "start_char": content_start + 6,
            "end_char": content_start + 17,
            "direction": "scam",
            "delta_logit": 0.7,
        }],
        _all_span_effects=[{
            "model_index": 0,
            "span_text": "part second",
            "start_char": content_start + 6,
            "end_char": content_start + 17,
            "direction": "scam",
            "delta_logit": 0.7,
        }],
    )

    payload = format_online_explanation(conversation, explanation, "cpu")

    assert payload["items"][0]["message"]["balloonIds"] == ["b1", "b2"]
    references = payload["items"][0]["span"]["balloonReferences"]
    assert references == [
        {"balloonId": "b1", "startChar": 6, "endChar": 10, "text": "part"},
        {"balloonId": "b2", "startChar": 0, "endChar": 6, "text": "second"},
    ]
    assert payload["items"][0]["dangerousSpan"]["balloonReferences"] == references


def test_dangerous_span_prefers_scam_when_strongest_absolute_span_is_ham() -> None:
    conversation = canonicalize_payload(_payload())
    turn = conversation.turns[0]
    content_start = turn.content_start
    ham = {
        "model_index": 0,
        "rank_within_message": 1,
        "span_text": "first part",
        "start_char": content_start,
        "end_char": content_start + 10,
        "direction": "ham",
        "delta_logit": -1.2,
    }
    scam = {
        "model_index": 0,
        "rank_within_message": 2,
        "span_text": "second part",
        "start_char": content_start + 11,
        "end_char": content_start + 22,
        "direction": "scam",
        "delta_logit": 0.6,
    }
    explanation = SimpleNamespace(
        baseline={"logit": 2.0, "probability_scam": 0.88, "predicted_label": 1},
        top_messages=[{"model_index": 0, "direction": "scam", "delta_logit": 1.2}],
        span_ranking=[ham],
        _all_span_effects=[ham, scam],
    )

    item = format_online_explanation(conversation, explanation, "cpu")["items"][0]

    assert item["span"]["direction"] == "ham"
    assert item["dangerousSpan"]["direction"] == "scam"
    assert item["dangerousSpan"]["text"] == "second part"


def test_dangerous_span_is_null_when_message_has_no_positive_scam_span() -> None:
    conversation = canonicalize_payload(_payload())
    turn = conversation.turns[0]
    ham = {
        "model_index": 0,
        "rank_within_message": 1,
        "span_text": "first part",
        "start_char": turn.content_start,
        "end_char": turn.content_start + 10,
        "direction": "ham",
        "delta_logit": -0.8,
    }
    explanation = SimpleNamespace(
        baseline={"logit": 2.0, "probability_scam": 0.88, "predicted_label": 1},
        top_messages=[{"model_index": 0, "direction": "scam", "delta_logit": 1.2}],
        span_ranking=[ham],
        _all_span_effects=[ham],
    )

    item = format_online_explanation(conversation, explanation, "cpu")["items"][0]

    assert item["dangerousSpan"] is None


def test_ham_does_not_create_explanation_job() -> None:
    queue = ExplanationJobQueue(ttl_seconds=30)
    service = OnlineExplainabilityService(FakeRuntime(0.2), queue=queue)
    try:
        result = service.analyse(_payload())
        assert result["isScam"] is False
        assert result["explanation"]["status"] == "not_requested"
    finally:
        queue.shutdown()


def test_identical_scam_requests_reuse_work_but_keep_analysis_ids() -> None:
    runtime = FakeRuntime(0.9)
    queue = ExplanationJobQueue(ttl_seconds=30)
    service = OnlineExplainabilityService(runtime, queue=queue)
    try:
        first = service.analyse(_payload("analysis-1"))
        first_status = _wait(service, first["explanation"]["jobId"])
        second = service.analyse(_payload("analysis-2"))
        second_status = _wait(service, second["explanation"]["jobId"])

        assert first["explanation"]["reused"] is False
        assert second["explanation"]["reused"] is True
        assert first_status["analysisId"] == "analysis-1"
        assert second_status["analysisId"] == "analysis-2"
        assert runtime.explain_count == 1
    finally:
        queue.shutdown()


def test_flask_contract_returns_202_while_job_is_available() -> None:
    queue = ExplanationJobQueue(ttl_seconds=30)
    service = OnlineExplainabilityService(FakeRuntime(0.9), queue=queue)
    app = create_app(service)
    client = app.test_client()
    try:
        response = client.post("/api/analyse_history", json=_payload())
        assert response.status_code == 200
        body = response.get_json()
        assert body["model"]["name"] == "bge-transformer"

        status = client.get(body["explanation"]["pollUrl"])
        assert status.status_code in {200, 202}
        assert status.get_json()["analysisId"] == "analysis-1"
    finally:
        queue.shutdown()
