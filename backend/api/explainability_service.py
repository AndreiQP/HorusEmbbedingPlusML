"""Online BGE inference with traceable WhatsApp balloon identifiers."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from machine_learning.studies.hierarchical_perturbation import (
    ConversationTurn,
    HierarchicalPerturbationExplainer,
    load_bge_encoder,
    load_bge_finalist_model,
    parse_conversation,
)


API_VERSION = "v1"
MAX_TURNS = 100
TOP_MESSAGES = 6
MAX_NGRAM = 5
DEFAULT_JOB_TTL_SECONDS = 30 * 60


class ContractError(ValueError):
    """Invalid browser/API contract."""


@dataclass(frozen=True)
class BalloonSegment:
    balloon_id: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class CanonicalTurn:
    source_index: int
    model_index: int
    turn_id: str
    speaker: str
    content: str
    model_text: str
    content_start: int
    conversation_start: int
    conversation_end: int
    balloons: tuple[BalloonSegment, ...]

    @property
    def balloon_ids(self) -> list[str]:
        return [segment.balloon_id for segment in self.balloons]

    def as_model_turn(self) -> ConversationTurn:
        return ConversationTurn(
            original_index=self.source_index,
            model_index=self.model_index,
            speaker=self.speaker,
            text=self.model_text,
            content_start=self.content_start,
            conversation_start=self.conversation_start,
            conversation_end=self.conversation_end,
        )


@dataclass(frozen=True)
class CanonicalConversation:
    api_version: str
    analysis_id: str
    chat_id: str
    conversation_version: int
    turns: tuple[CanonicalTurn, ...]
    conversation_text: str
    trigger: dict[str, Any] | None = None

    def model_turns(self) -> list[ConversationTurn]:
        return [turn.as_model_turn() for turn in self.turns]

    def cache_payload(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "chat_id": self.chat_id,
            "conversation_version": self.conversation_version,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "balloons": [
                        {"balloon_id": item.balloon_id, "text": item.text}
                        for item in turn.balloons
                    ],
                }
                for turn in self.turns
            ],
        }


@dataclass(frozen=True)
class PredictionSnapshot:
    logit: float
    probability: float
    embeddings: np.ndarray
    device: str


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} deve ser uma string não vazia")
    return value


def _canonical_turns(raw_turns: Any) -> tuple[tuple[CanonicalTurn, ...], str]:
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ContractError("turns deve ser uma lista não vazia")

    seen_turn_ids: set[str] = set()
    seen_balloon_ids: set[str] = set()
    parsed: list[tuple[int, str, str, str, tuple[BalloonSegment, ...]]] = []
    for source_index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            raise ContractError(f"turns[{source_index}] deve ser um objeto")
        turn_id = _required_string(raw_turn.get("turnId"), f"turns[{source_index}].turnId")
        if turn_id in seen_turn_ids:
            raise ContractError(f"turnId duplicado: {turn_id}")
        seen_turn_ids.add(turn_id)
        speaker = raw_turn.get("speaker")
        if speaker not in {"Innocent", "Suspect"}:
            raise ContractError(f"speaker inválido no turno {turn_id}: {speaker!r}")
        raw_balloons = raw_turn.get("balloons")
        if not isinstance(raw_balloons, list) or not raw_balloons:
            raise ContractError(f"turno {turn_id} deve conter balloons")

        segments: list[BalloonSegment] = []
        content_parts: list[str] = []
        cursor = 0
        for balloon_index, raw_balloon in enumerate(raw_balloons):
            if not isinstance(raw_balloon, dict):
                raise ContractError(f"balloon {balloon_index} do turno {turn_id} é inválido")
            balloon_id = _required_string(
                raw_balloon.get("balloonId"),
                f"turns[{source_index}].balloons[{balloon_index}].balloonId",
            )
            if balloon_id in seen_balloon_ids:
                raise ContractError(f"balloonId duplicado: {balloon_id}")
            seen_balloon_ids.add(balloon_id)
            text = _required_string(
                raw_balloon.get("text"),
                f"turns[{source_index}].balloons[{balloon_index}].text",
            )
            if content_parts:
                cursor += 1
            start = cursor
            cursor += len(text)
            segments.append(BalloonSegment(balloon_id, text, start, cursor))
            content_parts.append(text)

        content = " ".join(content_parts)
        model_text = f"{speaker}: {content}"
        parsed.append((source_index, turn_id, speaker, model_text, tuple(segments)))

    selected = parsed[-MAX_TURNS:]
    turns: list[CanonicalTurn] = []
    conversation_parts: list[str] = []
    conversation_cursor = 0
    for model_index, (source_index, turn_id, speaker, model_text, segments) in enumerate(selected):
        if conversation_parts:
            conversation_cursor += 1
        conversation_start = conversation_cursor
        conversation_cursor += len(model_text)
        content_start = len(f"{speaker}: ")
        turns.append(CanonicalTurn(
            source_index=source_index,
            model_index=model_index,
            turn_id=turn_id,
            speaker=speaker,
            content=model_text[content_start:],
            model_text=model_text,
            content_start=content_start,
            conversation_start=conversation_start,
            conversation_end=conversation_cursor,
            balloons=segments,
        ))
        conversation_parts.append(model_text)
    return tuple(turns), " ".join(conversation_parts)


def _legacy_turns(history: str) -> tuple[tuple[CanonicalTurn, ...], str]:
    """Temporary compatibility path for callers that still send ``historico``."""
    parsed = parse_conversation(history, max_turns=MAX_TURNS)
    raw_turns = []
    for turn in parsed:
        content = turn.text[turn.content_start:]
        balloon_id = f"legacy:{turn.original_index}"
        raw_turns.append({
            "turnId": f"turn:{balloon_id}",
            "speaker": turn.speaker,
            "balloons": [{"balloonId": balloon_id, "text": content}],
        })
    return _canonical_turns(raw_turns)


def canonicalize_payload(payload: Any) -> CanonicalConversation:
    if not isinstance(payload, dict):
        raise ContractError("Corpo JSON inválido")
    api_version = payload.get("apiVersion", API_VERSION)
    if api_version != API_VERSION:
        raise ContractError(f"apiVersion incompatível: {api_version!r}")
    analysis_id = _required_string(
        payload.get("analysisId") or str(uuid.uuid4()), "analysisId",
    )
    chat_id = _required_string(
        payload.get("chatId") or payload.get("contato") or "unknown-chat", "chatId",
    )
    raw_version = payload.get("conversationVersion", 0)
    if isinstance(raw_version, bool):
        raise ContractError("conversationVersion deve ser inteiro")
    try:
        conversation_version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ContractError("conversationVersion deve ser inteiro") from exc
    if conversation_version < 0:
        raise ContractError("conversationVersion deve ser >= 0")

    if payload.get("turns") is not None:
        turns, conversation_text = _canonical_turns(payload["turns"])
    else:
        history = _required_string(payload.get("historico"), "historico")
        turns, conversation_text = _legacy_turns(history)
    trigger = payload.get("trigger") if isinstance(payload.get("trigger"), dict) else None
    return CanonicalConversation(
        api_version=api_version,
        analysis_id=analysis_id,
        chat_id=chat_id,
        conversation_version=conversation_version,
        turns=turns,
        conversation_text=conversation_text,
        trigger=trigger,
    )


def _cuda_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text or "cudnn" in text or "cublas" in text


class BGERuntime:
    """Long-lived BGE/Transformer runtime with CUDA-to-CPU fallback."""

    seed = 42

    def __init__(
        self,
        *,
        checkpoint_root: str | None = None,
        model_name: str | None = None,
        device_preference: str = "auto",
    ):
        self.checkpoint_root = checkpoint_root
        self.model_name = model_name
        self.device_preference = device_preference.lower()
        if self.device_preference not in {"auto", "cuda", "cpu"}:
            raise ValueError("HORUS_DEVICE deve ser auto, cuda ou cpu")
        self._lock = threading.RLock()
        self.device = "cpu"
        self.explainer: HierarchicalPerturbationExplainer
        self.model_metadata: dict[str, Any]
        self._load_initial_runtime()

    @classmethod
    def from_environment(cls) -> "BGERuntime":
        return cls(
            checkpoint_root=os.getenv("HORUS_BGE_CHECKPOINT_ROOT") or None,
            model_name=os.getenv("HORUS_BGE_MODEL") or None,
            device_preference=os.getenv("HORUS_DEVICE", "auto"),
        )

    def _build(self, device: str) -> None:
        import torch

        torch_device = torch.device(device)
        model, metadata = load_bge_finalist_model(
            seed=self.seed,
            device=torch_device,
            checkpoint_root=self.checkpoint_root,
        )
        encoder = load_bge_encoder(device=torch_device, model_name=self.model_name)
        explainer = HierarchicalPerturbationExplainer(
            model,
            encoder,
            device=torch_device,
            seed=self.seed,
            model_metadata=metadata,
        )
        vector = explainer._encode(["Suspect: Horus device check"])
        explainer._infer_arrays(vector[None, :, :])
        self.explainer = explainer
        self.model_metadata = metadata
        self.device = device

    def _load_initial_runtime(self) -> None:
        import torch

        wants_cuda = self.device_preference != "cpu" and torch.cuda.is_available()
        if wants_cuda:
            try:
                self._build("cuda")
                return
            except RuntimeError as exc:
                if not _cuda_error(exc):
                    raise
                print(f"[horus] CUDA indisponível durante inicialização; usando CPU: {exc}")
        elif self.device_preference == "cuda":
            print("[horus] HORUS_DEVICE=cuda, mas CUDA não está disponível; usando CPU")
        self._build("cpu")

    def _switch_to_cpu(self, reason: BaseException) -> None:
        if self.device == "cpu":
            raise reason
        print(f"[horus] Falha CUDA em execução; recarregando em CPU: {reason}")
        self._build("cpu")

    def _with_fallback(self, operation: Callable[[], Any]) -> Any:
        with self._lock:
            try:
                return operation()
            except RuntimeError as exc:
                if self.device != "cuda" or not _cuda_error(exc):
                    raise
                self._switch_to_cpu(exc)
                return operation()

    def predict(self, conversation: CanonicalConversation) -> PredictionSnapshot:
        def operation() -> PredictionSnapshot:
            turns = conversation.model_turns()
            embeddings = self.explainer._encode([turn.text for turn in turns])
            logits, probabilities = self.explainer._infer_arrays(embeddings[None, :, :])
            return PredictionSnapshot(
                logit=float(logits[0]),
                probability=float(probabilities[0]),
                embeddings=embeddings,
                device=self.device,
            )

        return self._with_fallback(operation)

    def explain(
        self,
        conversation: CanonicalConversation,
        original_embeddings: np.ndarray,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self.explainer.prepare_turns(
                conversation.model_turns(),
                conversation=conversation.conversation_text,
                top_messages=TOP_MESSAGES,
                max_ngram=MAX_NGRAM,
                original_embeddings=original_embeddings,
            )
            explanation = self.explainer.explain_prepared(
                prepared,
                top_messages=TOP_MESSAGES,
                top_spans_per_message=1,
                analysis_scope="online",
                defer_span_fidelity=True,
                include_fidelity=False,
            )
            return format_online_explanation(conversation, explanation, self.device)

        return self._with_fallback(operation)


def _relative(value: float, maximum: float) -> float:
    return float(abs(value) / maximum) if maximum > 0 else 0.0


def _balloon_references(
    turn: CanonicalTurn,
    start_char: int,
    end_char: int,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for segment in turn.balloons:
        overlap_start = max(start_char, segment.start_char)
        overlap_end = min(end_char, segment.end_char)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - segment.start_char
        local_end = overlap_end - segment.start_char
        references.append({
            "balloonId": segment.balloon_id,
            "startChar": local_start,
            "endChar": local_end,
            "text": segment.text[local_start:local_end],
        })
    return references


def _online_span_payload(
    turn: CanonicalTurn,
    span_row: dict[str, Any] | None,
    maximum: float,
) -> dict[str, Any] | None:
    if span_row is None:
        return None
    start = max(0, int(span_row["start_char"]) - turn.content_start)
    end = min(len(turn.content), int(span_row["end_char"]) - turn.content_start)
    return {
        "text": str(span_row["span_text"]),
        "turnStartChar": start,
        "turnEndChar": end,
        "direction": str(span_row["direction"]),
        "deltaLogit": float(span_row["delta_logit"]),
        "relativeImpact": _relative(float(span_row["delta_logit"]), maximum),
        "balloonReferences": _balloon_references(turn, start, end),
    }


def format_online_explanation(
    conversation: CanonicalConversation,
    explanation,
    device: str,
) -> dict[str, Any]:
    turns_by_index = {turn.model_index: turn for turn in conversation.turns}
    spans_by_index = {
        int(row["model_index"]): row
        for row in explanation.span_ranking
        if int(row.get("rank_within_message", 0)) == 1
    }
    all_span_effects = getattr(explanation, "_all_span_effects", explanation.span_ranking)
    dangerous_spans_by_index: dict[int, dict[str, Any]] = {}
    for row in all_span_effects:
        if str(row.get("direction")) != "scam" or float(row.get("delta_logit", 0.0)) <= 1e-4:
            continue
        model_index = int(row["model_index"])
        current = dangerous_spans_by_index.get(model_index)
        if current is None or float(row["delta_logit"]) > float(current["delta_logit"]):
            dangerous_spans_by_index[model_index] = row
    selected = [
        row for row in explanation.top_messages
        if int(row["model_index"]) in turns_by_index
    ]
    message_max = max((abs(float(row["delta_logit"])) for row in selected), default=0.0)
    selected_spans = [
        spans_by_index[int(row["model_index"])]
        for row in selected
        if int(row["model_index"]) in spans_by_index
    ]
    span_max = max((abs(float(row["delta_logit"])) for row in selected_spans), default=0.0)
    selected_dangerous_spans = [
        dangerous_spans_by_index[int(row["model_index"])]
        for row in selected
        if str(row.get("direction")) == "scam"
        and int(row["model_index"]) in dangerous_spans_by_index
    ]
    dangerous_span_max = max(
        (float(row["delta_logit"]) for row in selected_dangerous_spans),
        default=0.0,
    )
    items: list[dict[str, Any]] = []
    for rank, message_row in enumerate(selected, start=1):
        model_index = int(message_row["model_index"])
        turn = turns_by_index[model_index]
        span_row = spans_by_index.get(model_index)
        span_payload = _online_span_payload(turn, span_row, span_max)
        dangerous_span_payload = _online_span_payload(
            turn,
            dangerous_spans_by_index.get(model_index),
            dangerous_span_max,
        )
        items.append({
            "rank": rank,
            "message": {
                "modelIndex": model_index,
                "turnId": turn.turn_id,
                "speaker": turn.speaker,
                "text": turn.content,
                "balloonIds": turn.balloon_ids,
                "direction": str(message_row["direction"]),
                "deltaLogit": float(message_row["delta_logit"]),
                "relativeImpact": _relative(float(message_row["delta_logit"]), message_max),
            },
            "span": span_payload,
            "dangerousSpan": dangerous_span_payload,
        })
    return {
        "device": device,
        "baseline": {
            "logit": float(explanation.baseline["logit"]),
            "probabilityScam": float(explanation.baseline["probability_scam"]),
            "predictedLabel": int(explanation.baseline["predicted_label"]),
        },
        "items": items,
    }


@dataclass
class _WorkRecord:
    work_id: str
    cache_key: str
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0


@dataclass(frozen=True)
class _JobAlias:
    job_id: str
    work_id: str
    analysis_id: str
    chat_id: str
    conversation_version: int
    created_at: float


class ExplanationJobQueue:
    """Single-worker in-memory queue with request aliases and 30-minute reuse."""

    def __init__(self, ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS):
        self.ttl_seconds = int(ttl_seconds)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="horus-explain")
        self._lock = threading.RLock()
        self._works: dict[str, _WorkRecord] = {}
        self._aliases: dict[str, _JobAlias] = {}
        self._cache: dict[str, str] = {}

    def _cleanup(self) -> None:
        now = time.time()
        expired = {
            work_id for work_id, work in self._works.items()
            if work.status not in {"queued", "running"} and work.expires_at <= now
        }
        for work_id in expired:
            work = self._works.pop(work_id)
            if self._cache.get(work.cache_key) == work_id:
                self._cache.pop(work.cache_key, None)
        self._aliases = {
            job_id: alias for job_id, alias in self._aliases.items()
            if alias.work_id in self._works
        }

    def submit(
        self,
        conversation: CanonicalConversation,
        cache_key: str,
        operation: Callable[[], dict[str, Any]],
    ) -> tuple[str, bool]:
        with self._lock:
            self._cleanup()
            work_id = self._cache.get(cache_key)
            reused = bool(work_id and work_id in self._works)
            if not reused:
                work_id = str(uuid.uuid4())
                self._works[work_id] = _WorkRecord(work_id=work_id, cache_key=cache_key)
                self._cache[cache_key] = work_id
                self._executor.submit(self._execute, work_id, operation)
            job_id = str(uuid.uuid4())
            self._aliases[job_id] = _JobAlias(
                job_id=job_id,
                work_id=str(work_id),
                analysis_id=conversation.analysis_id,
                chat_id=conversation.chat_id,
                conversation_version=conversation.conversation_version,
                created_at=time.time(),
            )
            return job_id, reused

    def _execute(self, work_id: str, operation: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            work = self._works[work_id]
            work.status = "running"
        try:
            result = operation()
        except Exception as exc:
            with self._lock:
                work = self._works[work_id]
                work.status = "failed"
                work.error = f"{type(exc).__name__}: {exc}"
                work.expires_at = time.time() + self.ttl_seconds
                if self._cache.get(work.cache_key) == work_id:
                    self._cache.pop(work.cache_key, None)
            return
        with self._lock:
            work = self._works[work_id]
            work.status = "completed"
            work.result = result
            work.expires_at = time.time() + self.ttl_seconds

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup()
            alias = self._aliases.get(job_id)
            if alias is None:
                return None
            work = self._works.get(alias.work_id)
            if work is None:
                return None
            payload: dict[str, Any] = {
                "apiVersion": API_VERSION,
                "jobId": job_id,
                "analysisId": alias.analysis_id,
                "chatId": alias.chat_id,
                "conversationVersion": alias.conversation_version,
                "status": work.status,
            }
            if work.status == "completed" and work.result is not None:
                payload.update(work.result)
            elif work.status == "failed":
                payload["error"] = work.error or "Falha desconhecida"
            return payload

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class OnlineExplainabilityService:
    def __init__(
        self,
        runtime: Any,
        *,
        queue: ExplanationJobQueue | None = None,
    ):
        self.runtime = runtime
        ttl = int(os.getenv("HORUS_EXPLANATION_TTL_SECONDS", DEFAULT_JOB_TTL_SECONDS))
        self.queue = queue or ExplanationJobQueue(ttl_seconds=ttl)

    @classmethod
    def from_environment(cls) -> "OnlineExplainabilityService":
        return cls(BGERuntime.from_environment())

    def _cache_key(self, conversation: CanonicalConversation) -> str:
        payload = {
            **conversation.cache_payload(),
            "seed": int(getattr(self.runtime, "seed", 42)),
            "top_messages": TOP_MESSAGES,
            "max_ngram": MAX_NGRAM,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def analyse(self, payload: Any) -> dict[str, Any]:
        conversation = canonicalize_payload(payload)
        prediction = self.runtime.predict(conversation)
        is_scam = prediction.probability >= 0.5
        response: dict[str, Any] = {
            "apiVersion": API_VERSION,
            "analysisId": conversation.analysis_id,
            "chatId": conversation.chat_id,
            "conversationVersion": conversation.conversation_version,
            "probabilidade": prediction.probability,
            "isScam": is_scam,
            "model": {
                "name": "bge-transformer",
                "seed": int(getattr(self.runtime, "seed", 42)),
                "device": prediction.device,
            },
        }
        if not is_scam:
            response["explanation"] = {
                "status": "not_requested",
                "reason": "prediction_ham",
            }
            return response

        embeddings = np.asarray(prediction.embeddings, dtype=np.float32).copy()
        job_id, reused = self.queue.submit(
            conversation,
            self._cache_key(conversation),
            lambda: self.runtime.explain(conversation, embeddings),
        )
        response["explanation"] = {
            "status": "queued",
            "jobId": job_id,
            "pollUrl": f"/api/explanations/{job_id}",
            "reused": reused,
        }
        return response

    def explanation_status(self, job_id: str) -> dict[str, Any] | None:
        return self.queue.get(job_id)
