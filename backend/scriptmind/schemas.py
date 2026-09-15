"""Dependency-light schemas used by annotation, training and inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .taxonomy import INTENTS, STAGES, validate_intent_stage


@dataclass
class TurnAnnotation:
    primary_intent: str
    stage: str
    rationale: str
    evidence_turn_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_intent_stage(self.primary_intent, self.stage)
        if not self.rationale.strip():
            raise ValueError("Annotation rationale cannot be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TurnAnnotation":
        return cls(
            primary_intent=str(value["primary_intent"]),
            stage=str(value["stage"]),
            rationale=str(value["rationale"]),
            evidence_turn_ids=[str(x) for x in value.get("evidence_turn_ids", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "primary_intent": {"type": "string", "enum": list(INTENTS)},
                "stage": {"type": "string", "enum": list(STAGES)},
                "rationale": {"type": "string"},
                "evidence_turn_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["primary_intent", "stage", "rationale", "evidence_turn_ids"],
            "additionalProperties": False,
        }


@dataclass
class ScriptMindPrediction:
    label: Literal["scam", "non_scam"]
    current_intent: str | None = None
    next_intent: str | None = None
    next_utterance: str | None = None
    rationale: str | None = None
    evidence_turn_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.label == "non_scam":
            if any((self.current_intent, self.next_intent, self.next_utterance, self.rationale)):
                raise ValueError("non_scam predictions must leave task-specific fields null")
        else:
            for value in (self.current_intent, self.next_intent):
                if value is not None and value not in INTENTS:
                    raise ValueError(f"Unknown intent: {value}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScriptMindPrediction":
        return cls(
            label=value["label"],
            current_intent=value.get("current_intent"),
            next_intent=value.get("next_intent"),
            next_utterance=value.get("next_utterance"),
            rationale=value.get("rationale"),
            evidence_turn_ids=[str(x) for x in value.get("evidence_turn_ids", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        nullable_intent = {"anyOf": [{"type": "string", "enum": list(INTENTS)}, {"type": "null"}]}
        nullable_text = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["scam", "non_scam"]},
                "current_intent": nullable_intent,
                "next_intent": nullable_intent,
                "next_utterance": nullable_text,
                "rationale": nullable_text,
                "evidence_turn_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["label", "current_intent", "next_intent", "next_utterance", "rationale", "evidence_turn_ids"],
            "additionalProperties": False,
        }


@dataclass
class JudgeAssessment:
    detection_correctness: int
    intent_consistency: int
    next_utterance_relevance: int
    rationale_grounding: int
    preventive_utility: int
    hallucination: bool
    explanation: str

    def __post_init__(self) -> None:
        for field_name in (
            "detection_correctness", "intent_consistency", "next_utterance_relevance",
            "rationale_grounding", "preventive_utility",
        ):
            value = getattr(self, field_name)
            if not 1 <= value <= 5:
                raise ValueError(f"{field_name} must be between 1 and 5")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JudgeAssessment":
        return cls(
            detection_correctness=int(value["detection_correctness"]),
            intent_consistency=int(value["intent_consistency"]),
            next_utterance_relevance=int(value["next_utterance_relevance"]),
            rationale_grounding=int(value["rationale_grounding"]),
            preventive_utility=int(value["preventive_utility"]),
            hallucination=bool(value["hallucination"]),
            explanation=str(value["explanation"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        score = {"type": "integer", "minimum": 1, "maximum": 5}
        return {
            "type": "object",
            "properties": {
                "detection_correctness": score,
                "intent_consistency": score,
                "next_utterance_relevance": score,
                "rationale_grounding": score,
                "preventive_utility": score,
                "hallucination": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": [
                "detection_correctness", "intent_consistency", "next_utterance_relevance",
                "rationale_grounding", "preventive_utility", "hallucination", "explanation",
            ],
            "additionalProperties": False,
        }
