"""Versioned Horus crime-script taxonomy."""
from __future__ import annotations

TAXONOMY_VERSION = "horus-crime-script-v1"

INTENT_TO_STAGE = {
    "identity_confirmation": "contact",
    "impersonation": "credibility",
    "problem_or_opportunity": "pretext",
    "false_proof_or_legitimacy": "credibility",
    "urgency_fear_or_threat": "pressure",
    "request_personal_information": "information_extraction",
    "request_credentials_or_otp": "information_extraction",
    "request_payment": "victim_action",
    "link_download_or_installation": "victim_action",
    "secrecy_or_isolation": "control",
    "objection_handling": "control",
    "follow_up_coordination": "closure",
    "legitimate_interaction": "benign_or_other",
    "other_or_insufficient": "benign_or_other",
}

INTENTS = tuple(INTENT_TO_STAGE)
STAGES = tuple(dict.fromkeys(INTENT_TO_STAGE.values()))


def validate_intent_stage(intent: str, stage: str) -> None:
    if intent not in INTENT_TO_STAGE:
        raise ValueError(f"Unknown ScriptMind intent: {intent}")
    expected = INTENT_TO_STAGE[intent]
    if stage != expected:
        raise ValueError(f"Intent {intent!r} belongs to stage {expected!r}, not {stage!r}")
