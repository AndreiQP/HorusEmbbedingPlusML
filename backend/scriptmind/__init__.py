"""Isolated ScriptMind research strategy for Horus."""

from .csid import build_csid
from .schemas import ScriptMindPrediction

__all__ = ["ScriptMindPrediction", "build_csid", "annotate_dialogues", "train_scriptmind", "evaluate_scriptmind"]


def annotate_dialogues(*args, **kwargs):
    from .annotation import annotate_dialogues as implementation

    return implementation(*args, **kwargs)


def train_scriptmind(*args, **kwargs):
    from .training import train_scriptmind as implementation

    return implementation(*args, **kwargs)


def evaluate_scriptmind(*args, **kwargs):
    from .evaluation import evaluate_scriptmind as implementation

    return implementation(*args, **kwargs)
