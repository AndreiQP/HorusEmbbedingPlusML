"""Predeclared statistical tests for paired ScriptMind comparisons."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def mcnemar_exact(y_true: Iterable[int], prediction_a: Iterable[int], prediction_b: Iterable[int]) -> dict[str, float | int]:
    """Exact paired McNemar test using discordant correctness counts."""
    from scipy.stats import binomtest

    truth = np.asarray(list(y_true))
    first = np.asarray(list(prediction_a))
    second = np.asarray(list(prediction_b))
    if not (len(truth) == len(first) == len(second)):
        raise ValueError("Paired arrays must have equal length")
    first_correct, second_correct = first == truth, second == truth
    a_only = int(np.sum(first_correct & ~second_correct))
    b_only = int(np.sum(~first_correct & second_correct))
    discordant = a_only + b_only
    p_value = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "discordant": discordant, "p_value": p_value}


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm step-down family-wise error correction in original order."""
    values = np.asarray(list(p_values), dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be between 0 and 1")
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()
