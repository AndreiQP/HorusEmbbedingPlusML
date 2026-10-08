"""Explicabilidade local hierárquica do Transformer BGE por perturbação.

O classificador e o BGE permanecem congelados. A etapa macro troca uma mensagem
por um embedding neutro; a etapa micro remove n-grams da mensagem, recodifica
somente esse texto e mantém os demais vetores da conversa inalterados.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..cache import ModelCache
from ..data import get_embedding_dim
from ..transformer.runner import TransformerScamClassifier
from .common import RESULTS_ROOT, TRANSFORMER_STUDIES, atomic_csv, atomic_json, stable_hash


BGE_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EFFECT_EPSILON = 1e-4
DEFAULT_REPRODUCTION_ATOL = 1e-5
DEFAULT_REPRODUCTION_RTOL = 1e-4
DEFAULT_MIN_MESSAGES = 7
TURN_MARKER_RE = re.compile(r"(?<!\S)(?P<speaker>Innocent|Suspect):")
WORD_RE = re.compile(r"\S+")


class ReproductionError(RuntimeError):
    """Indica que texto, embeddings e checkpoint não reproduzem o pipeline salvo."""


@dataclass(frozen=True)
class ConversationTurn:
    original_index: int
    model_index: int
    speaker: str
    text: str
    content_start: int
    conversation_start: int
    conversation_end: int


@dataclass
class HierarchicalExplanation:
    baseline: dict[str, Any]
    top_messages: list[dict[str, Any]]
    strongest_scam_span: dict[str, Any] | None
    strongest_ham_span: dict[str, Any] | None
    span_ranking: list[dict[str, Any]]
    message_ranking: list[dict[str, Any]]
    fidelity: list[dict[str, Any]]
    metadata: dict[str, Any]
    _all_span_effects: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self, *, include_all_spans: bool = False) -> dict[str, Any]:
        payload = {
            "baseline": self.baseline,
            "top_messages": self.top_messages,
            "strongest_scam_span": self.strongest_scam_span,
            "strongest_ham_span": self.strongest_ham_span,
            "span_ranking": self.span_ranking,
            "message_ranking": self.message_ranking,
            "fidelity": self.fidelity,
            "metadata": self.metadata,
        }
        if include_all_spans:
            payload["all_span_effects"] = self._all_span_effects
        return payload


@dataclass
class PreparedPerturbations:
    """Entradas BGE compartilhadas por todos os Transformers de uma conversa."""

    conversation: str
    turns: list[ConversationTurn]
    original_embeddings: np.ndarray
    neutral_vectors: np.ndarray
    candidates: list[dict[str, Any]]
    perturbed_embeddings: np.ndarray
    candidate_source_seed: int
    max_ngram: int
    source_baseline_logit: float
    source_baseline_probability: float
    source_message_ranking: list[dict[str, Any]]
    source_top_messages: list[dict[str, Any]]
    embedding_source: str


def parse_conversation(text: str, max_turns: int = 100) -> list[ConversationTurn]:
    """Separa turnos, preserva offsets e replica o corte dos últimos 100 turnos."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A conversa deve conter texto")
    if max_turns < 1:
        raise ValueError("max_turns deve ser >= 1")
    markers = list(TURN_MARKER_RE.finditer(text))
    if not markers:
        raise ValueError("Nenhum turno com prefixo Innocent: ou Suspect: foi encontrado")
    if text[:markers[0].start()].strip():
        raise ValueError("Há texto antes do primeiro prefixo Innocent:/Suspect:")

    parsed: list[tuple[int, str, str, int, int, int]] = []
    for original_index, marker in enumerate(markers):
        start = marker.start()
        raw_end = markers[original_index + 1].start() if original_index + 1 < len(markers) else len(text)
        end = raw_end
        while end > start and text[end - 1].isspace():
            end -= 1
        message = text[start:end]
        colon_end = marker.end() - start
        content_start = colon_end
        while content_start < len(message) and message[content_start].isspace():
            content_start += 1
        parsed.append((
            original_index, marker.group("speaker"), message, content_start, start, end,
        ))

    selected = parsed[-max_turns:]
    return [
        ConversationTurn(
            original_index=original_index,
            model_index=model_index,
            speaker=speaker,
            text=message,
            content_start=content_start,
            conversation_start=start,
            conversation_end=end,
        )
        for model_index, (original_index, speaker, message, content_start, start, end)
        in enumerate(selected)
    ]


def _delete_span(text: str, start: int, end: int) -> str:
    left = text[:start].rstrip()
    right = text[end:].lstrip()
    if left and right:
        return f"{left} {right}"
    return left or right


def generate_ngram_perturbations(
    turn: ConversationTurn,
    max_ngram: int = 5,
) -> list[dict[str, Any]]:
    """Gera remoções contíguas apenas no conteúdo textual do turno."""
    if max_ngram < 1:
        raise ValueError("max_ngram deve ser >= 1")
    content = turn.text[turn.content_start:]
    words = list(WORD_RE.finditer(content))
    rows: list[dict[str, Any]] = []
    for size in range(1, min(max_ngram, len(words)) + 1):
        for start_word in range(len(words) - size + 1):
            end_word = start_word + size
            start_char = turn.content_start + words[start_word].start()
            end_char = turn.content_start + words[end_word - 1].end()
            rows.append({
                "original_index": turn.original_index,
                "model_index": turn.model_index,
                "speaker": turn.speaker,
                "message_text": turn.text,
                "span_text": turn.text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
                "conversation_start_char": turn.conversation_start + start_char,
                "conversation_end_char": turn.conversation_start + end_char,
                "start_word": start_word,
                "end_word": end_word,
                "ngram_size": size,
                "perturbed_text": _delete_span(turn.text, start_char, end_char),
            })
    return rows


def _direction(delta_logit: float, epsilon: float) -> str:
    if delta_logit > epsilon:
        return "scam"
    if delta_logit < -epsilon:
        return "ham"
    return "neutral"


def _effect_sort_key(row: dict[str, Any]) -> tuple:
    return (
        -abs(float(row["delta_logit"])),
        -abs(float(row["delta_probability"])),
        int(row.get("ngram_size", 0)),
        int(row.get("original_index", 0)),
        int(row.get("start_word", 0)),
    )


def _supported_score(probability: float, direction: str, predicted_label: int) -> float:
    label = 1 if direction == "scam" else 0 if direction == "ham" else predicted_label
    return probability if label == 1 else 1.0 - probability


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Spearman sem dependência de scipy; retorna None para vetores degenerados."""
    if len(left) < 2 or len(left) != len(right):
        return None
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def encoder_provenance(encoder) -> dict[str, Any]:
    """Extrai metadados sem pressupor uma implementação manual de pooling."""
    pooling: dict[str, Any] = {}
    revision = None
    normalize_module = False
    modules = getattr(encoder, "_modules", {})
    for module in modules.values():
        normalize_module = normalize_module or type(module).__name__ == "Normalize"
        if hasattr(module, "pooling_mode"):
            pooling["pooling_mode"] = str(getattr(module, "pooling_mode"))
        if hasattr(module, "get_config_dict") and type(module).__name__ == "Pooling":
            pooling.update(module.get_config_dict())
        for name in (
            "pooling_mode_cls_token", "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens", "pooling_mode_weightedmean_tokens",
        ):
            if hasattr(module, name):
                pooling[name] = bool(getattr(module, name))
        auto_model = getattr(module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        revision = revision or getattr(config, "_commit_hash", None)
    return {
        "model_name": BGE_MODEL_NAME,
        "revision": revision,
        "pooling": pooling or "defined_by_sentence_transformer_checkpoint",
        "normalize_module": normalize_module,
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "transformers_version": _package_version("transformers"),
        "torch_version": _package_version("torch"),
    }


class HierarchicalPerturbationExplainer:
    """Executa triagem de mensagens e oclusão granular com modelos congelados."""

    def __init__(
        self,
        model,
        encoder,
        *,
        device=None,
        seed: int = 42,
        model_metadata: dict[str, Any] | None = None,
        max_turns: int = 100,
        embedding_batch_size: int = 64,
        inference_batch_size: int = 32,
        effect_epsilon: float = DEFAULT_EFFECT_EPSILON,
        reproduction_atol: float = DEFAULT_REPRODUCTION_ATOL,
        reproduction_rtol: float = DEFAULT_REPRODUCTION_RTOL,
    ):
        import torch

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.encoder = encoder
        self.seed = int(seed)
        self.model_metadata = dict(model_metadata or {})
        self.max_turns = int(max_turns)
        self.embedding_batch_size = int(embedding_batch_size)
        self.inference_batch_size = int(inference_batch_size)
        self.effect_epsilon = float(effect_epsilon)
        self.reproduction_atol = float(reproduction_atol)
        self.reproduction_rtol = float(reproduction_rtol)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, get_embedding_dim("bge")), dtype=np.float32)
        vectors = self.encoder.encode(
            list(texts),
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim == 1:
            result = result[None, :]
        if result.ndim != 2 or result.shape[0] != len(texts):
            raise ValueError(f"Encoder retornou shape inválido: {result.shape}")
        return result

    def _infer_arrays(self, arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if arrays.ndim != 3:
            raise ValueError(f"Esperado [batch, turnos, dimensão], recebido {arrays.shape}")
        logits: list[np.ndarray] = []
        for start in range(0, len(arrays), self.inference_batch_size):
            batch = torch.as_tensor(
                arrays[start:start + self.inference_batch_size],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                output = self.model(batch).reshape(-1)
            logits.append(output.detach().cpu().numpy())
        logit = np.concatenate(logits).astype(np.float64, copy=False)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0)))
        return logit, probability

    def _infer_replacements(
        self,
        original: np.ndarray,
        indices: Sequence[int],
        replacements: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monta cada lote sob demanda para não duplicar todas as sequências em RAM."""
        if len(indices) != len(replacements):
            raise ValueError("Índices e embeddings substitutos estão desalinhados")
        all_logits: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []
        for start in range(0, len(indices), self.inference_batch_size):
            end = start + self.inference_batch_size
            batch_indices = np.asarray(indices[start:end], dtype=int)
            batch_replacements = replacements[start:end]
            batch = np.repeat(original[None, :, :], len(batch_indices), axis=0)
            batch[np.arange(len(batch_indices)), batch_indices] = batch_replacements
            logits, probabilities = self._infer_arrays(batch)
            all_logits.append(logits)
            all_probabilities.append(probabilities)
        if not all_logits:
            return np.asarray([]), np.asarray([])
        return np.concatenate(all_logits), np.concatenate(all_probabilities)

    def _validate_reproduction(
        self,
        vectors: np.ndarray,
        logit: float,
        expected_embeddings: np.ndarray | None,
        expected_logit: float | None,
    ) -> dict[str, Any]:
        if expected_embeddings is None and expected_logit is None:
            return {
                "status": "not_applicable",
                "reason": "conversa sem embeddings/logit de referência",
            }
        metrics: dict[str, Any] = {
            "status": "passed",
            "atol": self.reproduction_atol,
            "rtol": self.reproduction_rtol,
        }
        if expected_embeddings is not None:
            expected = np.asarray(expected_embeddings, dtype=np.float32)
            if expected.shape != vectors.shape:
                raise ReproductionError(
                    f"Shape BGE não reproduzido: novo={vectors.shape}, salvo={expected.shape}"
                )
            difference = np.abs(vectors - expected)
            numerator = np.sum(vectors * expected, axis=1)
            denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(expected, axis=1)
            cosine = numerator / np.clip(denominator, 1e-12, None)
            metrics.update({
                "embedding_max_abs_error": float(difference.max(initial=0.0)),
                "embedding_mean_abs_error": float(difference.mean()),
                "embedding_min_cosine": float(cosine.min(initial=1.0)),
            })
            if not np.allclose(
                vectors, expected, atol=self.reproduction_atol, rtol=self.reproduction_rtol,
            ):
                raise ReproductionError(
                    "Os embeddings BGE atuais não reproduzem os vetores armazenados "
                    f"(erro máximo={metrics['embedding_max_abs_error']:.6g})"
                )
        if expected_logit is not None:
            error = abs(logit - float(expected_logit))
            metrics["logit_abs_error"] = error
            if not np.isclose(
                logit, float(expected_logit),
                atol=self.reproduction_atol, rtol=self.reproduction_rtol,
            ):
                raise ReproductionError(
                    f"Logit não reproduzido: novo={logit:.8g}, salvo={float(expected_logit):.8g}"
                )
        return metrics

    def _validated_original_embeddings(
        self,
        turns: Sequence[ConversationTurn],
        original_embeddings: np.ndarray,
    ) -> np.ndarray:
        original = np.asarray(original_embeddings, dtype=np.float32)
        if original.ndim != 2:
            raise ReproductionError(
                f"Embeddings originais devem ter shape [turnos, dimensão]: {original.shape}"
            )
        original = original[-self.max_turns:]
        if len(original) != len(turns):
            raise ReproductionError(
                "Alinhamento texto/embedding inválido: "
                f"{len(turns)} turnos textuais e {len(original)} vetores"
            )
        expected_dim = get_embedding_dim("bge")
        if original.shape[1] != expected_dim:
            raise ReproductionError(
                f"Dimensão BGE incompatível: {original.shape[1]} != {expected_dim}"
            )
        return original

    def _score_macro(
        self,
        turns: Sequence[ConversationTurn],
        original: np.ndarray,
        neutral_vectors: np.ndarray,
        top_messages: int,
    ) -> tuple[float, float, list[dict[str, Any]], list[dict[str, Any]]]:
        baseline_logits, baseline_probabilities = self._infer_arrays(original[None, :, :])
        baseline_logit = float(baseline_logits[0])
        baseline_probability = float(baseline_probabilities[0])
        predicted_label = int(baseline_probability >= 0.5)
        macro_logits, macro_probabilities = self._infer_replacements(
            original,
            [turn.model_index for turn in turns],
            neutral_vectors,
        )
        message_rows: list[dict[str, Any]] = []
        for turn, changed_logit, changed_probability in zip(
            turns, macro_logits, macro_probabilities,
        ):
            delta_logit = baseline_logit - float(changed_logit)
            delta_probability = baseline_probability - float(changed_probability)
            message_rows.append({
                "original_index": turn.original_index,
                "model_index": turn.model_index,
                "speaker": turn.speaker,
                "text": turn.text,
                "neutral_text": f"{turn.speaker}:",
                "perturbed_logit": float(changed_logit),
                "perturbed_probability": float(changed_probability),
                "delta_logit": delta_logit,
                "abs_delta_logit": abs(delta_logit),
                "delta_probability": delta_probability,
                "abs_delta_probability": abs(delta_probability),
                "direction": _direction(delta_logit, self.effect_epsilon),
                "prediction_changed": (
                    int(float(changed_probability) >= 0.5) != predicted_label
                ),
            })
        message_rows.sort(key=_effect_sort_key)
        selected = message_rows[:min(top_messages, len(message_rows))]
        return baseline_logit, baseline_probability, message_rows, selected

    def prepare_perturbations(
        self,
        conversation: str,
        *,
        top_messages: int = 6,
        max_ngram: int = 5,
        original_embeddings: np.ndarray | None = None,
        neutral_by_speaker: dict[str, np.ndarray] | None = None,
    ) -> PreparedPerturbations:
        """Codifica uma vez os candidatos definidos pelo Transformer canônico."""
        turns = parse_conversation(conversation, max_turns=self.max_turns)
        return self.prepare_turns(
            turns,
            conversation=conversation,
            top_messages=top_messages,
            max_ngram=max_ngram,
            original_embeddings=original_embeddings,
            neutral_by_speaker=neutral_by_speaker,
        )

    def prepare_turns(
        self,
        turns: Sequence[ConversationTurn],
        *,
        conversation: str,
        top_messages: int = 6,
        max_ngram: int = 5,
        original_embeddings: np.ndarray | None = None,
        neutral_by_speaker: dict[str, np.ndarray] | None = None,
    ) -> PreparedPerturbations:
        """Prepara perturbações para turnos já estruturados e alinhados.

        Esta entrada evita redescobrir fronteiras por regex quando o chamador já
        possui IDs e limites confiáveis, como na extensão do navegador.
        """
        if top_messages < 1:
            raise ValueError("top_messages deve ser >= 1")
        turns = list(turns)[-self.max_turns:]
        if not turns:
            raise ValueError("A conversa deve conter ao menos um turno")
        if [turn.model_index for turn in turns] != list(range(len(turns))):
            raise ValueError("Turnos estruturados devem usar model_index contíguo a partir de zero")
        if original_embeddings is None:
            original = self._encode([turn.text for turn in turns])
            embedding_source = "bge_encoded_for_new_conversation"
        else:
            original = self._validated_original_embeddings(turns, original_embeddings)
            embedding_source = "cached_dataset_embeddings"

        neutral_lookup = dict(neutral_by_speaker or {})
        missing_speakers = [
            speaker for speaker in dict.fromkeys(turn.speaker for turn in turns)
            if speaker not in neutral_lookup
        ]
        if missing_speakers:
            vectors = self._encode([f"{speaker}:" for speaker in missing_speakers])
            neutral_lookup.update(dict(zip(missing_speakers, vectors)))
        neutral_vectors = np.stack([
            np.asarray(neutral_lookup[turn.speaker], dtype=np.float32) for turn in turns
        ])
        if neutral_vectors.shape != original.shape:
            raise ReproductionError(
                f"Baselines neutros incompatíveis: {neutral_vectors.shape} != {original.shape}"
            )
        baseline_logit, baseline_probability, message_rows, selected_messages = (
            self._score_macro(turns, original, neutral_vectors, top_messages)
        )
        selected_indices = {int(row["model_index"]) for row in selected_messages}
        candidates: list[dict[str, Any]] = []
        for turn in turns:
            if turn.model_index in selected_indices:
                candidates.extend(generate_ngram_perturbations(turn, max_ngram=max_ngram))
        perturbed_vectors = self._encode([row["perturbed_text"] for row in candidates])
        return PreparedPerturbations(
            conversation=conversation,
            turns=list(turns),
            original_embeddings=original,
            neutral_vectors=neutral_vectors,
            candidates=candidates,
            perturbed_embeddings=perturbed_vectors,
            candidate_source_seed=self.seed,
            max_ngram=max_ngram,
            source_baseline_logit=baseline_logit,
            source_baseline_probability=baseline_probability,
            source_message_ranking=message_rows,
            source_top_messages=selected_messages,
            embedding_source=embedding_source,
        )

    def audit_cached_embeddings(
        self,
        conversation: str,
        original_embeddings: np.ndarray,
    ) -> dict[str, Any]:
        """Recodifica um único caso para provar compatibilidade do BGE do job."""
        turns = parse_conversation(conversation, max_turns=self.max_turns)
        expected = self._validated_original_embeddings(turns, original_embeddings)
        encoded = self._encode([turn.text for turn in turns])
        metrics = self._validate_reproduction(encoded, 0.0, expected, None)
        metrics.update({
            "audited_turns": len(turns),
            "policy": "once_per_study_job",
        })
        return metrics

    def explain_prepared(
        self,
        prepared: PreparedPerturbations,
        *,
        top_messages: int = 6,
        top_spans_per_message: int = 20,
        random_controls_per_span: int = 5,
        sample_id: str | None = None,
        label: int | None = None,
        split: str | None = None,
        analysis_scope: str = "primary",
        expected_probability: float | None = None,
        defer_span_fidelity: bool = False,
        include_fidelity: bool = True,
    ) -> HierarchicalExplanation:
        if top_messages < 1 or top_spans_per_message < 1:
            raise ValueError("top_messages e top_spans_per_message devem ser >= 1")
        original = prepared.original_embeddings
        if self.seed == prepared.candidate_source_seed:
            baseline_logit = prepared.source_baseline_logit
            baseline_probability = prepared.source_baseline_probability
            message_rows = [dict(row) for row in prepared.source_message_ranking]
            selected_messages = [dict(row) for row in prepared.source_top_messages]
        else:
            baseline_logit, baseline_probability, message_rows, selected_messages = (
                self._score_macro(
                    prepared.turns,
                    original,
                    prepared.neutral_vectors,
                    top_messages,
                )
            )
        predicted_label = int(baseline_probability >= 0.5)
        reproduction: dict[str, Any] = {
            "status": "passed" if expected_probability is not None else "not_applicable",
            "embedding_source": prepared.embedding_source,
            "structural_alignment": "passed",
            "atol": self.reproduction_atol,
            "rtol": self.reproduction_rtol,
        }
        if expected_probability is not None:
            expected_probability = float(expected_probability)
            probability_error = abs(baseline_probability - expected_probability)
            clipped = float(np.clip(expected_probability, 1e-7, 1.0 - 1e-7))
            expected_logit = float(np.log(clipped / (1.0 - clipped)))
            logit_error = abs(baseline_logit - expected_logit)
            logit_atol = self.reproduction_atol / max(
                clipped * (1.0 - clipped), 1e-6,
            )
            reproduction.update({
                "prediction_probability_abs_error": probability_error,
                "prediction_logit_abs_error": logit_error,
                "prediction_logit_atol": logit_atol,
            })
            if not np.isclose(
                baseline_probability,
                expected_probability,
                atol=self.reproduction_atol,
                rtol=self.reproduction_rtol,
            ):
                raise ReproductionError(
                    f"Probabilidade não reproduzida para {sample_id}, seed={self.seed} "
                    f"(erro={probability_error:.6g})"
                )
            if 1e-7 < expected_probability < 1.0 - 1e-7 and not np.isclose(
                baseline_logit,
                expected_logit,
                atol=logit_atol,
                rtol=self.reproduction_rtol,
            ):
                raise ReproductionError(
                    f"Logit não reproduzido para {sample_id}, seed={self.seed} "
                    f"(erro={logit_error:.6g})"
                )

        if prepared.candidates:
            micro_logits, micro_probabilities = self._infer_replacements(
                original,
                [int(row["model_index"]) for row in prepared.candidates],
                prepared.perturbed_embeddings,
            )
        else:
            micro_logits, micro_probabilities = np.asarray([]), np.asarray([])

        span_rows: list[dict[str, Any]] = []
        for candidate, changed_logit, changed_probability in zip(
            prepared.candidates, micro_logits, micro_probabilities,
        ):
            delta_logit = baseline_logit - float(changed_logit)
            delta_probability = baseline_probability - float(changed_probability)
            span_rows.append({
                **candidate,
                "perturbed_logit": float(changed_logit),
                "perturbed_probability": float(changed_probability),
                "delta_logit": delta_logit,
                "abs_delta_logit": abs(delta_logit),
                "delta_probability": delta_probability,
                "abs_delta_probability": abs(delta_probability),
                "delta_logit_per_word": delta_logit / int(candidate["ngram_size"]),
                "direction": _direction(delta_logit, self.effect_epsilon),
                "prediction_changed": int(float(changed_probability) >= 0.5) != predicted_label,
            })
        span_rows.sort(key=_effect_sort_key)

        ranked_spans: list[dict[str, Any]] = []
        strongest_by_message: list[dict[str, Any]] = []
        candidate_indices = list(dict.fromkeys(
            int(row["model_index"]) for row in prepared.candidates
        ))
        for model_index in candidate_indices:
            rows = [
                row for row in span_rows
                if int(row["model_index"]) == model_index
            ]
            rows.sort(key=_effect_sort_key)
            if rows:
                strongest_by_message.append(rows[0])
            for rank, row in enumerate(rows[:top_spans_per_message], start=1):
                ranked_spans.append({**row, "rank_within_message": rank})
        ranked_spans.sort(key=_effect_sort_key)

        fidelity_rows = (
            self._cumulative_fidelity(
                original=original,
                neutral_vectors=prepared.neutral_vectors,
                selected_messages=selected_messages,
                baseline_logit=baseline_logit,
                baseline_probability=baseline_probability,
                predicted_label=predicted_label,
            )
            if include_fidelity
            else []
        )
        scam_spans = [row for row in span_rows if row["direction"] == "scam"]
        ham_spans = [row for row in span_rows if row["direction"] == "ham"]
        category = None
        if label is not None:
            category = {
                (1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN",
            }[(int(label), predicted_label)]
        analysis_config = {
            "seed": self.seed,
            "top_messages": top_messages,
            "max_ngram": prepared.max_ngram,
            "top_spans_per_message": top_spans_per_message,
            "effect_epsilon": self.effect_epsilon,
            "max_turns": self.max_turns,
            "sample_id": sample_id,
            "split": split,
            "analysis_scope": analysis_scope,
            "candidate_source_seed": prepared.candidate_source_seed,
        }
        metadata = {
            **analysis_config,
            "analysis_hash": stable_hash({
                **analysis_config,
                "conversation_hash": hashlib.sha256(
                    prepared.conversation.encode("utf-8")
                ).hexdigest(),
                "model": self.model_metadata,
            }),
            "model": self.model_metadata,
            "encoder": encoder_provenance(self.encoder),
            "reproduction": reproduction,
            "label": int(label) if label is not None else None,
            "prediction_category": category,
            "analysis_scope": analysis_scope,
            "candidate_source_seed": prepared.candidate_source_seed,
            "embedding_source": prepared.embedding_source,
            "messages_considered": len(prepared.turns),
            "messages_discarded_by_window": max(
                0, (prepared.turns[0].original_index if prepared.turns else 0)
            ),
            "evaluated_span_count": len(span_rows),
            "macro_bge_policy": "cached originals; one shared neutral vector per speaker",
            "micro_bge_policy": "canonical perturbations encoded once and reused across seeds",
            "random_control_seed": 42,
        }
        explanation = HierarchicalExplanation(
            baseline={
                "logit": baseline_logit,
                "probability_scam": baseline_probability,
                "predicted_label": predicted_label,
                "n_turns": len(prepared.turns),
            },
            top_messages=selected_messages,
            strongest_scam_span=scam_spans[0] if scam_spans else None,
            strongest_ham_span=ham_spans[0] if ham_spans else None,
            span_ranking=ranked_spans,
            message_ranking=message_rows,
            fidelity=fidelity_rows,
            metadata=metadata,
            _all_span_effects=span_rows,
        )
        if not defer_span_fidelity and strongest_by_message:
            texts = [f"{row['speaker']}: {row['span_text']}" for row in strongest_by_message]
            unique_texts = list(dict.fromkeys(texts))
            vectors = self._encode(unique_texts)
            lookup = dict(zip(unique_texts, vectors))
            self.append_span_fidelity(
                explanation,
                prepared,
                lookup,
                random_controls_per_span=random_controls_per_span,
            )
        return explanation

    def explain(
        self,
        conversation: str,
        *,
        top_messages: int = 6,
        max_ngram: int = 5,
        top_spans_per_message: int = 20,
        random_controls_per_span: int = 5,
        original_embeddings: np.ndarray | None = None,
        expected_embeddings: np.ndarray | None = None,
        expected_logit: float | None = None,
        sample_id: str | None = None,
        label: int | None = None,
        split: str | None = None,
    ) -> HierarchicalExplanation:
        """Explica texto novo ou usa embeddings originais fornecidos pelo cache."""
        prepared = self.prepare_perturbations(
            conversation,
            top_messages=top_messages,
            max_ngram=max_ngram,
            original_embeddings=original_embeddings,
        )
        reproduction = None
        if expected_embeddings is not None:
            reproduction = self._validate_reproduction(
                prepared.original_embeddings,
                prepared.source_baseline_logit,
                expected_embeddings,
                expected_logit,
            )
        elif expected_logit is not None and not np.isclose(
            prepared.source_baseline_logit,
            float(expected_logit),
            atol=self.reproduction_atol,
            rtol=self.reproduction_rtol,
        ):
            raise ReproductionError(
                "Logit não reproduzido: "
                f"novo={prepared.source_baseline_logit:.8g}, salvo={float(expected_logit):.8g}"
            )
        explanation = self.explain_prepared(
            prepared,
            top_messages=top_messages,
            top_spans_per_message=top_spans_per_message,
            random_controls_per_span=random_controls_per_span,
            sample_id=sample_id,
            label=label,
            split=split,
        )
        if reproduction is not None:
            explanation.metadata["reproduction"] = reproduction
        return explanation

    def _cumulative_fidelity(
        self,
        *,
        original: np.ndarray,
        neutral_vectors: np.ndarray,
        selected_messages: list[dict[str, Any]],
        baseline_logit: float,
        baseline_probability: float,
        predicted_label: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if selected_messages:
            removal_sequences = []
            sufficiency_sequences = []
            selected_so_far: list[int] = []
            for message in selected_messages:
                selected_so_far.append(int(message["model_index"]))
                removed = original.copy()
                removed[selected_so_far] = neutral_vectors[selected_so_far]
                sufficient = neutral_vectors.copy()
                sufficient[selected_so_far] = original[selected_so_far]
                removal_sequences.append(removed)
                sufficiency_sequences.append(sufficient)
            removal_logits, removal_probabilities = self._infer_arrays(np.stack(removal_sequences))
            sufficient_logits, sufficient_probabilities = self._infer_arrays(
                np.stack(sufficiency_sequences)
            )
            original_class_score = (
                baseline_probability if predicted_label == 1 else 1.0 - baseline_probability
            )
            for rank, (removed_logit, removed_probability, sufficient_logit, sufficient_probability) in enumerate(
                zip(
                    removal_logits, removal_probabilities,
                    sufficient_logits, sufficient_probabilities,
                ),
                start=1,
            ):
                removed_class_score = (
                    float(removed_probability)
                    if predicted_label == 1 else 1.0 - float(removed_probability)
                )
                sufficient_class_score = (
                    float(sufficient_probability)
                    if predicted_label == 1 else 1.0 - float(sufficient_probability)
                )
                rows.append({
                    "level": "cumulative_messages",
                    "top_k": rank,
                    "selected_model_indices": [
                        int(row["model_index"]) for row in selected_messages[:rank]
                    ],
                    "removed_logit": float(removed_logit),
                    "removed_probability": float(removed_probability),
                    "delta_logit": baseline_logit - float(removed_logit),
                    "comprehensiveness": original_class_score - removed_class_score,
                    "sufficient_logit": float(sufficient_logit),
                    "sufficient_probability": float(sufficient_probability),
                    "sufficiency_gap": original_class_score - sufficient_class_score,
                    "supported_class": "scam" if predicted_label == 1 else "ham",
                })

        return rows

    def append_span_fidelity(
        self,
        explanation: HierarchicalExplanation,
        prepared: PreparedPerturbations,
        sufficient_vector_lookup: dict[str, np.ndarray],
        *,
        random_controls_per_span: int = 5,
    ) -> None:
        """Completa fidelity usando embeddings de sufficiency compartilháveis."""
        explanation.fidelity = [
            row for row in explanation.fidelity if row.get("level") != "span"
        ]
        strongest = [
            row for row in explanation.span_ranking
            if int(row.get("rank_within_message", 0)) == 1
        ]
        if not strongest:
            return
        texts = [f"{row['speaker']}: {row['span_text']}" for row in strongest]
        try:
            sufficient_vectors = np.stack([
                np.asarray(sufficient_vector_lookup[text], dtype=np.float32)
                for text in texts
            ])
        except KeyError as exc:
            raise ValueError(f"Embedding de sufficiency ausente: {exc.args[0]}") from exc
        sufficient_logits, sufficient_probabilities = self._infer_replacements(
            prepared.original_embeddings,
            [int(row["model_index"]) for row in strongest],
            sufficient_vectors,
        )
        baseline_probability = float(explanation.baseline["probability_scam"])
        predicted_label = int(explanation.baseline["predicted_label"])
        rng = np.random.default_rng(42)
        rows: list[dict[str, Any]] = []
        for span, sufficient_logit, sufficient_probability in zip(
            strongest, sufficient_logits, sufficient_probabilities,
        ):
            direction = str(span["direction"])
            original_supported = _supported_score(
                baseline_probability, direction, predicted_label,
            )
            removed_supported = _supported_score(
                float(span["perturbed_probability"]), direction, predicted_label,
            )
            sufficient_supported = _supported_score(
                float(sufficient_probability), direction, predicted_label,
            )
            controls = [
                row for row in explanation._all_span_effects
                if int(row["model_index"]) == int(span["model_index"])
                and int(row["ngram_size"]) == int(span["ngram_size"])
                and (
                    int(row["start_char"]), int(row["end_char"])
                ) != (int(span["start_char"]), int(span["end_char"]))
            ]
            count = min(max(0, random_controls_per_span), len(controls))
            chosen = (
                rng.choice(len(controls), size=count, replace=False).tolist()
                if count else []
            )
            random_effects = [abs(float(controls[index]["delta_logit"])) for index in chosen]
            random_mean = float(np.mean(random_effects)) if random_effects else None
            rows.append({
                "level": "span",
                "original_index": int(span["original_index"]),
                "model_index": int(span["model_index"]),
                "span_text": span["span_text"],
                "start_char": int(span["start_char"]),
                "end_char": int(span["end_char"]),
                "ngram_size": int(span["ngram_size"]),
                "supported_class": direction if direction != "neutral" else (
                    "scam" if predicted_label == 1 else "ham"
                ),
                "comprehensiveness": original_supported - removed_supported,
                "sufficient_logit": float(sufficient_logit),
                "sufficient_probability": float(sufficient_probability),
                "sufficiency_gap": original_supported - sufficient_supported,
                "random_control_count": count,
                "random_abs_delta_logit_mean": random_mean,
                "selected_to_random_ratio": (
                    abs(float(span["delta_logit"])) / random_mean
                    if random_mean not in {None, 0.0} else None
                ),
            })
        explanation.fidelity.extend(rows)


def _finalist_summary() -> dict[str, Any]:
    path = RESULTS_ROOT / "transformer" / "finalists" / "bge" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Resumo finalista BGE ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bge_finalist_seed_row(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retorna o finalista escolhido previamente, sem refazer seleção por teste."""
    summary = _finalist_summary()
    try:
        seed_row = next(row for row in summary["seeds"] if int(row["seed"]) == int(seed))
    except StopIteration as exc:
        available = [int(row["seed"]) for row in summary["seeds"]]
        raise ValueError(f"Seed {seed} ausente; disponíveis: {available}") from exc
    return summary, seed_row


def resolve_bge_finalist_checkpoint(
    seed: int = 42,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Localiza o checkpoint do finalista BGE pelo ``run_id`` já selecionado.

    ``checkpoint_root`` pode apontar para ``experiment_results``, para a pasta
    ``transformer`` ou para qualquer ancestral da pasta do run. Não há fallback
    para o melhor resultado de teste/validação externa nem para outro run com
    hiperparâmetros parecidos: isso evitaria uma nova seleção pós-hoc.
    """
    _, seed_row = _bge_finalist_seed_row(seed)
    run_id = str(seed_row["run_id"])
    root = (
        Path(checkpoint_root).expanduser()
        if checkpoint_root is not None
        else RESULTS_ROOT / "transformer"
    )
    direct_candidates = (
        root / run_id / "model.pth",
        root / "transformer" / run_id / "model.pth",
        root / "model.pth" if root.name == run_id else root / run_id / "model.pth",
    )
    checked: list[Path] = []
    for candidate in direct_candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate

    if root.is_dir():
        matches = sorted(
            (
                path.resolve()
                for path in root.rglob("model.pth")
                if path.parent.name == run_id
            ),
            key=str,
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            rendered = "\n  - ".join(str(path) for path in matches)
            raise RuntimeError(
                f"Mais de um checkpoint corresponde ao finalista BGE seed={seed}:\n"
                f"  - {rendered}\nInforme um --checkpoint-root mais específico."
            )

    discovered = len(list(root.rglob("model.pth"))) if root.is_dir() else 0
    expected = checked[0]
    raise FileNotFoundError(
        f"Checkpoint finalista BGE seed={seed} ausente. Run esperado: {run_id}. "
        f"Caminho principal: {expected}. Foram encontrados {discovered} arquivos "
        f"model.pth sob {root}. Informe checkpoint_root/--checkpoint-root apontando "
        "para os artefatos que contêm esse run."
    )


def load_bge_encoder(device=None, model_name: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name or BGE_MODEL_NAME,
        device=str(device) if device is not None else None,
    )


def load_bge_finalist_model(
    seed: int = 42,
    device=None,
    checkpoint_root: str | Path | None = None,
):
    import torch

    summary, seed_row = _bge_finalist_seed_row(seed)
    path = resolve_bge_finalist_checkpoint(seed=seed, checkpoint_root=checkpoint_root)
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    config = checkpoint.get("config", summary["configuration"])
    model = TransformerScamClassifier(
        embedding_dim=get_embedding_dim("bge"),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(state)
    metadata = {
        "embedding": "bge",
        "seed": int(seed),
        "run_id": seed_row["run_id"],
        "checkpoint": str(path),
        "configuration": config,
        "selection_metric": "internal_val_f1_macro",
        "checkpoint_role": "finalist_repetition",
    }
    return model, metadata


def explain_conversation(
    conversation: str,
    seed: int = 42,
    top_messages: int = 6,
    max_ngram: int = 5,
    top_spans_per_message: int = 20,
    *,
    device=None,
    encoder=None,
    original_embeddings: np.ndarray | None = None,
    expected_embeddings: np.ndarray | None = None,
    expected_logit: float | None = None,
    sample_id: str | None = None,
    label: int | None = None,
    split: str | None = None,
    checkpoint_root: str | Path | None = None,
) -> HierarchicalExplanation:
    """API pública para explicar uma conversa com um finalista BGE congelado."""
    model, model_metadata = load_bge_finalist_model(
        seed=seed, device=device, checkpoint_root=checkpoint_root,
    )
    encoder = encoder or load_bge_encoder(device=device)
    explainer = HierarchicalPerturbationExplainer(
        model, encoder, device=device, seed=seed, model_metadata=model_metadata,
    )
    return explainer.explain(
        conversation,
        top_messages=top_messages,
        max_ngram=max_ngram,
        top_spans_per_message=top_spans_per_message,
        original_embeddings=original_embeddings,
        expected_embeddings=expected_embeddings,
        expected_logit=expected_logit,
        sample_id=sample_id,
        label=label,
        split=split,
    )


def explain_cached_dataset_sample(
    sample_id: str,
    *,
    split: str = "test_internal",
    seeds: Sequence[int] = (42, 52, 62),
    top_messages: int = 6,
    max_ngram: int = 5,
    top_spans_per_message: int = 20,
    device=None,
    output_dir: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> list[HierarchicalExplanation]:
    """Explica uma amostra cacheada e reutiliza perturbações entre seeds."""
    from .explainability import _dataset

    seeds = tuple(dict.fromkeys(int(value) for value in seeds))
    if not seeds:
        raise ValueError("Informe ao menos uma seed")
    sequences, labels, ids, text_by_id = _dataset("bge", split)
    matches = np.flatnonzero(np.asarray(ids).astype(str) == str(sample_id))
    if len(matches) != 1:
        raise ValueError(f"sample_id deve ocorrer exatamente uma vez: {sample_id}")
    index = int(matches[0])
    original_embeddings = np.asarray(sequences[index], dtype=np.float32)[-100:]
    conversation = text_by_id[str(sample_id)]
    encoder = load_bge_encoder(device=device)
    primary_seed = 42 if 42 in seeds else int(seeds[0])
    explainers: dict[int, HierarchicalPerturbationExplainer] = {}
    bundles: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        model, model_metadata = load_bge_finalist_model(
            seed=seed, device=device, checkpoint_root=checkpoint_root,
        )
        explainers[seed] = HierarchicalPerturbationExplainer(
            model, encoder, device=device, seed=seed, model_metadata=model_metadata,
        )
        bundles[seed] = ModelCache.load_prediction_bundle(
            "transformer", model_metadata["run_id"], split,
        )

    primary = explainers[primary_seed]
    audit = primary.audit_cached_embeddings(conversation, original_embeddings)
    neutral_vectors = primary._encode(["Innocent:", "Suspect:"])
    neutral_by_speaker = dict(zip(("Innocent", "Suspect"), neutral_vectors))
    prepared = primary.prepare_perturbations(
        conversation,
        top_messages=top_messages,
        max_ngram=max_ngram,
        original_embeddings=original_embeddings,
        neutral_by_speaker=neutral_by_speaker,
    )
    explanations: list[HierarchicalExplanation] = []
    for seed in seeds:
        bundle = bundles[seed]
        bundle_ids = np.asarray(bundle["sample_ids"]).astype(str)
        bundle_matches = np.flatnonzero(bundle_ids == str(sample_id))
        if len(bundle_matches) != 1:
            raise ReproductionError(
                f"sample_id ausente ou duplicado nas predições: {sample_id}"
            )
        expected_probability = float(np.asarray(bundle["y_prob"])[bundle_matches[0]])
        explanation = explainers[seed].explain_prepared(
            prepared,
            top_messages=top_messages,
            top_spans_per_message=top_spans_per_message,
            sample_id=str(sample_id),
            label=int(labels[index]),
            split=split,
            analysis_scope="primary" if seed == primary_seed else "stability",
            expected_probability=expected_probability,
            defer_span_fidelity=True,
        )
        explanation.metadata["bge_audit"] = audit
        explanations.append(explanation)

    sufficient_texts = list(dict.fromkeys(
        f"{row['speaker']}: {row['span_text']}"
        for explanation in explanations
        for row in explanation.span_ranking
        if int(row.get("rank_within_message", 0)) == 1
    ))
    sufficient_lookup = dict(zip(sufficient_texts, primary._encode(sufficient_texts)))
    for explanation in explanations:
        explainers[int(explanation.metadata["seed"])].append_span_fidelity(
            explanation, prepared, sufficient_lookup,
        )
    if output_dir is not None:
        save_hierarchical_explanations(
            explanations,
            output_dir=output_dir,
            manifest_metadata={
                "primary_seed": primary_seed,
                "stability_seeds": list(seeds) if len(seeds) > 1 else [],
                "primary_sample_ids": [str(sample_id)],
                "stability_sample_ids": (
                    [str(sample_id)] if len(seeds) > 1 else []
                ),
                "bge_audit": audit,
            },
        )
    return explanations


def _select_samples_by_prediction_category(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    samples_per_category: int,
) -> dict[str, list[str]]:
    """Seleciona casos TP/TN/FP/FN de forma determinística e sem usar confiança."""
    if samples_per_category < 1:
        raise ValueError("samples_per_category deve ser >= 1")
    ids = np.asarray(sample_ids).astype(str)
    y_true = np.asarray(labels).astype(int)
    y_pred = np.asarray(predictions).astype(int)
    if not (len(ids) == len(y_true) == len(y_pred)):
        raise ValueError("IDs, labels e predições devem estar alinhados")
    masks = {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }
    selected: dict[str, list[str]] = {}
    for category, mask in masks.items():
        candidates = ids[mask].tolist()
        candidates.sort(key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        selected[category] = candidates[:samples_per_category]
    return selected


def _select_stability_subset(
    selected_by_category: dict[str, Sequence[str]],
    samples_per_category: int,
) -> set[str]:
    """Toma prefixos da seleção determinística principal em cada categoria."""
    if samples_per_category < 0:
        raise ValueError("samples_per_category deve ser >= 0")
    return {
        str(sample_id)
        for category in ("TP", "TN", "FP", "FN")
        for sample_id in selected_by_category.get(category, ())[:samples_per_category]
    }


def run_cached_dataset_study(
    *,
    split: str = "test_internal",
    primary_seed: int = 42,
    stability_seeds: Sequence[int] = (42, 52, 62),
    samples_per_category: int = 5,
    stability_samples_per_category: int = 2,
    min_messages: int = DEFAULT_MIN_MESSAGES,
    sample_ids: Sequence[str] | None = None,
    top_messages: int = 6,
    max_ngram: int = 5,
    top_spans_per_message: int = 20,
    device=None,
    output_dir: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> list[HierarchicalExplanation]:
    """Executa seed canônica completa e estabilidade reduzida sem repetir BGE.

    Apenas conversas com pelo menos ``min_messages`` turnos após o corte dos
    últimos 100 são elegíveis. O default 7 implementa o critério de "mais de
    seis mensagens", garantindo que a etapa macro possa exibir seis turnos.
    """
    from .explainability import _dataset

    if split not in {"test_internal", "validation"}:
        raise ValueError("split deve ser test_internal ou validation")
    if samples_per_category < 1:
        raise ValueError("samples_per_category deve ser >= 1")
    if stability_samples_per_category < 0:
        raise ValueError("stability_samples_per_category deve ser >= 0")
    if stability_samples_per_category > samples_per_category:
        raise ValueError(
            "stability_samples_per_category não pode exceder samples_per_category"
        )
    if min_messages < 1:
        raise ValueError("min_messages deve ser >= 1")
    sequences, labels, ids, text_by_id = _dataset("bge", split)
    ids = np.asarray(ids).astype(str)
    labels = np.asarray(labels).astype(int)
    primary_seed = int(primary_seed)
    stability_seeds = tuple(dict.fromkeys(int(seed) for seed in stability_seeds))
    required_seeds = [primary_seed]
    if stability_samples_per_category > 0:
        required_seeds.extend(seed for seed in stability_seeds if seed != primary_seed)
    required_seeds = list(dict.fromkeys(required_seeds))

    models: dict[int, tuple[Any, dict[str, Any]]] = {}
    bundles: dict[int, dict[str, np.ndarray]] = {}
    for seed in required_seeds:
        model, metadata = load_bge_finalist_model(
            seed=seed, device=device, checkpoint_root=checkpoint_root,
        )
        bundle = ModelCache.load_prediction_bundle("transformer", metadata["run_id"], split)
        bundle_ids = np.asarray(bundle["sample_ids"]).astype(str)
        if not np.array_equal(bundle_ids, ids):
            raise ReproductionError(
                f"IDs do dataset e das predições estão desalinhados para seed={seed}"
            )
        if not np.array_equal(np.asarray(bundle["y_true"]).astype(int), labels):
            raise ReproductionError(
                f"Labels do dataset e das predições estão desalinhados para seed={seed}"
            )
        models[seed] = (model, metadata)
        bundles[seed] = bundle

    message_counts = np.asarray([
        len(parse_conversation(text_by_id[sample_id]))
        for sample_id in ids
    ])
    eligible_mask = message_counts >= min_messages
    predictions = np.asarray(bundles[primary_seed]["y_pred"]).astype(int)

    if sample_ids is None:
        selected_by_category = _select_samples_by_prediction_category(
            ids[eligible_mask],
            labels[eligible_mask],
            predictions[eligible_mask],
            samples_per_category,
        )
        category_masks = {
            "TP": (labels == 1) & (predictions == 1),
            "TN": (labels == 0) & (predictions == 0),
            "FP": (labels == 0) & (predictions == 1),
            "FN": (labels == 1) & (predictions == 0),
        }
        available = {
            category: int(np.sum(mask & eligible_mask))
            for category, mask in category_masks.items()
        }
        reduced_categories = {
            category: count
            for category, count in available.items()
            if count < samples_per_category
        }
        if reduced_categories:
            print(
                "Amostragem reduzida por disponibilidade após aplicar "
                f"min_messages={min_messages}: solicitadas={samples_per_category}; "
                f"usando todas as elegíveis em {reduced_categories}"
            )
    else:
        selected_ids = [str(value) for value in sample_ids]
        missing = sorted(set(selected_ids) - set(ids.tolist()))
        if missing:
            raise ValueError(f"sample_ids ausentes no split {split}: {missing[:10]}")
        index_by_id = {sample_id: index for index, sample_id in enumerate(ids.tolist())}
        too_short = [
            sample_id for sample_id in selected_ids
            if message_counts[index_by_id[sample_id]] < min_messages
        ]
        if too_short:
            raise ValueError(
                f"sample_ids com menos de {min_messages} mensagens: {too_short[:10]}"
            )
        selected_by_category = {category: [] for category in ("TP", "TN", "FP", "FN")}
        for sample_id in selected_ids:
            index = index_by_id[sample_id]
            category = {
                (1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN",
            }[(int(labels[index]), int(predictions[index]))]
            selected_by_category[category].append(sample_id)
        for values in selected_by_category.values():
            values.sort(key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())

    selected_ids = [
        sample_id
        for category in ("TP", "TN", "FP", "FN")
        for sample_id in selected_by_category[category]
    ]
    if not selected_ids:
        raise ValueError("Nenhuma amostra selecionada para o estudo")
    stability_ids = _select_stability_subset(
        selected_by_category, stability_samples_per_category,
    )

    index_by_id = {sample_id: index for index, sample_id in enumerate(ids.tolist())}
    encoder = load_bge_encoder(device=device)
    explainers = {
        seed: HierarchicalPerturbationExplainer(
            model,
            encoder,
            device=device,
            seed=seed,
            model_metadata=metadata,
        )
        for seed, (model, metadata) in models.items()
    }
    primary = explainers[primary_seed]
    audit_sample_id = selected_ids[0]
    audit_index = index_by_id[audit_sample_id]
    bge_audit = primary.audit_cached_embeddings(
        text_by_id[audit_sample_id],
        np.asarray(sequences[audit_index], dtype=np.float32)[-100:],
    )
    bge_audit["sample_id"] = audit_sample_id
    neutral_embeddings = primary._encode(["Innocent:", "Suspect:"])
    neutral_by_speaker = dict(zip(("Innocent", "Suspect"), neutral_embeddings))
    explanations: list[HierarchicalExplanation] = []
    for sample_id in selected_ids:
        index = index_by_id[sample_id]
        original_embeddings = np.asarray(sequences[index], dtype=np.float32)[-100:]
        prepared = primary.prepare_perturbations(
            text_by_id[sample_id],
            top_messages=top_messages,
            max_ngram=max_ngram,
            original_embeddings=original_embeddings,
            neutral_by_speaker=neutral_by_speaker,
        )
        sample_seeds = [primary_seed]
        if sample_id in stability_ids:
            sample_seeds.extend(
                seed for seed in required_seeds if seed != primary_seed
            )
        sample_explanations: list[HierarchicalExplanation] = []
        for seed in sample_seeds:
            expected_probability = float(np.asarray(bundles[seed]["y_prob"])[index])
            explanation = explainers[seed].explain_prepared(
                prepared,
                top_messages=top_messages,
                top_spans_per_message=top_spans_per_message,
                sample_id=sample_id,
                label=int(labels[index]),
                split=split,
                analysis_scope="primary" if seed == primary_seed else "stability",
                expected_probability=expected_probability,
                defer_span_fidelity=True,
            )
            explanation.metadata["bge_audit"] = bge_audit
            sample_explanations.append(explanation)

        sufficient_texts = list(dict.fromkeys(
            f"{row['speaker']}: {row['span_text']}"
            for explanation in sample_explanations
            for row in explanation.span_ranking
            if int(row.get("rank_within_message", 0)) == 1
        ))
        sufficient_lookup = dict(zip(
            sufficient_texts,
            primary._encode(sufficient_texts),
        ))
        for explanation in sample_explanations:
            explainers[int(explanation.metadata["seed"])].append_span_fidelity(
                explanation,
                prepared,
                sufficient_lookup,
            )
        explanations.extend(sample_explanations)
    save_hierarchical_explanations(
        explanations,
        output_dir=output_dir or (
            TRANSFORMER_STUDIES / "hierarchical_explainability" / split
        ),
        manifest_metadata={
            "primary_seed": primary_seed,
            "stability_seeds": (
                required_seeds if stability_samples_per_category > 0 else []
            ),
            "samples_per_category": samples_per_category,
            "stability_samples_per_category": stability_samples_per_category,
            "min_messages": min_messages,
            "eligible_samples": int(eligible_mask.sum()),
            "eligible_category_counts": {
                category: int(np.sum(mask & eligible_mask))
                for category, mask in {
                    "TP": (labels == 1) & (predictions == 1),
                    "TN": (labels == 0) & (predictions == 0),
                    "FP": (labels == 0) & (predictions == 1),
                    "FN": (labels == 1) & (predictions == 0),
                }.items()
            },
            "primary_category_counts": {
                category: len(selected_by_category[category])
                for category in ("TP", "TN", "FP", "FN")
            },
            "stability_category_counts": {
                category: len(selected_by_category[category][
                    :stability_samples_per_category
                ])
                for category in ("TP", "TN", "FP", "FN")
            },
            "primary_sample_ids": selected_ids,
            "stability_sample_ids": sorted(stability_ids),
            "bge_audit": bge_audit,
        },
    )
    return explanations


def _flatten_with_metadata(
    explanation: HierarchicalExplanation,
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix = {
        "sample_id": explanation.metadata.get("sample_id"),
        "seed": explanation.metadata.get("seed"),
        "split": explanation.metadata.get("split"),
        "label": explanation.metadata.get("label"),
        "prediction_category": explanation.metadata.get("prediction_category"),
        "analysis_scope": explanation.metadata.get("analysis_scope"),
        "candidate_source_seed": explanation.metadata.get("candidate_source_seed"),
        "analysis_hash": explanation.metadata.get("analysis_hash"),
    }
    return [{**prefix, **row} for row in rows]


def _stability_rows(explanations: Sequence[HierarchicalExplanation]) -> list[dict[str, Any]]:
    import pandas as pd

    sample_seed_counts: dict[str, set[int]] = {}
    for explanation in explanations:
        sample_id = str(explanation.metadata.get("sample_id"))
        sample_seed_counts.setdefault(sample_id, set()).add(int(explanation.metadata["seed"]))
    stability_samples = {
        sample_id for sample_id, seeds in sample_seed_counts.items() if len(seeds) > 1
    }
    messages: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    stable_explanations = [
        explanation for explanation in explanations
        if str(explanation.metadata.get("sample_id")) in stability_samples
    ]
    for explanation in stable_explanations:
        selected = {int(row["model_index"]) for row in explanation.top_messages}
        for row in _flatten_with_metadata(explanation, explanation.message_ranking):
            messages.append({**row, "selected_top_message": int(row["model_index"]) in selected})
        ranked_span_keys = {
            (int(row["model_index"]), int(row["start_char"]), int(row["end_char"]))
            for row in explanation.span_ranking
        }
        for row in _flatten_with_metadata(explanation, explanation._all_span_effects):
            key = (int(row["model_index"]), int(row["start_char"]), int(row["end_char"]))
            row["selected_top_span"] = key in ranked_span_keys
            spans.append(row)
    result: list[dict[str, Any]] = []
    if messages:
        frame = pd.DataFrame(messages)
        keys = ["sample_id", "original_index", "speaker", "text"]
        for values, group in frame.groupby(keys, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            directions = group["direction"].value_counts()
            result.append({
                "item_type": "message",
                **dict(zip(keys, values)),
                "n_seeds": int(group["seed"].nunique()),
                "selection_count": int(group["selected_top_message"].sum()),
                "mean_abs_delta_logit": float(group["abs_delta_logit"].mean()),
                "std_abs_delta_logit": float(group["abs_delta_logit"].std(ddof=1)) if len(group) > 1 else 0.0,
                "consensus_direction": str(directions.index[0]),
                "direction_agreement": float(directions.iloc[0] / len(group)),
            })
    if spans:
        frame = pd.DataFrame(spans)
        keys = ["sample_id", "original_index", "start_char", "end_char", "span_text"]
        for values, group in frame.groupby(keys, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            directions = group["direction"].value_counts()
            result.append({
                "item_type": "span",
                **dict(zip(keys, values)),
                "n_seeds": int(group["seed"].nunique()),
                "selection_count": int(group["selected_top_span"].sum()),
                "mean_abs_delta_logit": float(group["abs_delta_logit"].mean()),
                "std_abs_delta_logit": float(group["abs_delta_logit"].std(ddof=1)) if len(group) > 1 else 0.0,
                "consensus_direction": str(directions.index[0]),
                "direction_agreement": float(directions.iloc[0] / len(group)),
            })

    by_sample: dict[str, list[HierarchicalExplanation]] = {}
    for explanation in stable_explanations:
        by_sample.setdefault(str(explanation.metadata.get("sample_id")), []).append(explanation)
    for sample_id, sample_explanations in by_sample.items():
        primary = next(
            explanation for explanation in sample_explanations
            if int(explanation.metadata["seed"])
            == int(explanation.metadata["candidate_source_seed"])
        )
        primary_messages = {
            int(row["model_index"]): row for row in primary.message_ranking
        }
        primary_top = {int(row["model_index"]) for row in primary.top_messages}
        primary_spans = {
            (int(row["model_index"]), int(row["start_char"]), int(row["end_char"])): row
            for row in primary._all_span_effects
        }
        for candidate in sample_explanations:
            seed = int(candidate.metadata["seed"])
            if candidate is primary:
                continue
            candidate_messages = {
                int(row["model_index"]): row for row in candidate.message_ranking
            }
            shared_message_indices = sorted(set(primary_messages) & set(candidate_messages))
            primary_message_values = np.asarray([
                primary_messages[index]["abs_delta_logit"] for index in shared_message_indices
            ], dtype=float)
            candidate_message_values = np.asarray([
                candidate_messages[index]["abs_delta_logit"] for index in shared_message_indices
            ], dtype=float)
            candidate_top = {int(row["model_index"]) for row in candidate.top_messages}
            union = primary_top | candidate_top
            message_direction_agreement = float(np.mean([
                primary_messages[index]["direction"] == candidate_messages[index]["direction"]
                for index in shared_message_indices
            ])) if shared_message_indices else None

            candidate_spans = {
                (int(row["model_index"]), int(row["start_char"]), int(row["end_char"])): row
                for row in candidate._all_span_effects
            }
            shared_span_keys = sorted(set(primary_spans) & set(candidate_spans))
            primary_span_values = np.asarray([
                primary_spans[key]["abs_delta_logit"] for key in shared_span_keys
            ], dtype=float)
            candidate_span_values = np.asarray([
                candidate_spans[key]["abs_delta_logit"] for key in shared_span_keys
            ], dtype=float)
            span_direction_agreement = float(np.mean([
                primary_spans[key]["direction"] == candidate_spans[key]["direction"]
                for key in shared_span_keys
            ])) if shared_span_keys else None
            result.append({
                "item_type": "summary",
                "sample_id": sample_id,
                "primary_seed": int(primary.metadata["seed"]),
                "compared_seed": seed,
                "candidate_source_seed": int(primary.metadata["candidate_source_seed"]),
                "message_top_k": len(primary_top),
                "message_top_k_overlap": len(primary_top & candidate_top),
                "message_top_k_jaccard": (
                    float(len(primary_top & candidate_top) / len(union)) if union else 1.0
                ),
                "message_direction_agreement": message_direction_agreement,
                "message_rank_spearman": _rank_correlation(
                    primary_message_values, candidate_message_values,
                ),
                "span_direction_agreement": span_direction_agreement,
                "span_rank_spearman": _rank_correlation(
                    primary_span_values, candidate_span_values,
                ),
                "shared_span_count": len(shared_span_keys),
            })
    return result


def save_hierarchical_explanations(
    explanations: Sequence[HierarchicalExplanation],
    *,
    output_dir: str | Path | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persiste tabelas completas, explicações resumidas e manifesto reproduzível."""
    import pandas as pd

    if not explanations:
        raise ValueError("Nenhuma explicação para salvar")
    destination = Path(output_dir) if output_dir else (
        TRANSFORMER_STUDIES / "hierarchical_explainability"
    )
    message_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    for explanation in explanations:
        message_rows.extend(_flatten_with_metadata(explanation, explanation.message_ranking))
        span_rows.extend(_flatten_with_metadata(explanation, explanation._all_span_effects))
        fidelity_rows.extend(_flatten_with_metadata(explanation, explanation.fidelity))
        sample_key = explanation.metadata.get("sample_id") or explanation.metadata["analysis_hash"][:16]
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_key))
        seed = explanation.metadata.get("seed")
        atomic_json(
            destination / "explanations" / safe_key / f"seed_{seed}.json",
            explanation.to_dict(),
        )
    stability = _stability_rows(explanations)
    atomic_csv(destination / "message_occlusion.csv", pd.DataFrame(message_rows))
    atomic_csv(destination / "span_occlusion.csv", pd.DataFrame(span_rows))
    atomic_csv(destination / "fidelity.csv", pd.DataFrame(fidelity_rows))
    stability_frame = pd.DataFrame(stability)
    if stability_frame.empty:
        stability_frame = pd.DataFrame(columns=[
            "item_type", "sample_id", "candidate_source_seed",
            "mean_abs_delta_logit", "direction_agreement", "selection_count",
        ])
    atomic_csv(destination / "stability.csv", stability_frame)
    manifest = {
        "analysis": "hierarchical_perturbation_v2",
        "analysis_hash": stable_hash([
            explanation.metadata.get("analysis_hash") for explanation in explanations
        ]),
        "seeds": sorted({int(explanation.metadata["seed"]) for explanation in explanations}),
        "sample_ids": sorted({
            str(explanation.metadata.get("sample_id")) for explanation in explanations
        }),
        "n_explanations": len(explanations),
        "external_validation_used_for_selection": False,
        "effect_definition": "original_logit - perturbed_logit",
        "effect_epsilon": DEFAULT_EFFECT_EPSILON,
        "primary_seed": next((
            int(explanation.metadata["seed"])
            for explanation in explanations
            if explanation.metadata.get("analysis_scope") == "primary"
        ), None),
        "stability_seeds": sorted({
            int(explanation.metadata["seed"])
            for explanation in explanations
            if explanation.metadata.get("analysis_scope") == "stability"
        }),
        "primary_sample_ids": sorted({
            str(explanation.metadata.get("sample_id"))
            for explanation in explanations
            if explanation.metadata.get("analysis_scope") == "primary"
        }),
        "stability_sample_ids": sorted({
            str(explanation.metadata.get("sample_id"))
            for explanation in explanations
            if explanation.metadata.get("analysis_scope") == "stability"
        }),
        "encoder": explanations[0].metadata.get("encoder"),
    }
    manifest.update(manifest_metadata or {})
    atomic_json(destination / "manifest.json", manifest)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicação hierárquica do Transformer BGE por oclusão textual"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--conversation")
    source.add_argument("--conversation-file", type=Path)
    source.add_argument("--sample-id")
    source.add_argument("--study", action="store_true")
    parser.add_argument("--split", choices=["test_internal", "validation"], default="test_internal")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--top-messages", type=int, default=6)
    parser.add_argument("--max-ngram", type=int, default=5)
    parser.add_argument("--top-spans-per-message", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-category", type=int, default=5)
    parser.add_argument("--min-messages", type=int, default=DEFAULT_MIN_MESSAGES)
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--stability-seed", type=int, action="append")
    parser.add_argument("--stability-samples-per-category", type=int, default=2)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = tuple(args.seed or [42])
    output_root = TRANSFORMER_STUDIES / "hierarchical_explainability"
    output_dir = args.output_dir or (
        output_root / args.split
        if args.study or args.sample_id is not None
        else output_root / "manual"
    )
    if args.study:
        explanations = run_cached_dataset_study(
            split=args.split,
            primary_seed=args.primary_seed,
            stability_seeds=tuple(args.stability_seed or (42, 52, 62)),
            samples_per_category=args.samples_per_category,
            min_messages=args.min_messages,
            stability_samples_per_category=(
                0 if args.skip_stability else args.stability_samples_per_category
            ),
            top_messages=args.top_messages,
            max_ngram=args.max_ngram,
            top_spans_per_message=args.top_spans_per_message,
            output_dir=output_dir,
            checkpoint_root=args.checkpoint_root,
        )
    elif args.sample_id is not None:
        explanations = explain_cached_dataset_sample(
            args.sample_id,
            split=args.split,
            seeds=seeds,
            top_messages=args.top_messages,
            max_ngram=args.max_ngram,
            top_spans_per_message=args.top_spans_per_message,
            output_dir=output_dir,
            checkpoint_root=args.checkpoint_root,
        )
    else:
        conversation = (
            args.conversation_file.read_text(encoding="utf-8")
            if args.conversation_file is not None else args.conversation
        )
        encoder = load_bge_encoder()
        explanations = [
            explain_conversation(
                conversation,
                seed=seed,
                top_messages=args.top_messages,
                max_ngram=args.max_ngram,
                top_spans_per_message=args.top_spans_per_message,
                encoder=encoder,
                checkpoint_root=args.checkpoint_root,
            )
            for seed in seeds
        ]
        save_hierarchical_explanations(explanations, output_dir=output_dir)
    summary = {
        "artifacts_dir": str(output_dir),
        "runs": [
            {
                "seed": explanation.metadata["seed"],
                "analysis_scope": explanation.metadata.get("analysis_scope"),
                "candidate_source_seed": explanation.metadata.get("candidate_source_seed"),
                "baseline": explanation.baseline,
                "top_message_indices": [
                    row["original_index"] for row in explanation.top_messages
                ],
                "strongest_scam_span": explanation.strongest_scam_span,
                "strongest_ham_span": explanation.strongest_ham_span,
            }
            for explanation in explanations
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
