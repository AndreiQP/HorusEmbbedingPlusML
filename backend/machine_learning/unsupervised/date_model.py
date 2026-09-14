"""
machine_learning/unsupervised/date_model.py
----------------------------------------------
DATE — Detecting Anomalies in Text via Self-Supervision of Transformers
(Manolache et al., 2021)

Ideia original: treinar um Transformer para resolver uma tarefa pretexto
auto-supervisionada (classificar QUAL transformação foi aplicada ao texto) e
usar a distância de Mahalanobis das representações da classe "identidade" em
relação à distribuição aprendida em dados normais (Ham) como score de anomalia
— textos fora da distribuição de treino (potenciais scams) tendem a produzir
representações distantes de todas as classes aprendidas.

Simplificações desta implementação (documentadas):
    - Backbone fixo: BGE (BAAI/bge-m3), com as últimas `unfreeze_last_n_layers`
      camadas do encoder fine-tunáveis (o resto congelado).
    - Transformações pretexto (K classes) usam apenas transformações offline
      (sem tradução automática): identidade, embaralhar palavras, remover
      palavras aleatórias, duplicar palavra aleatória, reverter ordem dos turnos.
    - Score de anomalia = distância de Mahalanobis mínima entre a representação
      [CLS] do texto original e as K distribuições Gaussianas (uma por classe
      pretexto) ajustadas nos dados de treino (Ham), com covariância
      compartilhada estimada via Ledoit-Wolf.

Hiperparâmetros:
    n_pretext_transforms:   K, número de classes da tarefa pretexto.
    unfreeze_last_n_layers: quantas camadas finais do BGE ficam treináveis.
    lr / n_epochs / batch_size: hiperparâmetros de otimização (Adam).
    max_tokens:             truncamento de tokens.
"""
from __future__ import annotations
import random
from typing import List
import numpy as np


def _word_shuffle(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def _word_dropout(text: str, rng: random.Random, p: float = 0.15) -> str:
    words = [w for w in text.split() if rng.random() > p]
    return " ".join(words) if words else text


def _word_duplicate(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    i = rng.randrange(len(words))
    words.insert(i, words[i])
    return " ".join(words)


def _turn_reverse(text: str, rng: random.Random) -> str:
    turns = text.split(". ")
    turns.reverse()
    return ". ".join(turns)


_TRANSFORMS = [
    lambda t, rng: t,          # 0: identidade
    _word_shuffle,             # 1
    _word_dropout,             # 2
    _word_duplicate,           # 3
    _turn_reverse,             # 4
]


class DATE:
    def __init__(
        self,
        n_pretext_transforms: int = 5,
        unfreeze_last_n_layers: int = 2,
        lr: float = 2e-5,
        n_epochs: int = 5,
        batch_size: int = 8,
        max_tokens: int = 256,
        bge_model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        random_state: int = 42,
    ):
        if n_pretext_transforms > len(_TRANSFORMS):
            raise ValueError(f"n_pretext_transforms deve ser <= {len(_TRANSFORMS)} (transformações disponíveis)")
        self.n_pretext_transforms = n_pretext_transforms
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.bge_model_name = bge_model_name
        self.device = device
        self.random_state = random_state

        self._tokenizer = None
        self._encoder = None
        self._head = None  # classificador linear (hidden_dim -> K)
        self._means = None       # (K, d)
        self._precision = None   # (d, d) inversa da covariância (Ledoit-Wolf)
        self._threshold = None

    def _load_backbone(self):
        if self._encoder is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
        self._encoder = AutoModel.from_pretrained(self.bge_model_name).to(self.device)

        for p in self._encoder.parameters():
            p.requires_grad_(False)
        if self.unfreeze_last_n_layers > 0:
            layers = self._encoder.encoder.layer
            for layer in layers[-self.unfreeze_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    def _pool(self, texts: List[str]):
        """Retorna representação [CLS] (B, d)."""
        import torch
        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt"
        ).to(self.device)
        out = self._encoder(**enc)
        return out.last_hidden_state[:, 0, :]  # [CLS]

    def _build_pretext_batch(self, texts: List[str], rng: random.Random):
        """Aplica uma transformação aleatória (0..K-1) a cada texto e retorna (textos, labels)."""
        aug_texts, labels = [], []
        for t in texts:
            k = rng.randrange(self.n_pretext_transforms)
            aug_texts.append(_TRANSFORMS[k](t, rng))
            labels.append(k)
        return aug_texts, labels

    def fit(self, texts: List[str]):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)
        rng = random.Random(self.random_state)
        self._load_backbone()

        hidden_dim = self._encoder.config.hidden_size
        self._head = nn.Linear(hidden_dim, self.n_pretext_transforms).to(self.device)
        trainable = [p for p in self._encoder.parameters() if p.requires_grad] + list(self._head.parameters())
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        n = len(texts)
        for epoch in range(self.n_epochs):
            perm = np.random.RandomState(self.random_state + epoch).permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_texts = [texts[i] for i in idx]
                aug_texts, labels = self._build_pretext_batch(batch_texts, rng)
                labels_t = torch.tensor(labels, device=self.device)

                optimizer.zero_grad()
                cls = self._pool(aug_texts)
                logits = self._head(cls)
                loss = criterion(logits, labels_t)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(idx)

            print(f"  [date] epoch {epoch + 1}/{self.n_epochs} pretext_loss={epoch_loss / n:.4f}")

        self._fit_gaussian(texts)
        scores = self.score(texts)
        self._threshold = float(np.quantile(scores, 1 - 0.1))
        return self

    def _fit_gaussian(self, texts: List[str]):
        """Ajusta uma Gaussiana por classe pretexto (identidade incluída) nas representações de treino (Ham)."""
        from sklearn.covariance import LedoitWolf
        import torch
        self._encoder.eval()
        reps_by_class = [[] for _ in range(self.n_pretext_transforms)]
        rng = random.Random(self.random_state)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                for k in range(self.n_pretext_transforms):
                    aug = [_TRANSFORMS[k](t, rng) for t in batch_texts]
                    cls = self._pool(aug).cpu().numpy()
                    reps_by_class[k].append(cls)

        all_reps, all_labels = [], []
        means = []
        for k in range(self.n_pretext_transforms):
            reps_k = np.concatenate(reps_by_class[k], axis=0)
            means.append(reps_k.mean(axis=0))
            all_reps.append(reps_k)
            all_labels.append(np.full(len(reps_k), k))

        self._means = np.stack(means)  # (K, d)
        X_centered = np.concatenate(
            [all_reps[k] - self._means[k] for k in range(self.n_pretext_transforms)], axis=0
        )
        cov_estimator = LedoitWolf().fit(X_centered)
        self._precision = cov_estimator.precision_  # (d, d)
        self._encoder.train()

    def score(self, texts: List[str]) -> np.ndarray:
        """Score contínuo de anomalia (maior = mais anômalo): distância de Mahalanobis mínima entre as
        K Gaussianas (uma por classe pretexto) ajustadas em Ham, aplicada ao texto ORIGINAL (sem transformação)."""
        import torch
        self._encoder.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                cls = self._pool(batch_texts).cpu().numpy()  # (B, d)
                dists = []
                for k in range(self.n_pretext_transforms):
                    diff = cls - self._means[k]
                    d = np.einsum("bi,ij,bj->b", diff, self._precision, diff)
                    dists.append(d)
                dists = np.stack(dists, axis=1)  # (B, K)
                scores.append(dists.min(axis=1))
        self._encoder.train()
        return np.concatenate(scores)

    def predict(self, texts: List[str]) -> np.ndarray:
        """0 = Ham, 1 = Scam."""
        return (self.score(texts) >= self._threshold).astype(int)

    def state_dict(self) -> dict:
        return {
            "encoder": self._encoder.state_dict(),
            "head": self._head.state_dict(),
            "means": self._means,
            "precision": self._precision,
            "threshold": self._threshold,
            "config": {
                "n_pretext_transforms": self.n_pretext_transforms,
                "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
                "max_tokens": self.max_tokens, "bge_model_name": self.bge_model_name,
            },
        }

    @classmethod
    def load_state_dict(cls, state: dict, device: str = "cpu") -> "DATE":
        import torch
        import torch.nn as nn
        cfg = state["config"]
        model = cls(device=device, **cfg)
        model._load_backbone()
        hidden_dim = model._encoder.config.hidden_size
        model._encoder.load_state_dict(state["encoder"])
        model._head = nn.Linear(hidden_dim, model.n_pretext_transforms).to(device)
        model._head.load_state_dict(state["head"])
        model._means = state["means"]
        model._precision = state["precision"]
        model._threshold = state["threshold"]
        return model
