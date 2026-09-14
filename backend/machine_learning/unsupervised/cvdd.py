"""
machine_learning/unsupervised/cvdd.py
----------------------------------------
CVDD — Context Vector Data Description
(Ruff et al., 2019 — "Self-Attentive, Multi-Context One-Class Classification
for Unsupervised Anomaly Detection on Text")

Diferente de OCSVM/IForest/LOF/LUNAR/SVDD, o CVDD opera sobre o TEXTO BRUTO
(não sobre embeddings pré-computados): cada token da conversa é representado
pelo embedding contextual do BGE (congelado), e um mecanismo de self-attention
com `n_contexts` vetores de contexto treináveis aprende múltiplas "perspectivas"
semânticas do texto normal (Ham). A anomalia é medida pela distância (cosseno)
entre a representação atendida do texto e o contexto mais próximo.

Hiperparâmetros:
    n_contexts:       número de vetores de contexto (K), cada um captura um
                       "tópico"/perspectiva distinta do texto normal.
    attention_hidden:  dimensão da camada oculta do mecanismo de self-attention.
    lambda_ortho:      peso da regularização de ortogonalidade entre contextos
                       (evita que os K contextos colapsem para o mesmo vetor).
    lr / n_epochs / batch_size: hiperparâmetros de otimização (Adam).
    max_tokens:        trunca sequências de tokens (BGE) neste limite.
    bge_model_name:    nome do modelo BGE usado como encoder de tokens (congelado).
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np


class CVDD:
    def __init__(
        self,
        n_contexts: int = 10,
        attention_hidden: int = 150,
        lambda_ortho: float = 1.0,
        lr: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 16,
        max_tokens: int = 128,
        nu: float = 0.1,
        bge_model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.n_contexts = n_contexts
        self.attention_hidden = attention_hidden
        self.lambda_ortho = lambda_ortho
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.nu = nu
        self.bge_model_name = bge_model_name
        self.device = device
        self.random_state = random_state

        self._tokenizer = None
        self._encoder = None       # BGE congelado (feature extractor)
        self._attention = None     # módulo treinável (self-attention)
        self.context_vectors_ = None
        self._threshold = None

    # ── Encoder BGE (congelado) ───────────────────────────────────

    def _load_bge(self):
        if self._encoder is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
        self._encoder = AutoModel.from_pretrained(self.bge_model_name)
        self._encoder.to(self.device).eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)

    def _encode_tokens(self, texts: List[str]):
        """Retorna (H, mask): embeddings de token (B, T, d) e máscara de padding (B, T)."""
        import torch
        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self._encoder(**enc)
        return out.last_hidden_state, enc["attention_mask"]

    # ── Módulo de self-attention multi-contexto ───────────────────

    def _build_attention(self, hidden_dim: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(hidden_dim, self.attention_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(self.attention_hidden, self.n_contexts, bias=False),
        )

    def _attended_reps(self, H, mask):
        """H: (B,T,d), mask: (B,T) -> M: (B,K,d) representações atendidas por contexto."""
        import torch
        scores = self._attention(H)  # (B, T, K)
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        A = torch.softmax(scores, dim=1)  # (B, T, K) — atenção por contexto ao longo de T
        M = torch.einsum("btk,btd->bkd", A, H)  # (B, K, d)
        return M, A

    def fit(self, texts: List[str]):
        import torch
        torch.manual_seed(self.random_state)
        self._load_bge()

        hidden_dim = self._encoder.config.hidden_size
        self._attention = self._build_attention(hidden_dim).to(self.device)
        self.context_vectors_ = torch.nn.Parameter(
            torch.randn(self.n_contexts, hidden_dim, device=self.device) * 0.05
        )
        params = list(self._attention.parameters()) + [self.context_vectors_]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        n = len(texts)
        for epoch in range(self.n_epochs):
            perm = np.random.RandomState(self.random_state + epoch).permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_texts = [texts[i] for i in idx]
                H, mask = self._encode_tokens(batch_texts)

                optimizer.zero_grad()
                M, _ = self._attended_reps(H, mask)  # (B, K, d)
                c = torch.nn.functional.normalize(self.context_vectors_, dim=1)  # (K, d)
                m_norm = torch.nn.functional.normalize(M, dim=2)  # (B, K, d)
                cos_sim = torch.einsum("bkd,kd->bk", m_norm, c)  # (B, K)
                dist = 1.0 - cos_sim  # (B, K)
                main_loss = dist.mean()

                ortho = c @ c.t() - torch.eye(self.n_contexts, device=self.device)
                ortho_loss = (ortho ** 2).sum() / (self.n_contexts ** 2)

                loss = main_loss + self.lambda_ortho * ortho_loss
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(idx)

            print(f"  [cvdd] epoch {epoch + 1}/{self.n_epochs} loss={epoch_loss / n:.4f}")

        scores = self.score(texts)
        self._threshold = float(np.quantile(scores, 1 - self.nu))
        return self

    def score(self, texts: List[str]) -> np.ndarray:
        """Score contínuo de anomalia (maior = mais anômalo): distância média ao contexto mais próximo."""
        import torch
        self._encoder.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                H, mask = self._encode_tokens(batch_texts)
                M, _ = self._attended_reps(H, mask)
                c = torch.nn.functional.normalize(self.context_vectors_, dim=1)
                m_norm = torch.nn.functional.normalize(M, dim=2)
                cos_sim = torch.einsum("bkd,kd->bk", m_norm, c)
                dist = 1.0 - cos_sim
                min_dist = dist.min(dim=1).values  # contexto mais próximo (mais "normal" possível)
                scores.append(min_dist.cpu().numpy())
        return np.concatenate(scores)

    def predict(self, texts: List[str]) -> np.ndarray:
        """0 = Ham, 1 = Scam."""
        return (self.score(texts) >= self._threshold).astype(int)

    def state_dict(self) -> dict:
        return {
            "attention": self._attention.state_dict(),
            "context_vectors": self.context_vectors_.detach().cpu(),
            "threshold": self._threshold,
            "config": {
                "n_contexts": self.n_contexts, "attention_hidden": self.attention_hidden,
                "lambda_ortho": self.lambda_ortho, "max_tokens": self.max_tokens,
                "nu": self.nu, "bge_model_name": self.bge_model_name,
            },
        }

    @classmethod
    def load_state_dict(cls, state: dict, device: str = "cpu") -> "CVDD":
        import torch
        cfg = state["config"]
        model = cls(device=device, **cfg)
        model._load_bge()
        hidden_dim = model._encoder.config.hidden_size
        model._attention = model._build_attention(hidden_dim).to(device)
        model._attention.load_state_dict(state["attention"])
        model.context_vectors_ = torch.nn.Parameter(state["context_vectors"].to(device))
        model._threshold = state["threshold"]
        return model
