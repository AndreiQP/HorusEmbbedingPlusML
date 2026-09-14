"""
machine_learning/unsupervised/svdd_torch.py
---------------------------------------------
Deep SVDD (Ruff et al., 2018 — "Deep One-Class Classification") implementado em
PyTorch com interface compatível com sklearn (fit/predict/decision_function),
para se integrar ao mesmo pipeline usado por OCSVM/IForest/LOF em runner.py.

Variante "soft-boundary": aprende um encoder que mapeia os embeddings (após
PCA/UMAP) para um espaço latente onde as amostras Ham ficam concentradas em uma
hiperesfera de raio R centrada em `c`. O raio é otimizado junto ao encoder para
que uma fração `nu` das amostras de treino possa ficar fora da esfera (outliers
de treino), o que evita colapso trivial (todos os pesos -> 0) sem a necessidade
de uma camada de bias.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class DeepSVDD:
    """
    Hiperparâmetros:
        latent_dim: dimensão do espaço latente da hiperesfera.
        hidden_dims: lista com tamanhos das camadas ocultas do encoder.
        nu: fração esperada de outliers no treino (soft-boundary), em (0, 1].
        lr: taxa de aprendizado (Adam).
        weight_decay: regularização L2 (essencial para evitar colapso trivial).
        n_epochs: número de épocas de treino.
        batch_size: tamanho do batch.
        warmup_epochs: épocas iniciais treinando apenas o encoder (R fixo em 0)
            antes de começar a otimizar o raio R.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        nu: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_epochs: int = 100,
        batch_size: int = 128,
        warmup_epochs: int = 10,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.random_state = random_state
        self.device = device

        self.encoder_ = None
        self.center_ = None
        self.radius_ = 0.0
        self._threshold = None

    def _build_encoder(self, input_dim: int):
        import torch.nn as nn
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h, bias=False), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, self.latent_dim, bias=False)]
        return nn.Sequential(*layers)

    def _init_center(self, Z, eps: float = 0.1):
        import numpy as np
        c = Z.mean(axis=0)
        # Evita centro exatamente em zero em alguma dimensão (colapso trivial)
        c[(np.abs(c) < eps) & (c < 0)] = -eps
        c[(np.abs(c) < eps) & (c >= 0)] = eps
        return c

    def fit(self, X: np.ndarray):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device)
        self.encoder_ = self._build_encoder(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(
            self.encoder_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        X_t = torch.from_numpy(X).to(device)
        n = X_t.shape[0]

        with torch.no_grad():
            Z0 = self.encoder_(X_t).cpu().numpy()
        self.center_ = self._init_center(Z0)
        c_t = torch.tensor(self.center_, dtype=torch.float32, device=device)

        R = torch.tensor(0.0, device=device)
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch = X_t[idx]
                optimizer.zero_grad()
                z = self.encoder_(batch)
                dist = torch.sum((z - c_t) ** 2, dim=1)
                if epoch < self.warmup_epochs:
                    loss = dist.mean()
                else:
                    scores = dist - R ** 2
                    loss = R ** 2 + (1.0 / self.nu) * torch.mean(torch.clamp(scores, min=0.0))
                loss.backward()
                optimizer.step()

            if epoch >= self.warmup_epochs:
                with torch.no_grad():
                    z = self.encoder_(X_t)
                    dist = torch.sum((z - c_t) ** 2, dim=1)
                    R = torch.sqrt(torch.quantile(dist, 1 - self.nu)).detach()

        self.radius_ = float(R.detach().cpu().item())
        self._threshold = self.radius_ ** 2
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        import torch
        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device)
        with torch.no_grad():
            z = self.encoder_(torch.from_numpy(X).to(device))
            c_t = torch.tensor(self.center_, dtype=torch.float32, device=device)
            dist = torch.sum((z - c_t) ** 2, dim=1)
        return dist.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Maior = mais normal (convenção sklearn OCSVM/IForest: +inlier / -outlier)."""
        return self._threshold - self._distances(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self._distances(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """+1 = normal (Ham), -1 = anomalia (Scam) — mesma convenção do sklearn."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
