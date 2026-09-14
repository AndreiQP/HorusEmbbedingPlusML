import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn

class FraudDataset(Dataset):
    def __init__(self, X, y):
        # Converte para Tensores Float32 (exigência do PyTorch para redes neurais)
        self.X = torch.tensor(X, dtype=torch.float32)
        # O CrossEntropyLoss no PyTorch exige labels no formato Long/Int64 para classificação
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ScamClassifierFCNN(nn.Module):
    def __init__(self, input_size):
        super(ScamClassifierFCNN, self).__init__()
        
        # O Sequential agrupa as camadas de forma limpa
        self.network = nn.Sequential(
            # Nível 2: 6528 -> 2048
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Nível 3: 2048 -> 1024
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Nível 4: 1024 -> 516
            nn.Linear(1024, 516),
            nn.BatchNorm1d(516),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer de Output: 516 -> 2 nodos (Scam, Ham)
            nn.Linear(516, 2)
        )

    def forward(self, x):
        # Apenas passa os dados pela rede sequencial
        return self.network(x)
    
