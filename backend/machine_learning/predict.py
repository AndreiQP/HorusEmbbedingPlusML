# machine-learning/predict.py
import joblib
import pandas as pd
import os
import torch
import torch.nn.functional as F
from machine_learning.fcnn_classifier import ScamClassifierFCNN

def predict(model_path, X_numpy):
    """
    Faz previsões usando um modelo salvo.

    Args:
        model_path (str): Caminho para o arquivo .pth do modelo.
        X (list): Lista de vetores de embedding para previsão.

    Returns:
        list: Lista de previsões (0 ou 1).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    input_dim = X_numpy.shape[1]
    # 1. Instancia o modelo "vazio"
    model = ScamClassifierFCNN(input_size=input_dim) 
    
    # 2. Carrega apenas os pesos treinados para dentro do modelo
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilidades = F.softmax(logits, dim=1)
        prob_scam = probabilidades[:, 1].tolist()
        predict_list = torch.argmax(logits, dim=1).tolist()

    return predict_list, prob_scam

