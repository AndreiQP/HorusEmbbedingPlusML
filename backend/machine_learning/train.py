# machine-learning/train.py
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from machine_learning.classifier import get_all_classifiers
from config import EMBEDDERS

MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)

models = get_all_classifiers()

def train_model(model, embedding, dataset):
    print(f"Iniciando o treinamento do modelo {type(model).__name__}...")

    parquet_file = f"../datasets/dataset_train/processed/all_data/{embedding}_chunks.parquet"

    if not os.path.exists(parquet_file):
        print(f"Arquivo {parquet_file} não encontrado. Pulando...")
        return
    
    print(f"Carregando dados de {parquet_file}...")

    df = pd.read_parquet(parquet_file)

    print(f"Dados carregados. Total de amostras: {len(df)}. 80% para treinamento, 20% para validação.")

    X_train, X_val, y_train, y_val = train_test_split(
        list(df['text_embedded']),
        list(df['label']),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"      F1-macro médio (5-fold CV) - treino: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    print(f"      Acurácia na validação: {val_accuracy:.4f} | F1-macro na validação: {val_f1_macro:.4f}")

    model_filename = f"{dataset}_{embedding}_{type(model).__name__}.joblib"
    model_path = f"../{MODELS_DIR}/{model_filename}"
    joblib.dump(model, model_path)
    print(f"      Modelo salvo em: {model_path}")

    return val_accuracy, val_f1_macro


def train_all_models():
    print("Iniciando o treinamento de todos os modelos...\n")

    final_stats = {}
    for emb_name, emb_func in EMBEDDERS.items():
        for alg_name, model in models.items():
            try:
                print(f"\nTreinando modelo: {alg_name.upper()} | Embedding: {emb_name}")
                accuracy, f1_macro = train_model(model, emb_name)
                final_stats[(emb_name, alg_name)] = (accuracy, f1_macro)
            except Exception as e:
                print(f"Erro ao treinar modelo {alg_name.upper()} para o dataset de treino com embedding {emb_name}: {e}")

    print("\nTreinamento concluído!")
    return final_stats
