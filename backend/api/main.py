import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from machine_learning.embedder import generate_analysis_embedding
from machine_learning.predict import predict
from config import EMBEDDERS

PREDICT_MODEL = os.path.join(PROJECT_ROOT, 'fcnn_trained', 'with_augmented_all_data_fcnn.pth')
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def prepare_predict_data(df_final: pd.DataFrame) -> np.ndarray:
    """
    Filtra o DataFrame em formato longo gerado na inferência e empilha
    os embeddings horizontalmente para alimentar o modelo Random Forest/SVM.
    """
    print("\n--- Preparando X para Predição (Ensembling Horizontal) ---")
    
    X_current_list = []
    
    for emb_name, _ in EMBEDDERS.items():
        df_filtered = df_final[df_final['embedding_model'] == emb_name]
        
        X_numpy = np.array(df_filtered['text_embedded'].tolist(), dtype=np.float32)
        print(f"Shape do X extraído para {emb_name}: {X_numpy.shape}")
        X_current_list.append(X_numpy)
        
    try:
        X_final = np.concatenate(X_current_list, axis=1)
    except ValueError as e:
        print("Erro de dimensão. Verifique se algum embedder gerou menos linhas que os outros (ex: falha na API ou dropna).")
        raise e
    
    print(f"Shape final do X: {X_final.shape} -> (Total de Textos, Total de Features Concatenadas)")
    return X_final


# =====================================================================
# FASE 1: Análise Superficial, apenas a primeira mensagem
# =====================================================================
@app.route('/api/analyse_message', methods=['POST'])
def analyse_message():
    print(f'Entrou em Analyse_message')
    
    data = request.get_json()
    msg = data.get('mensagem', '')
    contact = data.get('contato', 'Desconhecido')

    print(f"\n[Fase 1] Analisando mensagem isolada de {contact}: '{msg}'")

    text = f'Suspect: {msg}'
    df = pd.DataFrame({'text': [text], 'label': [None]})

    df_analysis = generate_analysis_embedding(df)
    
    X_predict = prepare_predict_data(df_analysis)

    is_scam, prob = predict(PREDICT_MODEL, X_predict)

    is_scam = is_scam[0]
    prob = prob[0]

    print('saida')
    print(is_scam, prob)

    is_suspect = prob > 0.4
    print(f" -> Resultado: Probabilidade {prob:.2f} ({'Requer Contexto' if is_suspect else 'Seguro'})")

    return jsonify({
        "probabilidade": prob,
        "isSuspect": is_suspect
    })

# =====================================================================
# FASE 2: Análise Profunda, todo o histórico
# =====================================================================
@app.route('/api/analyse_history', methods=['POST'])
def analyse_history():
    print(f'Entrou em Analyse_History')

    data = request.get_json()
    history = data.get('historico', '')
    contact = data.get('contato', 'Desconhecido')

    print(f"\n[Fase 2] Analisando HISTÓRICO COMPLETO de {contact}...")
    print(f" -> Contexto recebido: {history[:150]}...") 

    df = pd.DataFrame({'text': [history], 'label': [None]})

    df_analysis = generate_analysis_embedding(df)
    
    X_predict = prepare_predict_data(df_analysis)

    is_scam, prob = predict(PREDICT_MODEL, X_predict)

    is_scam = is_scam[0]
    prob = prob[0]

    print(f" -> Resultado: Probabilidade {prob:.2f} ({'É SCAM!' if is_scam else 'Seguro'})")

    return jsonify({
        "probabilidade": f"{prob:.2f}",
        "isScam": is_scam
    })

if __name__ == '__main__':
    print("👁️ Backend IC Horus Iniciado na porta 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)