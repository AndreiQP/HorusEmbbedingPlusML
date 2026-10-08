import requests

# Lembre-se de verificar se o seu Flask está rodando na porta 5000 ou 3000
# Altere aqui se necessário
BASE_URL = "http://localhost:5000" 

def testar_fase_2():
    print("\n--- TESTE FASE 2: Histórico Completo ---")
    url = f"{BASE_URL}/api/analyse_history"
    
    # Simulando a extensão enviando o histórico completo
    payload = {
        "apiVersion": "v1",
        "analysisId": "manual-api-test",
        "chatId": "Número Desconhecido",
        "conversationVersion": 1,
        "turns": [
            {"turnId": "turn:b1", "speaker": "Innocent", "balloons": [
                {"balloonId": "b1", "text": "Oi, quem é?"}
            ]},
            {"turnId": "turn:b2", "speaker": "Suspect", "balloons": [
                {"balloonId": "b2", "text": "Oi mãe, mudei de número!"},
                {"balloonId": "b3", "text": "Faz um PIX urgente para mim?"}
            ]}
        ]
    }
    
    try:
        print(f"Enviando POST para {url}...")
        response = requests.post(url, json=payload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Resposta JSON:", response.json())
        else:
            print("Erro no Servidor:", response.text)
            
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    testar_fase_2()
