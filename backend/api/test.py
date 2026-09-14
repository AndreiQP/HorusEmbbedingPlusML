import requests

# Lembre-se de verificar se o seu Flask está rodando na porta 5000 ou 3000
# Altere aqui se necessário
BASE_URL = "http://localhost:5000" 

def testar_fase_1():
    print("--- TESTE FASE 1: Mensagem Isolada ---")
    url = f"{BASE_URL}/api/analyse_message"
    
    # Simulando a extensão enviando uma mensagem suspeita
    payload = {
        "mensagem": "Oi mãe, meu celular quebrou. Faz um pix urgente pra mim?",
        "contato": "Número Desconhecido"
    }
    
    try:
        print(f"Enviando POST para {url}...")
        response = requests.post(url, json=payload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Resposta JSON:", response.json())
        else:
            print("Erro no Servidor:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ ERRO FATAL: Não consegui me conectar. O Flask está rodando?")
    except Exception as e:
        print(f"❌ Erro desconhecido: {e}")

def testar_fase_2():
    print("\n--- TESTE FASE 2: Histórico Completo ---")
    url = f"{BASE_URL}/api/analyse_history"
    
    # Simulando a extensão enviando o histórico completo
    payload = {
        "historico": "Innocent: Oi quem é? Suspect: Oi mãe, mudei de numero! Salva ai. Innocent: Ok filho. Suspect: Faz um pix urgente pra eu pagar o mecânico do celular?",
        "contato": "Número Desconhecido"
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
    # Garanta que você instalou a biblioteca requests: pip install requests
    testar_fase_1()
    testar_fase_2()