import os
import argparse
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import voyageai
from openai import OpenAI

from config import OPENAI_API_KEY, VOYAGE_API_KEY, EMBEDDERS

LOCAL_MODELS_CACHE = {}

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f'Arquivo {csv_path} não encontrado, skip')
        return None
    if csv_path.endswith('.csv'):
        return pd.read_csv(csv_path)
    return pd.read_parquet(csv_path)

def get_message_chunks(text: str) -> list:
    """Quebra o texto em mensagens usando regex e retorna TODAS as mensagens na ordem."""
    if not isinstance(text, str) or not text.strip():
        return [""]
    
    chunks = re.split(r'(?=Innocent: |Suspect: )', text)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks if chunks else [""]

def get_openai_embeddings(texts, embedding_name):
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []

    for i, text in enumerate(texts):
        chunks = get_message_chunks(text)
        try:
            response = client.embeddings.create(
                input=chunks,
                model=EMBEDDERS.get(embedding_name, embedding_name)
            )
            chunk_vectors = [item.embedding for item in response.data]
            all_embeddings.append(chunk_vectors)
        except Exception as e:
            print(f"Erro na API da OpenAI no doc local {i+1}: {e}")
            all_embeddings.append(None) # Mantém None para tentar de novo numa próxima rodada

    return all_embeddings

def get_voyage_embeddings(texts, embedding_name):
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    all_embeddings = []
    
    # Limite seguro da API Voyage para batch de embeddings
    MAX_API_BATCH_SIZE = 900 

    for i, text in enumerate(texts):
        chunks = get_message_chunks(text)
        doc_embeddings = []
        
        try:
            # Divide os chunks de um único texto em sub-lotes permitidos pela API
            for j in range(0, len(chunks), MAX_API_BATCH_SIZE):
                sub_chunks = chunks[j : j + MAX_API_BATCH_SIZE]
                
                result = vo.embed(
                    sub_chunks, 
                    model=EMBEDDERS.get(embedding_name, embedding_name), 
                    input_type="document"
                )
                
                # Acumula os embeddings deste sub-lote
                doc_embeddings.extend(result.embeddings)
            
            # Adiciona todos os embeddings reconstruídos para este texto
            all_embeddings.append(doc_embeddings)
            
        except Exception as e:
            print(f"Erro na API Voyage no doc local {i+1}: {e}")
            all_embeddings.append(None)

    return all_embeddings

def get_local_embedding(texts, embedding_name):
    # Carrega o modelo apenas se ele ainda não estiver na memória
    if embedding_name not in LOCAL_MODELS_CACHE:
        print(f"Carregando modelo local '{embedding_name}' na memória pela primeira vez...")
        LOCAL_MODELS_CACHE[embedding_name] = SentenceTransformer(EMBEDDERS[embedding_name])
    
    model = LOCAL_MODELS_CACHE[embedding_name]
    embeddings = []
    
    for i, text in enumerate(texts):
        chunks = get_message_chunks(text)
        if embedding_name == 'e5':
            chunks = [f'passage: {chunk}' for chunk in chunks]
        
        chunk_embeddings = model.encode(chunks, show_progress_bar=False)
        embeddings.append(chunk_embeddings.tolist())

    return embeddings

def _get_clean_dataframe(ds_path):
    df = load_data(ds_path)
    if df is None: 
        return None

    text_col = 'text_last_n_turns' if 'text_last_n_turns' in df.columns else 'text'
    
    if text_col not in df.columns:
        print(f"Nenhuma coluna de texto válida encontrada ('text_last_n_turns' ou 'text').")
        return None

    df_clean = df.dropna(subset=[text_col]).copy()
    if len(df_clean) == 0: 
        print("Nenhum texto válido encontrado após limpeza.")
        return None
    
    return df_clean, text_col

def generate_embedding(embedding_name, file_path, batch_size=50):
    print(f'\n--- Embeddando com [{embedding_name}] ---')

    output_filename = f'datasets/dataset_validation/processed/messages_transformer/{embedding_name}_chunks.parquet'

    # 1. Tentar carregar um checkpoint salvo anteriormente
    if os.path.exists(output_filename):
        print(f"Arquivo de checkpoint encontrado. Retomando o progresso de: {output_filename}")
        df_clean = pd.read_parquet(output_filename)
        text_col = 'text_last_n_turns' if 'text_last_n_turns' in df_clean.columns else 'text'
    else:
        # Se não houver checkpoint, carrega o arquivo original
        result = _get_clean_dataframe(file_path)
        if result is None:
            return
        df_clean, text_col = result

    # 2. Garantir que a coluna alvo existe e é do tipo object (para aceitar listas)
    if 'text_embedded' not in df_clean.columns:
        df_clean['text_embedded'] = pd.Series(dtype='object')

    # 3. Filtrar apenas as linhas que ainda NÃO foram processadas (onde text_embedded é nulo)
    def is_missing(val):
        return val is None or (isinstance(val, float) and pd.isna(val))

    mask_to_process = df_clean['text_embedded'].apply(is_missing)
    indices_to_process = df_clean[mask_to_process].index.tolist()

    if not indices_to_process:
        print(f"[{embedding_name}] Todos os {len(df_clean)} textos já possuem embeddings! Processo finalizado.")
        return

    print(f"Textos já processados: {len(df_clean) - len(indices_to_process)}/{len(df_clean)}")
    print(f"Faltam processar: {len(indices_to_process)}. Iniciando em lotes de {batch_size}...")

    # 4. Iniciar o processamento em lotes
    total_batches = (len(indices_to_process) - 1) // batch_size + 1

    for batch_num, i in enumerate(range(0, len(indices_to_process), batch_size), start=1):
        batch_indices = indices_to_process[i : i + batch_size]
        batch_texts = df_clean.loc[batch_indices, text_col].tolist()

        print(f"Processando lote {batch_num}/{total_batches} ({len(batch_texts)} conversas)...")

        # Obter embeddings
        if embedding_name == 'openai':
            batch_embeddings = get_openai_embeddings(batch_texts, embedding_name)
        elif embedding_name == 'voyage':
            batch_embeddings = get_voyage_embeddings(batch_texts, embedding_name)
        else:
            batch_embeddings = get_local_embedding(batch_texts, embedding_name)

        # Atualizar o DataFrame usando .at para inserir as listas com segurança
        for idx, emb in zip(batch_indices, batch_embeddings):
            if emb is not None:  # Só salva se não houve erro na API
                df_clean.at[idx, 'text_embedded'] = emb

        # 5. Salvar o progresso (Checkpoint)
        df_clean.to_parquet(output_filename)
        print(f"-> Checkpoint salvo! Lote {batch_num} concluído.")

    print(f'\nSucesso: Processamento de [{embedding_name}] finalizado. Arquivo salvo: {output_filename}')

def generate_embedding_all_embeddings(file_path, batch_size=50):
    print(f"Iniciando geração de embeddings para o arquivo: {file_path}\n")
    for embedding_name in EMBEDDERS.keys():
        generate_embedding(embedding_name, file_path, batch_size)