OPENAI_API_KEY="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"
VOYAGE_API_KEY="pa-R2UYufduP-vH3KTeOFS0HKL24moMeKuejqzqLLtPR9F"
TRAIN_DATASET_FILE = '../datasets/dataset_train/raw/all_data.csv'
VALIDATION_DATASET_FILE = '../datasets/dataset_validation/raw/all_data.csv'
EMBEDDERS = {
    'minilm': 'all-MiniLM-L6-v2',
    'voyage': 'voyage-3-large',
    'openai': 'text-embedding-3-large',
    'e5': 'intfloat/multilingual-e5-large',
    'bge': 'BAAI/bge-m3'
}