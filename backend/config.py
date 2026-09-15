"""Shared configuration.

API credentials intentionally come from the process environment.  Keeping the
historic constant names preserves compatibility with the existing embedding code.
"""
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
TRAIN_DATASET_FILE = "../datasets/dataset_train/raw/all_data.csv"
VALIDATION_DATASET_FILE = "../datasets/dataset_validation/raw/all_data.csv"
EMBEDDERS = {
    "minilm": "all-MiniLM-L6-v2",
    "voyage": "voyage-3-large",
    "openai": "text-embedding-3-large",
    "e5": "intfloat/multilingual-e5-large",
    "bge": "BAAI/bge-m3",
}
