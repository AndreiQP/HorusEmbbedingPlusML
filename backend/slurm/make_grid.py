"""
backend/slurm/make_grid.py
------------------------------
Gera arquivos temporários de grade em
backend/experiment_results/slurm_grids/:
  <model>_tuning.txt   -> um grid conjunto (dimensão + hiperparâmetros) por embedding.
CVDD/DATE não têm tuning automatizado (grid_search não suportado para
modelos de texto no CLI), então só geram <model>.txt.

Uso:
    python3 backend/slurm/make_grid.py
"""
import os

EMBEDDING_MODELS = ["ocsvm", "iforest", "lof", "lunar", "svdd"]
EMBEDDINGS = ["voyage", "openai", "e5", "bge", "minilm"]
DIMS = [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]
TEXT_MODELS = ["cvdd", "date"]
TRANSFORMER_EMBEDDINGS = ["voyage", "bge", "openai"]

_GRIDS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "experiment_results", "slurm_grids"
))

if __name__ == "__main__":
    os.makedirs(_GRIDS_DIR, exist_ok=True)

    for model in EMBEDDING_MODELS:
        dims = " ".join(str(dim) for dim in DIMS)
        tuning_lines = [
            f"--mode grid_search --model {model} --embedding {emb} --dims {dims}"
            for emb in EMBEDDINGS
        ]
        tuning_path = os.path.join(_GRIDS_DIR, f"{model}_tuning.txt")
        with open(tuning_path, "w") as f:
            f.write("\n".join(tuning_lines) + "\n")
        print(f"[make_grid] {tuning_path}: {len(tuning_lines)} combinações (tuning)")

    for model in TEXT_MODELS:
        path = os.path.join(_GRIDS_DIR, f"{model}.txt")
        with open(path, "w") as f:
            f.write(f"--mode train --model {model} --force-retrain\n")
        print(f"[make_grid] {path}: 1 combinação")

    transformer_path = os.path.join(_GRIDS_DIR, "transformer.txt")
    with open(transformer_path, "w") as f:
        f.write("\n".join(f"--mode finalist --embedding {emb}" for emb in TRANSFORMER_EMBEDDINGS) + "\n")
    with open(os.path.join(_GRIDS_DIR, "transformer_finalize.txt"), "w") as f:
        f.write("--mode finalize\n")
    print(f"[make_grid] {transformer_path}: {len(TRANSFORMER_EMBEDDINGS)} embeddings")
