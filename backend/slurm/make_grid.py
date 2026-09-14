"""
backend/slurm/make_grid.py
------------------------------
Gera, para CADA modelo implementado, arquivos de grade separados em
backend/slurm/grids/:
  <model>.txt          -> "--mode train" com hiperparâmetros PADRÃO
                           (embedding x dimensão; usado por submit_all_models.sh)
  <model>_tuning.txt    -> "--mode grid_search" (um por embedding, dim fixo em
                           TUNING_DIM; usado por submit_full_pipeline.sh, que
                           depois treina o modelo final com os hiperparâmetros
                           vencedores — ver collect_best_and_make_train_grid.py)
CVDD/DATE não têm tuning automatizado (grid_search não suportado para
modelos de texto no CLI), então só geram <model>.txt.

Uso:
    python3 backend/slurm/make_grid.py
"""
import os

EMBEDDING_MODELS = ["ocsvm", "iforest", "lof", "lunar", "svdd"]
EMBEDDINGS = ["voyage", "openai", "e5", "bge", "minilm"]
DIMS = [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]
TUNING_DIM = 300
TEXT_MODELS = ["cvdd", "date"]

_GRIDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids")

if __name__ == "__main__":
    os.makedirs(_GRIDS_DIR, exist_ok=True)

    for model in EMBEDDING_MODELS:
        lines = [
            f"--mode train --model {model} --embedding {emb} --dim {dim}"
            for emb in EMBEDDINGS for dim in DIMS
        ]
        path = os.path.join(_GRIDS_DIR, f"{model}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[make_grid] {path}: {len(lines)} combinações")

        tuning_lines = [
            f"--mode grid_search --model {model} --embedding {emb} --dim {TUNING_DIM} --force-retrain"
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
