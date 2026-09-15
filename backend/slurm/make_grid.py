"""
backend/slurm/make_grid.py
------------------------------
Gera grades agrupadas em backend/experiment_results/slurm_grids/.
Cada linha corresponde a um job completo por algoritmo ou embedding, evitando
um job SLURM por configuração individual.

Uso:
    python3 backend/slurm/make_grid.py
"""
import os

CPU_MODELS = ["ocsvm", "iforest", "lof"]
GPU_MODELS = ["lunar", "svdd"]
DIMS = [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]
TRANSFORMER_EMBEDDINGS = ["voyage", "bge", "openai"]

_GRIDS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "experiment_results", "slurm_grids"
))

if __name__ == "__main__":
    os.makedirs(_GRIDS_DIR, exist_ok=True)

    dims = " ".join(str(dim) for dim in DIMS)
    for filename, models in (("unsupervised_cpu.txt", CPU_MODELS), ("unsupervised_gpu.txt", GPU_MODELS)):
        path = os.path.join(_GRIDS_DIR, filename)
        lines = [f"--mode finalize --model {model} --dims {dims}" for model in models]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[make_grid] {path}: {len(lines)} pipelines completas")

    with open(os.path.join(_GRIDS_DIR, "unsupervised_finalize.txt"), "w") as f:
        f.write("--mode leaderboard\n")

    transformer_path = os.path.join(_GRIDS_DIR, "transformer.txt")
    with open(transformer_path, "w") as f:
        f.write("\n".join(f"--mode finalist --embedding {emb}" for emb in TRANSFORMER_EMBEDDINGS) + "\n")
    with open(os.path.join(_GRIDS_DIR, "transformer_finalize.txt"), "w") as f:
        f.write("--mode finalize\n")
    print(f"[make_grid] {transformer_path}: {len(TRANSFORMER_EMBEDDINGS)} embeddings")
