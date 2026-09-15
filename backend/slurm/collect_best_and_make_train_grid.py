"""
backend/slurm/collect_best_and_make_train_grid.py
------------------------------------------------------
Etapa 2 do pipeline com tuning automático (ver submit_full_pipeline.sh).

Para o modelo informado, lê o resultado do grid search (--mode grid_search,
já rodado e cacheado em experiment_results/unsupervised/gs_*/ pela etapa 1)
de CADA embedding e monta backend/slurm/grids/<model>_tuned.txt: três linhas
por embedding (seeds 42, 52 e 62), usando somente a dimensão e os parâmetros
vencedores na validação interna.

Como grid_search_anomaly() é cacheado por config, chamá-lo de novo aqui é
barato (cache hit) — não re-executa o tuning.

Uso:
    python3 backend/slurm/collect_best_and_make_train_grid.py <model>
"""
import argparse
import os
import sys

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _BACKEND_DIR)

from machine_learning.unsupervised.runner import PCA_DIMS, grid_search_anomaly  # noqa: E402

EMBEDDINGS = ["voyage", "openai", "e5", "bge", "minilm"]
SEEDS = [42, 52, 62]

_METRIC_COLS = {
    "internal_val_f1_macro", "internal_val_recall_scam", "internal_val_roc_auc", "internal_val_pr_auc",
    "_complexity",
}

_INTEGER_PARAMS = {
    "n_neighbors", "n_estimators", "n_epochs", "latent_dim", "batch_size", "warmup_epochs"
}


def _format_param(name, value):
    if name in _INTEGER_PARAMS and value is not None:
        return str(int(value))
    return str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()

    grids_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "experiment_results", "slurm_grids"
    ))
    os.makedirs(grids_dir, exist_ok=True)

    lines = []
    for emb in EMBEDDINGS:
        result = grid_search_anomaly(
            embedding=emb, model_type=args.model, pca_dims=PCA_DIMS,
            force_recompute=False,
        )
        if result.df_metrics.empty:
            print(f"[collect] Sem resultado de grid search para {args.model}/{emb} — pulando.", file=sys.stderr)
            continue
        best = result.df_metrics.iloc[0].to_dict()
        dim = int(best.pop("pca_dim"))
        params = {k: v for k, v in best.items() if k not in _METRIC_COLS}
        param_flags = " ".join(f"--param {k}={_format_param(k, v)}" for k, v in params.items())
        print(f"[collect] {args.model}/{emb}: melhores hiperparâmetros = {params}")
        for seed in SEEDS:
            lines.append(
                f"--mode train --model {args.model} --embedding {emb} "
                f"--dim {dim} --seed {seed} {param_flags}".strip()
            )

    path = os.path.join(grids_dir, f"{args.model}_tuned.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    finalize_path = os.path.join(grids_dir, f"{args.model}_finalize.txt")
    dims = " ".join(str(dim) for dim in PCA_DIMS)
    with open(finalize_path, "w") as f:
        f.write(f"--mode finalize --model {args.model} --dims {dims}\n")
    print(f"[collect] {path}: {len(lines)} combinações (treino final com hiperparâmetros tunados)")


if __name__ == "__main__":
    main()
