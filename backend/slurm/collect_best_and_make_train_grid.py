"""
backend/slurm/collect_best_and_make_train_grid.py
------------------------------------------------------
Etapa 2 do pipeline com tuning automático (ver submit_full_pipeline.sh).

Para o modelo informado, lê o resultado do grid search (--mode grid_search,
já rodado e cacheado em experiment_results/unsupervised/gs_*/ pela etapa 1)
de CADA embedding e monta backend/slurm/grids/<model>_tuned.txt: uma linha
"--mode train ... --param k=v ..." por (embedding x dimensão), já com os
hiperparâmetros vencedores daquele embedding aplicados em todas as dimensões.

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

from machine_learning.unsupervised.runner import grid_search_anomaly  # noqa: E402

EMBEDDINGS = ["voyage", "openai", "e5", "bge", "minilm"]
DIMS = [25, 60, 100, 150, 200, 250, 300, 500, 750, 1000]
TUNING_DIM = 300

_METRIC_COLS = {
    "internal_val_f1", "internal_val_recall_scam", "internal_val_roc_auc", "internal_val_pr_auc",
    "test_f1", "test_recall_scam", "test_roc_auc", "test_pr_auc",
    "external_val_f1", "external_val_recall_scam", "external_val_roc_auc", "external_val_pr_auc",
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

    grids_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids")
    os.makedirs(grids_dir, exist_ok=True)

    lines = []
    for emb in EMBEDDINGS:
        result = grid_search_anomaly(embedding=emb, model_type=args.model, pca_dim=TUNING_DIM, force_recompute=False)
        if result.df_metrics.empty:
            print(f"[collect] Sem resultado de grid search para {args.model}/{emb} — pulando.", file=sys.stderr)
            continue
        best = result.df_metrics.iloc[0].to_dict()
        params = {k: v for k, v in best.items() if k not in _METRIC_COLS}
        param_flags = " ".join(f"--param {k}={_format_param(k, v)}" for k, v in params.items())
        print(f"[collect] {args.model}/{emb}: melhores hiperparâmetros = {params}")
        for dim in DIMS:
            lines.append(f"--mode train --model {args.model} --embedding {emb} --dim {dim} --force-retrain {param_flags}".strip())

    path = os.path.join(grids_dir, f"{args.model}_tuned.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[collect] {path}: {len(lines)} combinações (treino final com hiperparâmetros tunados)")


if __name__ == "__main__":
    main()
