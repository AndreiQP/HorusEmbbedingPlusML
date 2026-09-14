"""
machine_learning/unsupervised/run_experiment.py
---------------------------------------------------
CLI único para treinar/avaliar qualquer modelo unsupervised a partir do terminal,
pensado para ser chamado de dentro de um job SLURM (ver backend/slurm/).

Exemplos:
    # Treino simples (embeddings + PCA/UMAP)
    python -m machine_learning.unsupervised.run_experiment \\
        --mode train --model lof --embedding bge --dim 300 --reducer pca

    # Curva de aprendizado
    python -m machine_learning.unsupervised.run_experiment \\
        --mode learning_curve --model svdd --embedding voyage --dim 200

    # Grid search
    python -m machine_learning.unsupervised.run_experiment \\
        --mode grid_search --model iforest --embedding e5 --dim 500

    # Todos os embeddings de uma vez
    python -m machine_learning.unsupervised.run_experiment \\
        --mode train_all --model lunar --dim 300

    # Modelos de texto bruto (BGE apenas)
    python -m machine_learning.unsupervised.run_experiment --mode train --model cvdd
    python -m machine_learning.unsupervised.run_experiment --mode train --model date

Hiperparâmetros específicos do modelo são passados como `--param chave=valor`
(pode ser repetido), e são convertidos automaticamente para int/float/bool quando possível.
"""
from __future__ import annotations
import argparse
import sys
import json

TEXT_MODELS = {"cvdd", "date"}
EMBEDDING_MODELS = {"ocsvm", "iforest", "lof", "lunar", "svdd"}


def _parse_value(raw: str):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        value = float(raw)
        # hiperparâmetros inteiros (ex: n_neighbors, n_estimators) podem vir como "10.0"
        # do grid search (colunas numéricas do pandas viram float) — sklearn valida o tipo
        return int(value) if value.is_integer() else value
    except ValueError:
        pass
    return raw


def _parse_params(pairs):
    kwargs = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"--param deve ser 'chave=valor', recebido: '{pair}'")
        key, raw_value = pair.split("=", 1)
        kwargs[key] = _parse_value(raw_value)
    return kwargs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark de modelos unsupervised de detecção de scam.")
    p.add_argument("--mode", required=True,
                    choices=["train", "learning_curve", "grid_search", "train_all", "compare_dimensions"])
    p.add_argument("--model", required=True,
                    choices=sorted(EMBEDDING_MODELS | TEXT_MODELS))
    p.add_argument("--embedding", default="bge", help="Embedder (ignorado para cvdd/date, que usam BGE fixo)")
    p.add_argument("--embeddings", nargs="*", default=None, help="Lista de embedders (para train_all/compare_dimensions)")
    p.add_argument("--dim", type=int, default=300, help="Dimensão do PCA/UMAP")
    p.add_argument("--dims", type=int, nargs="*", default=None, help="Lista de dimensões (para compare_dimensions)")
    p.add_argument("--reducer", default="pca", choices=["pca", "umap"])
    p.add_argument("--path-data", default="all_data", choices=["all_data", "suspect_turns"])
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--param", action="append", default=[],
                    help="Hiperparâmetro específico do modelo, formato chave=valor (repetível)")
    p.add_argument("--output-json", default=None, help="Se informado, salva um resumo das métricas neste caminho JSON")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_kwargs = _parse_params(args.param)
    is_text_model = args.model in TEXT_MODELS

    summary = {}
    try:
        if is_text_model:
            from . import text_runner
            train_fn = text_runner.train_cvdd if args.model == "cvdd" else text_runner.train_date
            if args.mode != "train":
                print(f"[run_experiment] --mode='{args.mode}' ainda não suportado para modelos de texto ({args.model}); "
                      f"use --mode train.", file=sys.stderr)
                return 2
            result = train_fn(
                test_size=args.test_size, val_size=args.val_size,
                force_retrain=args.force_retrain, **model_kwargs,
            )
            summary = result.df_metrics.to_dict(orient="records")
        else:
            from . import runner
            if args.mode == "train":
                result = runner.train_anomaly(
                    embedding=args.embedding, model_type=args.model, pca_dim=args.dim,
                    path_data=args.path_data, reducer=args.reducer,
                    test_size=args.test_size, val_size=args.val_size,
                    force_retrain=args.force_retrain, **model_kwargs,
                )
                summary = result.df_metrics.to_dict(orient="records")
            elif args.mode == "learning_curve":
                result = runner.learning_curve_anomaly(
                    embedding=args.embedding, model_type=args.model, pca_dim=args.dim,
                    path_data=args.path_data, reducer=args.reducer,
                    force_recompute=args.force_retrain, **model_kwargs,
                )
                summary = result.df_history.to_dict(orient="records")
            elif args.mode == "grid_search":
                result = runner.grid_search_anomaly(
                    embedding=args.embedding, model_type=args.model, pca_dim=args.dim,
                    path_data=args.path_data, force_recompute=args.force_retrain,
                    test_size=args.test_size, val_size=args.val_size,
                    param_grid=({k: [v] for k, v in model_kwargs.items()} or None),
                )
                # colunas de métrica não fazem parte dos hiperparâmetros do melhor combo
                _metric_cols = {
                    "internal_val_f1", "internal_val_recall_scam", "internal_val_roc_auc", "internal_val_pr_auc",
                    "test_f1", "test_recall_scam", "test_roc_auc", "test_pr_auc",
                    "external_val_f1", "external_val_recall_scam", "external_val_roc_auc", "external_val_pr_auc",
                }
                best_row = result.df_metrics.iloc[0].to_dict()
                best_params = {k: v for k, v in best_row.items() if k not in _metric_cols}
                summary = {
                    "best_params": best_params,
                    "best_metrics": {k: v for k, v in best_row.items() if k in _metric_cols},
                    "history": result.df_history.to_dict(orient="records"),
                }
            elif args.mode == "train_all":
                df = runner.train_all_anomaly(
                    model_type=args.model, pca_dim=args.dim, embeddings=args.embeddings,
                    path_data=args.path_data, reducer=args.reducer,
                    force_retrain=args.force_retrain, **model_kwargs,
                )
                summary = df.to_dict(orient="records")
            elif args.mode == "compare_dimensions":
                df = runner.compare_dimensions_anomaly(
                    model_type=args.model, embeddings=args.embeddings, dimensions=args.dims,
                    path_data=args.path_data, reducer=args.reducer,
                    force_retrain=args.force_retrain, **model_kwargs,
                )
                summary = df.to_dict(orient="records")
    except Exception as e:
        print(f"[run_experiment] ERRO: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"\n[run_experiment] OK — model={args.model} mode={args.mode}")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[run_experiment] Resumo salvo em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
