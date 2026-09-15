from __future__ import annotations

import argparse
import json

from .dataset_size import (
    aggregate_dataset_size_results, run_transformer_dataset_size_study,
    run_unsupervised_dataset_size_study,
)
from .explainability import (
    analyze_transformer_head_ablation, build_transformer_explainability_report,
    extract_transformer_head_attention,
)
from .manifest import select_study_finalists
from .report import build_studies_report
from .slurm import write_slurm_grids
from .transformer_analysis import analyze_transformer_complementarity, run_transformer_threshold_study


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Estudos cache-first de Transformer e Unsupervised")
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--strategy", choices=["all", "transformer", "unsupervised"], default="all")
    prepare.add_argument("--top-k", type=int, default=3)
    prepare.add_argument("--no-strict", action="store_true")

    size = sub.add_parser("dataset-size")
    size.add_argument("--strategy", required=True, choices=["transformer", "unsupervised"])
    size.add_argument("--candidate")
    size.add_argument("--fraction", type=float, action="append")
    size.add_argument("--seed", type=int, action="append")
    size.add_argument("--force", action="store_true")

    complementarity = sub.add_parser("complementarity")
    complementarity.add_argument("--force", action="store_true")
    threshold = sub.add_parser("threshold")
    threshold.add_argument("--force", action="store_true")
    explain = sub.add_parser("explainability")
    explain.add_argument("--embedding", choices=["voyage", "bge", "openai"])
    explain.add_argument("--seed", type=int, default=42)
    explain.add_argument("--split", choices=["test_internal", "validation", "all"], default="all")
    explain.add_argument("--attention-only", action="store_true")
    explain.add_argument("--force", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("report")
    grids = sub.add_parser("make-slurm-grids")
    grids.add_argument("--output-dir")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = select_study_finalists(args.strategy, args.top_k, strict=not args.no_strict).to_dict()
    elif args.command == "dataset-size":
        kwargs = {
            "candidate": args.candidate, "force": args.force,
            "fractions": args.fraction or (0.05, 0.10, 0.25, 0.50, 0.75, 1.0),
            "seeds": args.seed or (42, 52, 62),
        }
        experiment = run_transformer_dataset_size_study(**kwargs) if args.strategy == "transformer" else run_unsupervised_dataset_size_study(**kwargs)
        result = {"rows": len(experiment.df_history), "artifacts_dir": experiment.artifacts_dir}
    elif args.command == "complementarity":
        experiment = analyze_transformer_complementarity(force=args.force)
        result = {"rows": len(experiment.df_history), "artifacts_dir": experiment.artifacts_dir}
    elif args.command == "threshold":
        experiment = run_transformer_threshold_study(force=args.force)
        result = {"rows": len(experiment.df_metrics), "artifacts_dir": experiment.artifacts_dir}
    elif args.command == "explainability":
        splits = ("test_internal", "validation") if args.split == "all" else (args.split,)
        result = {"runs": []}
        for split in splits:
            attention = extract_transformer_head_attention(split=split, seed=args.seed, embedding=args.embedding, force=args.force)
            run = {"split": split, "attention_rows": len(attention.df_history)}
            if not args.attention_only:
                ablation = analyze_transformer_head_ablation(split=split, seed=args.seed, embedding=args.embedding, force=args.force)
                run["ablation_rows"] = len(ablation.df_history)
            result["runs"].append(run)
    elif args.command == "aggregate":
        result = aggregate_dataset_size_results()
        try:
            result["explainability"] = build_transformer_explainability_report().artifacts_dir
        except FileNotFoundError:
            result["explainability"] = "incomplete"
    elif args.command == "make-slurm-grids":
        result = write_slurm_grids(args.output_dir)
    else:
        result = build_studies_report()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
