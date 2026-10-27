from __future__ import annotations

import argparse
import json

from .common import TRANSFORMER_STUDIES, atomic_json
from .dataset_size import (
    aggregate_dataset_size_results, run_transformer_dataset_size_study,
    run_unsupervised_dataset_size_study,
)
from .explainability import (
    analyze_transformer_head_ablation, build_transformer_explainability_report,
    extract_transformer_head_attention,
)
from .manifest import select_study_finalists
from .hierarchical_perturbation import (
    explain_cached_dataset_sample,
    explain_conversation,
    load_bge_encoder,
    save_hierarchical_explanations,
)
from .hierarchical_global import (
    HIERARCHICAL_EXPLAINABILITY_ROOT,
    aggregate_global_results,
    build_global_inventory,
    run_global_shard,
)
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
    hierarchical = sub.add_parser("hierarchical-explain")
    hierarchical_source = hierarchical.add_mutually_exclusive_group(required=True)
    hierarchical_source.add_argument("--conversation")
    hierarchical_source.add_argument("--conversation-file")
    hierarchical_source.add_argument("--sample-id")
    hierarchical.add_argument("--split", choices=["test_internal", "validation"], default="test_internal")
    hierarchical.add_argument("--seed", type=int, action="append")
    hierarchical.add_argument("--top-messages", type=int, default=6)
    hierarchical.add_argument("--max-ngram", type=int, default=5)
    hierarchical.add_argument("--top-spans-per-message", type=int, default=20)
    hierarchical.add_argument("--output-dir")
    hierarchical.add_argument(
        "--checkpoint-root",
        help=(
            "Raiz que contém os model.pth finalistas BGE; aceita experiment_results, "
            "a pasta transformer ou um ancestral dos runs."
        ),
    )
    study = sub.add_parser("hierarchical-study")
    study.add_argument("--action", required=True, choices=["inventory", "shard", "aggregate"])
    study.add_argument("--split", choices=["test_internal", "validation", "all"], default="all")
    study.add_argument("--shard-index", type=int)
    study.add_argument("--shard-count", type=int, default=32)
    study.add_argument("--primary-seed", type=int, default=42)
    study.add_argument("--stability-seed", type=int, action="append")
    study.add_argument("--top-messages", type=int, default=6)
    study.add_argument("--max-ngram", type=int, default=5)
    study.add_argument("--top-spans-per-message", type=int, default=20)
    study.add_argument("--controls-per-span", type=int, default=5)
    study.add_argument("--output-dir", default=str(HIERARCHICAL_EXPLAINABILITY_ROOT))
    study.add_argument("--checkpoint-root")
    study.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    study.add_argument("--no-resume", action="store_true")
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
    elif args.command == "hierarchical-explain":
        from pathlib import Path

        seeds = tuple(args.seed or [42])
        output_root = TRANSFORMER_STUDIES / "hierarchical_explainability"
        output_dir = args.output_dir or str(output_root / "manual")
        common = {
            "top_messages": args.top_messages,
            "max_ngram": args.max_ngram,
            "top_spans_per_message": args.top_spans_per_message,
            "checkpoint_root": args.checkpoint_root,
        }
        if args.sample_id is not None:
            explanations = explain_cached_dataset_sample(
                args.sample_id,
                split=args.split,
                seeds=seeds,
                output_dir=output_dir,
                **common,
            )
        else:
            conversation = (
                Path(args.conversation_file).read_text(encoding="utf-8")
                if args.conversation_file is not None else args.conversation
            )
            encoder = load_bge_encoder()
            explanations = [
                explain_conversation(conversation, seed=seed, encoder=encoder, **common)
                for seed in seeds
            ]
            destination = save_hierarchical_explanations(
                explanations,
                output_dir=output_dir,
            )
            output_dir = str(destination)
        result = {
            "artifacts_dir": output_dir,
            "runs": [
                {
                    "seed": item.metadata["seed"],
                    "analysis_scope": item.metadata.get("analysis_scope"),
                    "candidate_source_seed": item.metadata.get("candidate_source_seed"),
                    "sample_id": item.metadata.get("sample_id"),
                    "baseline": item.baseline,
                    "top_message_indices": [row["original_index"] for row in item.top_messages],
                    "strongest_scam_span": item.strongest_scam_span,
                    "strongest_ham_span": item.strongest_ham_span,
                }
                for item in explanations
            ],
        }
    elif args.command == "hierarchical-study":
        from pathlib import Path
        import torch

        splits = ("test_internal", "validation") if args.split == "all" else (args.split,)
        if args.action == "inventory":
            frame = build_global_inventory(
                splits=splits,
                primary_seed=args.primary_seed,
                output_root=args.output_dir,
                checkpoint_root=args.checkpoint_root,
            )
            result = {
                "artifacts_dir": str(Path(args.output_dir)),
                "conversations": frame.groupby("split")["sample_id"].nunique().to_dict(),
            }
        elif args.action == "shard":
            if args.split == "all" or args.shard_index is None:
                raise ValueError("--action shard exige --split explícito e --shard-index")
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA foi exigida, mas não está disponível")
            device = (
                torch.device("cuda") if args.device == "cuda"
                else torch.device("cpu") if args.device == "cpu"
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            try:
                destination = run_global_shard(
                    split=args.split,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    primary_seed=args.primary_seed,
                    seeds=tuple(args.stability_seed or (42, 52, 62)),
                    top_messages=args.top_messages,
                    max_ngram=args.max_ngram,
                    top_spans_per_message=args.top_spans_per_message,
                    controls_per_span=args.controls_per_span,
                    device=device,
                    output_root=args.output_dir,
                    checkpoint_root=args.checkpoint_root,
                    resume=not args.no_resume,
                )
            except Exception as exc:
                import traceback
                failure_dir = (
                    Path(args.output_dir) / args.split / "shards"
                    / f"shard-{args.shard_index:05d}"
                )
                atomic_json(failure_dir / "failure.json", {
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                raise
            result = {"artifacts_dir": str(destination), "device": str(device)}
        else:
            destination = aggregate_global_results(
                output_root=args.output_dir,
                expected_shards=args.shard_count,
                splits=splits,
            )
            result = {"artifacts_dir": str(destination), "status": "complete"}
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
