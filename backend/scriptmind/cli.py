"""Command line interface used locally and by the SLURM jobs."""
from __future__ import annotations

import argparse
import json
from typing import Any

from .paths import ScriptMindPaths


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scriptmind", description="Horus ScriptMind research pipeline")
    parser.add_argument("--data-root", help="Override isolated ScriptMind data root")
    parser.add_argument("--results-root", help="Override isolated ScriptMind results root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-data", help="Create the read-only source inventory")
    audit.add_argument("--limit", type=int)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--similarity-threshold", type=float, default=0.93)
    audit.add_argument("--force", action="store_true")

    annotate = subparsers.add_parser("annotate", help="Run one annotation provider on one shard")
    annotate.add_argument("--provider", choices=["gemini", "local"], required=True)
    annotate.add_argument("--shard-index", type=int, default=0)
    annotate.add_argument("--num-shards", type=int, default=32)
    annotate.add_argument("--model-id")
    annotate.add_argument("--limit", type=int)
    annotate.add_argument("--force", action="store_true")

    adjudicate = subparsers.add_parser("adjudicate", help="Accept agreements and arbitrate disagreements")
    adjudicate.add_argument("--shard-index", type=int, default=0)
    adjudicate.add_argument("--num-shards", type=int, default=32)
    adjudicate.add_argument("--limit", type=int)
    adjudicate.add_argument("--force", action="store_true")

    quality = subparsers.add_parser("annotation-quality", help="Compute pilot inter-annotator agreement")
    quality.add_argument("--pilot-size", type=int, default=200)
    quality.add_argument("--enforce", action="store_true", help="Exit non-zero when taxonomy thresholds are not met")

    prepare_audit = subparsers.add_parser("prepare-human-audit", help="Create the deterministic 200-row audit sheet")
    prepare_audit.add_argument("--sample-size", type=int, default=200)
    prepare_audit.add_argument("--force", action="store_true")
    subparsers.add_parser("score-human-audit", help="Score a completed human annotation audit sheet")

    build = subparsers.add_parser("build-csid", help="Build parquet/JSONL CSID and transition matrices")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--allow-partial", action="store_true")
    build.add_argument("--minimum-annotation-coverage", type=float, default=0.95)
    build.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train", help="QLoRA fine-tuning")
    train.add_argument("--model-key", choices=["llama1b", "llama8b"], default="llama8b")
    train.add_argument("--seed", type=int, choices=[42, 52, 62], default=42)
    train.add_argument("--ablation", choices=["multitask", "detection_only", "no_next_utterance"], default="multitask")
    train.add_argument("--origin-mode", choices=["all", "external"], default="all")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate zero-shot or fine-tuned generation")
    evaluate.add_argument("--model-key", choices=["llama1b", "llama8b"], default="llama8b")
    evaluate.add_argument("--seed", type=int, choices=[42, 52, 62], default=42)
    evaluate.add_argument("--split", choices=["test", "external_validation"], default="test")
    evaluate.add_argument("--mode", choices=["finetuned", "zero_shot"], default="finetuned")
    evaluate.add_argument("--ablation", choices=["multitask", "detection_only", "no_next_utterance"], default="multitask")
    evaluate.add_argument("--origin-mode", choices=["all", "external"], default="all")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--force", action="store_true")

    judge = subparsers.add_parser("judge-evaluation", help="Run Gemini rubric on one evaluation directory")
    judge.add_argument("--evaluation-dir", required=True)
    judge.add_argument("--sample-size", type=int, default=200)
    judge.add_argument("--model-id")
    judge.add_argument("--force", action="store_true")

    judge_audit = subparsers.add_parser("score-judge-audit", help="Correlate completed human and LLM judge scores")
    judge_audit.add_argument("--evaluation-dir", required=True)

    subparsers.add_parser("report", help="Consolidate metrics and existing read-only baselines")
    return parser


def _paths(args: argparse.Namespace) -> ScriptMindPaths:
    defaults = ScriptMindPaths.defaults()
    from pathlib import Path
    return ScriptMindPaths(
        Path(args.data_root).resolve() if args.data_root else defaults.data_root,
        Path(args.results_root).resolve() if args.results_root else defaults.results_root,
    ).ensure()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.command == "audit-data":
        from .data import audit_data
        _print(audit_data(paths, limit=args.limit, seed=args.seed, similarity_threshold=args.similarity_threshold, force=args.force))
    elif args.command == "annotate":
        from .annotation import annotate_dialogues
        print(annotate_dialogues(
            args.provider, paths=paths, shard_index=args.shard_index, num_shards=args.num_shards,
            model_id=args.model_id, limit=args.limit, force=args.force,
        ))
    elif args.command == "adjudicate":
        from .annotation import adjudicate_annotations
        print(adjudicate_annotations(
            paths=paths, shard_index=args.shard_index, num_shards=args.num_shards,
            limit=args.limit, force=args.force,
        ))
    elif args.command == "annotation-quality":
        from .annotation import annotation_quality
        result = annotation_quality(paths, pilot_size=args.pilot_size)
        _print(result)
        if args.enforce and not result["taxonomy_ready"]:
            raise SystemExit("Annotation pilot failed: require kappa >= 0.75 and arbitration rate < 0.15")
    elif args.command == "prepare-human-audit":
        from .annotation import prepare_human_audit
        print(prepare_human_audit(paths, sample_size=args.sample_size, force=args.force))
    elif args.command == "score-human-audit":
        from .annotation import score_human_audit
        _print(score_human_audit(paths))
    elif args.command == "build-csid":
        from .csid import build_csid
        _print(build_csid(
            paths, seed=args.seed, force=args.force, allow_partial=args.allow_partial,
            minimum_annotation_coverage=args.minimum_annotation_coverage,
        ))
    elif args.command == "train":
        from .training import train_scriptmind
        result = train_scriptmind(
            paths=paths, model_key=args.model_key, seed=args.seed, ablation=args.ablation,
            origin_mode=args.origin_mode, epochs=args.epochs, force=args.force,
        )
        _print({"config": result.config, "metrics": result.df_metrics.to_dict(orient="records"), "extra": result.extra})
    elif args.command == "evaluate":
        from .evaluation import evaluate_scriptmind
        result = evaluate_scriptmind(
            paths=paths, model_key=args.model_key, seed=args.seed, split=args.split, mode=args.mode,
            ablation=args.ablation, origin_mode=args.origin_mode, limit=args.limit, force=args.force,
        )
        _print({"config": result.config, "metrics": result.df_metrics.to_dict(orient="records"), "extra": result.extra})
    elif args.command == "judge-evaluation":
        from .evaluation import judge_evaluation
        _print(judge_evaluation(
            args.evaluation_dir, paths=paths, sample_size=args.sample_size,
            model_id=args.model_id, force=args.force,
        ))
    elif args.command == "score-judge-audit":
        from .evaluation import score_judge_audit
        _print(score_judge_audit(args.evaluation_dir, paths=paths))
    elif args.command == "report":
        from .report import generate_report
        _print(generate_report(paths))
    else:
        parser.error(f"Unsupported command: {args.command}")
