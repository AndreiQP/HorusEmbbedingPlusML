from __future__ import annotations

from pathlib import Path

from .common import TRANSFORMER_STUDIES, atomic_json
from .manifest import load_study_manifest


def write_slurm_grids(output_dir: str | Path | None = None) -> dict[str, int]:
    manifest = load_study_manifest()
    destination = Path(output_dir) if output_dir else TRANSFORMER_STUDIES / "slurm_grids"
    destination.mkdir(parents=True, exist_ok=True)

    transformer_lines = [
        f"{candidate['candidate']}\t{fraction}\t{seed}"
        for candidate in manifest.transformer for fraction in manifest.fractions for seed in manifest.seeds
    ]
    cpu_lines, gpu_lines = [], []
    for candidate in manifest.unsupervised:
        target = gpu_lines if candidate["model_type"] in {"lunar", "svdd"} else cpu_lines
        target.extend(
            f"{candidate['candidate']}\t{fraction}\t{seed}"
            for fraction in manifest.fractions for seed in manifest.seeds
        )
    explainability_lines = [
        f"{candidate['embedding']}\t{seed}"
        for candidate in manifest.transformer for seed in manifest.seeds
    ]
    grids = {
        "transformer_size.tsv": transformer_lines,
        "unsupervised_size_cpu.tsv": cpu_lines,
        "unsupervised_size_gpu.tsv": gpu_lines,
        "explainability.tsv": explainability_lines,
    }
    for name, lines in grids.items():
        path = destination / name
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    counts = {name: len(lines) for name, lines in grids.items()}
    atomic_json(destination / "grid_manifest.json", {"study_manifest_hash": manifest.manifest_hash, "counts": counts})
    return counts
