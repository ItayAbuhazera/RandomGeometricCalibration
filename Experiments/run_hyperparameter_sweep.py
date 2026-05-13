#!/usr/bin/env python3
"""
Run the RGC_L hyperparameter sweep used for Figure 1 of the IJCAI 2026 paper.

This script loops over the paper grid L x d and invokes
Experiments/run_rgc_experiments.py for each cell, writing one
paper_results.json per (model, dataset, seed, L, d) configuration. The output
JSONs are consumed by Experiments/generate_hyperparameter_table.py.

"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_paper_experiments import PAPER_CONFIGS
from utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_LAYERS = "2,4,6,8,10,12"
DEFAULT_DIMS = "64,128,256,512,1024"
DEFAULT_MODELS = "resnet18,resnet50,resnet101,resnet152,densenet121,dinov2_large"
DEFAULT_DATASETS = "cifar10,cifar100,tiny_imagenet"
DEFAULT_SEEDS = "11,12,13"
TRAINING_METHOD = "baseline"


def parse_int_list(raw: str, name: str) -> List[int]:
    """Parse a comma-separated integer list while preserving user order."""
    values: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            values.append(int(piece))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be a comma-separated integer list; got {raw!r}"
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return values


def parse_name_list(raw: str, name: str) -> List[str]:
    """Parse a comma-separated list of non-empty names."""
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return values


def format_duration(seconds: float) -> str:
    """Format a wall-clock duration for summary output."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}m {remainder}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def output_file_for_cell(
    output_dir: Path,
    model: str,
    dataset: str,
    seed: int,
    layer: int,
    dim: int,
) -> Path:
    """Return the per-cell paper_results.json path."""
    return (
        output_dir
        / model
        / dataset
        / TRAINING_METHOD
        / str(seed)
        / f"L{layer}_d{dim}"
        / "paper_results.json"
    )


def has_populated_rgcl(output_file: Path) -> bool:
    """Return True when an existing result file already contains RGCL output."""
    if not output_file.exists():
        return False
    try:
        with output_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read existing result %s: %s", output_file, exc)
        return False

    if not isinstance(data, Mapping):
        return False
    rgcl = data.get("rgcl")
    if not isinstance(rgcl, Mapping) or not rgcl or rgcl.get("skipped", False):
        return False
    return rgcl.get("ece") is not None


def build_cells(
    paper_configs: Sequence[Tuple[str, str]],
    models: Iterable[str],
    datasets: Iterable[str],
    seeds: Sequence[int],
    layers: Sequence[int],
    dims: Sequence[int],
) -> List[Tuple[str, str, int, int, int]]:
    """Build the deterministic sweep cell list."""
    requested_models = set(models)
    requested_datasets = set(datasets)
    filtered_configs = [
        (model, dataset)
        for model, dataset in paper_configs
        if model in requested_models and dataset in requested_datasets
    ]

    cells: List[Tuple[str, str, int, int, int]] = []
    for model, dataset in filtered_configs:
        for seed in sorted(seeds):
            for dim in sorted(dims):
                for layer in sorted(layers):
                    cells.append((model, dataset, seed, layer, dim))
    return cells


def build_command(
    *,
    model: str,
    dataset: str,
    seed: int,
    layer: int,
    dim: int,
    results_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    force: bool,
) -> List[str]:
    """Build one run_rgc_experiments.py invocation."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "Experiments" / "run_rgc_experiments.py"),
        "--model-name",
        model,
        "--dataset",
        dataset,
        "--training-method",
        TRAINING_METHOD,
        "--seed",
        str(seed),
        "--num-layers",
        str(layer),
        "--target-dimension",
        str(dim),
        "--skip-tost-methods",
        "--skip-standard-baselines",
        "--results-base-dir",
        str(results_dir),
        "--output-dir",
        str(output_dir),
        "--output-subdir",
        f"L{layer}_d{dim}",
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    if force:
        cmd.append("--force-recompute")
    return cmd


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the Figure 1 RGC_L hyperparameter sweep."
    )
    parser.add_argument(
        "--layers",
        type=lambda raw: parse_int_list(raw, "layers"),
        default=parse_int_list(DEFAULT_LAYERS, "layers"),
        help=f"Comma-separated L values to run (default: {DEFAULT_LAYERS}).",
    )
    parser.add_argument(
        "--dims",
        type=lambda raw: parse_int_list(raw, "dims"),
        default=parse_int_list(DEFAULT_DIMS, "dims"),
        help=f"Comma-separated d values to run (default: {DEFAULT_DIMS}).",
    )
    parser.add_argument(
        "--models",
        type=lambda raw: parse_name_list(raw, "models"),
        default=parse_name_list(DEFAULT_MODELS, "models"),
        help=f"Comma-separated model filter (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--datasets",
        type=lambda raw: parse_name_list(raw, "datasets"),
        default=parse_name_list(DEFAULT_DATASETS, "datasets"),
        help=f"Comma-separated dataset filter (default: {DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--seeds",
        type=lambda raw: parse_int_list(raw, "seeds"),
        default=parse_int_list(DEFAULT_SEEDS, "seeds"),
        help=f"Comma-separated seed values to run (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/models"),
        help="Directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_results"),
        help="Directory for calibration JSON outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip cells whose per-cell paper_results.json already has RGCL output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override --skip-existing and re-run every cell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cells without launching experiments.",
    )
    return parser


def print_cell(prefix: str, index: int, total: int, cell: Tuple[str, str, int, int, int]) -> None:
    """Print one planned or executed sweep cell."""
    model, dataset, seed, layer, dim = cell
    print(
        f"[{index}/{total}] {prefix} "
        f"model={model} dataset={dataset} seed={seed} L={layer} d={dim}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep."""
    setup_logging(console_level=logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)

    cells = build_cells(
        paper_configs=PAPER_CONFIGS,
        models=args.models,
        datasets=args.datasets,
        seeds=args.seeds,
        layers=args.layers,
        dims=args.dims,
    )
    total = len(cells)

    print(f"Planned hyperparameter sweep cells: {total}")
    if total == 0:
        logger.warning("No cells match the requested model/dataset filters.")
        return 1

    if args.dry_run:
        for index, cell in enumerate(cells, start=1):
            print_cell("PLAN", index, total, cell)
        return 0

    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for index, cell in enumerate(cells, start=1):
        model, dataset, seed, layer, dim = cell
        output_file = output_file_for_cell(
            args.output_dir, model, dataset, seed, layer, dim
        )

        if args.skip_existing and not args.force and has_populated_rgcl(output_file):
            skipped += 1
            print_cell("SKIP", index, total, cell)
            continue

        print_cell("RUN", index, total, cell)
        cmd = build_command(
            model=model,
            dataset=dataset,
            seed=seed,
            layer=layer,
            dim=dim,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            force=args.force,
        )
        try:
            result = subprocess.run(cmd, check=False)
        except Exception as exc:
            failed += 1
            logger.exception(
                "Failed to launch cell model=%s dataset=%s seed=%s L=%s d=%s: %s",
                model,
                dataset,
                seed,
                layer,
                dim,
                exc,
            )
            continue

        if result.returncode == 0:
            completed += 1
        else:
            failed += 1
            logger.error(
                "Cell failed with return code %s: model=%s dataset=%s seed=%s L=%s d=%s",
                result.returncode,
                model,
                dataset,
                seed,
                layer,
                dim,
            )

    elapsed = time.time() - start_time
    print("\nHyperparameter sweep summary")
    print(f"Completed cells: {completed}")
    print(f"Skipped cells:   {skipped}")
    print(f"Failed cells:    {failed}")
    print(f"Total wall-clock: {format_duration(elapsed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
