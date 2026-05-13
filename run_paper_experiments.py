#!/usr/bin/env python3
"""
Run Random Geometric Calibration (RGC) experiment sweeps.

This script provides a clean entry point for running the available experiment
sweep with sensible defaults and comprehensive error handling.

Usage:
    python run_paper_experiments.py --quick         # Quick smoke test
    python run_paper_experiments.py --cifar-full    # CIFAR-only convenience sweep
    python run_paper_experiments.py --full          # Full paper Table 1 sweep
    python run_paper_experiments.py --model resnet50 --dataset cifar100

Examples:
    # Quick test with single configuration
    python run_paper_experiments.py --quick

    # CIFAR-only convenience sweep (11 model-dataset pairs x 10 seeds = 110 runs)
    python run_paper_experiments.py --cifar-full

    # Full Table 1 sweep (14 model-dataset pairs x 10 seeds = 140 runs)
    python run_paper_experiments.py --full

    # Specific configuration
    python run_paper_experiments.py --model resnet50 --dataset cifar100 --seed 12

    # Resume interrupted full sweep (skips completed experiments)
    python run_paper_experiments.py --full

    # Force re-run even if results exist
    python run_paper_experiments.py --full --force

    # Skip training, only run calibration (assumes models exist)
    python run_paper_experiments.py --full --skip-training
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PAPER CONFIGURATIONS
# =============================================================================

MODELS = ["resnet18", "resnet50", "resnet101", "resnet152", "densenet121", "dinov2_large"]
DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
SEEDS = list(range(11, 21))
TINY_IMAGENET_DIR = PROJECT_ROOT / "data" / "tiny-imagenet-200"
TINY_IMAGENET_REQUIRED_PATHS = ("wnids.txt", "train", "val", "val/val_annotations.txt")

TABLE1_CONFIGS = [
    # CIFAR-10
    ("densenet121", "cifar10"),
    ("resnet18", "cifar10"),
    ("resnet50", "cifar10"),
    ("resnet101", "cifar10"),
    ("resnet152", "cifar10"),
    # CIFAR-100
    ("densenet121", "cifar100"),
    ("dinov2_large", "cifar100"),
    ("resnet18", "cifar100"),
    ("resnet50", "cifar100"),
    ("resnet101", "cifar100"),
    ("resnet152", "cifar100"),
    # Tiny-ImageNet
    ("resnet50", "tiny_imagenet"),
    ("resnet101", "tiny_imagenet"),
    ("resnet152", "tiny_imagenet"),
]

CIFAR_CONFIGS = [
    (model, dataset)
    for model, dataset in TABLE1_CONFIGS
    if dataset in {"cifar10", "cifar100"}
]

TOST_CONFIGS = [
    (model, dataset)
    for model, dataset in TABLE1_CONFIGS
    if model != "dinov2_large"
]

# Backward-compatible import name used by existing helper scripts.
PAPER_CONFIGS = TABLE1_CONFIGS

# Default hyperparameters (from paper Section 4)
DEFAULT_NUM_LAYERS = 6       # L in paper
DEFAULT_TARGET_DIM = 256     # d for RGCL, K for RGCC
DEFAULT_K_NEIGHBORS = 1      # k-NN parameter
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAINING_METHOD = "baseline"

# Number of classes per dataset
DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}

# Dataset-specific target accuracies (for reference)
DATASET_TARGET_ACCURACY = {
    "cifar10": 96.0,
    "cifar100": 80.0,
    "tiny_imagenet": 65.0,
}


# =============================================================================
# RESULT TRACKING
# =============================================================================

@dataclass
class ExperimentResult:
    """Store results from a single experiment run."""
    model: str
    dataset: str
    seed: int
    status: str  # "OK", "FAILED", "SKIPPED"
    rgcl_ece: Optional[float] = None
    rgcc_ece: Optional[float] = None
    uncal_ece: Optional[float] = None
    ts_ece: Optional[float] = None
    accuracy: Optional[float] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "seed": self.seed,
            "status": self.status,
            "rgcl_ece": self.rgcl_ece,
            "rgcc_ece": self.rgcc_ece,
            "uncal_ece": self.uncal_ece,
            "ts_ece": self.ts_ece,
            "accuracy": self.accuracy,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return f"cuda ({device_name})"
        return "cpu"
    except ImportError:
        return "cpu"
    except Exception as e:
        print(f"Warning: Error detecting device: {e}")
        return "cpu"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def format_ece(ece: Optional[float]) -> str:
    """Format ECE value as percentage string."""
    if ece is None:
        return "---"
    return f"{ece * 100:.2f}%"


def construct_model_path(
    results_dir: str,
    training_method: str,
    dataset: str,
    model_name: str,
    seed: int,
) -> Path:
    """Construct the expected path to a trained model checkpoint."""
    # training_method is kept in the signature for compatibility with callers.
    # Match the structure from train_model.py
    return (
        Path(results_dir)
        / dataset
        / model_name
        / f"seed{seed}"
        / "best_model.pth"
    )


def check_model_exists(
    results_dir: str, model: str, dataset: str, seed: int
) -> Tuple[bool, Path]:
    """Check if a trained model exists at the expected path."""
    model_path = construct_model_path(
        results_dir, DEFAULT_TRAINING_METHOD, dataset, model, seed
    )
    return model_path.exists(), model_path


def construct_output_path(
    output_dir: str, model: str, dataset: str, seed: int
) -> Path:
    """Construct the output path for calibration results."""
    return Path(output_dir) / model / dataset / DEFAULT_TRAINING_METHOD / str(seed)


def check_results_exist(
    output_dir: str, model: str, dataset: str, seed: int
) -> Tuple[bool, Path]:
    """Check if calibration results already exist."""
    # Check all possible output path conventions
    candidates = [
        Path(output_dir) / model / dataset / DEFAULT_TRAINING_METHOD / str(seed) / "paper_results.json",
        Path(output_dir) / model / dataset / f"seed{seed}" / "paper_results.json",
        Path(output_dir) / model / dataset / str(seed) / "paper_results.json",
    ]
    for results_file in candidates:
        if results_file.exists():
            return True, results_file
    return False, candidates[0]


def load_existing_results(results_file: Path) -> Optional[Dict[str, Any]]:
    """Load existing results from JSON file."""
    if not results_file.exists():
        return None
    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics_from_results(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract key metrics from calibration results."""
    metrics = {
        "rgcl_ece": None,
        "rgcc_ece": None,
        "uncal_ece": None,
        "ts_ece": None,
        "accuracy": None,
    }

    # Extract RGCL ECE
    if "rgcl" in results and isinstance(results["rgcl"], dict):
        metrics["rgcl_ece"] = results["rgcl"].get("ece")

    # Extract RGCC ECE
    if "rgcc" in results and isinstance(results["rgcc"], dict):
        metrics["rgcc_ece"] = results["rgcc"].get("ece")

    # Extract baselines
    baselines = results.get("standard_baselines", {})
    if "uncalibrated" in baselines:
        metrics["uncal_ece"] = baselines["uncalibrated"].get("ece")
        metrics["accuracy"] = baselines["uncalibrated"].get("acc")
    if "temperature_scaling" in baselines:
        metrics["ts_ece"] = baselines["temperature_scaling"].get("ece")

    return metrics


def is_cuda_oom_error(error_message: str) -> bool:
    """Check if error message indicates CUDA out of memory."""
    oom_indicators = [
        "CUDA out of memory",
        "cuda out of memory",
        "OutOfMemoryError",
        "RuntimeError: CUDA",
        "cudaErrorMemoryAllocation",
    ]
    return any(indicator in error_message for indicator in oom_indicators)


def validate_required_datasets(configs: List[Tuple[str, str, int]]) -> None:
    """Fail early with an actionable message if a manual dataset is missing."""
    needs_tiny_imagenet = any(dataset == "tiny_imagenet" for _, dataset, _ in configs)
    if not needs_tiny_imagenet:
        return

    missing_paths = [
        relative_path
        for relative_path in TINY_IMAGENET_REQUIRED_PATHS
        if not (TINY_IMAGENET_DIR / relative_path).exists()
    ]
    if not missing_paths:
        return

    expected = "data/tiny-imagenet-200/"
    missing = ", ".join(missing_paths)
    raise FileNotFoundError(
        "Tiny-ImageNet is required for this run but is not available. "
        f"Expected an extracted dataset at {expected} with train/, val/, wnids.txt, and val/val_annotations.txt. "
        f"Missing under {expected}: {missing}. "
        "CIFAR-10/100 are downloaded automatically by torchvision; Tiny-ImageNet must be downloaded manually. "
        "For a CIFAR-only run, use --cifar-full."
    )


# =============================================================================
# TRAINING
# =============================================================================

def run_training(
    model: str,
    dataset: str,
    seed: int,
    output_dir: str,
    batch_size: int = 64,
    epochs: int = 200,
    verbose: bool = True,
) -> Tuple[bool, str]:
    """
    Train a single model.

    Returns:
        Tuple of (success, error_message)
    """
    train_script = PROJECT_ROOT / "Experiments" / "train_model.py"
    
    if not train_script.exists():
        return False, f"Training script not found: {train_script}"

    cmd = [
        sys.executable,
        str(train_script),
        "--model-name", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--output-root", output_dir,
    ]

    try:
        if verbose:
            # Run with output visible
            result = subprocess.run(cmd, check=True)
        else:
            # Capture output
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = ""
        if hasattr(e, 'stderr') and e.stderr:
            error_msg = e.stderr[:500]
        elif hasattr(e, 'stdout') and e.stdout:
            error_msg = e.stdout[:500]
        else:
            error_msg = str(e)
        return False, error_msg
    except Exception as e:
        return False, str(e)


# =============================================================================
# CALIBRATION
# =============================================================================

def run_calibration(
    model: str,
    dataset: str,
    seed: int,
    results_dir: str,
    output_dir: str,
    batch_size: int = 128,
    target_dim: int = DEFAULT_TARGET_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    device: str = "cuda",
    eval_ensemble: bool = False,
    ensemble_seeds: Optional[List[int]] = None,
    skip_tost_methods: bool = False,
    emit_beyond_ece: bool = False,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run calibration experiments for a single configuration.

    Returns:
        Tuple of (success, results_dict)
    """
    calib_script = PROJECT_ROOT / "Experiments" / "run_rgc_experiments.py"
    
    if not calib_script.exists():
        return False, {"error": f"Calibration script not found: {calib_script}"}

    # Extract just the device type (cuda or cpu)
    device_type = device.split()[0] if " " in device else device

    cmd = [
        sys.executable,
        str(calib_script),
        "--model-name", model,
        "--dataset", dataset,
        "--training-method", DEFAULT_TRAINING_METHOD,
        "--seed", str(seed),
        "--target-dimension", str(target_dim),
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--results-base-dir", results_dir,
        "--output-dir", output_dir,
        "--device", device_type,
    ]

    if eval_ensemble and ensemble_seeds:
        cmd.extend([
            "--eval-ensemble",
            "--ensemble-seeds", " ".join(map(str, ensemble_seeds))
        ])

    if skip_tost_methods:
        cmd.append("--skip-tost-methods")

    if emit_beyond_ece:
        cmd.append("--emit-beyond-ece")

    try:
        if verbose:
            result = subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

        # Try to find and load the generated results
        possible_paths = [
            Path(output_dir) / model / dataset / f"seed{seed}" / "paper_results.json",
            Path(output_dir) / model / dataset / DEFAULT_TRAINING_METHOD / str(seed) / "paper_results.json",
            Path(output_dir) / model / dataset / str(seed) / "paper_results.json",
        ]

        for results_file in possible_paths:
            if results_file.exists():
                with open(results_file, "r") as f:
                    return True, json.load(f)

        # No results file found but command succeeded
        return True, {"warning": "Command succeeded but no results file found"}

    except subprocess.CalledProcessError as e:
        error_msg = ""
        if hasattr(e, 'stderr') and e.stderr:
            error_msg = e.stderr[:1000]
        elif hasattr(e, 'stdout') and e.stdout:
            error_msg = e.stdout[:1000]
        else:
            error_msg = str(e)
        return False, {"error": error_msg}
    except Exception as e:
        return False, {"error": str(e)}


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(results: List[ExperimentResult], output_dir: str):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    header = f"{'Model':<15} {'Dataset':<15} {'RGCL ECE':<12} {'RGCC ECE':<12} {'Status':<10}"
    print(header)
    print("-" * 80)

    # Group results by model and dataset
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r.model, r.dataset)].append(r)

    # Calculate averages and print
    all_rgcl = []
    all_rgcc = []

    for (model, dataset), exp_results in sorted(grouped.items()):
        rgcl_vals = [r.rgcl_ece for r in exp_results if r.rgcl_ece is not None]
        rgcc_vals = [r.rgcc_ece for r in exp_results if r.rgcc_ece is not None]

        if rgcl_vals:
            all_rgcl.extend(rgcl_vals)
        if rgcc_vals:
            all_rgcc.extend(rgcc_vals)

        # Calculate mean for this config
        rgcl_mean = sum(rgcl_vals) / len(rgcl_vals) if rgcl_vals else None
        rgcc_mean = sum(rgcc_vals) / len(rgcc_vals) if rgcc_vals else None

        # Count statuses
        ok_count = sum(1 for r in exp_results if r.status == "OK")
        total = len(exp_results)
        status = f"{ok_count}/{total} OK"

        row = f"{model:<15} {dataset:<15} {format_ece(rgcl_mean):<12} {format_ece(rgcc_mean):<12} {status:<10}"
        print(row)

    print("-" * 80)

    # Overall statistics
    total_ok = sum(1 for r in results if r.status == "OK")
    total_failed = sum(1 for r in results if r.status == "FAILED")
    total_skipped = sum(1 for r in results if r.status == "SKIPPED")

    if all_rgcl:
        avg_rgcl = sum(all_rgcl) / len(all_rgcl)
        print(f"Average RGCL ECE: {format_ece(avg_rgcl)}")
    if all_rgcc:
        avg_rgcc = sum(all_rgcc) / len(all_rgcc)
        print(f"Average RGCC ECE: {format_ece(avg_rgcc)}")

    print(f"\nExperiments: {total_ok} OK, {total_failed} Failed, {total_skipped} Skipped")
    print(f"Results saved to: {output_dir}/")
    print("=" * 80)


def save_summary_csv(results: List[ExperimentResult], output_dir: str):
    """Save summary results to CSV file."""
    csv_path = Path(output_dir) / "summary_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "dataset", "seed", "status",
        "rgcl_ece", "rgcc_ece", "uncal_ece", "ts_ece",
        "accuracy", "duration_seconds", "error_message"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            # Format ECE values as percentages for CSV
            for key in ["rgcl_ece", "rgcc_ece", "uncal_ece", "ts_ece"]:
                if row[key] is not None:
                    row[key] = f"{row[key] * 100:.4f}"
            writer.writerow(row)

    print(f"Summary CSV saved to: {csv_path}")


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(
    model: str,
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    experiment_num: int,
    total_experiments: int,
) -> ExperimentResult:
    """Run a single experiment configuration."""
    start_time = time.time()
    device_str = args.device if args.device else get_device()

    print(f"\n[{experiment_num}/{total_experiments}] {model} / {dataset} / seed{seed}")

    # Check if results already exist (resume capability)
    results_exist, results_file = check_results_exist(
        args.output_dir, model, dataset, seed
    )
    
    if results_exist and not args.force:
        print("      Results: EXISTS (use --force to re-run)")
        existing_results = load_existing_results(results_file)
        if existing_results:
            metrics = extract_metrics_from_results(existing_results)
            duration = time.time() - start_time
            print(f"      ECE: {format_ece(metrics['rgcl_ece'])} (RGCL), {format_ece(metrics['rgcc_ece'])} (RGCC)")
            return ExperimentResult(
                model=model,
                dataset=dataset,
                seed=seed,
                status="SKIPPED",
                rgcl_ece=metrics["rgcl_ece"],
                rgcc_ece=metrics["rgcc_ece"],
                uncal_ece=metrics["uncal_ece"],
                ts_ece=metrics["ts_ece"],
                accuracy=metrics["accuracy"],
                duration_seconds=duration,
            )
        # Results file exists but couldn't be loaded
        return ExperimentResult(
            model=model,
            dataset=dataset,
            seed=seed,
            status="SKIPPED",
            duration_seconds=time.time() - start_time,
        )

    # Check if model exists
    model_exists, model_path = check_model_exists(
        args.results_dir, model, dataset, seed
    )

    # Training phase
    if not args.skip_training:
        if model_exists:
            print("      Training: SKIPPED (model exists)")
        else:
            print("      Training: Running...")
            success, error_msg = run_training(
                model=model,
                dataset=dataset,
                seed=seed,
                output_dir=args.results_dir,
                batch_size=max(32, args.batch_size // 2),  # Smaller batch for training
                verbose=True,
            )
            if not success:
                duration = time.time() - start_time
                print(f"      Training FAILED: {error_msg[:200]}")
                return ExperimentResult(
                    model=model,
                    dataset=dataset,
                    seed=seed,
                    status="FAILED",
                    duration_seconds=duration,
                    error_message=f"Training failed: {error_msg[:500]}",
                )
            print("      Training: COMPLETE")
    else:
        if not model_exists:
            print("      Training: SKIPPED (--skip-training)")
            print(f"      ERROR: Model not found at {model_path}")
            print("      Run without --skip-training or train the model manually.")
            duration = time.time() - start_time
            return ExperimentResult(
                model=model,
                dataset=dataset,
                seed=seed,
                status="FAILED",
                duration_seconds=duration,
                error_message=f"Model checkpoint not found at: {model_path}",
            )
        else:
            print("      Training: SKIPPED (model exists)")

    # Calibration phase
    print("      Calibration: Running...")
    try:
        success, cal_results = run_calibration(
            model=model,
            dataset=dataset,
            seed=seed,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            target_dim=DEFAULT_TARGET_DIM,
            num_layers=DEFAULT_NUM_LAYERS,
            device=device_str,
            eval_ensemble=getattr(args, 'eval_ensemble', False),
            ensemble_seeds=[int(s) for s in args.ensemble_seeds.split()],
            skip_tost_methods=args.quick,
            emit_beyond_ece=args.emit_beyond_ece,
            verbose=True,
        )

        duration = time.time() - start_time

        if success:
            metrics = extract_metrics_from_results(cal_results)
            print(f"      Results: ECE={format_ece(metrics['rgcl_ece'])} (RGCL), ECE={format_ece(metrics['rgcc_ece'])} (RGCC)")
            print(f"      Time: {format_duration(duration)}")

            return ExperimentResult(
                model=model,
                dataset=dataset,
                seed=seed,
                status="OK",
                rgcl_ece=metrics["rgcl_ece"],
                rgcc_ece=metrics["rgcc_ece"],
                uncal_ece=metrics["uncal_ece"],
                ts_ece=metrics["ts_ece"],
                accuracy=metrics["accuracy"],
                duration_seconds=duration,
            )
        else:
            error_msg = cal_results.get("error", "Unknown error")

            # Check for CUDA OOM
            if is_cuda_oom_error(error_msg):
                print("      ERROR: CUDA out of memory!")
                print(f"      Suggestion: Reduce batch size with --batch-size {args.batch_size // 2}")
            else:
                print(f"      ERROR: {error_msg[:200]}")

            return ExperimentResult(
                model=model,
                dataset=dataset,
                seed=seed,
                status="FAILED",
                duration_seconds=duration,
                error_message=error_msg[:500],
            )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e) + "\n" + traceback.format_exc()

        if is_cuda_oom_error(error_msg):
            print("      ERROR: CUDA out of memory!")
            print(f"      Suggestion: Reduce batch size with --batch-size {args.batch_size // 2}")
        else:
            print(f"      ERROR: {str(e)[:200]}")

        return ExperimentResult(
            model=model,
            dataset=dataset,
            seed=seed,
            status="FAILED",
            duration_seconds=duration,
            error_message=error_msg[:500],
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Random Geometric Calibration (RGC) experiment sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_paper_experiments.py --quick
        Quick smoke test with ResNet-18 on CIFAR-10 / seed 11

    python run_paper_experiments.py --cifar-full
        Run the CIFAR-only convenience sweep (11 model-dataset pairs x 10 seeds = 110 runs)

    python run_paper_experiments.py --full
        Run the full paper Table 1 sweep (14 model-dataset pairs x 10 seeds = 140 runs)

    python run_paper_experiments.py --model resnet50 --dataset cifar100
        Run specific configuration

    python run_paper_experiments.py --full --force
        Re-run all experiments even if results exist

    python run_paper_experiments.py --full --skip-training
        Only run calibration (assumes trained models exist)
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: ResNet-18 on CIFAR-10 with seed 11",
    )
    mode_group.add_argument(
        "--cifar-full",
        action="store_true",
        help="CIFAR-only convenience sweep: 11 model-dataset pairs x 10 seeds (110 configurations)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full paper Table 1 sweep: all 14 model-dataset pairs x 10 seeds (140 configurations), including Tiny-ImageNet",
    )

    # Specific configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        help=f"Specific model to run. Choices: {MODELS}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        help=f"Specific dataset to use. Choices: {DATASETS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed (default: 11)",
    )

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}, reduce if OOM)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )

    # Paths
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/models",
        help="Directory containing trained models (default: results/models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="calibration_results",
        help="Output directory for calibration results (default: calibration_results)",
    )

    # Options
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Assume models exist, only run calibration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even if results exist",
    )
    parser.add_argument(
        "--eval-ensemble",
        action="store_true",
        help="Evaluate deep ensemble baseline (requires multiple seeds)",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default="11 12 13",
        help="Space-separated seeds for deep ensemble evaluation (default: '11 12 13')",
    )
    parser.add_argument(
        "--emit-beyond-ece",
        action="store_true",
        default=False,
        help="Forward --emit-beyond-ece to run_rgc_experiments.py for opt-in MCE reporting.",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_device()

    # Determine configurations to run
    if args.quick:
        configs = [("resnet18", "cifar10", 11)]
        mode_name = "QUICK"
        mode_desc = "Testing ResNet-18 on CIFAR-10 / seed 11"
    elif args.cifar_full:
        configs = [(m, d, s) for (m, d) in CIFAR_CONFIGS for s in SEEDS]
        mode_name = "CIFAR-FULL"
        mode_desc = f"{len(configs)} configurations ({len(CIFAR_CONFIGS)} CIFAR model-dataset pairs x {len(SEEDS)} seeds)"
    elif args.full:
        configs = [(m, d, s) for (m, d) in TABLE1_CONFIGS for s in SEEDS]
        mode_name = "FULL"
        mode_desc = f"{len(configs)} configurations ({len(TABLE1_CONFIGS)} Table 1 model-dataset pairs x {len(SEEDS)} seeds)"
    elif args.model and args.dataset:
        configs = [(args.model, args.dataset, args.seed)]
        mode_name = "CUSTOM"
        mode_desc = f"{args.model} on {args.dataset} (seed={args.seed})"
    else:
        parser.print_help()
        print("\nPlease specify --quick, --cifar-full, --full, or both --model and --dataset")
        sys.exit(1)

    try:
        validate_required_datasets(configs)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    # Print header
    print("\n" + "=" * 60)
    print("RGC Experiment Sweep - IJCAI 2026")
    print("=" * 60)
    print(f"Mode:          {mode_name} ({mode_desc})")
    print(f"Device:        {args.device}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Results dir:   {args.results_dir}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Skip training: {args.skip_training}")
    print(f"Force re-run:  {args.force}")
    print("=" * 60)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run experiments
    results: List[ExperimentResult] = []
    total_start_time = time.time()

    for i, (model, dataset, seed) in enumerate(configs, 1):
        try:
            result = run_single_experiment(
                model=model,
                dataset=dataset,
                seed=seed,
                args=args,
                experiment_num=i,
                total_experiments=len(configs),
            )
            results.append(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            traceback.print_exc()
            results.append(ExperimentResult(
                model=model,
                dataset=dataset,
                seed=seed,
                status="FAILED",
                error_message=str(e),
            ))
            if mode_name in {"QUICK", "CUSTOM"}:
                # In single-run modes, stop on first error.
                break

    # Calculate total time
    total_duration = time.time() - total_start_time

    # Print and save summary
    if results:
        print_summary_table(results, args.output_dir)
        save_summary_csv(results, args.output_dir)

    print(f"\nTotal time: {format_duration(total_duration)}")

    # Exit with appropriate code
    failed_count = sum(1 for r in results if r.status == "FAILED")
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
