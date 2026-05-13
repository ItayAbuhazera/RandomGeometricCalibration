#!/usr/bin/env python3
"""
Generate opt-in Beyond-ECE artifacts.

This script aggregates ``paper_results.json`` files produced by
``run_rgc_experiments.py --emit-beyond-ece`` and emits:
    beyond_ece_results.csv
    beyond_ece_table.tex
    beyond_ece_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

PAPER_CONFIG_COUNT = 14


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display: str
    section: str
    rank_in_paper: bool = True


METHOD_SPECS = [
    MethodSpec("uncalibrated", "Uncal", "standard_baselines.uncalibrated"),
    MethodSpec("temperature_scaling", "TS", "standard_baselines.temperature_scaling"),
    MethodSpec("platt_scaling", "Platt", "standard_baselines.platt_scaling"),
    MethodSpec("isotonic_toplabel", "Isotonic", "standard_baselines.isotonic_toplabel"),
    MethodSpec("beta_calibration", "Beta", "standard_baselines.beta_calibration"),
    MethodSpec("density_aware_calibration", "DAC", "standard_baselines.density_aware_calibration"),
    MethodSpec("geometric_physical_space", "Fast Separation", "standard_baselines.geometric_physical_space"),
    MethodSpec("gc_tulip", "GC(TULIP)", "gc_tulip"),
    MethodSpec("deep_ensemble", "D.Ens", "deep_ensemble"),
    MethodSpec("gc_dac", "GC(DAC)", "gc_dac", rank_in_paper=False),
    MethodSpec("rgcl", "RGC$_L$", "rgcl"),
    MethodSpec("rgcc", "RGC$_C$", "rgcc"),
]

METHOD_ORDER = {spec.key: idx for idx, spec in enumerate(METHOD_SPECS)}
RANKING_METHOD_KEYS = [spec.key for spec in METHOD_SPECS if spec.rank_in_paper]


def normalize_training_method(training_method: Any) -> str:
    """Return the canonical training-method name used for aggregation."""
    if training_method is None:
        return "unknown"
    training_method = str(training_method)
    if training_method == "baseline":
        return "baseline_cross_entropy"
    return training_method


def is_clean_result(result: Dict[str, Any]) -> bool:
    """Check if result is from clean non-corrupted data."""
    exp_info = result.get("experiment_info", result.get("experiment_config", {}))
    return exp_info.get("corruption_type") is None and exp_info.get("corruption_severity") is None


def as_float(value: Any) -> Optional[float]:
    """Convert finite numeric values to float, otherwise return None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def get_nested_section(result: Dict[str, Any], dotted_path: str) -> Optional[Dict[str, Any]]:
    """Return a nested method section, supporting the old baselines alias."""
    parts = dotted_path.split(".")
    current: Any = result

    if parts and parts[0] == "standard_baselines" and "standard_baselines" not in result:
        current = result.get("baselines", {})
        parts = parts[1:]

    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)

    if not isinstance(current, dict) or current.get("skipped", False):
        return None
    return current


def discover_result_files(input_dir: Path) -> List[Path]:
    """Find nested paper_results.json files."""
    return sorted(input_dir.rglob("paper_results.json"))


def load_results(result_files: List[Path]) -> List[Dict[str, Any]]:
    """Load result JSON files, skipping parse failures."""
    results = []
    for result_file in result_files:
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["_source_file"] = str(result_file)
            results.append(result)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", result_file, exc)
    return results


def extract_rows(results: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract per-seed Beyond-ECE rows and audit information."""
    rows: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {
        "files_scanned": [],
        "skipped_files": [],
        "fields_found": {},
        "missing_mce": [],
        "field_counts": {"adaptive_ece": 0, "mce": 0},
    }

    clean_results = [result for result in results if is_clean_result(result)]
    if not clean_results:
        clean_results = results

    for result in clean_results:
        source_file = result.get("_source_file", "<unknown>")
        audit["files_scanned"].append(source_file)
        audit["fields_found"][source_file] = {}

        exp_info = result.get("experiment_info", result.get("experiment_config", {}))
        training_method = normalize_training_method(exp_info.get("training_method", "unknown"))
        dataset = exp_info.get("dataset", "unknown")
        model_name = exp_info.get("model_name", "unknown")
        seed = exp_info.get("seed")

        output_subdir = exp_info.get("output_subdir")
        if output_subdir:
            audit["skipped_files"].append({
                "file": source_file,
                "reason": f"non-empty output_subdir ({output_subdir})",
            })
            continue

        for spec in METHOD_SPECS:
            section = get_nested_section(result, spec.section)
            if section is None:
                continue

            fields = sorted(k for k in ("ece", "adaptive_ece", "mce") if k in section)
            audit["fields_found"][source_file][spec.key] = fields

            adaptive_ece = as_float(section.get("adaptive_ece"))
            mce = as_float(section.get("mce"))

            if adaptive_ece is not None:
                audit["field_counts"]["adaptive_ece"] += 1
            if mce is not None:
                audit["field_counts"]["mce"] += 1
            elif as_float(section.get("ece")) is not None:
                audit["missing_mce"].append({
                    "file": source_file,
                    "dataset": dataset,
                    "model_name": model_name,
                    "seed": seed,
                    "method": spec.display,
                })

            if adaptive_ece is None and mce is None:
                continue

            rows.append({
                "training_method": training_method,
                "dataset": dataset,
                "model_name": model_name,
                "seed": seed,
                "method_key": spec.key,
                "method": spec.display,
                "adaptive_ece": adaptive_ece,
                "mce": mce,
            })

    return rows, audit


def ci95(values: pd.Series) -> float:
    """Compute the normal-approximation 95% CI half-width."""
    values = values.dropna()
    if len(values) <= 1:
        return np.nan
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-seed rows to one row per dataset, model, and method."""
    columns = [
        "dataset",
        "model_name",
        "method_key",
        "method",
        "adaptive_ece_mean",
        "adaptive_ece_ci95",
        "adaptive_ece_count",
        "mce_mean",
        "mce_ci95",
        "mce_count",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    baseline_df = df[df["training_method"] == "baseline_cross_entropy"].copy()
    if not baseline_df.empty:
        df = baseline_df

    aggregated_rows = []
    group_cols = ["dataset", "model_name", "method_key", "method"]
    for (dataset, model_name, method_key, method), group in df.groupby(group_cols, dropna=False):
        adaptive = group["adaptive_ece"].dropna()
        mce = group["mce"].dropna()
        aggregated_rows.append({
            "dataset": dataset,
            "model_name": model_name,
            "method_key": method_key,
            "method": method,
            "adaptive_ece_mean": float(adaptive.mean()) if len(adaptive) else np.nan,
            "adaptive_ece_ci95": ci95(adaptive),
            "adaptive_ece_count": int(len(adaptive)),
            "mce_mean": float(mce.mean()) if len(mce) else np.nan,
            "mce_ci95": ci95(mce),
            "mce_count": int(len(mce)),
        })

    aggregated = pd.DataFrame(aggregated_rows)
    aggregated["method_order"] = aggregated["method_key"].map(METHOD_ORDER).fillna(999)
    aggregated = aggregated.sort_values(["dataset", "model_name", "method_order"]).drop(columns=["method_order"])
    return aggregated


def rank_maps_for_config(config_rows: pd.DataFrame, metric_prefix: str) -> Dict[str, float]:
    """Return method ranks for one dataset/model config."""
    metric_col = f"{metric_prefix}_mean"
    valid = config_rows[["method_key", metric_col]].dropna()
    if valid.empty:
        return {}
    ranks = valid.set_index("method_key")[metric_col].rank(method="average", ascending=True)
    return {method_key: float(rank) for method_key, rank in ranks.items()}


def format_metric(mean_val: Any, ci_val: Any, count_val: Any, rank: Optional[float]) -> str:
    """Format a calibration metric as percent with optional rank coloring."""
    if pd.isna(mean_val):
        return "---"

    if pd.isna(ci_val) or count_val is None or int(count_val) <= 1:
        formatted = f"{mean_val * 100:.2f}"
    else:
        formatted = f"{mean_val * 100:.2f} $\\pm$ {ci_val * 100:.2f}"

    if rank == 1:
        return f"\\textcolor{{blue}}{{\\textbf{{{formatted}}}}}"
    if rank == 2:
        return f"\\textcolor{{teal}}{{\\textbf{{{formatted}}}}}"
    if rank == 3:
        return f"\\textcolor{{olive}}{{\\textbf{{{formatted}}}}}"
    return formatted


def generate_latex_table(aggregated: pd.DataFrame) -> str:
    """Generate a LaTeX table in the same booktabs/color style as Table 1."""
    if aggregated.empty:
        return "% No Beyond-ECE results found."

    lines = []
    lines.append("% Beyond-ECE results")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lllcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & Method & Adaptive ECE & MCE \\\\")
    lines.append("\\midrule")

    aggregated = aggregated.copy()
    aggregated["method_order"] = aggregated["method_key"].map(METHOD_ORDER).fillna(999)
    aggregated = aggregated.sort_values(["dataset", "model_name", "method_order"])
    current_config = None
    rank_cache: Dict[tuple[str, str], Dict[str, Dict[str, float]]] = {}

    for _, row in aggregated.iterrows():
        config_key = (row["dataset"], row["model_name"])
        if config_key not in rank_cache:
            config_rows = aggregated[
                (aggregated["dataset"] == config_key[0])
                & (aggregated["model_name"] == config_key[1])
            ]
            rank_cache[config_key] = {
                "adaptive_ece": rank_maps_for_config(config_rows, "adaptive_ece"),
                "mce": rank_maps_for_config(config_rows, "mce"),
            }

        if current_config is not None and current_config != config_key:
            lines.append("\\midrule")
        show_config = current_config != config_key
        current_config = config_key

        method_key = row["method_key"]
        adaptive_rank = rank_cache[config_key]["adaptive_ece"].get(method_key)
        mce_rank = rank_cache[config_key]["mce"].get(method_key)

        cells = [
            str(row["dataset"]).upper() if show_config else "",
            str(row["model_name"]) if show_config else "",
            row["method"],
            format_metric(row["adaptive_ece_mean"], row["adaptive_ece_ci95"], row["adaptive_ece_count"], adaptive_rank),
            format_metric(row["mce_mean"], row["mce_ci95"], row["mce_count"], mce_rank),
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Beyond-ECE calibration metrics (\\%; lower is better). Values are mean $\\pm$ 95\\% CI. "
                 "\\textcolor{blue}{\\textbf{Best}}, \\textcolor{teal}{\\textbf{2nd}}, \\textcolor{olive}{\\textbf{3rd}}.}")
    lines.append("\\label{tab:beyond_ece}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def compute_rank_statistics(aggregated: pd.DataFrame, metric_prefix: str) -> List[Dict[str, Any]]:
    """Compute average rank and top-three counts for RGC variants."""
    if aggregated.empty:
        return []

    stats_rows = []
    metric_col = f"{metric_prefix}_mean"

    for target_key, excluded_key in [("rgcl", "rgcc"), ("rgcc", "rgcl")]:
        ranks = []
        top_three = 0
        observed_configs = 0

        for (_, _), group in aggregated.groupby(["dataset", "model_name"]):
            allowed = set(RANKING_METHOD_KEYS)
            allowed.discard(excluded_key)
            rank_group = group[group["method_key"].isin(allowed)]
            valid = rank_group[["method_key", metric_col]].dropna()
            if target_key not in set(valid["method_key"]):
                continue

            observed_configs += 1
            method_ranks = valid.set_index("method_key")[metric_col].rank(method="average", ascending=True)
            rank = float(method_ranks.loc[target_key])
            ranks.append(rank)
            if rank <= 3:
                top_three += 1

        display = next(spec.display for spec in METHOD_SPECS if spec.key == target_key)
        stats_rows.append({
            "metric": metric_prefix,
            "method": display,
            "average_rank": float(np.mean(ranks)) if ranks else np.nan,
            "top_three_count": int(top_three),
            "observed_config_count": int(observed_configs),
            "paper_config_count": PAPER_CONFIG_COUNT,
        })

    return stats_rows


def print_rank_statistics(aggregated: pd.DataFrame) -> None:
    """Print Beyond-ECE ranking statistics to stdout."""
    print("\nBeyond-ECE ranking statistics")
    print("=" * 80)
    print("Ranks exclude the other RGC variant for RGC_L and RGC_C.")

    for metric_prefix, label in [("adaptive_ece", "Adaptive ECE"), ("mce", "MCE")]:
        rows = compute_rank_statistics(aggregated, metric_prefix)
        print(f"\n{label}:")
        for row in rows:
            avg_rank = row["average_rank"]
            avg_rank_text = "N/A" if pd.isna(avg_rank) else f"{avg_rank:.2f}"
            observed = row["observed_config_count"]
            print(
                f"  {row['method']}: average rank {avg_rank_text}; "
                f"top-three {row['top_three_count']} out of {PAPER_CONFIG_COUNT} paper configurations "
                f"({observed} observed with this metric)"
            )


def write_audit(audit_path: Path, audit: Dict[str, Any], result_files: List[Path], aggregated: pd.DataFrame) -> None:
    """Write the audit log."""
    audit = {
        **audit,
        "result_files_discovered": [str(path) for path in result_files],
        "num_result_files_discovered": len(result_files),
        "num_rows_written": int(len(aggregated)),
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Beyond-ECE CSV, LaTeX table, and audit log."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("calibration_results"),
        help="Directory containing nested paper_results.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("latex_tables_statistics/beyond_ece"),
        help="Directory for Beyond-ECE artifacts.",
    )
    args = parser.parse_args()

    result_files = discover_result_files(args.input_dir)
    logger.info("Discovered %d paper_results.json files under %s", len(result_files), args.input_dir)

    results = load_results(result_files)
    rows, audit = extract_rows(results)
    aggregated = aggregate_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "beyond_ece_results.csv"
    latex_path = args.output_dir / "beyond_ece_table.tex"
    audit_path = args.output_dir / "beyond_ece_audit.json"

    aggregated.to_csv(csv_path, index=False)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_table(aggregated))
    write_audit(audit_path, audit, result_files, aggregated)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote LaTeX table: {latex_path}")
    print(f"Wrote audit log: {audit_path}")

    if audit["field_counts"]["mce"] == 0:
        print(
            "\nNo MCE fields were found. Re-run with: "
            "python run_paper_experiments.py --full --emit-beyond-ece"
        )

    print_rank_statistics(aggregated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
