#!/usr/bin/env python3
"""
Generate the hyperparameter sensitivity table for Random Geometric Calibration.

The script scans calibration result JSON files, extracts real RGC_L metric
values indexed by number of layers (L) and random feature dimension (d), and
writes three reproducibility artifacts:

* a CSV summary over the requested L x d grid,
* a compact LaTeX table with rows as L and columns as d,
* a JSON audit describing scanned files, detected keys, filters, and summary
  statistics.

No values are synthesized. Missing grid cells stay empty, and if no usable
hyperparameter records are found the script writes an audit JSON and exits
nonzero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


METHOD_KEYS = (
    "rgcl",
    "random_geometric",
    "random_geometric_calibration",
)
SWEEP_KEYS = (
    "results_by_hyperparameter",
    "hyperparameter_sweep",
    "layer_dim_sweep",
)
EXCLUDED_METHOD_TOKENS = (
    "rgcc",
    "gc_dac",
    "gc_tulip",
    "density_aware",
    "temperature",
    "platt",
    "beta",
    "isotonic",
    "uncalibrated",
    "physical_space",
)
LAYER_KEYS = ("num_layers", "n_layers", "layers", "layer", "L", "l")
DIM_KEYS = (
    "target_dim",
    "target_dimension",
    "dim",
    "dimension",
    "projection_dim",
    "random_dimension",
    "d",
)


@dataclass(frozen=True)
class HyperparameterRecord:
    """One observed metric value for a specific (L, d) configuration."""

    layer: int
    dim: int
    value: float
    source_file: str
    key_path: str
    dataset: Optional[str]
    model: Optional[str]
    training_method: Optional[str]
    seed: Optional[int]


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


def normalize_training_method(training_method: Any) -> str:
    """Use the same baseline naming convention as generate_paper_tables.py."""
    if training_method is None:
        return "unknown"
    training_method = str(training_method)
    if training_method == "baseline":
        return "baseline_cross_entropy"
    return training_method


def parse_optional_int(value: Any) -> Optional[int]:
    """Return an integer when value clearly represents one."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    return None


def parse_optional_float(value: Any) -> Optional[float]:
    """Return a finite float, accepting percentage strings as fractions."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            result = float(text)
        except ValueError:
            return None
        if is_percent:
            result /= 100.0
    else:
        return None
    if not math.isfinite(result):
        return None
    return result


def metric_aliases(metric: str) -> Tuple[str, ...]:
    """Return field names that commonly hold the requested metric."""
    metric = metric.lower()
    aliases = {
        metric,
        f"mean_{metric}",
        f"rgcl_{metric}",
        f"random_geometric_{metric}",
        f"geometric_{metric}",
        f"{metric}_geometric",
    }
    if metric == "ece":
        aliases.update(
            {
                "ECE",
                "ece_rgcl",
                "ece_random_geometric",
                "ece_geometric",
            }
        )
    return tuple(aliases)


def get_metric_value(data: Mapping[str, Any], metric: str) -> Optional[float]:
    """Extract a metric value from a mapping using known aliases."""
    for key in metric_aliases(metric):
        if key in data:
            value = parse_optional_float(data[key])
            if value is not None:
                return value
    return None


def get_first_int(data: Mapping[str, Any], keys: Sequence[str]) -> Optional[int]:
    """Extract the first integer value found under any candidate key."""
    for key in keys:
        if key in data:
            value = parse_optional_int(data[key])
            if value is not None:
                return value
    return None


def parse_layer_dim_from_key(key: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse layer/dimension hints from a dictionary key."""
    text = str(key).strip()
    layer: Optional[int] = None
    dim: Optional[int] = None

    layer_match = re.search(
        r"(?:^|[^a-zA-Z])(?:L|layers?|num_layers|n_layers)\s*[:=_-]?\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    dim_match = re.search(
        r"(?:^|[^a-zA-Z])(?:d|dim|target_dim|target_dimension|dimension)\s*[:=_-]?\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if layer_match:
        layer = int(layer_match.group(1))
    if dim_match:
        dim = int(dim_match.group(1))

    if layer is None or dim is None:
        pair_match = re.fullmatch(r"\(?\s*(\d+)\s*[,xX/_-]\s*(\d+)\s*\)?", text)
        if pair_match:
            layer = layer if layer is not None else int(pair_match.group(1))
            dim = dim if dim is not None else int(pair_match.group(2))

    return layer, dim


def experiment_info(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the metadata block used by current and older result JSONs."""
    info = result.get("experiment_info")
    if isinstance(info, Mapping):
        return dict(info)
    config = result.get("experiment_config")
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def source_metadata(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract common source metadata used in audit records."""
    info = experiment_info(result)
    training_method = normalize_training_method(info.get("training_method"))
    return {
        "dataset": info.get("dataset"),
        "model": info.get("model_name", info.get("model")),
        "training_method": training_method,
        "seed": parse_optional_int(info.get("seed")),
    }


def matches_filters(
    metadata: Mapping[str, Any],
    dataset: Optional[str],
    model: Optional[str],
    training_method: Optional[str],
) -> bool:
    """Return True when source metadata passes CLI filters."""
    if dataset and metadata.get("dataset") != dataset:
        return False
    if model and metadata.get("model") != model:
        return False
    if training_method:
        requested = normalize_training_method(training_method)
        if metadata.get("training_method") != requested:
            return False
    return True


def is_clean_result(result: Mapping[str, Any]) -> bool:
    """Exclude corrupted-result variants from the paper sensitivity table."""
    info = experiment_info(result)
    return info.get("corruption_type") is None and info.get("corruption_severity") is None


def should_skip_path(key_path: str) -> bool:
    """Avoid pulling non-RGC_L methods from mixed result dictionaries."""
    lowered = key_path.lower()
    return any(token in lowered for token in EXCLUDED_METHOD_TOKENS)


def add_record_if_valid(
    records: List[HyperparameterRecord],
    *,
    layer: Optional[int],
    dim: Optional[int],
    value: Optional[float],
    source_file: Path,
    key_path: str,
    metadata: Mapping[str, Any],
    layer_grid: Sequence[int],
    dim_grid: Sequence[int],
) -> bool:
    """Append a record only when it is complete and belongs to the requested grid."""
    if value is None or layer is None or dim is None:
        return False
    if layer not in layer_grid or dim not in dim_grid:
        return False
    if should_skip_path(key_path):
        return False
    records.append(
        HyperparameterRecord(
            layer=layer,
            dim=dim,
            value=value,
            source_file=source_file.as_posix(),
            key_path=key_path,
            dataset=metadata.get("dataset"),
            model=metadata.get("model"),
            training_method=metadata.get("training_method"),
            seed=metadata.get("seed"),
        )
    )
    return True


def walk_result_container(
    container: Any,
    *,
    metric: str,
    source_file: Path,
    metadata: Mapping[str, Any],
    layer_grid: Sequence[int],
    dim_grid: Sequence[int],
    key_path: str,
    context_layer: Optional[int] = None,
    context_dim: Optional[int] = None,
) -> List[HyperparameterRecord]:
    """
    Recursively extract metric records from a known RGC_L or sweep container.

    The walker is intentionally narrow: once a mapping yields a direct
    (L, d, metric) record, it stops descending into that mapping so calibration
    parameters and timing fields cannot be mistaken for metric observations.
    """
    records: List[HyperparameterRecord] = []

    if isinstance(container, Mapping):
        layer = get_first_int(container, LAYER_KEYS)
        dim = get_first_int(container, DIM_KEYS)
        layer = layer if layer is not None else context_layer
        dim = dim if dim is not None else context_dim
        value = get_metric_value(container, metric)

        if add_record_if_valid(
            records,
            layer=layer,
            dim=dim,
            value=value,
            source_file=source_file,
            key_path=key_path,
            metadata=metadata,
            layer_grid=layer_grid,
            dim_grid=dim_grid,
        ):
            return records

        for raw_key, child in container.items():
            child_path = f"{key_path}.{raw_key}" if key_path else str(raw_key)
            key_layer, key_dim = parse_layer_dim_from_key(str(raw_key))
            child_layer = key_layer if key_layer is not None else layer
            child_dim = key_dim if key_dim is not None else dim

            key_as_int = parse_optional_int(raw_key)
            if key_as_int is not None:
                if child_layer is None and key_as_int in layer_grid:
                    child_layer = key_as_int
                elif child_layer is not None and child_dim is None and key_as_int in dim_grid:
                    child_dim = key_as_int

            child_value = parse_optional_float(child)
            if child_value is not None:
                metric_like_key = str(raw_key) in metric_aliases(metric)
                complete_cell = child_layer is not None and child_dim is not None
                if metric_like_key or complete_cell:
                    add_record_if_valid(
                        records,
                        layer=child_layer,
                        dim=child_dim,
                        value=child_value,
                        source_file=source_file,
                        key_path=child_path,
                        metadata=metadata,
                        layer_grid=layer_grid,
                        dim_grid=dim_grid,
                    )
                continue

            if isinstance(child, (Mapping, list, tuple)):
                records.extend(
                    walk_result_container(
                        child,
                        metric=metric,
                        source_file=source_file,
                        metadata=metadata,
                        layer_grid=layer_grid,
                        dim_grid=dim_grid,
                        key_path=child_path,
                        context_layer=child_layer,
                        context_dim=child_dim,
                    )
                )

    elif isinstance(container, (list, tuple)):
        numeric_values = [parse_optional_float(item) for item in container]
        if (
            context_layer is not None
            and context_dim is not None
            and any(value is not None for value in numeric_values)
        ):
            for index, value in enumerate(numeric_values):
                add_record_if_valid(
                    records,
                    layer=context_layer,
                    dim=context_dim,
                    value=value,
                    source_file=source_file,
                    key_path=f"{key_path}[{index}]",
                    metadata=metadata,
                    layer_grid=layer_grid,
                    dim_grid=dim_grid,
                )
            return records

        for index, child in enumerate(container):
            records.extend(
                walk_result_container(
                    child,
                    metric=metric,
                    source_file=source_file,
                    metadata=metadata,
                    layer_grid=layer_grid,
                    dim_grid=dim_grid,
                    key_path=f"{key_path}[{index}]",
                    context_layer=context_layer,
                    context_dim=context_dim,
                )
            )

    return records


def extract_hyperparameter_records(
    result: Mapping[str, Any],
    *,
    source_file: Path,
    metric: str,
    layer_grid: Sequence[int],
    dim_grid: Sequence[int],
) -> List[HyperparameterRecord]:
    """Extract all usable RGC_L hyperparameter records from one result JSON."""
    metadata = source_metadata(result)
    records: List[HyperparameterRecord] = []

    for key in METHOD_KEYS:
        if key in result:
            records.extend(
                walk_result_container(
                    result[key],
                    metric=metric,
                    source_file=source_file,
                    metadata=metadata,
                    layer_grid=layer_grid,
                    dim_grid=dim_grid,
                    key_path=key,
                )
            )

    for key in SWEEP_KEYS:
        if key in result:
            records.extend(
                walk_result_container(
                    result[key],
                    metric=metric,
                    source_file=source_file,
                    metadata=metadata,
                    layer_grid=layer_grid,
                    dim_grid=dim_grid,
                    key_path=key,
                )
            )

    direct_layer = get_first_int(result, LAYER_KEYS)
    direct_dim = get_first_int(result, DIM_KEYS)
    if direct_layer is not None and direct_dim is not None:
        records.extend(
            walk_result_container(
                result,
                metric=metric,
                source_file=source_file,
                metadata=metadata,
                layer_grid=layer_grid,
                dim_grid=dim_grid,
                key_path="root",
            )
        )

    return deduplicate_records(records)


def deduplicate_records(records: Iterable[HyperparameterRecord]) -> List[HyperparameterRecord]:
    """Remove exact duplicate observations from overlapping readers."""
    seen: set[Tuple[int, int, float, str, str]] = set()
    unique: List[HyperparameterRecord] = []
    for record in records:
        marker = (
            record.layer,
            record.dim,
            record.value,
            record.source_file,
            record.key_path,
        )
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(record)
    return unique


def discover_result_files(input_dir: Path) -> List[Path]:
    """Discover current nested and older flat JSON result layouts."""
    candidates = []
    candidates.extend(input_dir.rglob("paper_results.json"))
    candidates.extend(input_dir.glob("ablation_*.json"))
    candidates.extend(input_dir.rglob("ablation_*.json"))
    candidates.extend(input_dir.rglob("*.json"))

    unique_by_path: Dict[Path, Path] = {}
    for path in candidates:
        if path.is_file():
            try:
                unique_by_path[path.resolve()] = path
            except OSError:
                unique_by_path[path.absolute()] = path
    return [unique_by_path[key] for key in sorted(unique_by_path)]


def t_confidence_interval(values: Sequence[float]) -> Optional[float]:
    """Return the two-sided 95% t-distribution CI half-width."""
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    sample_std = float(np.std(arr, ddof=1))
    if not math.isfinite(sample_std):
        return None
    sem = sample_std / math.sqrt(len(values))
    return float(t_critical_975(len(values) - 1) * sem)


def t_critical_975(df: int) -> float:
    """Return the 97.5th percentile of Student's t distribution."""
    return float(scipy_stats.t.ppf(0.975, df=df))


def summarize_grid(
    records: Sequence[HyperparameterRecord],
    layers: Sequence[int],
    dims: Sequence[int],
    metric: str,
) -> List[Dict[str, Any]]:
    """Aggregate records into the requested CSV row schema."""
    values_by_cell: Dict[Tuple[int, int], List[float]] = {
        (layer, dim): [] for layer in layers for dim in dims
    }
    for record in records:
        values_by_cell.setdefault((record.layer, record.dim), []).append(record.value)

    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    ci_col = f"ci95_{metric}"
    rows: List[Dict[str, Any]] = []

    for layer in layers:
        for dim in dims:
            values = values_by_cell.get((layer, dim), [])
            if values:
                arr = np.asarray(values, dtype=float)
                mean_value: Optional[float] = float(np.mean(arr))
                std_value: Optional[float] = (
                    float(np.std(arr, ddof=1)) if len(values) >= 2 else None
                )
                ci_value = t_confidence_interval(values)
            else:
                mean_value = None
                std_value = None
                ci_value = None

            rows.append(
                {
                    "layer": layer,
                    "dim": dim,
                    mean_col: mean_value,
                    std_col: std_value,
                    ci_col: ci_value,
                    "count": len(values),
                    "values": values,
                }
            )

    return rows


def json_safe(value: Any) -> Any:
    """Convert NumPy and NaN-like values into strict JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, metric: str) -> None:
    """Write the summary CSV with the required schema."""
    fieldnames = [
        "layer",
        "dim",
        f"mean_{metric}",
        f"std_{metric}",
        f"ci95_{metric}",
        "count",
        "values",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row: Dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if field == "values":
                    output_row[field] = json.dumps(json_safe(value))
                elif value is None:
                    output_row[field] = ""
                else:
                    output_row[field] = value
            writer.writerow(output_row)


def format_cell(
    mean_value: Optional[float],
    ci_value: Optional[float],
    *,
    metric: str,
    is_chosen: bool,
) -> str:
    """Format one LaTeX cell."""
    if mean_value is None:
        return "--"

    multiplier = 100.0 if metric == "ece" else 1.0
    decimals = 2 if metric == "ece" else 4
    mean_text = f"{mean_value * multiplier:.{decimals}f}"
    if ci_value is not None:
        cell = f"{mean_text} $\\pm$ {ci_value * multiplier:.{decimals}f}"
    else:
        cell = mean_text
    if is_chosen:
        return f"\\textbf{{{cell}}}"
    return cell


def generate_latex_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    dims: Sequence[int],
    metric: str,
    chosen_layer: int,
    chosen_dim: int,
) -> str:
    """Render a compact LaTeX table with rows as L and columns as d."""
    mean_col = f"mean_{metric}"
    ci_col = f"ci95_{metric}"
    by_cell = {(int(row["layer"]), int(row["dim"])): row for row in rows}
    layer_list = ",".join(str(value) for value in layers)
    dim_list = ",".join(str(value) for value in dims)
    unit_note = " in percent" if metric == "ece" else ""
    column_spec = "l" + "c" * len(dims)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        (
            "\\caption{Hyperparameter sensitivity for $L \\in \\{"
            f"{layer_list}"
            "\\}$ and $d \\in \\{"
            f"{dim_list}"
            "\\}$. Cells report "
            f"{metric.upper()} mean $\\pm$ 95\\% CI{unit_note}; "
            f"the chosen configuration $L={chosen_layer}, d={chosen_dim}$ is bold.}}"
        ),
        "\\label{tab:hyperparameter_sensitivity}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        "$L \\backslash d$ & " + " & ".join(str(dim) for dim in dims) + " \\\\",
        "\\midrule",
    ]

    for layer in layers:
        cells = [str(layer)]
        for dim in dims:
            row = by_cell[(layer, dim)]
            cells.append(
                format_cell(
                    row.get(mean_col),
                    row.get(ci_col),
                    metric=metric,
                    is_chosen=(layer == chosen_layer and dim == chosen_dim),
                )
            )
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def summary_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    chosen_layer: int,
    chosen_dim: int,
) -> Dict[str, Any]:
    """Compute the audit summary line requested for the paper artifact."""
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    ci_col = f"ci95_{metric}"
    populated = [row for row in rows if row.get("count", 0) > 0]
    means = [row.get(mean_col) for row in populated if row.get(mean_col) is not None]
    stds = [row.get(std_col) for row in populated if row.get(std_col) is not None]
    chosen = next(
        (
            row
            for row in rows
            if row.get("layer") == chosen_layer and row.get("dim") == chosen_dim
        ),
        {},
    )
    return {
        "min_mean": min(means) if means else None,
        "max_mean": max(means) if means else None,
        "min_std": min(stds) if stds else None,
        "max_std": max(stds) if stds else None,
        "chosen_mean": chosen.get(mean_col),
        "chosen_std": chosen.get(std_col),
        "chosen_ci95": chosen.get(ci_col),
        "chosen_count": chosen.get("count", 0),
    }


def write_audit(path: Path, audit: Mapping[str, Any]) -> None:
    """Write a strict JSON audit file."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(audit)), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate RGC_L hyperparameter sensitivity CSV/LaTeX/audit outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("calibration_results"),
        help="Directory containing calibration result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("latex_tables_statistics/hyperparameter_sensitivity"),
        help="Directory for generated CSV, LaTeX, and audit JSON outputs.",
    )
    parser.add_argument(
        "--layers",
        type=lambda raw: parse_int_list(raw, "layers"),
        default=parse_int_list("2,4,6,8,10,12", "layers"),
        help="Comma-separated L values to include.",
    )
    parser.add_argument(
        "--dims",
        type=lambda raw: parse_int_list(raw, "dims"),
        default=parse_int_list("64,128,256,512,1024", "dims"),
        help="Comma-separated d values to include.",
    )
    parser.add_argument("--dataset", default=None, help="Optional dataset filter.")
    parser.add_argument("--model", default=None, help="Optional model-name filter.")
    parser.add_argument(
        "--training-method",
        default="baseline_cross_entropy",
        help="Training-method filter; 'baseline' is normalized to baseline_cross_entropy.",
    )
    parser.add_argument("--metric", default="ece", help="Metric to aggregate.")
    parser.add_argument("--chosen-layer", type=int, default=6, help="Chosen paper L.")
    parser.add_argument("--chosen-dim", type=int, default=256, help="Chosen paper d.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the hyperparameter sensitivity aggregation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    metric = args.metric.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_result_files(args.input_dir)
    audit: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": args.input_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "layers": list(args.layers),
        "dims": list(args.dims),
        "dataset_filter": args.dataset,
        "model_filter": args.model,
        "training_method_filter": normalize_training_method(args.training_method),
        "metric": metric,
        "chosen_layer": args.chosen_layer,
        "chosen_dim": args.chosen_dim,
        "files_scanned": [],
        "parse_failures": [],
        "records_found": 0,
        "records_used": [],
        "missing_cells": [],
        "outputs": {},
        "summary": {},
    }

    all_records: List[HyperparameterRecord] = []
    filtered_out = 0

    for path in files:
        file_entry: Dict[str, Any] = {"path": path.as_posix(), "top_level_keys": []}
        try:
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except Exception as exc:
            audit["parse_failures"].append({"path": path.as_posix(), "error": str(exc)})
            audit["files_scanned"].append(file_entry)
            continue

        if isinstance(result, Mapping):
            file_entry["top_level_keys"] = sorted(str(key) for key in result.keys())
            metadata = source_metadata(result)
            file_entry["metadata"] = dict(metadata)
            if not is_clean_result(result) or not matches_filters(
                metadata,
                dataset=args.dataset,
                model=args.model,
                training_method=args.training_method,
            ):
                filtered_out += 1
                audit["files_scanned"].append(file_entry)
                continue

            records = extract_hyperparameter_records(
                result,
                source_file=path,
                metric=metric,
                layer_grid=args.layers,
                dim_grid=args.dims,
            )
            file_entry["records_found"] = len(records)
            all_records.extend(records)
        else:
            file_entry["top_level_type"] = type(result).__name__

        audit["files_scanned"].append(file_entry)

    all_records = deduplicate_records(all_records)
    audit["records_found"] = len(all_records)
    audit["filtered_files"] = filtered_out

    audit_path = args.output_dir / "hyperparameter_sensitivity_audit.json"
    if not all_records:
        write_audit(audit_path, audit)
        print(
            "ERROR: no usable RGC_L hyperparameter sweep records were found. "
            f"Scanned {len(files)} JSON file(s); audit written to {audit_path}.",
            file=sys.stderr,
        )
        return 1

    rows = summarize_grid(all_records, args.layers, args.dims, metric)
    missing_cells = [
        {"layer": row["layer"], "dim": row["dim"]}
        for row in rows
        if row.get("count", 0) == 0
    ]
    audit["missing_cells"] = missing_cells
    audit["records_used"] = [asdict(record) for record in all_records]
    audit["summary"] = summary_statistics(
        rows,
        metric=metric,
        chosen_layer=args.chosen_layer,
        chosen_dim=args.chosen_dim,
    )

    csv_path = args.output_dir / "hyperparameter_sensitivity_summary.csv"
    latex_path = args.output_dir / "hyperparameter_sensitivity_table.tex"
    write_csv(rows, csv_path, metric)
    latex = generate_latex_table(
        rows,
        layers=args.layers,
        dims=args.dims,
        metric=metric,
        chosen_layer=args.chosen_layer,
        chosen_dim=args.chosen_dim,
    )
    latex_path.write_text(latex, encoding="utf-8")

    audit["outputs"] = {
        "csv": csv_path.as_posix(),
        "latex": latex_path.as_posix(),
        "audit": audit_path.as_posix(),
    }
    write_audit(audit_path, audit)

    populated = len(rows) - len(missing_cells)
    print(
        f"Wrote {csv_path}, {latex_path}, and {audit_path}. "
        f"Populated {populated}/{len(rows)} grid cells from {len(all_records)} record(s)."
    )
    if missing_cells:
        print(
            f"WARNING: {len(missing_cells)} requested grid cell(s) had no data; "
            "they were left blank in the LaTeX table.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
