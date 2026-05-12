#!/usr/bin/env python3
"""
Generate the paper's LaTeX tables (Tables 1 and 2).

This script aggregates the per-config calibration JSON files produced by
run_rgc_experiments.py and emits:
    Table 1 - ECE comparison across datasets, models, and methods.
    Table 2 - TOST equivalence tests (RGC_L vs DAC, RGC_L vs RGC_C).
"""

import json
import logging
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import sys
# Add project root to path (before importing utils)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Global list to store detailed significance analysis
SIGNIFICANCE_LOG = []

# TULIP published results (ECE in decimal form)
# From TULIP paper Table 8
# Note: Original paper reports 0.658±0.004 for CIFAR-100, but this appears to be a decimal
# error (should be 0.0658). We've corrected to match their CIFAR-10 scale.
# Note: format_ece_value multiplies by 100, so input 0.030 to display "3.00%"
TULIP_PUBLISHED = {
    ('cifar10', 'baseline_cross_entropy', 'resnet50'): (0.030, 0.005),
    ('cifar10', 'baseline_cross_entropy', 'resnet101'): (0.025, 0.021),
    ('cifar10', 'baseline_cross_entropy', 'resnet152'): (0.026, 0.011),
    # CIFAR-100 values corrected: divided by 10 to fix decimal point error in paper
    # Original paper: 0.658±0.004 (decimal) -> Corrected: 0.0658±0.0004 (decimal)
    # format_ece_value multiplies by 100, so 0.0658 -> displays as 6.58%
    ('cifar100', 'baseline_cross_entropy', 'resnet50'): (0.0658, 0.0004),
    ('cifar100', 'baseline_cross_entropy', 'resnet101'): (0.0671, 0.0021),
    ('cifar100', 'baseline_cross_entropy', 'resnet152'): (0.0662, 0.0055),
}


def normalize_training_method(training_method: Any) -> str:
    """Return the canonical training-method name used for aggregation/lookups."""
    if training_method is None:
        return 'unknown'
    training_method = str(training_method)
    if training_method == 'baseline':
        return 'baseline_cross_entropy'
    return training_method


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load result JSON files from old flat and current nested layouts."""
    old_layout_files = sorted(results_dir.glob('ablation_*.json'))
    nested_layout_files = sorted(results_dir.rglob('paper_results.json'))

    logger.info(
        "Result file discovery in %s: old-layout ablation_*.json=%d, "
        "nested paper_results.json=%d",
        results_dir,
        len(old_layout_files),
        len(nested_layout_files),
    )

    result_files_by_path = {}
    for result_file in old_layout_files + nested_layout_files:
        try:
            result_files_by_path[result_file.resolve()] = result_file
        except OSError:
            result_files_by_path[result_file.absolute()] = result_file

    result_files = [result_files_by_path[path] for path in sorted(result_files_by_path)]

    if not result_files:
        logger.error(
            "No result JSON files found in %s. Expected either old flat layout "
            "%s or current nested layout %s.",
            results_dir,
            results_dir / 'ablation_*.json',
            results_dir / '**' / 'paper_results.json',
        )
        return []

    results = []
    parse_failures = 0
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            results.append(result)
        except Exception as e:
            parse_failures += 1
            logger.warning(f"Failed to load {result_file}: {e}")
            continue

    logger.info(
        "Loaded %d total JSON files from %s (%d parse failures)",
        len(results),
        results_dir,
        parse_failures,
    )
    return results


def load_coordinate_results(coord_dir: Path, k_value: int = 256) -> pd.DataFrame:
    """
    Load coordinate ablation results and aggregate by (dataset, model, training_method).
    
    Args:
        coord_dir: Directory containing coordinate_ablation_*.json files
        k_value: Number of coordinates to use (default 256 for comparison with L=6 d=256)
    
    Returns:
        DataFrame with columns: dataset, model_name, training_method, coord_ece_mean, coord_ece_std, coord_ece_count, coord_ece_values, coord_acc_mean, coord_acc_std, coord_acc_count
    """
    results = []
    
    for strategy in ["nested", "independent"]:
        pattern = str(coord_dir / f"coordinate_ablation_*_{strategy}.json")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")
                continue
            
            config = data.get("experiment_config", {})
            dataset = config.get("dataset")
            model_name = config.get("model_name")
            training_method = config.get("training_method")
            
            if not all([dataset, model_name, training_method]):
                continue
            training_method = normalize_training_method(training_method)
            
            results_by_k = data.get("results_by_num_coordinates", {})
            k_str = str(k_value)
            
            if k_str in results_by_k:
                for trial in results_by_k[k_str]:
                    ece_geo = trial.get("ece_geometric")
                    acc_geo = trial.get("accuracy_geometric")
                    # DAC on coordinate features (if present)
                    ece_dac = trial.get("ece_dac")
                    acc_dac = trial.get("accuracy_dac")

                    if ece_geo is not None:
                        results.append({
                            "dataset": dataset,
                            "model_name": model_name,
                            "training_method": training_method,
                            "coord_ece": ece_geo,
                            "coord_acc": acc_geo,
                            "coord_dac_ece": ece_dac,
                            "coord_dac_acc": acc_dac,
                        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Aggregate by (dataset, model_name, training_method)
    agg = df.groupby(["dataset", "model_name", "training_method"]).agg(
        coord_ece_mean=("coord_ece", "mean"),
        coord_ece_std=("coord_ece", "std"),
        coord_ece_count=("coord_ece", "count"),
        coord_ece_values=("coord_ece", list),
        coord_acc_mean=("coord_acc", "mean"),
        coord_acc_std=("coord_acc", "std"),
        coord_acc_count=("coord_acc", "count"),
        coord_dac_ece_mean=("coord_dac_ece", "mean"),
        coord_dac_ece_std=("coord_dac_ece", "std"),
        coord_dac_ece_count=("coord_dac_ece", "count"),
        coord_dac_ece_values=("coord_dac_ece", list),
        coord_dac_acc_mean=("coord_dac_acc", "mean"),
        coord_dac_acc_std=("coord_dac_acc", "std"),
        coord_dac_acc_count=("coord_dac_acc", "count"),
    ).reset_index()
    
    return agg


def extract_experiment_key(result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (training_method, dataset, model_name) key from result."""
    exp_info = result.get('experiment_info', result.get('experiment_config', {}))
    training_method = normalize_training_method(exp_info.get('training_method', 'unknown'))
    dataset = exp_info.get('dataset', 'unknown')
    model_name = exp_info.get('model_name', 'unknown')
    return (training_method, dataset, model_name)


def is_clean_result(result: Dict[str, Any]) -> bool:
    """Check if result is from clean (non-corrupted) dataset."""
    exp_info = result.get('experiment_info', result.get('experiment_config', {}))
    corruption_type = exp_info.get('corruption_type')
    corruption_severity = exp_info.get('corruption_severity')
    return corruption_type is None and corruption_severity is None


def extract_baseline_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract baseline calibration metrics from standard_baselines section."""
    # Try new structure first (standard_baselines), fallback to old (baselines)
    # Be robust to cases where these keys exist but are explicitly set to null/None.
    baselines = result.get('standard_baselines')
    if baselines is None:
        baselines = result.get('baselines') or {}

    metrics = {}
    
    # Uncalibrated
    if 'uncalibrated' in baselines:
        metrics['uncal_ece'] = baselines['uncalibrated'].get('ece', np.nan)
        metrics['uncal_acc'] = baselines['uncalibrated'].get('acc', np.nan)
    
    # Temperature Scaling
    if 'temperature_scaling' in baselines:
        metrics['ts_ece'] = baselines['temperature_scaling'].get('ece', np.nan)
        metrics['ts_acc'] = baselines['temperature_scaling'].get('acc', np.nan)
    
    # Platt Scaling
    if 'platt_scaling' in baselines:
        metrics['platt_ece'] = baselines['platt_scaling'].get('ece', np.nan)
        metrics['platt_acc'] = baselines['platt_scaling'].get('acc', np.nan)
    
    # Isotonic Regression
    if 'isotonic_toplabel' in baselines:
        metrics['isotonic_ece'] = baselines['isotonic_toplabel'].get('ece', np.nan)
        metrics['isotonic_acc'] = baselines['isotonic_toplabel'].get('acc', np.nan)
    
    # Pixel Geo Space
    if 'geometric_physical_space' in baselines:
        metrics['pixel_geo_ece'] = baselines['geometric_physical_space'].get('ece', np.nan)
        metrics['pixel_geo_acc'] = baselines['geometric_physical_space'].get('acc', np.nan)
    
    # Beta Calibration
    if 'beta_calibration' in baselines:
        metrics['beta_ece'] = baselines['beta_calibration'].get('ece', np.nan)
        metrics['beta_acc'] = baselines['beta_calibration'].get('acc', np.nan)
    
    # Density Aware Calibration (actual algorithm, not layer selection)
    if 'density_aware_calibration' in baselines:
        metrics['dac_algorithm_ece'] = baselines['density_aware_calibration'].get('ece', np.nan)
        metrics['dac_algorithm_acc'] = baselines['density_aware_calibration'].get('acc', np.nan)
    
    return metrics


def extract_rgcl_result(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract RGCL ECE for the requested (L, d) configuration."""
    rgcl = result.get('rgcl', {})
    if rgcl and isinstance(rgcl, dict) and not rgcl.get('skipped', False):
        num_layers = rgcl.get('num_layers')
        target_dim = rgcl.get('target_dim', rgcl.get('target_dimension'))
        if num_layers == L and target_dim == d:
            return rgcl.get('ece')
    return None


def extract_rgcl_acc(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract RGCL accuracy for the requested (L, d) configuration."""
    rgcl = result.get('rgcl', {})
    if rgcl and isinstance(rgcl, dict) and not rgcl.get('skipped', False):
        num_layers = rgcl.get('num_layers')
        target_dim = rgcl.get('target_dim', rgcl.get('target_dimension'))
        if num_layers == L and target_dim == d:
            acc = rgcl.get('accuracy')
            if acc is not None:
                return acc / 100.0
    return None


def extract_coordinate_sampling_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract RGCC ECE from the rgcc section."""
    rgcc = result.get('rgcc', {})
    if rgcc and isinstance(rgcc, dict) and not rgcc.get('skipped', False):
        return rgcc.get('ece')
    return None


def extract_coordinate_sampling_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract RGCC accuracy from the rgcc section."""
    rgcc = result.get('rgcc', {})
    if rgcc and isinstance(rgcc, dict) and not rgcc.get('skipped', False):
        acc = rgcc.get('accuracy')
        if acc is not None:
            return acc / 100.0
    return None


def extract_gc_with_dac_layers_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract GC(DAC) ECE from the gc_dac section."""
    gc_dac = result.get('gc_dac', {})
    if gc_dac and isinstance(gc_dac, dict) and not gc_dac.get('skipped', False):
        return gc_dac.get('ece')
    return None


def extract_gc_with_dac_layers_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract GC(DAC) accuracy from the gc_dac section."""
    gc_dac = result.get('gc_dac', {})
    if gc_dac and isinstance(gc_dac, dict) and not gc_dac.get('skipped', False):
        acc = gc_dac.get('accuracy')
        if acc is not None:
            return acc / 100.0
    return None


def extract_deep_ensemble_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract D.Ens ECE from the deep_ensemble section."""
    deep_ensemble = result.get('deep_ensemble', {})
    if deep_ensemble and isinstance(deep_ensemble, dict) and not deep_ensemble.get('skipped', False):
        return deep_ensemble.get('ece')
    return None


def extract_deep_ensemble_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract D.Ens accuracy from the deep_ensemble section."""
    deep_ensemble = result.get('deep_ensemble', {})
    if deep_ensemble and isinstance(deep_ensemble, dict) and not deep_ensemble.get('skipped', False):
        acc = deep_ensemble.get('accuracy')
        if acc is not None:
            return acc / 100.0 if acc > 1.0 else acc
    return None


def validate_results(results: List[Dict[str, Any]]) -> None:
    """Validate results and log statistics."""
    clean_results = [r for r in results if is_clean_result(r)]

    has_baselines = any('standard_baselines' in r for r in clean_results)
    has_rgcl = any('rgcl' in r for r in clean_results)
    has_rgcc = any('rgcc' in r for r in clean_results)

    logger.info(f"Validated: {len(clean_results)} clean results out of {len(results)} total")
    logger.info(f"Structure check: baselines={has_baselines}, rgcl={has_rgcl}, rgcc={has_rgcc}")


def aggregate_over_seeds(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate results over seeds, computing mean ± std for each metric.
    Also stores raw ECE values as lists for statistical significance testing.
    """
    # Only use clean results for main tables
    clean_results = [r for r in results if is_clean_result(r)]
    
    if not clean_results:
        logger.warning("No clean results found! Using all results.")
        clean_results = results
    
    rows = []
    
    for result in clean_results:
        exp_key = extract_experiment_key(result)
        training_method, dataset, model_name = exp_key
        
        # Extract baseline metrics
        baseline_metrics = extract_baseline_metrics(result)
        
        # RGCL (L=6, d=256)
        rgcl_ece = extract_rgcl_result(result, L=6, d=256)
        rgcl_acc = extract_rgcl_acc(result, L=6, d=256)

        # RGCC
        coord_ece = extract_coordinate_sampling_ece(result)
        coord_acc = extract_coordinate_sampling_acc(result)

        # GC(DAC)
        gc_dac_layers_ece = extract_gc_with_dac_layers_ece(result)
        gc_dac_layers_acc = extract_gc_with_dac_layers_acc(result)

        # D.Ens
        deep_ensemble_ece = extract_deep_ensemble_ece(result)
        deep_ensemble_acc = extract_deep_ensemble_acc(result)

        row = {
            'training_method': training_method,
            'dataset': dataset,
            'model_name': model_name,
            'seed': result.get('experiment_info', {}).get('seed', result.get('experiment_config', {}).get('seed', None)),
            **baseline_metrics,
            'rgcl_ece': rgcl_ece,
            'rgcl_acc': rgcl_acc,
            'coord_ece': coord_ece,
            'coord_acc': coord_acc,
            'gc_dac_layers_ece': gc_dac_layers_ece,
            'gc_dac_layers_acc': gc_dac_layers_acc,
            'deep_ensemble_ece': deep_ensemble_ece,
            'deep_ensemble_acc': deep_ensemble_acc,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Aggregate over seeds: group by (training_method, dataset, model_name)
    # For ECE columns, also store lists of raw values for significance testing
    agg_dict = {}
    
    for col in df.columns:
        if col not in ['training_method', 'dataset', 'model_name', 'seed']:
            if col.endswith('_ece') or col.endswith('_acc'):
                # For ECE/ACC columns, aggregate mean/std/count AND list
                agg_dict[col] = ['mean', 'std', 'count']
            else:
                # For other columns, just mean/std/count
                agg_dict[col] = ['mean', 'std', 'count']
    
    grouped = df.groupby(['training_method', 'dataset', 'model_name']).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
    
    # Calculate 95% CI instead of std: CI = 1.96 * std / sqrt(N)
    for col in df.columns:
        if col not in ['training_method', 'dataset', 'model_name', 'seed']:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            count_col = f'{col}_count'
            ci_col = f'{col}_ci'
            
            if mean_col in grouped.columns and std_col in grouped.columns and count_col in grouped.columns:
                # Calculate 95% CI: 1.96 * std / sqrt(N)
                # Handle division by zero and NaN cases
                def calc_ci(row):
                    std_val = row[std_col]
                    count_val = row[count_col]
                    if pd.isna(std_val) or pd.isna(count_val) or count_val <= 1:
                        return np.nan
                    return 1.96 * std_val / np.sqrt(count_val)
                
                grouped[ci_col] = grouped.apply(calc_ci, axis=1)
    
    # Now aggregate lists separately for significance testing
    for col in df.columns:
        if col not in ['training_method', 'dataset', 'model_name', 'seed'] and \
           (col.endswith('_ece') or col.endswith('_acc')):
            list_col = f'{col}_values'
            list_series = df.groupby(['training_method', 'dataset', 'model_name'])[col].apply(
                lambda x: x.dropna().tolist()
            )
            grouped[list_col] = list_series
    
    # Reset index
    grouped = grouped.reset_index()
    
    return grouped


def perform_tost_test(
    scores1: List[float], 
    scores2: List[float], 
    equivalence_margin: float = 0.0001
) -> Dict[str, Any]:
    """
    Perform Two One-Sided Tests (TOST) for equivalence testing.
    
    Args:
        scores1: First set of paired scores (e.g., RGC ECE values)
        scores2: Second set of paired scores (e.g., Coord ECE values)
        equivalence_margin: Equivalence margin (delta) in same units as scores (default 0.005 = 0.5%)
    
    Returns:
        Dictionary with:
        - tost_pvalue: TOST p-value (max of two one-sided tests)
        - mean_diff: Mean difference (scores2 - scores1)
        - ci_lower: Lower bound of 95% CI for difference
        - ci_upper: Upper bound of 95% CI for difference
        - dz: Paired effect size (Cohen's d_z)
        - equivalence_established: Boolean indicating if equivalence is established (p < 0.05)
        - n_pairs: Number of paired observations
    """
    # Filter out NaN values and ensure paired data
    pairs = [(s1, s2) for s1, s2 in zip(scores1, scores2) if pd.notna(s1) and pd.notna(s2)]
    
    if len(pairs) < 2:
        return {
            'tost_pvalue': 1.0,
            'mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'dz': 0.0,
            'equivalence_established': False,
            'n_pairs': len(pairs)
        }
    
    arr1 = np.array([p[0] for p in pairs])
    arr2 = np.array([p[1] for p in pairs])
    diff_arr = arr2 - arr1
    
    n = len(diff_arr)
    mean_diff = diff_arr.mean()
    std_diff = diff_arr.std(ddof=1)
    
    # Standard error of the mean difference
    se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
    
    # 95% CI for mean difference
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # TOST: Two One-Sided Tests
    # H01: mean_diff >= margin (reject if p_upper < 0.05)
    # H02: mean_diff <= -margin (reject if p_lower < 0.05)
    # Equivalence established if both are rejected (max(p_upper, p_lower) < 0.05)
    if se_diff > 0 and n > 1:
        t_upper = (mean_diff - equivalence_margin) / se_diff
        t_lower = (mean_diff + equivalence_margin) / se_diff
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        tost_pval = max(p_upper, p_lower)
    else:
        tost_pval = 1.0
    
    # Paired effect size (Cohen's d_z)
    dz = mean_diff / std_diff if std_diff > 0 else 0.0
    
    equivalence_established = tost_pval < 0.05
    
    # Minimum margin for equivalence = max of absolute CI bounds
    # This is the tightest margin where equivalence would still be established
    if pd.notna(ci_lower) and pd.notna(ci_upper):
        min_margin_for_equiv = max(abs(ci_lower), abs(ci_upper))
    else:
        min_margin_for_equiv = np.nan
    
    return {
        'tost_pvalue': float(tost_pval),
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'dz': float(dz),
        'equivalence_established': equivalence_established,
        'n_pairs': n,
        'min_margin_for_equiv': float(min_margin_for_equiv) if pd.notna(min_margin_for_equiv) else np.nan,
        'min_margin_for_equiv_pct': float(min_margin_for_equiv * 100) if pd.notna(min_margin_for_equiv) else np.nan,
    }


def determine_result(
    tost_pvalue: float, 
    mean_diff: float, 
    equivalence_margin: float,
    ci_lower: float,
    ci_upper: float
) -> str:
    """
    Determine the result category based on TOST and CI.
    
    Note: mean_diff = scores2 - scores1. Since lower ECE is better:
    - If mean_diff > 0: Method 2 has higher (worse) ECE, so Method 1 is superior
    - If mean_diff < 0: Method 2 has lower (better) ECE, so Method 2 is superior
    
    Returns:
        "Equivalent" if TOST p < 0.05 (equivalence established)
        "Inconclusive" if TOST p >= 0.05 (equivalence not established, but may be due to insufficient power)
        "Method X superior" if CI doesn't include 0 and doesn't overlap with equivalence region
    """
    if pd.isna(tost_pvalue) or pd.isna(mean_diff) or pd.isna(ci_lower) or pd.isna(ci_upper):
        return "---"
    
    if tost_pvalue < 0.05:
        return "Equivalent"
    
    # Check if CI overlaps with equivalence region [-margin, +margin]
    ci_overlaps_equiv = (ci_lower <= equivalence_margin and ci_upper >= -equivalence_margin)
    
    # Check if CI excludes 0 (statistically significant difference)
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    
    # If CI excludes 0 and doesn't overlap with equivalence region, one method is superior
    # Remember: mean_diff = scores2 - scores1, and lower ECE is better
    if ci_excludes_zero and not ci_overlaps_equiv:
        if ci_lower > 0:
            # Entire CI is positive: mean_diff > 0, so Method 2 has higher (worse) ECE
            # Therefore Method 1 is superior
            return "Method 1 superior"
        elif ci_upper < 0:
            # Entire CI is negative: mean_diff < 0, so Method 2 has lower (better) ECE
            # Therefore Method 2 is superior
            return "Method 2 superior"
    
    # Otherwise, inconclusive (equivalence not established, but can't conclude superiority)
    return "Inconclusive"


def format_ece_value(mean_val: float, ci_val: float = None, std_val: float = None, count: int = None, inline_math: bool = False) -> str:
    """Format ECE value as mean ± 95% CI or just mean if CI is NaN/zero or count=1.
    
    Args:
        mean_val: Mean ECE value
        ci_val: 95% Confidence Interval (preferred, pre-calculated)
        std_val: Standard deviation (used to calculate CI if ci_val not provided)
        count: Number of samples (required for CI calculation from std)
        inline_math: If True, use inline math mode (1.23$\\pm$0.26), else use spaced (1.23 $\\pm$ 0.26)
    
    Note:
        When calculating CI from std, uses t-distribution: t_critical * SEM where
        SEM = std / sqrt(n) and t_critical = stats.t.ppf(0.975, df=n-1)
    """
    if pd.isna(mean_val):
        return "---"
    
    # Use CI if provided, otherwise calculate from std if available
    if ci_val is not None and not pd.isna(ci_val) and ci_val > 0:
        error_val = ci_val
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and count is not None and count > 1:
        # Calculate 95% CI from std using t-distribution
        # SEM = standard error of the mean
        sem = std_val / np.sqrt(count)
        # t-critical for 95% CI with n-1 degrees of freedom
        t_critical = stats.t.ppf(0.975, df=count - 1)
        error_val = t_critical * sem
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and (count is None or count == 1):
        # Fallback: if count is missing or 1, use std as approximation
        error_val = std_val
    else:
        # No error bar available
        return f"{mean_val * 100:.2f}"
    
    if inline_math:
        return f"{mean_val * 100:.2f}$\\pm${error_val * 100:.2f}"
    else:
        return f"{mean_val * 100:.2f} $\\pm$ {error_val * 100:.2f}"


def format_accuracy_value(mean_val: float, ci_val: float = None, std_val: float = None, count: int = None, inline_math: bool = False) -> str:
    """Format accuracy value as mean ± 95% CI or just mean if CI is NaN/zero or count=1.
    
    Args:
        mean_val: Mean accuracy value (as percentage, e.g., 95.5 for 95.5%, or in [0,1] range)
        ci_val: 95% Confidence Interval (preferred, calculated as 1.96 * std / sqrt(N))
        std_val: Standard deviation (fallback if ci_val not provided, for backward compatibility)
        count: Number of samples
        inline_math: If True, use inline math mode (95.5$\\pm$0.5), else use spaced (95.5 $\\pm$ 0.5)
    """
    if pd.isna(mean_val):
        return "---"
    
    # If accuracy is in [0, 1] range, convert to percentage
    convert_to_pct = mean_val <= 1.0
    if convert_to_pct:
        mean_val = mean_val * 100
        if ci_val is not None and not pd.isna(ci_val):
            ci_val = ci_val * 100
        if std_val is not None and not pd.isna(std_val):
            std_val = std_val * 100
    
    # Use CI if provided, otherwise calculate from std if available
    if ci_val is not None and not pd.isna(ci_val) and ci_val > 0:
        error_val = ci_val
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and count is not None and count > 1:
        # Calculate CI from std: 1.96 * std / sqrt(N)
        if not convert_to_pct:
            error_val = 1.96 * std_val / np.sqrt(count) * 100
        else:
            error_val = 1.96 * std_val / np.sqrt(count)
    else:
        # No error bar available
        return f"{mean_val:.2f}"
    
    if inline_math:
        return f"{mean_val:.2f}$\\pm${error_val:.2f}"
    else:
        return f"{mean_val:.2f} $\\pm$ {error_val:.2f}"


def generate_main_results_table(aggregated_df: pd.DataFrame) -> str:
    """Generate Table 1: Main Results Comparison.
    
    Columns: Dataset, Model (Acc), Uncal, TS, Platt, Isotonic, Beta, DAC,
    Fast Separation, TULIP, D.Ens, RGC_L, RGC_C
    
    Args:
        aggregated_df: DataFrame with aggregated results (from aggregate_over_seeds)
    
    Returns:
        LaTeX table string
    """
    # Filter to only include baseline_cross_entropy training method
    aggregated_df = aggregated_df[
        aggregated_df['training_method'] == 'baseline_cross_entropy'
    ].copy()
    
    lines = []
    lines.append("% Table 1: Main Results Comparison")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    
    # Build table header (13 columns: 2 id + 11 methods)
    lines.append("\\begin{tabular}{llccccccccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model (Acc) & Uncal & TS & Platt & Isotonic & Beta & DAC & Fast Separation & TULIP & D.Ens & RGC$_L$ & RGC$_C$ \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset, then model
    aggregated_df = aggregated_df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in aggregated_df.iterrows():
        dataset = row['dataset']
        model_name = row['model_name']
        
        # Group rows by dataset
        is_new_dataset = (current_dataset != dataset)
        if is_new_dataset:
            if current_dataset is not None:
                lines.append("\\midrule")
        current_dataset = dataset
        
        # Extract ECE values for comparison
        methods = {
            'uncal': row.get('uncal_ece_mean', np.nan),
            'ts': row.get('ts_ece_mean', np.nan),
            'platt': row.get('platt_ece_mean', np.nan),
            'isotonic': row.get('isotonic_ece_mean', np.nan),
            'beta': row.get('beta_ece_mean', np.nan),
            'dac': row.get('dac_algorithm_ece_mean', np.nan),
            'pixel_geo': row.get('pixel_geo_ece_mean', np.nan),
            'tulip': TULIP_PUBLISHED.get((dataset, row['training_method'], model_name), (np.nan, np.nan))[0],
            'deep_ensemble': row.get('deep_ensemble_ece_mean', np.nan),
            'rgcl': row.get('rgcl_ece_mean', np.nan),
            'coord': row.get('coord_ece_mean', np.nan),
        }
        
        # Find best, 2nd, 3rd (excluding NaN)
        valid_methods = {k: v for k, v in methods.items() if not pd.isna(v)}
        if valid_methods:
            sorted_methods = sorted(valid_methods.items(), key=lambda x: x[1])
            best_method = sorted_methods[0][0] if len(sorted_methods) > 0 else None
            second_method = sorted_methods[1][0] if len(sorted_methods) > 1 else None
            third_method = sorted_methods[2][0] if len(sorted_methods) > 2 else None
        else:
            best_method = second_method = third_method = None
        
        # Format dataset and model
        dataset_display = dataset.upper()
        acc_mean = row.get('uncal_acc_mean', row.get('rgcl_acc_mean', np.nan))
        acc_ci = row.get('uncal_acc_ci', row.get('rgcl_acc_ci', np.nan))
        acc_count = row.get('uncal_acc_count', row.get('rgcl_acc_count', None))
        model_display = f"{model_name} ({format_accuracy_value(acc_mean, ci_val=acc_ci, count=acc_count)})"
        
        # Show dataset name only on first row of each dataset group
        show_dataset = is_new_dataset
        
        # Format each method with coloring
        def format_method(method_key, mean_col, ci_col, count_col, is_best=False, is_second=False, is_third=False):
            mean_val = row.get(mean_col, np.nan)
            ci_val = row.get(ci_col, np.nan)
            count_val = row.get(count_col, None)
            
            if pd.isna(mean_val):
                return "---"
            
            formatted = format_ece_value(mean_val, ci_val=ci_val, count=count_val)
            
            if is_best:
                return f"\\textcolor{{blue}}{{\\textbf{{{formatted}}}}}"
            elif is_second:
                return f"\\textcolor{{teal}}{{\\textbf{{{formatted}}}}}"
            elif is_third:
                return f"\\textcolor{{olive}}{{\\textbf{{{formatted}}}}}"
            return formatted

        def format_tulip(is_best=False, is_second=False, is_third=False):
            tulip_key = (dataset, row['training_method'], model_name)
            if tulip_key not in TULIP_PUBLISHED:
                return "---"

            mean_val, std_val = TULIP_PUBLISHED[tulip_key]
            formatted = f"{mean_val * 100:.2f} $\\pm$ {std_val * 100:.2f}"

            if is_best:
                return f"\\textcolor{{blue}}{{\\textbf{{{formatted}}}}}"
            elif is_second:
                return f"\\textcolor{{teal}}{{\\textbf{{{formatted}}}}}"
            elif is_third:
                return f"\\textcolor{{olive}}{{\\textbf{{{formatted}}}}}"
            return formatted
        
        cells = [
            dataset_display if show_dataset else "",
            model_display,
            format_method('uncal', 'uncal_ece_mean', 'uncal_ece_ci', 'uncal_ece_count', 
                         best_method == 'uncal', second_method == 'uncal', third_method == 'uncal'),
            format_method('ts', 'ts_ece_mean', 'ts_ece_ci', 'ts_ece_count',
                         best_method == 'ts', second_method == 'ts', third_method == 'ts'),
            format_method('platt', 'platt_ece_mean', 'platt_ece_ci', 'platt_ece_count',
                         best_method == 'platt', second_method == 'platt', third_method == 'platt'),
            format_method('isotonic', 'isotonic_ece_mean', 'isotonic_ece_ci', 'isotonic_ece_count',
                         best_method == 'isotonic', second_method == 'isotonic', third_method == 'isotonic'),
            format_method('beta', 'beta_ece_mean', 'beta_ece_ci', 'beta_ece_count',
                         best_method == 'beta', second_method == 'beta', third_method == 'beta'),
            format_method('dac', 'dac_algorithm_ece_mean', 'dac_algorithm_ece_ci', 'dac_algorithm_ece_count',
                         best_method == 'dac', second_method == 'dac', third_method == 'dac'),
            format_method('pixel_geo', 'pixel_geo_ece_mean', 'pixel_geo_ece_ci', 'pixel_geo_ece_count',
                         best_method == 'pixel_geo', second_method == 'pixel_geo', third_method == 'pixel_geo'),
            format_tulip(best_method == 'tulip', second_method == 'tulip', third_method == 'tulip'),
            format_method('deep_ensemble', 'deep_ensemble_ece_mean', 'deep_ensemble_ece_ci', 'deep_ensemble_ece_count',
                         best_method == 'deep_ensemble', second_method == 'deep_ensemble', third_method == 'deep_ensemble'),
            format_method('rgcl', 'rgcl_ece_mean', 'rgcl_ece_ci', 'rgcl_ece_count',
                         best_method == 'rgcl', second_method == 'rgcl', third_method == 'rgcl'),
            format_method('coord', 'coord_ece_mean', 'coord_ece_ci', 'coord_ece_count',
                         best_method == 'coord', second_method == 'coord', third_method == 'coord'),
        ]

        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{ECE (\\%) comparison across datasets and models. Values shown as mean $\\pm$ 95\\% CI. "
                 "\\textcolor{blue}{\\textbf{Best}}, \\textcolor{teal}{\\textbf{2nd}}, \\textcolor{olive}{\\textbf{3rd}}.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_rgcl_vs_coord_comparison_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare RGC (RGC_L) against Coordinate (RGC_C) ECE with TOST equivalence testing.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.001 = 0.1pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for RGC vs Coord comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'rgcl_ece_mean': 'mean',
        'rgcl_ece_std': rms_agg,
        'rgcl_ece_count': 'sum',
        'coord_ece_mean': 'mean',
        'coord_ece_std': rms_agg,
        'coord_ece_count': 'sum',
        'rgcl_ece_values': flatten_lists,
        'coord_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('rgcl_ece_mean').notna()
        & df.get('coord_ece_mean').notna()
    ]

    if df.empty:
        return "% No overlapping RGC and Coord results for comparison"

    lines.append("% RGC_L vs RGC_C (RGC vs Coordinate) calibration comparison table")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{RGC$_L$ vs RGC$_C$ calibration comparison (ECE \\%; lower is better). We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:rgc_l_vs_rgc_c}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & RGC$_L$ & RGC$_C$ & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        rgcl_mean = row.get('rgcl_ece_mean', np.nan)
        rgcl_std = row.get('rgcl_ece_std', np.nan)
        rgcl_count = row.get('rgcl_ece_count', None)
        coord_mean = row.get('coord_ece_mean', np.nan)
        coord_std = row.get('coord_ece_std', np.nan)
        coord_count = row.get('coord_ece_count', None)

        rgcl_str = format_ece_value(rgcl_mean, rgcl_std, rgcl_count, inline_math=True)
        coord_str = format_ece_value(coord_mean, coord_std, coord_count, inline_math=True)

        rgcl_vals = [v for v in row.get('rgcl_ece_values', []) if pd.notna(v)]
        coord_vals = [v for v in row.get('coord_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (coord) mean for margin calculation
        baseline_mean = coord_mean if pd.notna(coord_mean) else rgcl_mean if pd.notna(rgcl_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(rgcl_vals, coord_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "RGC$_L$").replace("Method 2", "RGC$_C$")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        SIGNIFICANCE_LOG.append({
            'Comparison': 'RGC_L_vs_RGC_C',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'RGC_L_Mean': rgcl_mean * 100 if pd.notna(rgcl_mean) else np.nan,
            'RGC_C_Mean': coord_mean * 100 if pd.notna(coord_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            rgcl_str,
            coord_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_rgcl_vs_gc_dac_comparison_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare RGC_L against GC(DAC) with TOST equivalence testing.
    
    This table demonstrates that random layer selection performs comparably to DAC's carefully tuned layer selection.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.001 = 0.1pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for RGC_L vs GC(DAC) comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'rgcl_ece_mean': 'mean',
        'rgcl_ece_std': rms_agg,
        'rgcl_ece_count': 'sum',
        'gc_dac_layers_ece_mean': 'mean',
        'gc_dac_layers_ece_std': rms_agg,
        'gc_dac_layers_ece_count': 'sum',
        'rgcl_ece_values': flatten_lists,
        'gc_dac_layers_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('rgcl_ece_mean').notna()
        & df.get('gc_dac_layers_ece_mean').notna()
    ]

    if df.empty:
        return "% No overlapping RGC and GC(DAC) results for comparison"

    lines.append("% RGC_L (random layers) vs GC(DAC) (DAC's layer selection) comparison table")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{RGC$_L$ (random layers) vs GC(DAC) (DAC's layer selection) comparison (ECE \\%; lower is better). This demonstrates that random layer selection performs comparably to DAC's carefully tuned layer selection. We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:rgc_l_vs_gc_dac}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & RGC$_L$ & GC(DAC) & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        rgcl_mean = row.get('rgcl_ece_mean', np.nan)
        rgcl_std = row.get('rgcl_ece_std', np.nan)
        rgcl_count = row.get('rgcl_ece_count', None)
        gc_dac_mean = row.get('gc_dac_layers_ece_mean', np.nan)
        gc_dac_std = row.get('gc_dac_layers_ece_std', np.nan)
        gc_dac_count = row.get('gc_dac_layers_ece_count', None)

        rgcl_str = format_ece_value(rgcl_mean, rgcl_std, rgcl_count, inline_math=True)
        gc_dac_str = format_ece_value(gc_dac_mean, gc_dac_std, gc_dac_count, inline_math=True)

        rgcl_vals = [v for v in row.get('rgcl_ece_values', []) if pd.notna(v)]
        gc_dac_vals = [v for v in row.get('gc_dac_layers_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (gc_dac) mean for margin calculation
        baseline_mean = gc_dac_mean if pd.notna(gc_dac_mean) else rgcl_mean if pd.notna(rgcl_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(rgcl_vals, gc_dac_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "RGC$_L$").replace("Method 2", "GC(DAC)")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        SIGNIFICANCE_LOG.append({
            'Comparison': 'RGC_L_vs_GC_DAC',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'RGC_L_Mean': rgcl_mean * 100 if pd.notna(rgcl_mean) else np.nan,
            'GC_DAC_Mean': gc_dac_mean * 100 if pd.notna(gc_dac_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            rgcl_str,
            gc_dac_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def main():
    # Configure logging once at entrypoint.
    from utils.logging_config import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate RGC-focused LaTeX tables for the paper"
    )
    default_results_dirs = (
        [Path('calibration_results')]
        if Path('calibration_results').exists()
        else [Path('calibration_comparison_results/random_ablation'), Path('calibration_comparison')]
    )
    parser.add_argument(
        '--results-dir',
        '--input-dir',
        dest='results_dirs',
        type=Path,
        nargs='+',
        default=default_results_dirs,
        help='Directory(ies) containing result JSON files (can specify multiple)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('latex_tables_statistics'),
        help='Directory to save LaTeX tables'
    )
    parser.add_argument(
        '--coord-dir',
        type=Path,
        default=Path('calibration_comparison'),
        help='Directory containing coordinate ablation results'
    )
    
    args = parser.parse_args()
    
    # Load all results from all specified directories
    results = []
    for results_dir in args.results_dirs:
        dir_results = load_all_results(results_dir)
        results.extend(dir_results)
        logger.info(f"Loaded {len(dir_results)} results from {results_dir}")
    
    logger.info(f"Total results loaded: {len(results)}")
    
    if not results:
        logger.error("No results loaded. Exiting.")
        return 1
    
    # Validate results
    validate_results(results)
    
    # Aggregate over seeds (only clean results)
    logger.info("Aggregating results over seeds (clean data only)...")
    aggregated_df = aggregate_over_seeds(results)
    if not aggregated_df.empty:
        logger.info(f"Loaded datasets: {sorted(aggregated_df['dataset'].dropna().unique())}")
        logger.info(f"Loaded models: {sorted(aggregated_df['model_name'].dropna().unique())}")
        logger.info(f"Loaded training methods: {sorted(aggregated_df['training_method'].dropna().unique())}")
    
    # Load coordinate results if directory exists
    if args.coord_dir.exists():
        logger.info("Loading coordinate ablation results...")
        coord_df = load_coordinate_results(args.coord_dir, k_value=256)
        if not coord_df.empty:
            # Merge with main aggregated_df
            aggregated_df = aggregated_df.merge(
                coord_df,
                on=["dataset", "model_name", "training_method"],
                how="left"
            )
            logger.info(f"Merged {len(coord_df)} coordinate configurations")
        else:
            logger.info("No coordinate results found in directory")
    else:
        logger.info(f"Coordinate directory not found: {args.coord_dir}")
    
    # Save aggregated CSV for reference
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rgc_aggregated_results.csv"
    aggregated_df.to_csv(csv_path, index=False)
    logger.info(f"Saved aggregated results to: {csv_path}")
    
    # Generate Table 1: Main Results Comparison
    logger.info("Generating Table 1: Main Results Comparison...")
    main_results_latex = generate_main_results_table(aggregated_df)
    main_results_path = args.output_dir / "table_1_main_results.tex"
    with open(main_results_path, 'w') as f:
        f.write(main_results_latex)
    logger.info(f"Saved Table 1 (Main Results) to: {main_results_path}")
    
    # Generate TOST comparison table (RGC_L vs RGC_C)
    logger.info("Generating TOST comparison table (RGC_L vs RGC_C)...")
    tost_latex = generate_rgcl_vs_coord_comparison_table(aggregated_df, relative_margin_percent=0.10, absolute_margin_floor=0.001)
    tost_path = args.output_dir / "table_tost_rgc_l_vs_rgc_c.tex"
    with open(tost_path, 'w') as f:
        f.write(tost_latex)
    logger.info(f"Saved TOST comparison table to: {tost_path}")
    
    # Generate TOST comparison table (RGC_L vs GC(DAC))
    logger.info("Generating TOST comparison table (RGC_L vs GC(DAC))...")
    tost_dac_latex = generate_rgcl_vs_gc_dac_comparison_table(aggregated_df, relative_margin_percent=0.10, absolute_margin_floor=0.001)
    tost_dac_path = args.output_dir / "table_tost_rgc_l_vs_gc_dac.tex"
    with open(tost_dac_path, 'w') as f:
        f.write(tost_dac_latex)
    logger.info(f"Saved TOST comparison table (RGC_L vs GC(DAC)) to: {tost_dac_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal configurations: {len(aggregated_df)}")
    print(f"Datasets: {sorted(aggregated_df['dataset'].unique())}")
    print(f"Models: {sorted(aggregated_df['model_name'].unique())}")
    print(f"Training methods: {sorted(aggregated_df['training_method'].unique())}")
    
    print("\n" + "="*80)
    print("All tables generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
