#!/usr/bin/env python3
"""
Generate the paper's computational-efficiency figures (Figures 4 and 5).

- Figure 4: Preprocessing time comparison (RGCL, RGCC, DAC, Fast Separation).
- Figure 5: Throughput (confidence estimates per second).

Reads the per-config calibration JSON files produced by run_rgc_experiments.py
and produces the publication-quality plots used in the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Test set sizes for throughput calculation
TEST_SET_SIZES = {
    "cifar10": 10000,
    "cifar100": 10000,
    "svhn": 26032,
    "tiny_imagenet": 10000,
    "tinyimagenet": 10000,
    "tiny-imagenet": 10000,
}

# =============================================================================
# Configuration
# =============================================================================

# Input/output paths
TIMING_CSV = Path("calibration_results/timing_aggregated.csv")
OUTPUT_DIR = Path("figures")

# Method names (adjust based on your CSV)
RGCL_METHOD = "RGCL (RGC global random)"
RGCC_METHOD = "RGCC (coordinate sampling)"
BASELINE_METHOD = "DAC_Orig"
GEOMETRIC_PIXEL_METHOD = "Geometric Pixel"

METHOD_SECTION_GROUPS = [
    (RGCL_METHOD, [("top", "rgcl")]),
    (RGCC_METHOD, [("top", "rgcc")]),
    (BASELINE_METHOD, [("standard_baselines", "density_aware_calibration")]),
    (GEOMETRIC_PIXEL_METHOD, [("standard_baselines", "geometric_physical_space")]),
]

# Dataset abbreviations
DATASET_ABBREV = {
    "cifar10": "C10",
    "cifar100": "C100",
    "tiny_imagenet": "TINY",
    "tinyimagenet": "TINY",
    "tiny-imagenet": "TINY",  # Hyphenated variation
}

# Model abbreviations  
MODEL_ABBREV = {
    "densenet121": "DN121",
    "resnet18": "R18",
    "resnet50": "R50",
    "resnet101": "R101",
    "resnet152": "R152",
}

# Colors - using a professional, accessible palette
COLOR_RGCL = "#2171b5"   # Strong blue
COLOR_RGCC = "#fd8d3c"   # Vibrant orange
COLOR_DAC = "#cb181d"    # Strong red for baseline
COLOR_GEOMETRIC_PIXEL = "#6a3d9a"  # Purple for geometric pixel

# =============================================================================
# Publication-Quality Styling
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for IJCAI-style figures with large fonts and compact layout."""
    
    # Check for available serif fonts (IJCAI uses serif fonts like Times)
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    preferred_fonts = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Nimbus Roman']
    
    selected_font = 'DejaVu Serif'  # fallback
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    plt.rcParams.update({
        # Font settings - IJCAI: body text is 10pt, so figure fonts should be 9-11pt
        'font.family': 'serif',
        'font.serif': [selected_font, 'DejaVu Serif', 'Times New Roman'],
        'font.size': 11,  # Increased from 8
        'axes.labelsize': 11,  # Increased from 9 - matches body text
        'axes.titlesize': 12,  # Increased from 10 - slightly larger for titles
        'xtick.labelsize': 10,  # Increased from 7 - close to body text
        'ytick.labelsize': 10,  # Increased from 7
        'legend.fontsize': 10,  # Increased from 7 - matches body text
        'figure.titlesize': 12,  # Increased from 11
        
        # Line and edge settings for sharpness (slightly thicker for visibility)
        'axes.linewidth': 1.0,  # Increased from 0.8
        'grid.linewidth': 0.6,  # Increased from 0.5
        'lines.linewidth': 1.2,  # Increased from 1.0
        'patch.linewidth': 0.6,  # Increased from 0.5
        
        # Output quality
        'figure.dpi': 150,
        'savefig.dpi': 600,  # High DPI for sharp output
        'pdf.fonttype': 42,  # TrueType fonts in PDF
        'ps.fonttype': 42,
        
        # Remove spines clutter
        'axes.spines.top': True,
        'axes.spines.right': True,
        
        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        
        # Legend settings
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        
        # Tick settings
        'xtick.major.width': 1.0,  # Increased from 0.8
        'ytick.major.width': 1.0,  # Increased from 0.8
        'xtick.minor.width': 0.6,  # Increased from 0.5
        'ytick.minor.width': 0.6,  # Increased from 0.5
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })

# =============================================================================
# Data Loading
# =============================================================================

def normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset name to handle variations (tiny_imagenet, tinyimagenet, tiny-imagenet, etc.)."""
    if not dataset:
        return dataset
    normalized = str(dataset).lower().replace('-', '_').replace(' ', '_')
    # Map common variations to canonical form for lookup
    if normalized in ['tiny_imagenet', 'tinyimagenet']:
        return 'tiny_imagenet'
    return normalized


def normalize_training_method(training_method: Any) -> str:
    """Return the canonical training-method name used by table/timing summaries."""
    if training_method is None:
        return 'unknown'
    training_method = str(training_method)
    if training_method == 'baseline':
        return 'baseline_cross_entropy'
    return training_method


def _as_float(value: Any) -> Optional[float]:
    """Convert a JSON numeric value to float, returning None for missing/non-numeric values."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _first_float(section: Dict[str, Any], keys: List[str]) -> Optional[float]:
    """Return the first finite numeric field present in a method section."""
    for key in keys:
        if key in section:
            value = _as_float(section.get(key))
            if value is not None:
                return value
    return None


def extract_timing(section: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract only explicit timing values from a method section.

    Offline timing is pre-test setup. For the current JSON shape this is
    extraction plus fit when both component fields are present. Total timing is
    an explicit total when available, otherwise a sum of real component fields.
    """
    if not isinstance(section, dict):
        return {}

    timing: Dict[str, float] = {}

    total_s = _first_float(section, [
        'timing_seconds',
        'timing_total_seconds',
        'total_time_seconds',
        'total_seconds',
        'elapsed_seconds',
    ])

    explicit_offline_s = _first_float(section, [
        'offline_time_seconds',
        'preprocessing_time_seconds',
        'feature_extraction_time_seconds',
    ])
    extraction_s = _first_float(section, ['extraction_time_s'])
    fit_s = _first_float(section, ['fit_time_s'])
    init_s = _first_float(section, ['timing_init_seconds', 'init_time_seconds'])
    timing_fit_s = _first_float(section, ['timing_fit_seconds'])

    offline_components: List[float] = []
    if explicit_offline_s is not None:
        offline_s = explicit_offline_s
        offline_components.append(explicit_offline_s)
    elif extraction_s is not None:
        offline_s = extraction_s + fit_s if fit_s is not None else extraction_s
        offline_components.append(extraction_s)
        if fit_s is not None:
            offline_components.append(fit_s)
    else:
        if init_s is not None:
            offline_components.append(init_s)
        if timing_fit_s is not None:
            offline_components.append(timing_fit_s)
        offline_s = sum(offline_components) if offline_components else None

    if offline_s is not None:
        timing['offline_s'] = offline_s

    online_ms = _first_float(section, ['online_mean_ms'])
    online_seconds = _first_float(section, [
        'online_time_seconds',
        'inference_time_seconds',
        'calibration_time_seconds',
        'calibrate_time_s',
        'timing_calibrate_seconds',
    ])
    if online_ms is None and online_seconds is not None:
        online_ms = online_seconds * 1000.0
    if online_ms is not None:
        timing['online_ms'] = online_ms

    if total_s is None:
        total_components = list(offline_components)
        if online_seconds is not None:
            total_components.append(online_seconds)
        elif online_ms is not None:
            total_components.append(online_ms / 1000.0)
        total_s = sum(total_components) if total_components else None
    if total_s is not None:
        timing['total_s'] = total_s

    return timing


def _get_method_section(result: Dict[str, Any], location: str, key: str) -> Optional[Dict[str, Any]]:
    """Fetch a top-level or standard-baseline method section from a result JSON."""
    if location == 'top':
        section = result.get(key)
    elif location == 'standard_baselines':
        section = (result.get('standard_baselines') or {}).get(key)
    else:
        section = None
    return section if isinstance(section, dict) and not section.get('skipped', False) else None


def _section_label(location: str, key: str) -> str:
    return key if location == 'top' else f"{location}.{key}"


def build_timing_data_from_results(input_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the timing CSV data from real nested paper_results.json files."""
    audit = {
        'json_files_found': 0,
        'loaded_json_count': 0,
        'malformed_json_count': 0,
        'extracted_rows_per_method': Counter(),
        'skipped_missing_timing': Counter(),
    }
    rows: List[Dict[str, Any]] = []

    result_files = sorted(input_dir.rglob('paper_results.json'))
    audit['json_files_found'] = len(result_files)

    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except Exception as exc:
            audit['malformed_json_count'] += 1
            print(f"Warning: failed to load {result_file}: {exc}")
            continue

        audit['loaded_json_count'] += 1
        config = result.get('experiment_config', result.get('experiment_info', {}))
        dataset = config.get('dataset')
        model = config.get('model_name')
        seed = config.get('seed')
        training_method = normalize_training_method(config.get('training_method'))

        if not dataset or not model:
            print(f"Warning: skipping {result_file}; missing dataset or model_name in experiment_config")
            continue

        for method_name, section_specs in METHOD_SECTION_GROUPS:
            for location, section_key in section_specs:
                section = _get_method_section(result, location, section_key)
                if section is None:
                    continue

                timing = extract_timing(section)
                section_label = _section_label(location, section_key)
                if not timing:
                    audit['skipped_missing_timing'][section_label] += 1
                    continue

                rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Method': method_name,
                    'training_method': training_method,
                    'seed': seed,
                    'offline_s': timing.get('offline_s', np.nan),
                    'online_ms': timing.get('online_ms', np.nan),
                    'total_s': timing.get('total_s', np.nan),
                })
                audit['extracted_rows_per_method'][method_name] += 1
                break

    if not rows:
        return pd.DataFrame(columns=[
            'Dataset',
            'Model',
            'Method',
            'offline_mean_s',
            'offline_std_s',
            'online_mean_ms',
            'online_std_ms',
            'total_mean_s',
            'total_std_s',
            'n_seeds',
        ]), audit

    raw_df = pd.DataFrame(rows)

    def count_seeds(values: pd.Series) -> int:
        non_missing = [v for v in values if pd.notna(v)]
        return len(set(non_missing)) if non_missing else len(values)

    aggregated = raw_df.groupby(['Dataset', 'Model', 'Method']).agg(
        offline_mean_s=('offline_s', 'mean'),
        offline_std_s=('offline_s', 'std'),
        online_mean_ms=('online_ms', 'mean'),
        online_std_ms=('online_ms', 'std'),
        total_mean_s=('total_s', 'mean'),
        total_std_s=('total_s', 'std'),
        n_seeds=('seed', count_seeds),
    ).reset_index()

    return aggregated, audit


def load_timing_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess timing data from CSV."""
    df = pd.read_csv(csv_path)
    
    # Standardize column names (handle variations)
    df.columns = df.columns.str.strip()
    
    # Add abbreviations with robust normalization
    df['dataset_abbrev'] = df['Dataset'].apply(
        lambda x: DATASET_ABBREV.get(normalize_dataset_name(x), str(x).upper()[:4])
    )
    df['model_abbrev'] = df['Model'].apply(
        lambda x: MODEL_ABBREV.get(str(x).lower(), str(x))
    )
    
    return df


def create_sample_data() -> pd.DataFrame:
    """
    Create demo-only sample data matching the plotting schema.

    These estimated values are not paper-valid and are used only when the user
    explicitly passes --allow-sample-data.
    """
    data = []
    
    # Demo-only values estimated from earlier timing plots.
    # Format: (Dataset, Model, RGCL_feat, RGCC_feat, RGCL_calib, RGCC_calib, DAC_total)
    # Note: Calibration times are similar for both methods
    
    timing_data = [
        # CIFAR-10 (all 5 models)
        ('cifar10', 'densenet121', 550, 22, 15, 8, 25),
        ('cifar10', 'resnet18', 320, 6, 8, 5, 33),
        ('cifar10', 'resnet50', 1100, 18, 22, 22, 17),
        ('cifar10', 'resnet101', 1250, 32, 22, 24, 62),
        ('cifar10', 'resnet152', 1300, 45, 22, 24, 50),
        
        # CIFAR-100 (all 5 models)
        ('cifar100', 'densenet121', 580, 28, 15, 10, 26),
        ('cifar100', 'resnet18', 340, 8, 10, 6, 35),
        ('cifar100', 'resnet50', 1150, 38, 25, 24, 22),
        ('cifar100', 'resnet101', 1300, 55, 26, 26, 65),
        ('cifar100', 'resnet152', 1350, 65, 27, 28, 55),
        
        # Tiny-ImageNet (only R50, R101, R152 based on the paper's figures)
        ('tiny_imagenet', 'resnet50', 2500, 80, 34, 34, None),
        ('tiny_imagenet', 'resnet101', 2800, 145, 36, 36, None),
        ('tiny_imagenet', 'resnet152', 3200, 190, 38, 38, None),
    ]
    
    for dataset, model, rgcl_feat, rgcc_feat, rgcl_calib, rgcc_calib, dac_total in timing_data:
        # RGCL entry
        rgcl_total = rgcl_feat + rgcl_calib
        data.append({
            'Dataset': dataset,
            'Model': model,
            'Method': RGCL_METHOD,
            'offline_mean_s': rgcl_feat,
            'offline_std_s': rgcl_feat * 0.03,
            'online_mean_ms': rgcl_calib * 1000,
            'online_std_ms': rgcl_calib * 30,
            'total_mean_s': rgcl_total,
            'total_std_s': rgcl_total * 0.03,
        })
        
        # RGCC entry
        rgcc_total = rgcc_feat + rgcc_calib
        data.append({
            'Dataset': dataset,
            'Model': model,
            'Method': RGCC_METHOD,
            'offline_mean_s': rgcc_feat,
            'offline_std_s': rgcc_feat * 0.05,
            'online_mean_ms': rgcc_calib * 1000,
            'online_std_ms': rgcc_calib * 50,
            'total_mean_s': rgcc_total,
            'total_std_s': rgcc_total * 0.05,
        })
        
        # DAC entry (if available - only for CIFAR datasets based on paper)
        if dac_total is not None:
            data.append({
                'Dataset': dataset,
                'Model': model,
                'Method': BASELINE_METHOD,
                'offline_mean_s': dac_total * 0.75,
                'offline_std_s': dac_total * 0.03,
                'online_mean_ms': dac_total * 250,
                'online_std_ms': dac_total * 15,
                'total_mean_s': dac_total,
                'total_std_s': dac_total * 0.04,
            })
    
    df = pd.DataFrame(data)
    
    # Add abbreviations
    df['dataset_abbrev'] = df['Dataset'].apply(lambda x: DATASET_ABBREV.get(x, x.upper()))
    df['model_abbrev'] = df['Model'].apply(lambda x: MODEL_ABBREV.get(x, x))
    
    return df


# =============================================================================
# Plotting Functions
# =============================================================================

def get_plot_positions(df: pd.DataFrame, datasets: List[str], models: List[str]):
    """
    Calculate x-positions for grouped bar charts.
    Returns positions and group separators.
    """
    positions = []
    labels = []
    group_centers = []
    group_labels = []
    
    current_pos = 0
    gap_between_groups = 0.8
    
    for dataset in datasets:
        dataset_data = df[df['dataset_abbrev'] == dataset]
        available_models = [m for m in models if m in dataset_data['model_abbrev'].values]
        
        if not available_models:
            continue
            
        group_start = current_pos
        for model in available_models:
            positions.append(current_pos)
            labels.append(model)
            current_pos += 1
        
        group_end = current_pos - 1
        group_centers.append((group_start + group_end) / 2)
        group_labels.append(dataset)
        current_pos += gap_between_groups
    
    return positions, labels, group_centers, group_labels


def plot_timing_comparison(ax, df: pd.DataFrame, time_col: str, std_col: str,
                          title: str, ylabel: str, show_dac: bool = False,
                          show_geometric_pixel: bool = False,
                          show_legend: bool = True, legend_loc: str = 'upper right',
                          show_ylabel: bool = True):
    """
    Create a single timing comparison subplot with clean, publication-quality style.
    
    Parameters:
    -----------
    ax : matplotlib axis
    df : DataFrame with timing data
    time_col : column name for mean time values
    std_col : column name for std values
    title : subplot title
    ylabel : y-axis label
    show_dac : whether to include DAC baseline
    show_geometric_pixel : whether to include Geometric Pixel calibration
    show_legend : whether to show legend
    """
    datasets = ['C10', 'C100', 'TINY']
    models = ['DN121', 'R18', 'R50', 'R101', 'R152']
    
    # Filter data
    methods = [RGCL_METHOD, RGCC_METHOD]
    if show_dac:
        methods.append(BASELINE_METHOD)
    if show_geometric_pixel:
        methods.append(GEOMETRIC_PIXEL_METHOD)
    
    df_filtered = df[df['Method'].isin(methods)]
    
    # Build position mapping with proper grouping
    positions = []
    labels = []
    group_boundaries = [0]  # Track where each dataset group ends
    group_centers = []
    group_labels = []
    
    current_pos = 0
    gap_between_groups = 1.0  # Gap between dataset groups
    
    for dataset in datasets:
        dataset_data = df_filtered[df_filtered['dataset_abbrev'] == dataset]
        available_models = [m for m in models if m in dataset_data['model_abbrev'].values]
        
        if not available_models:
            continue
        
        group_start = current_pos
        for model in available_models:
            positions.append(current_pos)
            labels.append(model)
            current_pos += 1
        
        group_end = current_pos - 1
        group_centers.append((group_start + group_end) / 2)
        group_labels.append(dataset)
        group_boundaries.append(current_pos - 0.5 + gap_between_groups / 2)
        current_pos += gap_between_groups
    
    if not positions:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes)
        return
    
    # Bar width calculation
    n_methods = 2
    if show_dac:
        n_methods += 1
    if show_geometric_pixel:
        n_methods += 1
    bar_width = 0.7 / n_methods
    
    x = np.array(positions)
    
    # Extract values for each method
    def get_values(method_name):
        values, errors = [], []
        for dataset in datasets:
            dataset_data = df_filtered[df_filtered['dataset_abbrev'] == dataset]
            available_models = [m for m in models if m in dataset_data['model_abbrev'].values]
            for model in available_models:
                row = dataset_data[(dataset_data['model_abbrev'] == model) & 
                                   (dataset_data['Method'] == method_name)]
                if len(row) > 0:
                    val = row[time_col].iloc[0]
                    std = row[std_col].iloc[0] if std_col in row.columns and pd.notna(row[std_col].iloc[0]) else 0
                    values.append(val if pd.notna(val) else np.nan)
                    errors.append(std if pd.notna(std) else 0)
                else:
                    values.append(np.nan)
                    errors.append(0)
        return values, errors
    
    # Plot bars with clean styling
    rgcl_vals, rgcl_errs = get_values(RGCL_METHOD)
    rgcc_vals, rgcc_errs = get_values(RGCC_METHOD)
    
    # Filter out positions where only one method has data (not a meaningful comparison)
    # Count how many methods have valid (non-NaN) data at each position
    valid_counts = []
    for i in range(len(rgcl_vals)):
        count = 0
        if pd.notna(rgcl_vals[i]):
            count += 1
        if pd.notna(rgcc_vals[i]):
            count += 1
        if show_dac:
            dac_vals, _ = get_values(BASELINE_METHOD)
            if pd.notna(dac_vals[i]):
                count += 1
        if show_geometric_pixel:
            geo_pixel_vals, _ = get_values(GEOMETRIC_PIXEL_METHOD)
            if pd.notna(geo_pixel_vals[i]):
                count += 1
        valid_counts.append(count)
    
    # Only keep positions where at least 2 methods have data
    keep_mask = [count >= 2 for count in valid_counts]
    
    if not any(keep_mask):
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes)
        return
    
    # Filter positions, values, and labels - only keep where keep_mask is True
    keep_indices = [i for i in range(len(keep_mask)) if keep_mask[i]]
    x_filtered = x[keep_mask]
    rgcl_vals_filtered = [rgcl_vals[i] for i in keep_indices]
    rgcl_errs_filtered = [rgcl_errs[i] for i in keep_indices]
    rgcc_vals_filtered = [rgcc_vals[i] for i in keep_indices]
    rgcc_errs_filtered = [rgcc_errs[i] for i in keep_indices]
    labels_filtered = [labels[i] for i in keep_indices]
    
    # Update positions for filtered data
    positions_filtered = [positions[i] for i in keep_indices]
    
    error_kw = {'elinewidth': 0.7, 'capthick': 0.7}
    
    # Get values for all methods (needed for filtering)
    dac_vals, dac_errs = get_values(BASELINE_METHOD) if show_dac else ([], [])
    geo_pixel_vals, geo_pixel_errs = get_values(GEOMETRIC_PIXEL_METHOD) if show_geometric_pixel else ([], [])
    
    # Filter all value arrays - only keep where keep_mask is True
    if show_dac:
        dac_vals_filtered = [dac_vals[i] for i in keep_indices]
        dac_errs_filtered = [dac_errs[i] for i in keep_indices]
    if show_geometric_pixel:
        geo_pixel_vals_filtered = [geo_pixel_vals[i] for i in keep_indices]
        geo_pixel_errs_filtered = [geo_pixel_errs[i] for i in keep_indices]
    
    # Calculate offsets based on number of methods
    if n_methods == 2:
        # Only RGCL and RGCC
        offset = bar_width / 2
        ax.bar(x_filtered - offset, rgcl_vals_filtered, bar_width, yerr=rgcl_errs_filtered,
               label=r'RGC$_L$', color=COLOR_RGCL, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
        ax.bar(x_filtered + offset, rgcc_vals_filtered, bar_width, yerr=rgcc_errs_filtered,
               label=r'RGC$_C$', color=COLOR_RGCC, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
    elif n_methods == 3:
        # RGCL, RGCC, and one additional (DAC or Geometric Pixel)
        offset = bar_width
        ax.bar(x_filtered - offset, rgcl_vals_filtered, bar_width, yerr=rgcl_errs_filtered,
               label=r'RGC$_L$', color=COLOR_RGCL, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
        ax.bar(x_filtered, rgcc_vals_filtered, bar_width, yerr=rgcc_errs_filtered,
               label=r'RGC$_C$', color=COLOR_RGCC, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
        if show_dac:
            ax.bar(x_filtered + offset, dac_vals_filtered, bar_width, yerr=dac_errs_filtered,
                   label='DAC', color=COLOR_DAC, edgecolor='none',
                   capsize=1.5, error_kw=error_kw)
        elif show_geometric_pixel:
            ax.bar(x_filtered + offset, geo_pixel_vals_filtered, bar_width, yerr=geo_pixel_errs_filtered,
                   label='Fast Separation', color=COLOR_GEOMETRIC_PIXEL, edgecolor='none',
                   capsize=1.5, error_kw=error_kw)
    else:
        # n_methods == 4: RGCL, RGCC, DAC, and Geometric Pixel
        # Position bars: RGCL, RGCC, DAC, Fast Separation (evenly spaced)
        spacing = bar_width
        ax.bar(x_filtered - 1.5 * spacing, rgcl_vals_filtered, bar_width, yerr=rgcl_errs_filtered,
               label=r'RGC$_L$', color=COLOR_RGCL, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
        ax.bar(x_filtered - 0.5 * spacing, rgcc_vals_filtered, bar_width, yerr=rgcc_errs_filtered,
               label=r'RGC$_C$', color=COLOR_RGCC, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
        ax.bar(x_filtered + 0.5 * spacing, dac_vals_filtered, bar_width, yerr=dac_errs_filtered,
               label='DAC', color=COLOR_DAC, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
        ax.bar(x_filtered + 1.5 * spacing, geo_pixel_vals_filtered, bar_width, yerr=geo_pixel_errs_filtered,
               label='Fast Separation', color=COLOR_GEOMETRIC_PIXEL, edgecolor='none',
               capsize=1.5, error_kw=error_kw)
    
    # Configure y-axis (log scale)
    ax.set_yscale('log')
    # Only show ylabel if requested (for shared y-axis, only show on leftmost subplot)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11)  # Increased from 8 to match IJCAI guidelines
    
    # Title - only show if provided (empty string means no title)
    if title:
        ax.set_title(title, fontweight='bold', fontsize=12, pad=3)  # Increased from 9
    
    # X-axis labels (model names) - rotated for readability, pushed down to avoid dataset labels
    ax.set_xticks(x_filtered)
    ax.set_xticklabels(labels_filtered, rotation=45, ha='right', fontsize=10)  # Increased from 7
    ax.tick_params(axis='x', which='major', pad=8)  # Increased padding to push model names down
    
    # Set x-axis limits with padding
    if len(positions_filtered) > 0:
        ax.set_xlim(min(positions_filtered) - 0.8, max(positions_filtered) + 0.8)
    else:
        ax.set_xlim(-1, 1)
    
    # Recalculate group boundaries and centers based on filtered positions
    # Map original positions to their dataset groups
    position_to_dataset = {}
    for dataset_idx, dataset_label in enumerate(group_labels):
        # Find the range of positions for this dataset
        if dataset_idx == 0:
            start_idx = 0
        else:
            # Find where previous dataset ended
            prev_center = group_centers[dataset_idx - 1]
            start_idx = next((i for i, p in enumerate(positions) if p > prev_center), len(positions))
        
        if dataset_idx < len(group_centers) - 1:
            end_idx = next((i for i, p in enumerate(positions) if p > group_centers[dataset_idx]), len(positions))
        else:
            end_idx = len(positions)
        
        for i in range(start_idx, end_idx):
            position_to_dataset[i] = dataset_label
    
    # Build filtered groups
    group_boundaries_filtered = []
    group_centers_filtered = []
    group_labels_filtered = []
    
    current_dataset = None
    current_group_positions = []
    
    for i, pos in enumerate(positions):
        if not keep_mask[i]:
            continue
        
        dataset = position_to_dataset.get(i)
        if dataset != current_dataset:
            if current_dataset is not None and len(current_group_positions) > 0:
                # Close previous group
                group_centers_filtered.append(np.mean(current_group_positions))
                group_labels_filtered.append(current_dataset)
                if len(group_boundaries_filtered) > 0:
                    group_boundaries_filtered.append((group_boundaries_filtered[-1] + current_group_positions[0]) / 2)
                else:
                    group_boundaries_filtered.append(current_group_positions[0] - 0.5)
            
            # Start new group
            current_dataset = dataset
            current_group_positions = [pos]
        else:
            current_group_positions.append(pos)
    
    # Close last group
    if current_dataset is not None and len(current_group_positions) > 0:
        group_centers_filtered.append(np.mean(current_group_positions))
        group_labels_filtered.append(current_dataset)
    
    # Add vertical separator lines between dataset groups
    for sep_x in group_boundaries_filtered[1:]:  # Skip first boundary
        ax.axvline(x=sep_x, color='#888888', linestyle='-', linewidth=0.8, alpha=0.8,
                   zorder=0)
    
    # Add dataset labels BELOW the x-axis using a secondary x-axis approach
    # Note: x-axis tick padding is set earlier to push model names down
    
    # Use text below the axis for dataset labels - push down further to avoid model names
    # Position lowered by ~2 pixels (figure height 3.2in, bottom margin 0.28, so plot area ~2.3in = ~690px at 300dpi)
    # 2 pixels / 690px ~ 0.003 in axes coordinates
    trans = ax.get_xaxis_transform()
    for center, label in zip(group_centers_filtered, group_labels_filtered):
        ax.text(center, -0.33, label, transform=trans,
                ha='center', va='top', fontsize=10, fontweight='bold')  # Lowered from -0.30 by ~2 pixels
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Grid - only horizontal lines, subtle
    ax.yaxis.grid(True, linestyle='-', alpha=0.25, which='major', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, which='minor', linewidth=0.3)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    # Legend - positioned on top, horizontal layout (one row)
    if show_legend:
        # Count number of methods to determine ncol
        n_methods_legend = 2  # RGCL and RGCC always present
        if show_dac:
            n_methods_legend += 1
        if show_geometric_pixel:
            n_methods_legend += 1
        
        # Place legend on top, horizontal layout
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), 
                       ncol=n_methods_legend, framealpha=0.95, edgecolor='0.7',
                       fontsize=10, handlelength=1.2, handletextpad=0.4,
                       borderpad=0.3, columnspacing=0.8)
        leg.get_frame().set_linewidth(0.6)  # Increased from 0.5


def create_preprocessing_figure(df: pd.DataFrame, output_path: Path):
    """
    Create figure for preprocessing time (in seconds).
    IJCAI style: smaller figure size with large fonts for readability.
    """
    setup_publication_style()
    
    # Create single subplot figure - smaller size for IJCAI (column width ~3.25 inches)
    # Using 5 inches width to allow for margins when inserted
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))  # Reduced from (6, 4.2)
    
    # Adjust spacing - leave room on top for legend, more bottom for labels
    plt.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.28)  # More top margin for legend, more bottom for labels
    
    # Prepare data columns - Feature extraction time (offline/preprocessing)
    if 'offline_mean_s' in df.columns:
        feat_col, feat_std = 'offline_mean_s', 'offline_std_s'
    else:
        feat_col, feat_std = 'feature_extraction_s', 'feature_extraction_std_s'
    
    # Pre processing - RGCL vs RGCC vs DAC vs Geometric Pixel
    plot_timing_comparison(
        ax, df,
        feat_col, feat_std,
        '',  # No title - will be in caption
        'Seconds (log scale)',
        show_dac=True,
        show_geometric_pixel=True,
        show_legend=True,
        legend_loc='upper left',
        show_ylabel=True
    )
    
    # Save figure in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # High quality PDF (vector graphics, ideal for papers)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf',
                facecolor='white', edgecolor='none')
    
    # High quality PNG for preview
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    
    return fig


def create_online_figure(df: pd.DataFrame, output_path: Path):
    """
    Create figure for online time (in samples per second - throughput).
    IJCAI style: smaller figure size with large fonts for readability.
    """
    setup_publication_style()
    
    # Create single subplot figure - smaller size for IJCAI (column width ~3.25 inches)
    # Using 5 inches width to allow for margins when inserted
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))  # Reduced from (6, 4.2)
    
    # Adjust spacing - leave room on top for legend, more bottom for labels
    plt.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.28)  # More top margin for legend, more bottom for labels
    
    # Prepare data columns - Use throughput (samples per second) instead of time
    if 'throughput_mean' in df.columns:
        throughput_col, throughput_std = 'throughput_mean', 'throughput_std'
    else:
        # Compute throughput from online time if not available
        if 'online_mean_ms' in df.columns:
            # Convert ms to seconds, then compute throughput
            df['online_s'] = df['online_mean_ms'] / 1000
            df['online_std_s'] = df['online_std_ms'] / 1000 if 'online_std_ms' in df.columns else 0
            # Compute throughput: samples per second = n_test / time_in_seconds
            # We need n_test for each row - approximate from dataset
            def compute_throughput(row):
                dataset = row.get('Dataset', 'cifar10')
                normalized = str(dataset).lower().replace('-', '_').replace(' ', '_')
                n_test = TEST_SET_SIZES.get(normalized, 10000)
                time_s = row.get('online_s')
                if pd.notna(time_s) and time_s > 0:
                    return n_test / time_s
                return np.nan
            
            def compute_throughput_std(row):
                dataset = row.get('Dataset', 'cifar10')
                normalized = str(dataset).lower().replace('-', '_').replace(' ', '_')
                n_test = TEST_SET_SIZES.get(normalized, 10000)
                time_s = row.get('online_s')
                time_std = row.get('online_std_s', 0)
                if pd.notna(time_s) and time_s > 0 and pd.notna(time_std) and time_std > 0:
                    # Error propagation: if throughput = n_test / time, then
                    # d(throughput) = -n_test * d(time) / time^2
                    # For std: std(throughput) ~ n_test * std(time) / time^2
                    return n_test * time_std / (time_s ** 2)
                return 0.0
            
            df['throughput'] = df.apply(compute_throughput, axis=1)
            df['throughput_std'] = df.apply(compute_throughput_std, axis=1)
            throughput_col, throughput_std = 'throughput', 'throughput_std'
        else:
            # Fallback: use calibration time column if available
            throughput_col, throughput_std = 'calibration_s', 'calibration_std_s'
    
    # Online (throughput) - RGCL vs RGCC vs DAC vs Geometric Pixel
    plot_timing_comparison(
        ax, df,
        throughput_col, throughput_std,
        '',  # No title - will be in caption
        'Ops per second (log scale)',
        show_dac=True,
        show_geometric_pixel=True,
        show_legend=True,
        legend_loc='upper left',
        show_ylabel=True
    )
    
    # Limit y-axis to 10^3 (1000) - higher values will be cut off
    # User will mention in text that DAC was faster for those cases
    ax.set_ylim(top=1e3)
    
    # Save figure in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # High quality PDF (vector graphics, ideal for papers)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf',
                facecolor='white', edgecolor='none')
    
    # High quality PNG for preview
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    
    return fig


def create_end_to_end_figure(df: pd.DataFrame, output_path: Path):
    """
    Create a separate figure for end-to-end time comparison.
    IJCAI style: smaller figure size with large fonts for readability.
    """
    setup_publication_style()
    
    # Create single subplot figure - smaller size for IJCAI (column width ~3.25 inches)
    # Using 5 inches width to allow for margins when inserted
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))  # Reduced from (6, 4.2)
    
    # Adjust spacing - leave room on top for legend, more bottom for labels
    plt.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.28)  # More top margin for legend, more bottom for labels
    
    # Prepare data columns
    if 'total_mean_s' in df.columns:
        total_col, total_std = 'total_mean_s', 'total_std_s'
    else:
        # Compute if not available
        if 'offline_mean_s' in df.columns:
            feat_col = 'offline_mean_s'
        else:
            feat_col = 'feature_extraction_s'
        
        if 'online_mean_ms' in df.columns:
            df['calib_s'] = df['online_mean_ms'] / 1000
            calib_col = 'calib_s'
        else:
            calib_col = 'calibration_s'
        
        df['total_s'] = df[feat_col] + df.get(calib_col, 0)
        total_col, total_std = 'total_s', 'total_std_s'
    
    # End-to-end time - RGCL vs RGCC vs DAC vs Geometric Pixel
    plot_timing_comparison(
        ax, df,
        total_col, total_std,
        '',  # No title - will be in caption
        'Seconds (log scale)',
        show_dac=True,
        show_geometric_pixel=True,
        show_legend=True,
        legend_loc='upper left',
        show_ylabel=True
    )
    
    # Save figure in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # High quality PDF (vector graphics, ideal for papers)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf',
                facecolor='white', edgecolor='none')
    
    # High quality PNG for preview
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    
    return fig


def create_timing_alternative_layout(df: pd.DataFrame, output_path: Path):
    """
    Alternative layout: 1 row of 4 subplots for wider figures.
    Better suited for single-column journal formats.
    """
    setup_publication_style()
    
    # Create 1x4 figure (wide)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    # ... similar logic as above but arranged horizontally
    # This is an alternative if the 2x2 doesn't fit well
    
    pass  # Implement if needed


# =============================================================================
# Main
# =============================================================================

def _column_has_real_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def generate_available_figures(df: pd.DataFrame, output_dir: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate every timing figure whose required real timing column is available."""
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []

    figure_specs = [
        (
            "figure4_preprocessing.pdf",
            "offline_mean_s",
            create_preprocessing_figure,
            "offline_mean_s is missing or all NaN",
        ),
        (
            "figure5_throughput.pdf",
            "online_mean_ms",
            create_online_figure,
            "online_mean_ms is missing or all NaN",
        ),
        (
            "figure4_5_end_to_end.pdf",
            "total_mean_s",
            create_end_to_end_figure,
            "total_mean_s is missing or all NaN",
        ),
    ]

    for filename, required_column, figure_func, skip_reason in figure_specs:
        output_path = output_dir / filename
        if not _column_has_real_data(df, required_column):
            print(f"Skipping {filename}: {skip_reason}")
            skipped.append((filename, skip_reason))
            continue

        figure_func(df.copy(), output_path)
        generated.append(str(output_path))

    return generated, skipped


def print_timing_audit(
    audit: Dict[str, Any],
    generated_figures: List[str],
    skipped_figures: List[Tuple[str, str]],
    csv_path: Optional[Path],
) -> None:
    """Print a compact audit trail for timing extraction and plotting."""
    print("\n" + "=" * 60)
    print("TIMING EXTRACTION AUDIT")
    print("=" * 60)

    if audit.get('source') == 'existing_csv':
        print("Source: existing timing CSV")
        print(f"Loaded CSV rows: {audit.get('loaded_rows', 0)}")
    elif audit.get('source') == 'sample_data':
        print("Source: demo-only sample data")
    else:
        print(f"JSON files found: {audit.get('json_files_found', 0)}")
        print(f"Loaded JSON count: {audit.get('loaded_json_count', 0)}")
        print(f"Malformed JSON count: {audit.get('malformed_json_count', 0)}")

        extracted = audit.get('extracted_rows_per_method', Counter())
        print("Extracted rows per method:")
        if extracted:
            for method, count in sorted(extracted.items()):
                print(f"  {method}: {count}")
        else:
            print("  none")

        skipped_missing = audit.get('skipped_missing_timing', Counter())
        print("Skipped sections due to missing timing fields:")
        if skipped_missing:
            for section, count in sorted(skipped_missing.items()):
                print(f"  {section}: {count}")
        else:
            print("  none")

    print("Generated figures:")
    if generated_figures:
        for figure in generated_figures:
            print(f"  {figure}")
    else:
        print("  none")

    print("Skipped figures:")
    if skipped_figures:
        for figure, reason in skipped_figures:
            print(f"  {figure}: {reason}")
    else:
        print("  none")

    print(f"Final CSV path: {csv_path if csv_path is not None else 'not written'}")
    print("=" * 60)


def main() -> int:
    """Main function to generate timing figures from real result JSONs."""
    parser = argparse.ArgumentParser(
        description="Generate paper timing figures from real calibration result JSON files."
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('calibration_results'),
        help='Directory containing nested paper_results.json files.'
    )
    parser.add_argument(
        '--timing-csv',
        type=Path,
        default=TIMING_CSV,
        help='Path to write/read the aggregated timing CSV.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Directory to save generated timing figures.'
    )
    parser.add_argument(
        '--use-existing-timing-csv',
        action='store_true',
        help='Use --timing-csv instead of rebuilding it from --input-dir.'
    )
    parser.add_argument(
        '--allow-sample-data',
        action='store_true',
        help='Use demo-only hardcoded sample data. Not valid for paper figures.'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Figures 4 and 5 - Timing Comparison")
    print("=" * 60)

    audit: Dict[str, Any]
    csv_path_for_audit: Optional[Path] = args.timing_csv

    if args.allow_sample_data:
        print("\n" + "!" * 72)
        print("WARNING: USING DEMO-ONLY SAMPLE TIMING DATA.")
        print("These figures are NOT paper-valid and must not be used as results.")
        print("!" * 72 + "\n")
        df = create_sample_data()
        audit = {'source': 'sample_data'}
        csv_path_for_audit = None
    elif args.use_existing_timing_csv:
        if not args.timing_csv.exists():
            print(f"Error: requested --use-existing-timing-csv but file does not exist: {args.timing_csv}")
            return 1
        print(f"Loading existing timing data from: {args.timing_csv}")
        df = load_timing_data(args.timing_csv)
        audit = {'source': 'existing_csv', 'loaded_rows': len(df)}
    else:
        print(f"Rebuilding timing CSV from real JSON files under: {args.input_dir}")
        timing_df, audit = build_timing_data_from_results(args.input_dir)
        if timing_df.empty:
            generated_figures: List[str] = []
            skipped_figures: List[Tuple[str, str]] = [
                ("all", "no real timing fields could be extracted from paper_results.json files")
            ]
            print_timing_audit(audit, generated_figures, skipped_figures, args.timing_csv)
            print("Error: no real timing data could be extracted. No figures generated.")
            return 1

        args.timing_csv.parent.mkdir(parents=True, exist_ok=True)
        timing_df.to_csv(args.timing_csv, index=False)
        print(f"Saved real aggregated timing CSV to: {args.timing_csv}")
        df = load_timing_data(args.timing_csv)

    if df.empty:
        print("Error: timing DataFrame is empty. No figures generated.")
        return 1

    # Print data summary
    print(f"\nData summary:")
    print(f"  Methods: {df['Method'].unique().tolist()}")
    print(f"  Datasets: {df['dataset_abbrev'].unique().tolist()}")
    print(f"  Models: {df['model_abbrev'].unique().tolist()}")
    print(f"  Total rows: {len(df)}")

    generated_figures, skipped_figures = generate_available_figures(df, args.output_dir)
    print_timing_audit(audit, generated_figures, skipped_figures, csv_path_for_audit)
    
    print("\n" + "=" * 60)
    if generated_figures:
        print("Timing figure generation completed.")
    else:
        print("No timing figures were generated; see skipped figure reasons above.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
