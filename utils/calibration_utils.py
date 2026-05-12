"""
Calibration utilities for geometric calibration methods.

This module consolidates calibration-related utilities including:
- ECE and other calibration metrics
- UncertaintyMetrics class for standard metrics
- Standard baselines (Temperature Scaling, Isotonic, etc.)
- SPP+JL compression utilities
"""

import numpy as np
import torch
import os
import json
import time
import logging
from typing import Tuple, List, Dict, Optional, Any
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from utils.logging_config import get_logger
logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

ECE_BIN_COUNT = 15  # unified bin count for scalar ECE calculations


def print_phase_header(method: str, phase: str) -> None:
    """Print a console-visible method phase header without re-enabling INFO logs."""
    tqdm.write(f"\n[{method}] {phase}")


# =============================================================================
# JSON ENCODING UTILITIES
# =============================================================================

def _format_number_7digits(num):
    """Format a number to have at most 7 significant digits."""
    if not isinstance(num, (int, float, np.integer, np.floating)) or not np.isfinite(num):
        return num
    
    val = float(num)
    if val == 0:
        return 0
    
    abs_val = abs(val)
    if abs_val >= 1e6 or abs_val < 1e-6:
        return float(f"{val:.6e}")
    
    if abs_val >= 1:
        digits_before_decimal = len(str(int(abs_val)))
        digits_after_decimal = max(0, 7 - digits_before_decimal)
        return round(val, digits_after_decimal)
    else:
        return round(val, 6)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types and limits numbers to 7 digits max."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return _format_number_7digits(float(obj))
        elif isinstance(obj, np.ndarray):
            return [_format_number_7digits(x) if isinstance(x, (int, float, np.integer, np.floating)) else x for x in obj.tolist()]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            item = obj.item()
            return _format_number_7digits(item) if isinstance(item, (int, float)) else item
        elif isinstance(obj, (int, float)):
            return _format_number_7digits(obj)
        return super().default(obj)


def format_numbers_7digits(obj):
    """Recursively format all numeric values to at most 7 significant digits."""
    if isinstance(obj, dict):
        return {k: format_numbers_7digits(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_numbers_7digits(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(format_numbers_7digits(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [
            _format_number_7digits(float(x)) if isinstance(x, (int, float, np.integer, np.floating)) else x
            for x in obj.tolist()
        ]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _format_number_7digits(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# =============================================================================
# BASIC CALIBRATION METRICS
# =============================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute the Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels of shape (n_samples,)
        n_bins: Number of bins for probability discretization
        
    Returns:
        Expected Calibration Error
    """
    pred_probs = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(pred_probs >= bin_lower, pred_probs < bin_upper)
        if np.sum(in_bin) > 0:
            accuracy = np.mean(pred_labels[in_bin] == labels[in_bin])
            confidence = np.mean(pred_probs[in_bin])
            ece += np.abs(accuracy - confidence) * np.sum(in_bin) / len(labels)
    
    return ece


def compute_error_detection_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC for error detection using model confidence as the score.
    """
    pred_labels = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    errors = (pred_labels != labels).astype(int)
    scores = 1.0 - max_probs

    if errors.sum() == 0 or errors.sum() == len(errors):
        return float("nan")

    return float(roc_auc_score(errors, scores))


def compute_risk_coverage_curve(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve for selective prediction.
    """
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = (pred_labels == labels).astype(int)
    n = len(labels)

    if n == 0:
        return np.array([]), np.array([])

    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    cumulative_errors = np.cumsum(1 - correct_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cumulative_errors / k

    return coverage, risk


def compute_aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under Risk-Coverage curve (AURC).
    """
    coverage, risk = compute_risk_coverage_curve(probs, labels)
    if coverage.size == 0:
        return float("nan")
    return float(np.trapz(risk, coverage))


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
                           title: str = "Reliability Diagram") -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a reliability diagram.
    """
    pred_probs = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(pred_probs >= bin_lower, pred_probs < bin_upper)
        if np.sum(in_bin) > 0:
            accuracy = np.mean(pred_labels[in_bin] == labels[in_bin])
            confidence = np.mean(pred_probs[in_bin])
            count = np.sum(in_bin)
            
            accuracies.append(accuracy)
            confidences.append(confidence)
            counts.append(count)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(confidences, accuracies, 'bo-', label='Model')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    return fig, ax


def evaluate_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """
    Evaluate calibration using ECE.
    """
    return {'ece': compute_ece(probs, labels, n_bins)}


# =============================================================================
# UNCERTAINTY METRICS CLASS
# =============================================================================

class UncertaintyMetrics:
    """A helper class to compute common uncertainty and calibration metrics."""
    
    @staticmethod
    def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BIN_COUNT) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            if i == n_bins - 1:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    @staticmethod
    def calculate_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Accuracy."""
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == labels)


# =============================================================================
# OOD DETECTION UTILITIES
# =============================================================================

def compute_ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute AUROC for OOD detection.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    
    if len(np.unique(labels)) < 2:
        return float('nan')
    
    return float(roc_auc_score(labels, scores))


def compute_fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, 
                       tpr_threshold: float = 0.95) -> float:
    """
    Compute FPR@95%TPR for OOD detection.
    """
    from sklearn.metrics import roc_curve
    
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    
    if len(np.unique(labels)) < 2:
        return float('nan')
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    idx = np.where(tpr >= tpr_threshold)[0]
    if len(idx) == 0:
        return 1.0
    
    return float(fpr[idx[0]])


# =============================================================================
# DATA UTILITIES
# =============================================================================

def get_all_data_as_numpy(loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Iterates through a DataLoader and returns the entire dataset as NumPy arrays."""
    all_images = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extracting NumPy data from loader"):
        all_images.append(images.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)


# =============================================================================
# SPP+JL COMPRESSION
# =============================================================================

def compress_with_spp_jl(
    X_np: np.ndarray,
    final_output_dim: int = 1024,
    pyramid_levels = [4, 2, 1],
    seed: int = 42,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compress images [N, C, H, W] -> [N, final_output_dim] using SPP + fixed JL projection.
    """
    from utils.compression_utils import FixedSizeSPP_JL
    
    assert X_np.ndim == 4, f"Expected [N,C,H,W], got {X_np.shape}"
    N, C, H, W = X_np.shape

    projector = FixedSizeSPP_JL(
        in_channels=C,
        final_output_dim=final_output_dim,
        pyramid_levels=pyramid_levels,
        seed=seed,
    ).to(device).eval()

    ds = TensorDataset(torch.from_numpy(X_np))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device).float()
            z = projector(xb)
            outs.append(z.cpu())
    Z = torch.cat(outs, dim=0).numpy()
    return Z


# =============================================================================
# RELIABILITY CURVE UTILITIES
# =============================================================================

def _compute_reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Compute reliability bins for plotting."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_idx = np.digitize(conf, bins, right=True) - 1
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = bucket_idx == b
        if np.any(m):
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ws.append(m.mean())
        else:
            xs.append((bins[b] + bins[b+1]) * 0.5)
            ys.append(np.nan)
            ws.append(0.0)
    return np.array(xs), np.array(ys), np.array(ws)


def plot_reliability_curves_baselines(curves: dict, out_path: str, n_bins: int = 15):
    """Plot reliability curves for multiple methods."""
    plt.figure(figsize=(8, 6))
    xs = np.linspace(0, 1, 101)
    plt.plot(xs, xs, linestyle='--', linewidth=1.5, label='Perfect')
    for name, (probs, labels) in curves.items():
        x, y, _ = _compute_reliability_bins(probs, labels, n_bins=n_bins)
        plt.plot(x, y, marker='o', linewidth=2, label=name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Reliability Curves (Baselines)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# STANDARD BASELINES
# =============================================================================

def run_standard_baselines(model_adapter,
                           val_raw, val_labels,
                           test_raw, test_labels,
                           batch_size: int,
                           output_dir: str,
                           train_raw: Optional[np.ndarray] = None,
                           train_labels: Optional[np.ndarray] = None,
                           candidate_layers_all: Optional[List[int]] = None,
                           seed: int = 42,
                           device: str = "cuda"):
    """
    Runs standard calibration baselines: Uncalibrated, Temperature Scaling, Isotonic Top-Label,
    Platt Scaling, Beta Calibration, Density-Aware Calibration, and Geometric Calibration.
    
    Returns metrics dict for each method.
    """
    from Calibrators.temperature_scaling import TemperatureScaling
    from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
    from Calibrators.platt_scaling import PlattScaling
    from Calibrators.beta_calibration import BetaCalibration
    from Calibrators.geometric_calibrator import GeometricCalibrator
    
    os.makedirs(output_dir, exist_ok=True)

    print_phase_header("Standard Baselines", "start")
    print_phase_header("Standard Baselines", "precompute model probabilities")
    probs_val = model_adapter.predict_proba(val_raw, batch_size=min(256, batch_size))
    probs_test = model_adapter.predict_proba(test_raw, batch_size=min(256, batch_size))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get dataset name from model_adapter
    dataset_name = getattr(model_adapter, 'dataset_name', 'cifar10')
    
    def _metrics(probs):
        return dict(
            ece=float(UncertaintyMetrics.calculate_ece(probs, test_labels, n_bins=ECE_BIN_COUNT)),
            acc=float(UncertaintyMetrics.calculate_accuracy(probs, test_labels)),
        )

    results = {}

    # 1) Uncalibrated
    print_phase_header("Uncalibrated", "evaluate")
    uncal_start = time.perf_counter()
    probs_uncal = np.clip(probs_test, 1e-8, 1-1e-8)
    probs_uncal /= probs_uncal.sum(axis=1, keepdims=True)
    uncal_elapsed = time.perf_counter() - uncal_start
    results["uncalibrated"] = _metrics(probs_uncal)
    results["uncalibrated"]["timing_seconds"] = uncal_elapsed

    # 2) Temperature Scaling
    logger.info("Calculating Temperature Scaling (post hoc) ...")
    ts_start = time.perf_counter()
    try:
        print_phase_header("Temperature Scaling", "extract logits")
        logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, batch_size))
        logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        
        # Check if logits look like probabilities
        lv = np.asarray(logits_val)
        if lv.min() >= 0.0 and lv.max() <= 1.0:
            row_sums = lv.sum(axis=1)
            if np.allclose(row_sums, np.ones_like(row_sums), atol=1e-3, rtol=1e-3):
                logger.warning("TS: logits_val look like probabilities. Converting to log-probabilities.")
                eps = 1e-8
                logits_val = np.log(np.clip(lv, eps, 1.0))
        
        print_phase_header("Temperature Scaling", "fit")
        ts = TemperatureScaling()
        ts.fit(logits_val, val_labels)
        
        lt = np.asarray(logits_test)
        if lt.min() >= 0.0 and lt.max() <= 1.0:
            row_sums = lt.sum(axis=1)
            if np.allclose(row_sums, np.ones_like(row_sums), atol=1e-3, rtol=1e-3):
                eps = 1e-8
                logits_test = np.log(np.clip(lt, eps, 1.0))
        
        print_phase_header("Temperature Scaling", "calibrate")
        probs_ts = ts.calibrate(logits_test)
        ts_elapsed = time.perf_counter() - ts_start
        results["temperature_scaling"] = _metrics(probs_ts)
        results["temperature_scaling"]["timing_seconds"] = ts_elapsed
        results["temperature_scaling"]["T"] = float(ts.temperature)
    except Exception as e:
        ts_elapsed = time.perf_counter() - ts_start
        logger.warning(f"Temperature scaling failed: {e}")
        results["temperature_scaling"] = {"ece": None, "acc": None, "error": str(e), "timing_seconds": ts_elapsed}

    # 3) Isotonic (Top-Label)
    logger.info("Calculating Isotonic (Top-Label) Calibration (post hoc) ...")
    iso_start = time.perf_counter()
    try:
        if 'logits_val' not in locals() or logits_val is None:
            print_phase_header("Isotonic Top-Label", "extract validation logits")
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, batch_size))
        if 'logits_test' not in locals() or logits_test is None:
            print_phase_header("Isotonic Top-Label", "extract test logits")
            logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        print_phase_header("Isotonic Top-Label", "fit")
        iso = TopLabelIsotonicCalibrator()
        iso.fit(logits_val, val_labels)
        print_phase_header("Isotonic Top-Label", "calibrate")
        probs_iso = iso.calibrate(logits_test)
        iso_elapsed = time.perf_counter() - iso_start
        results["isotonic_toplabel"] = _metrics(probs_iso)
        results["isotonic_toplabel"]["timing_seconds"] = iso_elapsed
    except Exception as e:
        iso_elapsed = time.perf_counter() - iso_start
        logger.warning(f"Isotonic regression failed: {e}")
        results["isotonic_toplabel"] = {"ece": None, "acc": None, "error": str(e), "timing_seconds": iso_elapsed}

    # 4) Platt Scaling
    logger.info("Calculating Platt Scaling (post hoc) ...")
    platt_start = time.perf_counter()
    try:
        if 'logits_val' not in locals() or logits_val is None:
            print_phase_header("Platt Scaling", "extract validation logits")
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, batch_size))
        if 'logits_test' not in locals() or logits_test is None:
            print_phase_header("Platt Scaling", "extract test logits")
            logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        print_phase_header("Platt Scaling", "fit")
        platt = PlattScaling()
        platt.fit(logits_val, val_labels)
        print_phase_header("Platt Scaling", "calibrate")
        probs_platt = platt.calibrate(logits_test)
        platt_elapsed = time.perf_counter() - platt_start
        results["platt_scaling"] = _metrics(probs_platt)
        results["platt_scaling"]["timing_seconds"] = platt_elapsed
    except Exception as e:
        platt_elapsed = time.perf_counter() - platt_start
        logger.warning(f"Platt scaling failed: {e}")
        results["platt_scaling"] = {"ece": None, "acc": None, "error": str(e), "timing_seconds": platt_elapsed}

    # 5) Beta Calibration
    logger.info("Calculating Beta Calibration (post hoc) ...")
    beta_start = time.perf_counter()
    try:
        if 'logits_val' not in locals() or logits_val is None:
            print_phase_header("Beta Calibration", "extract validation logits")
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, batch_size))
        if 'logits_test' not in locals() or logits_test is None:
            print_phase_header("Beta Calibration", "extract test logits")
            logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        print_phase_header("Beta Calibration", "fit")
        beta = BetaCalibration()
        beta.fit(logits_val, val_labels)
        print_phase_header("Beta Calibration", "calibrate")
        probs_beta = beta.calibrate(logits_test)
        beta_elapsed = time.perf_counter() - beta_start
        results["beta_calibration"] = _metrics(probs_beta)
        results["beta_calibration"]["timing_seconds"] = beta_elapsed
    except Exception as e:
        beta_elapsed = time.perf_counter() - beta_start
        logger.warning(f"Beta calibration failed: {e}")
        results["beta_calibration"] = {"ece": None, "acc": None, "error": str(e), "timing_seconds": beta_elapsed}

    # 6) Density-Aware Calibration (DAC)
    logger.info("Calculating Density-Aware Calibration (DAC) ...")
    dac_start = time.perf_counter()
    try:
        from Calibrators.density_aware_calibration import (
            DensityAwareCalibrator,
            get_dac_target_layers,
            extract_dac_features,
            get_dac_k_value
        )
        
        raw_model = model_adapter.model
        dev = torch.device(device)
        
        # Get model name from model class
        model_name = raw_model.__class__.__name__.lower()
        if 'resnet' in model_name:
            if hasattr(raw_model, 'layer1'):
                num_blocks = [len(raw_model.layer1), len(raw_model.layer2), 
                             len(raw_model.layer3), len(raw_model.layer4)]
                if num_blocks == [2, 2, 2, 2]:
                    model_name = 'resnet18'
                elif num_blocks == [3, 4, 6, 3]:
                    model_name = 'resnet50'
                elif num_blocks == [3, 4, 23, 3]:
                    model_name = 'resnet101'
                elif num_blocks == [3, 8, 36, 3]:
                    model_name = 'resnet152'
                else:
                    model_name = 'resnet50'
        elif 'densenet' in model_name:
            model_name = 'densenet121'
        
        layer_names = get_dac_target_layers(model_name, raw_model)
        logger.info(f"   DAC target layers for {model_name}: {layer_names}")
        
        # Create data loaders
        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size, shuffle=False
        )
        va_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size, shuffle=False
        )
        te_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        # Extract features
        logger.info("   Extracting DAC features from intermediate layers...")
        dac_extract_start = time.perf_counter()
        print_phase_header("Density-Aware Calibration", "extract train features")
        train_feats_list, _, train_labels_dac = extract_dac_features(raw_model, tr_loader, layer_names, dev)
        print_phase_header("Density-Aware Calibration", "extract validation features")
        val_feats_list, logits_val_dac, val_labels_dac = extract_dac_features(raw_model, va_loader, layer_names, dev)
        print_phase_header("Density-Aware Calibration", "extract test features")
        test_feats_list, logits_test_dac, _ = extract_dac_features(raw_model, te_loader, layer_names, dev)
        dac_extract_elapsed = time.perf_counter() - dac_extract_start
        
        train_feats_list_np = [feat.numpy() for feat in train_feats_list]
        val_feats_list_np = [feat.numpy() for feat in val_feats_list]
        test_feats_list_np = [feat.numpy() for feat in test_feats_list]
        
        k_value = get_dac_k_value(dataset_name)
        logger.info(f"   Using k={k_value} for {dataset_name}")
        
        print_phase_header("Density-Aware Calibration", "fit")
        dac_fit_start = time.perf_counter()
        dac = DensityAwareCalibrator(k=k_value, use_gpu=(device == 'cuda'))
        dac.fit(
            train_features_list=train_feats_list_np,
            val_features_list=val_feats_list_np,
            val_logits=logits_val_dac.numpy(),
            val_labels=val_labels_dac.numpy()
        )
        dac_fit_elapsed = time.perf_counter() - dac_fit_start
        
        print_phase_header("Density-Aware Calibration", "calibrate")
        dac_calibrate_start = time.perf_counter()
        probs_dac = dac.calibrate(
            test_features_list=test_feats_list_np,
            test_logits=logits_test_dac.numpy()
        )
        dac_calibrate_elapsed = time.perf_counter() - dac_calibrate_start
        
        dac_elapsed = time.perf_counter() - dac_start
        results["density_aware_calibration"] = _metrics(probs_dac)
        results["density_aware_calibration"]["timing_seconds"] = dac_elapsed
        results["density_aware_calibration"]["extraction_time_s"] = dac_extract_elapsed
        results["density_aware_calibration"]["fit_time_s"] = dac_fit_elapsed
        results["density_aware_calibration"]["calibrate_time_s"] = dac_calibrate_elapsed
        results["density_aware_calibration"]["selected_layers"] = layer_names
        logger.info(f"DAC calibration completed - total: {dac_elapsed:.2f}s")

    except Exception as e:
        dac_elapsed = time.perf_counter() - dac_start
        logger.warning(f"Density-Aware Calibration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        results["density_aware_calibration"] = {"ece": None, "acc": None, "error": str(e), "timing_seconds": dac_elapsed}

    # 7) Geometric Calibrator (Physical Space)
    logger.info("Geometric Calibrator (Physical Space with SPP+JL compression) ...")
    geo_total_start = time.perf_counter()
    try:
        FINAL_DIM = 512
        PYR = [4, 2, 1]

        geo_init_start = time.perf_counter()
        print_phase_header("Geometric Physical Space", "extract train features")
        Xtr_c = compress_with_spp_jl(
            train_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
            seed=seed, batch_size=min(512, batch_size), device=device
        )
        print_phase_header("Geometric Physical Space", "extract validation features")
        Xva_c = compress_with_spp_jl(
            val_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
            seed=seed, batch_size=min(512, batch_size), device=device
        )
        print_phase_header("Geometric Physical Space", "extract test features")
        Xte_c = compress_with_spp_jl(
            test_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
            seed=seed, batch_size=min(512, batch_size), device=device
        )
        logger.info(f"Compressed data shapes: train={Xtr_c.shape}, val={Xva_c.shape}, test={Xte_c.shape}")
        
        geo = GeometricCalibrator(
            model=model_adapter,
            X_train_embed=Xtr_c,
            y_train=train_labels,
            library="fast_separation",
            device=device,
        )
        geo_init_elapsed = time.perf_counter() - geo_init_start

        print_phase_header("Geometric Physical Space", "fit")
        geo_fit_start = time.perf_counter()
        geo.fit(X_val_embed=Xva_c, X_val_original=val_raw, y_val=val_labels, fit_batch_size=batch_size)
        geo_fit_elapsed = time.perf_counter() - geo_fit_start

        print_phase_header("Geometric Physical Space", "calibrate")
        geo_calibrate_start = time.perf_counter()
        probs_geo = geo.calibrate_batched(X_test_embed=Xte_c, X_test_original=test_raw, batch_size=batch_size)
        geo_calibrate_elapsed = time.perf_counter() - geo_calibrate_start
        
        geo_total_elapsed = time.perf_counter() - geo_total_start
        geo_extraction_time = geo_init_elapsed + geo_fit_elapsed
        
        results["geometric_physical_space"] = dict(
            ece=float(UncertaintyMetrics.calculate_ece(probs_geo, test_labels, n_bins=ECE_BIN_COUNT)),
            acc=float(UncertaintyMetrics.calculate_accuracy(probs_geo, test_labels)),
            timing_init_seconds=geo_init_elapsed,
            timing_fit_seconds=geo_fit_elapsed,
            timing_calibrate_seconds=geo_calibrate_elapsed,
            timing_total_seconds=geo_total_elapsed,
            extraction_time_s=geo_extraction_time,
            calibrate_time_s=geo_calibrate_elapsed,
        )
        logger.info(f"Geometric calibrator completed - total: {geo_total_elapsed:.2f}s")

    except Exception as e:
        geo_total_elapsed = time.perf_counter() - geo_total_start
        logger.warning(f"Geometric calibrator (SPP+JL) failed: {e}")
        results["geometric_physical_space"] = {
            "ece": None, "acc": None, "error": str(e),
            "timing_total_seconds": geo_total_elapsed,
        }

    return results
