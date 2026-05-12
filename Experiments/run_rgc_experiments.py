"""
Per-config calibration pipeline for the IJCAI paper.

Produces the per-config JSON consumed by ``Experiments/generate_paper_tables.py``
and the timing plots. The methods emitted here match the paper:
- Table 1: Uncal, TS, Platt, Isotonic, Beta, DAC, Fast Separation, TULIP, D.Ens, RGCL, RGCC
- Table 2: TOST equivalence tests (RGCL vs GC(DAC), RGCL vs RGCC)
- Figure 3 (PDF): ECE normalised vs RGCL (RGCL, RGCC, GC(DAC), GC(TULIP))
- Figure 4 (PDF): Preprocessing time (RGCL, RGCC, DAC, Fast Separation)
- Figure 5 (PDF): Throughput (RGCL, RGCC, DAC, Fast Separation)
"""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so we can import project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import os
import logging
import argparse
import json
import time
from typing import Dict, Any, List, Union, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import DAC implementation
from Calibrators.density_aware_calibration import (
    DensityAwareCalibrator,
    extract_dac_features,
    extract_dac_features_memory_efficient,
    get_dac_target_layers,
    get_dac_k_value,
    find_layer_module,
)

# Import TULIP layer selection
from utils.layer_utils import select_tulip_layers

# Import existing utilities
from utils.model_utils import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
    construct_model_path,
)
from utils.calibration_utils import (
    run_standard_baselines,
    get_all_data_as_numpy,
    UncertaintyMetrics,
    print_phase_header,
)
from utils.utils import discover_model_layers
from Calibrators.geometric_calibrator import GeometricCalibrator
from Metrics.metrics import expected_calibration_error
from utils.coordinate_extraction import (
    discover_coordinate_space,
    plan_coordinate_extraction,
    extract_coordinate_features,
    coerce_to_tensor,
)
from utils.layer_utils import filter_feature_layers, is_feature_layer
from sklearn.preprocessing import normalize

# Configure logging
from utils.logging_config import setup_logging, get_logger

setup_logging()

logger = get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Expected Calibration Error."""
    preds = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)
    confs = np.max(probs, axis=1) if probs.ndim > 1 else probs
    return expected_calibration_error(confs, preds, labels, num_bins=n_bins)


def calculate_adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Adaptive ECE (equal-mass bins)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]
    
    # Equal-mass bins
    n_samples = len(confidences)
    samples_per_bin = n_samples // n_bins
    
    ece = 0.0
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = start_idx + samples_per_bin if i < n_bins - 1 else n_samples
        
        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]
        
        if len(bin_confidences) > 0:
            avg_conf = np.mean(bin_confidences)
            avg_acc = np.mean(bin_accuracies)
            ece += (len(bin_confidences) / n_samples) * abs(avg_acc - avg_conf)
    
    return ece


def calculate_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculate accuracy."""
    predictions = np.argmax(probs, axis=1)
    return 100.0 * np.mean(predictions == labels)


def make_json_serializable(obj):
    """Recursively convert numpy arrays and remove non-serializable objects."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if key in [
                "calibrator",
                "val_calibrated_probs",
                "model_probs",
                "X_thresholds",
                "y_thresholds",
            ]:
                continue
            cleaned[key] = make_json_serializable(value)
        return cleaned
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    else:
        return obj


def save_results_incrementally(results_dict: Dict[str, Any], output_file: str):
    """Save results to JSON file incrementally."""
    try:
        cleaned = make_json_serializable(results_dict)
        with open(output_file, "w") as f:
            json.dump(cleaned, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def load_existing_results(output_file: str) -> Union[Dict[str, Any], None]:
    """Load existing results if file exists."""
    if not os.path.exists(output_file):
        return None
    try:
        with open(output_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load existing results from {output_file}: {e}")
        return None


def is_section_complete(results: Dict[str, Any], section_name: str, required_fields: List[str] = None) -> bool:
    """Check if a section exists and has required fields."""
    if required_fields is None:
        required_fields = ["ece"]
    
    parts = section_name.split(".")
    current = results
    
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    
    if current is None or (isinstance(current, (list, dict)) and not current):
        return False
    
    if isinstance(current, dict) and current.get("skipped", False):
        return True
    
    if isinstance(current, dict):
        return all(field in current for field in required_fields)
    
    return bool(current)


def should_rerun_section(results: Dict[str, Any], section_name: str, required_fields: List[str] = None) -> bool:
    """Check if section needs to be rerun."""
    return not is_section_complete(results, section_name, required_fields)


GEO_REQUIRED_FIELDS = ["ece", "calibrator_params"]
DAC_REQUIRED_FIELDS = ["ece", "layer_weights"]


# ============================================================================
# LAYER DISCOVERY AND FILTERING
# ============================================================================

def normalize_discovered_layers(model, discovered_layers):
    """Normalize discovered layers to consistent format."""
    normalized = []
    for layer in discovered_layers:
        if isinstance(layer, dict):
            normalized.append(layer)
        elif isinstance(layer, str):
            normalized.append({"name": layer, "shape": None})
        else:
            normalized.append({"name": str(layer), "shape": None})
    return normalized


def filter_non_feature_layers(discovered_layers, model):
    """Filter out non-feature layers (activations, dropout, etc.)."""
    filtered = []
    excluded = []
    
    for layer_info in discovered_layers:
        layer_name = layer_info["name"] if isinstance(layer_info, dict) else layer_info
        
        # Skip activation, dropout, identity layers
        skip_patterns = ["relu", "gelu", "dropout", "identity", "flatten", "act"]
        should_skip = any(pattern in layer_name.lower() for pattern in skip_patterns)
        
        if should_skip:
            excluded.append(layer_name)
        else:
            filtered.append(layer_info)
    
    return filtered, excluded


# ============================================================================
# FEATURE EXTRACTION FOR RGC (RGCL)
# ============================================================================

def extract_and_aggregate_rgc_features(
    model: torch.nn.Module,
    layer_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    target_dim: int = 256,
    seed: int = 42,
    pooling_mode: str = "max",
    method_label: str = "RGC",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract and aggregate features using RGC pipeline (RGCL).
    
    Pipeline:
    1. SPP pooling per layer (4x4, 2x2, 1x1)
    2. Random projection per layer
    3. Project-then-sum aggregation
    4. L2 normalization
    """
    start_time = time.perf_counter()
    
    # Extract raw features from selected layers
    model.eval()
    
    def extract_with_spp(loader, split_name: str):
        """Extract features with SPP pooling."""
        all_features = []
        
        with torch.no_grad():
            desc = f"{method_label} extract {split_name} features"
            for batch_data in tqdm(loader, desc=desc):
                if isinstance(batch_data, (tuple, list)):
                    x = batch_data[0]
                else:
                    x = batch_data
                x = x.to(device).float()
                
                # Get activations from each layer
                batch_features = []
                handles = []
                activations = {}
                
                def make_hook(name):
                    def hook(module, input, output):
                        activations[name] = output
                    return hook
                
                # Register hooks
                for name in layer_names:
                    try:
                        module = dict(model.named_modules())[name]
                        handle = module.register_forward_hook(make_hook(name))
                        handles.append(handle)
                    except KeyError:
                        logger.warning(f"Layer {name} not found in model")
                
                # Forward pass
                _ = model(x)
                
                # Remove hooks
                for handle in handles:
                    handle.remove()
                
                # Apply SPP and aggregate
                layer_features = []
                for name in layer_names:
                    if name in activations:
                        feat = activations[name]
                        if isinstance(feat, torch.Tensor):
                            # Apply SPP (4x4, 2x2, 1x1)
                            feat_spp = apply_spp(feat, pooling_mode=pooling_mode)
                            layer_features.append(feat_spp.cpu().numpy())
                
                if layer_features:
                    batch_features = np.concatenate(layer_features, axis=1)
                    all_features.append(batch_features)
        
        return np.concatenate(all_features, axis=0) if all_features else np.array([])
    
    # Extract from all loaders
    train_features = extract_with_spp(train_loader, "train") if train_loader else None
    val_features = extract_with_spp(val_loader, "val") if val_loader else None
    test_features = extract_with_spp(test_loader, "test") if test_loader else None
    
    # Random projection to target_dim
    np.random.seed(seed)
    if train_features is not None and train_features.shape[1] > target_dim:
        # Create random projection matrix
        proj_matrix = np.random.randn(train_features.shape[1], target_dim) / np.sqrt(target_dim)
        train_features = train_features @ proj_matrix
        if val_features is not None:
            val_features = val_features @ proj_matrix
        if test_features is not None:
            test_features = test_features @ proj_matrix
    
    # L2 normalize
    if train_features is not None:
        train_features = normalize(train_features, norm="l2", axis=1)
    if val_features is not None:
        val_features = normalize(val_features, norm="l2", axis=1)
    if test_features is not None:
        test_features = normalize(test_features, norm="l2", axis=1)
    
    extraction_time = time.perf_counter() - start_time
    
    extraction_info = {
        "extraction_time_s": extraction_time,
        "num_layers": len(layer_names),
        "target_dim": target_dim,
        "layer_contributions": {name: 1.0/len(layer_names) for name in layer_names},
        "total_spp_dims": train_features.shape[1] if train_features is not None else 0,
    }
    
    return train_features, val_features, test_features, extraction_info


def apply_spp(feature_tensor: torch.Tensor, pooling_mode: str = "max") -> torch.Tensor:
    """Apply Spatial Pyramid Pooling (4x4, 2x2, 1x1)."""
    import torch.nn.functional as F
    
    if feature_tensor.dim() == 2:
        # Already flattened (e.g., from linear layer)
        return feature_tensor
    
    if feature_tensor.dim() == 3:
        # Transformer output: (batch, seq, channels) -> treat as (batch, channels, seq, 1)
        feature_tensor = feature_tensor.permute(0, 2, 1).unsqueeze(-1)
    
    batch_size = feature_tensor.shape[0]
    channels = feature_tensor.shape[1]
    
    pooled_features = []
    
    for grid_size in [4, 2, 1]:
        if pooling_mode == "max":
            pooled = F.adaptive_max_pool2d(feature_tensor, (grid_size, grid_size))
        else:
            pooled = F.adaptive_avg_pool2d(feature_tensor, (grid_size, grid_size))
        pooled_features.append(pooled.view(batch_size, -1))
    
    return torch.cat(pooled_features, dim=1)


# ============================================================================
# COORDINATE EXTRACTION FOR RGCC
# ============================================================================

def run_coordinate_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_coordinates: int = 256,
    batch_size: int = 128,
    output_dir: str = None,
    seed: int = 42,
    training_method: str = None,
    precomputed_test_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
) -> Dict[str, Any]:
    """
    RGCC: Coordinate-based random sampling calibration.
    
    Instead of sampling layers, samples K random coordinates from the
    global activation space across all layers.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING COORDINATE SAMPLING CALIBRATION (K={num_coordinates})")
    logger.info("=" * 80)
    print_phase_header("RGCC", f"start (K={num_coordinates})")
    
    # Discover coordinate space
    print_phase_header("RGCC", "discover coordinate space")
    is_dinov2 = model_name is not None and "dinov2" in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    # Get coordinate space info
    layer_map, total_coords = discover_coordinate_space(model, input_shape, device)
    
    # Sample coordinates
    np.random.seed(seed)
    sampling_plan, global_order = plan_coordinate_extraction(total_coords, num_coordinates, layer_map, seed=seed)

    
    logger.info(f"Total coordinates available: {total_coords}")
    logger.info(f"Sampling {num_coordinates} coordinates")
    
    # Create loaders if not provided
    if train_loader is None:
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size,
            shuffle=False,
        )
    if val_loader is None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size,
            shuffle=False,
        )
    if test_loader is None:
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
        )
    
    # Extract coordinate features
    print_phase_header("RGCC", "extract train features")
    start_extraction = time.perf_counter()
    
    train_coords = extract_coordinate_features(
        model, train_loader, sampling_plan, layer_map, global_order, device=device
    )
    print_phase_header("RGCC", "extract validation features")
    val_coords = extract_coordinate_features(
        model, val_loader, sampling_plan, layer_map, global_order, device=device
    )
    print_phase_header("RGCC", "extract test features")
    test_coords = extract_coordinate_features(
        model, test_loader, sampling_plan, layer_map, global_order, device=device
    )
    
    extraction_time = time.perf_counter() - start_extraction
    
    # L2 normalize
    train_coords = normalize(train_coords, norm="l2", axis=1)
    val_coords = normalize(val_coords, norm="l2", axis=1)
    test_coords = normalize(test_coords, norm="l2", axis=1)
    
    logger.info(f"Coordinate feature shape: {train_coords.shape}")
    logger.info(f"Extraction time: {extraction_time:.2f}s")
    
    # Fit geometric calibrator
    logger.info("Fitting geometric calibrator...")
    print_phase_header("RGCC", "fit")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_coords,
        y_train=train_labels,
        library="fast_separation",
        device=str(device),
    )
    
    geo_cal.fit(
        X_val_embed=val_coords,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set
    logger.info("Calibrating test set...")
    print_phase_header("RGCC", "calibrate")
    
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
    else:
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_coords,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    calibrate_time = time.perf_counter() - start_calibrate
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)

    throughput = len(test_labels) / calibrate_time if calibrate_time > 0 else 0
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    logger.info(f"\nRGCC Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")

    return {
        "method": f"RGCC (K={num_coordinates})",
        "ece": float(ece),
        "adaptive_ece": float(adaptive_ece),
        "accuracy": float(accuracy),
        "num_coordinates": num_coordinates,
        "extraction_time_s": float(extraction_time),
        "fit_time_s": float(fit_time),
        "calibrate_time_s": float(calibrate_time),
        "throughput_samples_per_sec": float(throughput),
        "peak_memory_mb": float(peak_memory_mb),
        "calibrator_params": geo_cal.get_params(),
    }


# ============================================================================
# RGCL: GLOBAL RANDOM LAYER CALIBRATION
# ============================================================================

def run_global_random_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_layers: int = 6,
    target_dim: int = 256,
    output_dir: str = None,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = "max",
    output_file: str = None,
    results_dict: Dict[str, Any] = None,
    precomputed_train_features: np.ndarray = None,
    precomputed_val_features: np.ndarray = None,
    precomputed_test_features: np.ndarray = None,
    precomputed_extraction_info: Dict[str, Any] = None,
    precomputed_selected_layers: List[str] = None,
    precomputed_test_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    training_method: str = None,
) -> Dict[str, Any]:
    """
    RGCL: Random Geometric Calibration with Layer sampling.
    
    Samples L random layers, applies SPP pooling, random projection,
    and geometric calibration.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING RGCL (L={num_layers}, d={target_dim})")
    logger.info("=" * 80)
    print_phase_header("RGCL", f"start (L={num_layers}, d={target_dim})")
    
    # Use precomputed features or extract new ones
    if precomputed_train_features is not None:
        logger.info("Using precomputed features")
        train_features = precomputed_train_features
        val_features = precomputed_val_features
        test_features = precomputed_test_features
        extraction_info = precomputed_extraction_info
        selected_layers = precomputed_selected_layers
    else:
        # Discover and filter layers
        is_dinov2 = model_name is not None and "dinov2" in model_name.lower()
        if is_dinov2:
            input_shape = (1, 3, 224, 224)
        elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
            input_shape = (1, 3, 64, 64)
        else:
            input_shape = (1, 3, 32, 32)
        
        discovered_layers = normalize_discovered_layers(
            model, discover_model_layers(model, device=device, input_shape=input_shape)
        )
        filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
        all_layer_names = [d["name"] for d in filtered_layers]
        
        # Random layer selection
        np.random.seed(seed)
        if num_layers > len(all_layer_names):
            selected_layers = all_layer_names
        else:
            selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
        
        logger.info(f"Selected {len(selected_layers)} layers: {selected_layers}")
        
        # Create loaders
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size,
                shuffle=False,
            )
        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size,
                shuffle=False,
            )
        if test_loader is None:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=batch_size,
                shuffle=False,
            )
        
        # Extract features
        print_phase_header("RGCL", "extract features")
        train_features, val_features, test_features, extraction_info = extract_and_aggregate_rgc_features(
            model=model,
            layer_names=selected_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode=pooling_mode,
            method_label="RGCL",
        )
    
    logger.info(f"Feature shape: {train_features.shape}")
    
    # Fit geometric calibrator
    logger.info("Fitting geometric calibrator...")
    print_phase_header("RGCL", "fit")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        device=str(device),
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set
    logger.info("Calibrating test set...")
    print_phase_header("RGCL", "calibrate")
    
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
    else:
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    calibrate_time = time.perf_counter() - start_calibrate
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)

    throughput = len(test_labels) / calibrate_time if calibrate_time > 0 else 0
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    logger.info(f"\nRGCL Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")

    return {
        "method": f"RGCL (L={num_layers}, d={target_dim})",
        "ece": float(ece),
        "adaptive_ece": float(adaptive_ece),
        "accuracy": float(accuracy),
        "num_layers": num_layers,
        "target_dim": target_dim,
        "selected_layers": selected_layers,
        "extraction_time_s": extraction_info.get("extraction_time_s", 0),
        "fit_time_s": float(fit_time),
        "calibrate_time_s": float(calibrate_time),
        "throughput_samples_per_sec": float(throughput),
        "peak_memory_mb": float(peak_memory_mb),
        "calibrator_params": geo_cal.get_params(),
    }


# ============================================================================
# GC(DAC): GC with DAC layer selection (for TOST comparison)
# ============================================================================

def run_gc_with_dac_layers(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = "max",
    precomputed_test_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
) -> Dict[str, Any]:
    """
    GC(DAC): Geometric calibration using DAC's layer selection.
    For TOST equivalence comparison in Table 2.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING GC WITH DAC LAYER SELECTION (d={target_dim})")
    logger.info("=" * 80)
    print_phase_header("GC(DAC)", f"start (d={target_dim})")
    
    # Get DAC's layer selection
    model_name_dac = model_name.lower() if model_name else model.__class__.__name__.lower()
    
    if "resnet" in model_name_dac and hasattr(model, "layer1"):
        num_blocks = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
        if num_blocks == [2, 2, 2, 2]:
            model_name_dac = "resnet18"
        elif num_blocks == [3, 4, 6, 3]:
            model_name_dac = "resnet50"
        elif num_blocks == [3, 4, 23, 3]:
            model_name_dac = "resnet101"
        elif num_blocks == [3, 8, 36, 3]:
            model_name_dac = "resnet152"
    elif "densenet" in model_name_dac:
        model_name_dac = "densenet121"
    
    selected_layers = get_dac_target_layers(model_name_dac, model)
    logger.info(f"Using DAC layer selection: {selected_layers}")
    
    num_layers = len(selected_layers)
    
    # Create loaders
    if train_loader is None:
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size,
            shuffle=False,
        )
    if val_loader is None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size,
            shuffle=False,
        )
    if test_loader is None:
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
        )
    
    # Extract features using RGC pipeline
    print_phase_header("GC(DAC)", "extract features")
    train_features, val_features, test_features, extraction_info = extract_and_aggregate_rgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_dim=target_dim,
        seed=seed,
        pooling_mode=pooling_mode,
        method_label="GC(DAC)",
    )
    
    logger.info(f"Feature shape: {train_features.shape}")
    
    # Fit geometric calibrator
    print_phase_header("GC(DAC)", "fit")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        device=str(device),
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate
    print_phase_header("GC(DAC)", "calibrate")
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
    else:
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size,
    )
    
    calibrate_time = time.perf_counter() - start_calibrate
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)

    logger.info(f"\nGC(DAC) Results:")
    logger.info(f"  ECE: {ece:.6f}")

    return {
        "method": f"GC(DAC) (L={num_layers}, d={target_dim})",
        "ece": float(ece),
        "adaptive_ece": float(adaptive_ece),
        "accuracy": float(accuracy),
        "num_layers": num_layers,
        "target_dim": target_dim,
        "selected_layers": selected_layers,
        "layer_selection": "dac",
        "extraction_time_s": extraction_info.get("extraction_time_s", 0),
        "fit_time_s": float(fit_time),
        "calibrate_time_s": float(calibrate_time),
        "calibrator_params": geo_cal.get_params(),
    }


# ============================================================================
# GC(TULIP): RGC with TULIP layer selection (for TOST comparison)
# ============================================================================

def run_rgc_with_tulip_layers(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = "max",
    precomputed_test_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
) -> Dict[str, Any]:
    """
    GC(TULIP): RGC using TULIP's layer selection instead of random selection.
    For TOST equivalence comparison in Figure 3.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING RGC WITH TULIP LAYER SELECTION (d={target_dim})")
    logger.info("=" * 80)
    print_phase_header("GC(TULIP)", f"start (d={target_dim})")
    
    # Get TULIP's layer selection
    selected_layers = select_tulip_layers(model, model_name)
    logger.info(f"Using TULIP layer selection: {selected_layers}")
    
    num_layers = len(selected_layers)
    
    # Create loaders
    if train_loader is None:
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size,
            shuffle=False,
        )
    if val_loader is None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size,
            shuffle=False,
        )
    if test_loader is None:
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
        )
    
    # Extract features using RGC pipeline
    print_phase_header("GC(TULIP)", "extract features")
    train_features, val_features, test_features, extraction_info = extract_and_aggregate_rgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_dim=target_dim,
        seed=seed,
        pooling_mode=pooling_mode,
        method_label="GC(TULIP)",
    )
    
    logger.info(f"Feature shape: {train_features.shape}")
    
    # Fit geometric calibrator
    print_phase_header("GC(TULIP)", "fit")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        device=str(device),
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate
    print_phase_header("GC(TULIP)", "calibrate")
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
    else:
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size,
    )
    
    calibrate_time = time.perf_counter() - start_calibrate
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)

    logger.info(f"\nGC(TULIP) Results:")
    logger.info(f"  ECE: {ece:.6f}")

    return {
        "method": f"GC(TULIP) (L={num_layers}, d={target_dim})",
        "ece": float(ece),
        "adaptive_ece": float(adaptive_ece),
        "accuracy": float(accuracy),
        "num_layers": num_layers,
        "target_dim": target_dim,
        "selected_layers": selected_layers,
        "layer_selection": "tulip",
        "extraction_time_s": extraction_info.get("extraction_time_s", 0),
        "fit_time_s": float(fit_time),
        "calibrate_time_s": float(calibrate_time),
        "calibrator_params": geo_cal.get_params(),
    }


# ============================================================================
# DEEP ENSEMBLE BASELINE
# ============================================================================

def run_deep_ensemble_baseline(
    model_name: str,
    dataset: str,
    training_method: str,
    ensemble_seeds: List[int],
    results_base_dir: str,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Run deep ensemble baseline (D.Ens in Table 1).
    Averages predictions from multiple models trained with different seeds.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING DEEP ENSEMBLE BASELINE (seeds: {ensemble_seeds})")
    logger.info("=" * 80)
    print_phase_header("Deep Ensemble", f"start (seeds={ensemble_seeds})")
    
    from utils.model_utils import get_data_loaders
    
    # Get num_classes
    _, _, _, num_classes = get_data_loaders(dataset, batch_size, seed=ensemble_seeds[0])
    
    ensemble_probs = []
    
    for seed in ensemble_seeds:
        print_phase_header("Deep Ensemble", f"load and predict seed {seed}")
        model_path = construct_model_path(
            results_base_dir, training_method, dataset, model_name, seed
        )
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, skipping seed {seed}")
            continue
        
        logger.info(f"Loading model with seed {seed}...")
        model = load_trained_model(model_path, model_name, num_classes, device, dataset=dataset)
        adapter = PyTorchModelAdapter(model, device, dataset)
        
        probs = adapter.predict_proba(test_raw, batch_size=batch_size)
        ensemble_probs.append(probs)
    
    if len(ensemble_probs) < 2:
        raise ValueError(f"Need at least 2 models for ensemble, got {len(ensemble_probs)}")
    
    # Average predictions
    avg_probs = np.mean(ensemble_probs, axis=0)
    
    # Metrics
    ece = calculate_ece(avg_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(avg_probs, test_labels)
    accuracy = calculate_accuracy(avg_probs, test_labels)

    logger.info(f"\nDeep Ensemble Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Ensemble size: {len(ensemble_probs)}")

    return {
        "method": f"Deep Ensemble (n={len(ensemble_probs)})",
        "ece": float(ece),
        "adaptive_ece": float(adaptive_ece),
        "accuracy": float(accuracy),
        "ensemble_size": len(ensemble_probs),
        "ensemble_seeds": ensemble_seeds[:len(ensemble_probs)],
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare DAC vs Geometric Calibration (Paper Version)"
    )
    
    # Model configuration
    parser.add_argument("--model-name", type=str, required=True, help="Model architecture")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--training-method", type=str, default="baseline", help="Training method")
    parser.add_argument("--target-dimension", type=int, default=256, help="Target dimension d (paper: 256)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of random layers L (paper: 6)")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--force-recompute", action="store_true", help="Force recompute all results")
    
    # Paths
    parser.add_argument("--results-base-dir", type=str, default="results/results", help="Base directory for results")
    parser.add_argument("--output-dir", type=str, default="calibration_comparison", help="Output directory")
    
    # Device and batch size
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--spp-pooling-mode", type=str, default="max", choices=["avg", "max"], help="SPP pooling mode")
    
    # Deep ensemble
    parser.add_argument("--eval-ensemble", action="store_true", help="Evaluate deep ensemble baseline")
    parser.add_argument("--ensemble-seeds", type=str, default="11 12 13", help="Ensemble seeds")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup output
    is_dinov2 = args.model_name and "dinov2" in args.model_name.lower()
    output_subdir = os.path.join(
        args.output_dir,
        args.model_name,
        args.dataset,
        args.training_method,
        str(args.seed),
    )
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, "paper_results.json")
    
    # Load existing results or start fresh
    if os.path.exists(output_file) and not args.force_recompute:
        results = load_existing_results(output_file) or {}
    else:
        results = {}
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, seed=args.seed
    )
    
    # Load model
    model_path = construct_model_path(
        args.results_base_dir, args.training_method, args.dataset, args.model_name, args.seed
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    
    # Extract raw data
    def extract_raw_data(loader):
        raw_data, labels = [], []
        for data, target in loader:
            raw_data.append(data.numpy())
            labels.append(target.numpy())
        return np.concatenate(raw_data), np.concatenate(labels)
    
    train_raw, train_labels = extract_raw_data(train_loader)
    val_raw, val_labels = extract_raw_data(val_loader)
    test_raw, test_labels = extract_raw_data(test_loader)
        
    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)
    
    # Pre-compute predictions
    logger.info("Pre-computing model predictions...")
    prediction_cache = {
        "test_probs": model_adapter.predict_proba(test_raw, batch_size=args.batch_size),
        "val_probs": model_adapter.predict_proba(val_raw, batch_size=args.batch_size),
    }
    
    # Save experiment config
    results["experiment_config"] = {
        **vars(args),
        "num_classes": num_classes,
        "train_size": len(train_labels),
        "val_size": len(val_labels),
        "test_size": len(test_labels),
    }
    save_results_incrementally(results, output_file)
    
    # ===== 1. STANDARD BASELINES (Table 1: Uncal, TS, Platt, Isotonic, Beta, DAC, Fast Separation) =====
    if should_rerun_section(results, "standard_baselines", ["uncalibrated"]):
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING STANDARD BASELINES")
        logger.info("=" * 80)
        
        baseline_results = run_standard_baselines(
            model_adapter=model_adapter,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            train_raw=train_raw,
            train_labels=train_labels,
            seed=args.seed,
            device=str(device),
        )
        
        # Skip DAC for DINOv2
        if is_dinov2:
            baseline_results["density_aware_calibration"] = {
                "skipped": True,
                "reason": "DINOv2 not supported by DAC",
            }
        
        results["standard_baselines"] = baseline_results
        save_results_incrementally(results, output_file)
    else:
        logger.info("Standard baselines already computed")
    
    # ===== 2. DEEP ENSEMBLE (Table 1: D.Ens) =====
    if args.eval_ensemble:
        if should_rerun_section(results, "deep_ensemble", ["ece"]):
            ensemble_seeds = [int(s) for s in args.ensemble_seeds.split()]
            try:
                ensemble_results = run_deep_ensemble_baseline(
                    model_name=args.model_name,
                    dataset=args.dataset,
                    training_method=args.training_method,
                    ensemble_seeds=ensemble_seeds,
                    results_base_dir=args.results_base_dir,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    device=device,
                    batch_size=args.batch_size,
                )
                results["deep_ensemble"] = ensemble_results
                save_results_incrementally(results, output_file)
            except Exception as e:
                logger.error(f"Deep ensemble failed: {e}")
                results["deep_ensemble"] = {"error": str(e)}
        else:
            logger.info("Deep ensemble already computed")
    
    # ===== 3. RGCL (Table 1, Table 2, Figures 2-4) =====
    if should_rerun_section(results, "rgcl", GEO_REQUIRED_FIELDS):
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING RGCL (Global Random Layer Calibration)")
        logger.info("=" * 80)
        
        rgcl_results = run_global_random_calibration(
            model=model,
            model_name=args.model_name,
            dataset_name=args.dataset,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            device=device,
            num_layers=args.num_layers,
            target_dim=args.target_dimension,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            pooling_mode=args.spp_pooling_mode,
            precomputed_test_probs=prediction_cache["test_probs"],
            training_method=args.training_method,
        )
        results["rgcl"] = rgcl_results
        save_results_incrementally(results, output_file)
    else:
        logger.info("RGCL already computed")
    
    # ===== 4. RGCC (Table 1, Table 2, Figures 2-4) =====
    if should_rerun_section(results, "rgcc", GEO_REQUIRED_FIELDS):
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING RGCC (Coordinate Sampling Calibration)")
        logger.info("=" * 80)
        
        rgcc_results = run_coordinate_calibration(
            model=model,
            model_name=args.model_name,
            dataset_name=args.dataset,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            device=device,
            num_coordinates=args.target_dimension,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            seed=args.seed,
            training_method=args.training_method,
            precomputed_test_probs=prediction_cache["test_probs"],
        )
        results["rgcc"] = rgcc_results
        save_results_incrementally(results, output_file)
    else:
        logger.info("RGCC already computed")
    
    # ===== 5. GC(DAC) - For TOST comparison (Table 2, Figure 3) =====
    if not is_dinov2 and should_rerun_section(results, "gc_dac", GEO_REQUIRED_FIELDS):
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING GC(DAC) - GC with DAC layer selection")
        logger.info("=" * 80)
        
        try:
            gc_dac_results = run_gc_with_dac_layers(
                model=model,
                model_name=args.model_name,
                dataset_name=args.dataset,
                model_adapter=model_adapter,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                device=device,
                target_dim=args.target_dimension,
                batch_size=args.batch_size,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode,
                    precomputed_test_probs=prediction_cache["test_probs"],
            )
            results["gc_dac"] = gc_dac_results
            save_results_incrementally(results, output_file)
        except Exception as e:
            logger.error(f"GC(DAC) failed: {e}")
            results["gc_dac"] = {"error": str(e)}
            save_results_incrementally(results, output_file)
    elif is_dinov2:
        results["gc_dac"] = {"skipped": True, "reason": "DINOv2 not supported"}
        save_results_incrementally(results, output_file)
    else:
        logger.info("GC(DAC) already computed")
    
    # ===== 6. GC(TULIP) - For TOST comparison (Figure 3) =====
    if should_rerun_section(results, "gc_tulip", GEO_REQUIRED_FIELDS):
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING GC(TULIP) - RGC with TULIP layer selection")
        logger.info("=" * 80)
        
        try:
            gc_tulip_results = run_rgc_with_tulip_layers(
                model=model,
                model_name=args.model_name,
                dataset_name=args.dataset,
                model_adapter=model_adapter,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                device=device,
                target_dim=args.target_dimension,
                batch_size=args.batch_size,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode,
                    precomputed_test_probs=prediction_cache["test_probs"],
            )
            results["gc_tulip"] = gc_tulip_results
            save_results_incrementally(results, output_file)
        except Exception as e:
            logger.error(f"GC(TULIP) failed: {e}")
            results["gc_tulip"] = {"error": str(e)}
            save_results_incrementally(results, output_file)
    else:
        logger.info("GC(TULIP) already computed")
    
    # ===== SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    
    # Print ECE for all methods
    methods_to_print = [
        ("Uncalibrated", "standard_baselines.uncalibrated.ece"),
        ("Temperature Scaling", "standard_baselines.temperature_scaling.ece"),
        ("Isotonic", "standard_baselines.isotonic_toplabel.ece"),
        ("DAC", "standard_baselines.density_aware_calibration.ece"),
        ("Deep Ensemble", "deep_ensemble.ece"),
        ("RGCL", "rgcl.ece"),
        ("RGCC", "rgcc.ece"),
        ("GC(DAC)", "gc_dac.ece"),
        ("GC(TULIP)", "gc_tulip.ece"),
    ]
    
    for name, path in methods_to_print:
        parts = path.split(".")
        value = results
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        
        if value is not None:
            logger.info(f"  {name}: ECE = {value:.4f}")
        else:
            logger.info(f"  {name}: N/A")
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
