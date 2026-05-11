"""
Density Aware Calibration (DAC) Implementation.

This module provides DAC using PyTorch for kNN computation.

Reference: DAC Paper (Table 6, Appendix C.1)
"""

import numpy as np
from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import time
import logging

# Try to import project logger, fall back to standard logging
try:
    from utils.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# =============================================================================
# Common Utilities
# =============================================================================


def np_softmax(x):
    """Numerically stable softmax."""
    max_val = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_val)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def get_dac_target_layers(model_name: str, model: nn.Module) -> List[str]:
    """
    Get target layers for DAC feature extraction according to DAC paper (Table 6, Appendix C.1).

    IMPORTANT: This function returns INTERMEDIATE layers only.
    The logits layer is extracted separately and added later in the pipeline.

    For DenseNet: PRE-BLOCK + BLOCK-1-3 + PENULTIMATE = 5 intermediate layers
                  Logits added later -> Total 6 layers

    For ResNet: PRE-BLOCK + BLOCK-1-4 = 5 intermediate layers
                Logits added later -> Total 6 layers

    Args:
        model_name: Architecture name (e.g., 'resnet18', 'densenet121')
        model: PyTorch model instance

    Returns:
        List of INTERMEDIATE layer names (logits added separately)
    """
    model_name_lower = model_name.lower()

    if "resnet" in model_name_lower:
        # According to DAC paper Table 6: PRE-BLOCK + BLOCK-1-4 (logits added later)
        target_layers = []

        # PRE-BLOCK: Try maxpool first (ImageNet), then conv1 (CIFAR)
        if find_layer_module(model, "maxpool") is not None:
            target_layers.append("maxpool")
        elif find_layer_module(model, "conv1") is not None:
            target_layers.append("conv1")

        # BLOCK-1 to BLOCK-4
        target_layers.extend(["layer1", "layer2", "layer3", "layer4"])

        logger.info(f"DAC intermediate layers for {model_name}: {target_layers}")
        logger.info(f"  Total with logits: {len(target_layers) + 1} layers")
        return target_layers

    elif "densenet" in model_name_lower:
        layers = []

        # Try standard torchvision structure first
        if hasattr(model, "features"):
            features_module = model.features
            features_modules = features_module._modules

            # PRE-BLOCK: Programmatically find the module immediately before 'denseblock1'
            pre_block_name = None
            if "denseblock1" in features_modules:
                module_names = list(features_modules.keys())
                denseblock1_idx = module_names.index("denseblock1")
                if denseblock1_idx > 0:
                    pre_block_name = module_names[denseblock1_idx - 1]
                    layers.append(f"features.{pre_block_name}")
                    logger.info(
                        f"  Found PRE-BLOCK: features.{pre_block_name} (immediately before denseblock1)"
                    )
                else:
                    raise ValueError(
                        f"DenseNet {model_name}: denseblock1 is the first module in features, "
                        f"cannot determine PRE-BLOCK. Expected a module before denseblock1."
                    )
            else:
                raise ValueError(
                    f"DenseNet {model_name}: 'denseblock1' not found in model.features"
                )

            # BLOCK-1 to BLOCK-3: Add denseblocks only (per paper Table 6)
            for block_idx in [1, 2, 3]:
                block_name = f"denseblock{block_idx}"
                if block_name in features_modules:
                    layers.append(f"features.{block_name}")
                    logger.info(f"  Found BLOCK-{block_idx}: features.{block_name}")
                else:
                    raise ValueError(
                        f"DenseNet {model_name}: Required block 'features.{block_name}' not found"
                    )

            # PENULTIMATE: Last module in model.features (typically 'norm5')
            module_names = list(features_modules.keys())
            if module_names:
                penultimate_name = module_names[-1]
                layers.append(f"features.{penultimate_name}")
                logger.info(
                    f"  Found PENULTIMATE: features.{penultimate_name} (last module in features)"
                )
            else:
                raise ValueError(
                    f"DenseNet {model_name}: model.features has no modules"
                )

            # Verify no transitions were included
            for layer_name in layers:
                if "transition" in layer_name.lower() or "trans" in layer_name.lower():
                    raise ValueError(
                        f"DenseNet {model_name}: Transition layer '{layer_name}' should not be included. "
                        f"Paper specifies: PRE-BLOCK + 3 dense blocks + PENULTIMATE only."
                    )

        # Custom DenseNet implementation (no .features attribute)
        else:
            logger.info(f"Using custom DenseNet structure for {model_name}")

            module_names = list(model._modules.keys())

            # Find first dense block
            first_dense_idx = None
            first_dense_name = None
            for idx, name in enumerate(module_names):
                name_lower = name.lower()
                if ("dense" in name_lower and "block" in name_lower) or (
                    name_lower.startswith("dense") and name_lower[5:6].isdigit()
                ):
                    if "trans" not in name_lower:
                        first_dense_idx = idx
                        first_dense_name = name
                        break

            if first_dense_idx is None:
                raise ValueError(
                    f"DenseNet {model_name}: Could not find first dense block in model structure"
                )

            # PRE-BLOCK
            if first_dense_idx > 0:
                pre_block_name = module_names[first_dense_idx - 1]
                layers.append(pre_block_name)
                logger.info(
                    f"  Found PRE-BLOCK: {pre_block_name} (immediately before {first_dense_name})"
                )
            else:
                raise ValueError(
                    f"DenseNet {model_name}: First dense block '{first_dense_name}' is the first module, "
                    f"cannot determine PRE-BLOCK. Expected a module before it."
                )

            # BLOCK-1 to BLOCK-3
            dense_blocks_found = []
            for name in module_names:
                name_lower = name.lower()
                if (
                    ("dense" in name_lower and "block" in name_lower)
                    or (name_lower.startswith("dense") and name_lower[5:6].isdigit())
                ) and "trans" not in name_lower:
                    dense_blocks_found.append(name)
                    if len(dense_blocks_found) >= 3:
                        break

            if len(dense_blocks_found) < 3:
                raise ValueError(
                    f"DenseNet {model_name}: Found only {len(dense_blocks_found)} dense blocks, "
                    f"need at least 3 for BLOCK-1..BLOCK-3"
                )

            for i, block_name in enumerate(dense_blocks_found[:3], 1):
                layers.append(block_name)
                logger.info(f"  Found BLOCK-{i}: {block_name}")

            # PENULTIMATE
            penultimate_name = None
            for name in reversed(module_names):
                name_lower = name.lower()
                if (
                    "classifier" in name_lower
                    or "fc" in name_lower
                    or "linear" in name_lower
                ):
                    continue
                if "norm" in name_lower or name_lower.startswith("bn"):
                    penultimate_name = name
                    break

            if penultimate_name is None:
                for name in reversed(module_names):
                    name_lower = name.lower()
                    if (
                        "classifier" not in name_lower
                        and "fc" not in name_lower
                        and "linear" not in name_lower
                    ):
                        penultimate_name = name
                        break

            if penultimate_name is None:
                raise ValueError(
                    f"DenseNet {model_name}: Could not determine PENULTIMATE layer"
                )

            layers.append(penultimate_name)
            logger.info(f"  Found PENULTIMATE: {penultimate_name}")

            # Verify no transitions
            for layer_name in layers:
                if "transition" in layer_name.lower() or "trans" in layer_name.lower():
                    raise ValueError(
                        f"DenseNet {model_name}: Transition layer '{layer_name}' should not be included."
                    )

        # Enforce exactly 5 intermediate layers
        if len(layers) != 5:
            raise ValueError(
                f"DenseNet {model_name}: Expected exactly 5 intermediate layers "
                f"(PRE-BLOCK + BLOCK-1..BLOCK-3 + PENULTIMATE), got {len(layers)}. "
                f"Layers found: {layers}"
            )

        logger.info(f"DAC intermediate layers for {model_name}: {layers}")
        logger.info(f"  Expected: 5 intermediate + 1 logits = 6 total")
        logger.info(f"  Actual intermediate: {len(layers)}")

        return layers

    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}. "
            f"Supported: ResNet, DenseNet"
        )


def find_layer_module(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """
    Find a module in the model by its name (supports dot notation for nested modules).
    Also handles DINOv2 block names like 'block_0', 'block_4', etc.

    Args:
        model: PyTorch model
        layer_name: Layer name, can use dot notation (e.g., 'features.denseblock1')
                    or DINOv2 block format (e.g., 'block_0')

    Returns:
        The module if found, None otherwise
    """
    # Handle DINOv2 block names (e.g., 'block_0', 'block_4')
    if layer_name.startswith("block_"):
        try:
            block_index = int(layer_name.split("_")[1])

            # Pattern 1: timm-style (model.backbone.blocks[index])
            if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
                blocks = model.backbone.blocks
                if isinstance(
                    blocks, (nn.ModuleList, nn.Sequential)
                ) and block_index < len(blocks):
                    return blocks[block_index]

            # Pattern 2: Direct blocks access
            if hasattr(model, "blocks"):
                blocks = model.blocks
                if isinstance(
                    blocks, (nn.ModuleList, nn.Sequential)
                ) and block_index < len(blocks):
                    return blocks[block_index]

            # Pattern 3: HuggingFace-style
            if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):
                encoder = model.backbone.encoder
                if hasattr(encoder, "layer") and block_index < len(encoder.layer):
                    return encoder.layer[block_index]
                elif hasattr(encoder, "layers") and block_index < len(encoder.layers):
                    return encoder.layers[block_index]

            # Pattern 4: Direct encoder access
            if hasattr(model, "encoder"):
                encoder = model.encoder
                if hasattr(encoder, "layer") and block_index < len(encoder.layer):
                    return encoder.layer[block_index]
                elif hasattr(encoder, "layers") and block_index < len(encoder.layers):
                    return encoder.layers[block_index]

        except (ValueError, IndexError):
            pass

    # Standard handling for dot notation or direct attributes
    if "." in layer_name:
        parts = layer_name.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
    else:
        return getattr(model, layer_name) if hasattr(model, layer_name) else None


def extract_dac_features(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Extract intermediate features and logits separately for DAC.

    NOTE:
    - Extracts features for INTERMEDIATE layers (not logits)
    - Extracts logits separately by running full forward pass
    - Returns: (intermediate_features_list, logits, labels)

    To match the paper's experimental setup, append logits to the feature list:
        features_with_logits = intermediate_features_list + [logits]

    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        layer_names: List of intermediate layer names (from get_dac_target_layers)
        device: Device to run inference on

    Returns:
        Tuple of:
        - intermediate_features_list: List of tensors, one per intermediate layer
        - logits: Final layer logits (N, C)
        - labels: Ground truth labels (N,)
    """
    model.eval()

    layer_features = {name: [] for name in layer_names}
    logits_list = []
    labels_list = []

    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                feat = output
            elif isinstance(output, (tuple, list)):
                feat = output[0] if len(output) > 0 else None
            else:
                logger.warning(
                    f"Unexpected output type for {layer_name}: {type(output)}"
                )
                return

            if feat is not None:
                if feat.ndim == 4:
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1))
                    feat = pooled.squeeze(-1).squeeze(-1)
                elif feat.ndim == 3:
                    feat = feat.mean(dim=1)

                layer_features[layer_name].append(feat.detach().cpu())

        return hook_fn

    for layer_name in layer_names:
        module = find_layer_module(model, layer_name)
        if module is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        hook = module.register_forward_hook(make_hook(layer_name))
        hooks.append(hook)

    try:
        from tqdm import tqdm

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Extracting intermediate layer features"
            ):
                if len(batch) == 2:
                    data, labels = batch
                else:
                    data = batch[0]
                    labels = torch.zeros(len(data), dtype=torch.long)

                data = data.to(device)
                labels_list.append(labels)

                logits = model(data)
                logits_list.append(logits.detach().cpu())

    finally:
        for hook in hooks:
            hook.remove()

    intermediate_features_list = []
    for layer_name in layer_names:
        if not layer_features[layer_name]:
            raise ValueError(f"No features extracted for layer '{layer_name}'")
        features = torch.cat(layer_features[layer_name], dim=0)

        if features.ndim != 2:
            raise ValueError(
                f"Features from layer '{layer_name}' must be 2D [N, C], got shape {features.shape}"
            )

        intermediate_features_list.append(features)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    logger.info(f"Extracted features from {len(layer_names)} layers:")
    for i, (layer_name, feat) in enumerate(
        zip(layer_names, intermediate_features_list)
    ):
        logger.info(f"  Layer {i+1} ({layer_name}): {feat.shape}")
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    return intermediate_features_list, logits, labels


def extract_dac_features_memory_efficient(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    max_layers_per_batch: int = 4,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Extract DAC features with memory-efficient batching.
    Processes layers in groups to avoid OOM.

    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        layer_names: List of intermediate layer names (from get_dac_target_layers)
        device: Device to run inference on
        max_layers_per_batch: Maximum number of layers to process simultaneously

    Returns:
        Tuple of:
        - intermediate_features_list: List of tensors, one per layer
        - logits: Final layer logits (N, C)
        - labels: Ground truth labels (N,)
    """
    model.eval()

    layer_chunks = [
        layer_names[i : i + max_layers_per_batch]
        for i in range(0, len(layer_names), max_layers_per_batch)
    ]

    all_features = {name: [] for name in layer_names}
    logits_list = []
    labels_list = []

    logger.info(
        f"Processing {len(layer_names)} layers in {len(layer_chunks)} chunks of {max_layers_per_batch}"
    )

    for chunk_idx, layer_chunk in enumerate(layer_chunks, 1):
        logger.info(
            f"Processing layer chunk {chunk_idx}/{len(layer_chunks)}: {layer_chunk}"
        )

        chunk_features, chunk_logits, chunk_labels = extract_dac_features(
            model, dataloader, layer_chunk, device
        )

        for layer_name, feats in zip(layer_chunk, chunk_features):
            all_features[layer_name] = feats

        if chunk_idx == 1:
            logits_list = chunk_logits
            labels_list = chunk_labels

        if device.type == "cuda":
            torch.cuda.empty_cache()

    features_list = [all_features[name] for name in layer_names]

    return features_list, logits_list, labels_list


def get_dac_k_value(dataset_name: str) -> int:
    """
    Get the k value for KNN based on dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100', 'tiny_imagenet')

    Returns:
        k value: 50 for CIFAR-10, 200 for CIFAR-100 and Tiny ImageNet
    """
    dataset_lower = dataset_name.lower()
    # Check CIFAR-100 before CIFAR-10 (substring issue)
    if "cifar100" in dataset_lower or "cifar-100" in dataset_lower:
        return 200
    elif "cifar10" in dataset_lower or "cifar-10" in dataset_lower:
        return 50
    elif "tiny_imagenet" in dataset_lower or "tinyimagenet" in dataset_lower:
        return 200
    else:
        logger.warning(f"Unknown dataset {dataset_name}, defaulting to k=200")
        return 200


# =============================================================================
# PyTorch-based kNN Scorer
# =============================================================================


class LayerKNNScorer:
    """
    Computes density scores (s_l) for a specific layer using k-nearest neighbors.

    Uses PyTorch for GPU-accelerated kNN computation.

    Implements the preprocessing described in DAC Section 3.3:
    "average over spatial dimensions as well as normalize it."
    """

    def __init__(self, k: int = 50, use_gpu: bool = True):
        """
        Args:
            k: Number of nearest neighbors
            use_gpu: Whether to use GPU (if available)
        """
        self.k = k
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.train_features = None

    def _preprocess(self, features) -> torch.Tensor:
        """
        1. Spatial Averaging: If features are (N, C, H, W), average to (N, C).
        2. Normalization: L2 normalize.

        Returns:
            Preprocessed features as torch.Tensor on self.device
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        elif isinstance(features, torch.Tensor):
            features = features.float()

        features = features.to(self.device)

        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features.mean(dim=1)

        features = F.normalize(features, p=2, dim=-1)

        return features

    def fit(self, train_features):
        """
        Stores preprocessed training features for kNN queries.

        Args:
            train_features: Training features (N, C, H, W) or (N, C)
        """
        self.train_features = self._preprocess(train_features)

    def score(self, test_features) -> np.ndarray:
        """
        Calculates s_l: The distance to the k-th nearest neighbor.

        Uses PyTorch for GPU-accelerated distance computation.
        For L2-normalized vectors: ||a - b||^2 = 2 - 2*<a,b>

        Args:
            test_features: Test features (N, C, H, W) or (N, C)

        Returns:
            knn_distances: Distance to k-th nearest neighbor for each sample (N,)
        """
        if self.train_features is None:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        test_feats = self._preprocess(test_features)

        batch_size = 1024
        n_test = test_feats.shape[0]
        knn_distances = []

        for i in range(0, n_test, batch_size):
            batch = test_feats[i : i + batch_size]

            similarity = torch.mm(batch, self.train_features.t())

            distances_sq = 2.0 - 2.0 * similarity
            distances_sq = torch.clamp(distances_sq, min=0.0)
            distances = torch.sqrt(distances_sq)

            if self.k <= distances.shape[1]:
                kth_distances, _ = torch.kthvalue(distances, self.k, dim=1)
            else:
                kth_distances, _ = distances.max(dim=1)

            knn_distances.append(kth_distances.cpu().numpy())

        return np.concatenate(knn_distances)


# =============================================================================
# DAC Calibrator
# =============================================================================


class DensityAwareCalibrator:
    """
    Implements the DAC Algorithm exactly as per the paper.

    Model:
        S(x, w) = sum_{l=1}^L (w_l * s_l) + w_0  (Eq. 7)
        Q(x, w) = softmax(z_L / S(x, w))         (Eq. 6)

    Optimization:
        Minimize Squared Error Loss (Eq. 9)
        Constraints: w_l >= 0 (positive weights for layers)
    """

    def __init__(self, k: int = 50, use_gpu: bool = True):
        """
        Args:
            k: Number of nearest neighbors for density estimation
            use_gpu: Whether to use GPU (if available)
        """
        self.k = k
        self.use_gpu = use_gpu
        self.layer_scorers: List[LayerKNNScorer] = []
        self.weights = None
        self.num_layers = 0

        # Timing instrumentation
        self.timing = {
            "fit_knn_time": 0.0,
            "score_val_time": 0.0,
            "optimize_time": 0.0,
            "total_fit_time": 0.0,
        }

    def fit(self, train_features_list, val_features_list, val_logits, val_labels):
        """
        Fit DAC calibrator using layer features and logits.

        Args:
            train_features_list: List of arrays/tensors, one per layer
            val_features_list: List of arrays/tensors, one per layer
            val_logits: Final layer logits (N, C)
            val_labels: Ground truth labels (N,) integer labels OR (N, C) one-hot
        """
        total_start = time.perf_counter()

        self.num_layers = len(train_features_list)

        if isinstance(val_logits, torch.Tensor):
            val_logits = val_logits.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().cpu().numpy()

        train_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f for f in train_features_list
        ]
        val_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f for f in val_features_list
        ]

        if len(val_features_list) != self.num_layers:
            raise ValueError("Mismatch in number of layers between train and val.")

        # 1. Initialize and Fit KNN for each layer
        logger.info(f"Fitting KNN for {self.num_layers} layers...")
        knn_start = time.perf_counter()

        self.layer_scorers = []
        for i in range(self.num_layers):
            scorer = LayerKNNScorer(k=self.k, use_gpu=self.use_gpu)
            scorer.fit(train_features_list[i])
            self.layer_scorers.append(scorer)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timing["fit_knn_time"] = time.perf_counter() - knn_start

        # 2. Compute density scores for validation set
        logger.info("Computing validation density scores...")
        score_start = time.perf_counter()

        val_scores_list = []
        for i, (scorer, val_feats) in enumerate(
            zip(self.layer_scorers, val_features_list)
        ):
            s_l = scorer.score(val_feats)
            val_scores_list.append(s_l)

        val_knn_scores = np.column_stack(val_scores_list)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timing["score_val_time"] = time.perf_counter() - score_start

        # 3. Prepare labels as one-hot
        if val_labels.ndim == 1:
            num_classes = val_logits.shape[1]
            val_labels_onehot = np.eye(num_classes)[val_labels.astype(int)]
        else:
            val_labels_onehot = val_labels

        # 4. Optimization
        logger.info("Optimizing DAC weights using Squared Error Loss...")
        opt_start = time.perf_counter()

        x0 = np.ones(self.num_layers + 1)
        x0[-1] = 0.0

        bounds = [(0.0, None)] * self.num_layers + [(None, None)]

        result = optimize.minimize(
            fun=self._squared_error_loss,
            x0=x0,
            args=(val_logits, val_labels_onehot, val_knn_scores),
            method="L-BFGS-B",
            bounds=bounds,
            tol=1e-12,
        )

        self.timing["optimize_time"] = time.perf_counter() - opt_start

        self.weights = result.x
        self.timing["total_fit_time"] = time.perf_counter() - total_start

        logger.info(f"Optimization finished. Loss: {result.fun:.6f}")
        logger.info(f"Learned Weights: {self.weights[:-1]}, Bias: {self.weights[-1]}")
        logger.info(
            f"Timing breakdown: kNN={self.timing['fit_knn_time']:.2f}s, "
            f"score={self.timing['score_val_time']:.2f}s, "
            f"opt={self.timing['optimize_time']:.2f}s, "
            f"total={self.timing['total_fit_time']:.2f}s"
        )

    def _get_temperature(self, knn_scores_matrix):
        """Computes Eq (7): S(x,w) = sum(w_l * s_l) + w_0"""
        layer_weights = self.weights[:-1]
        bias = self.weights[-1]

        T = np.dot(knn_scores_matrix, layer_weights) + bias
        return np.maximum(T, 1e-12)

    def _squared_error_loss(self, weights, logits, labels_onehot, knn_scores):
        """Eq (9): L_w = sum_{c=1}^C (I_c - sigma(z/S)^(c))^2"""
        current_layer_weights = weights[:-1]
        current_bias = weights[-1]

        T = np.dot(knn_scores, current_layer_weights) + current_bias
        T = np.clip(T, 1e-12, None)

        scaled_logits = logits / T[:, np.newaxis]
        probs = np_softmax(scaled_logits)

        squared_diff = (labels_onehot - probs) ** 2
        loss = np.sum(squared_diff)

        return loss

    def get_scaled_logits(self, features_list, logits):
        """
        Returns the logits scaled by the density temperature S(x,w).
        Used for chaining with other calibrators (e.g., TS + DAC).
        """
        if self.weights is None:
            raise ValueError("Calibrator not fitted.")

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f for f in features_list
        ]

        if len(features_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} feature layers, got {len(features_list)}"
            )

        scores_list = []
        for i, scorer in enumerate(self.layer_scorers):
            s_l = scorer.score(features_list[i])
            scores_list.append(s_l)

        knn_scores = np.column_stack(scores_list)
        T = self._get_temperature(knn_scores)
        scaled_logits = logits / T[:, np.newaxis]

        return scaled_logits

    def calibrate(self, test_features_list, test_logits):
        """
        Applies the learned calibration to test data.
        Returns: Calibrated probabilities
        """
        if self.weights is None:
            raise ValueError("Calibrator not fitted.")

        if isinstance(test_logits, torch.Tensor):
            test_logits = test_logits.detach().cpu().numpy()

        test_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f for f in test_features_list
        ]

        if len(test_features_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} feature layers, got {len(test_features_list)}"
            )

        scaled_logits = self.get_scaled_logits(test_features_list, test_logits)
        calibrated_probs = np_softmax(scaled_logits)

        return calibrated_probs

    def get_timing(self) -> dict:
        """Returns timing breakdown from last fit() call."""
        return self.timing.copy()
