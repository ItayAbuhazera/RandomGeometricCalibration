"""
Shared utilities for layer discovery and filtering.

Provides consistent filtering of activation and non-feature-bearing layers
across all layer discovery mechanisms.
"""
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Patterns to exclude from feature layer selection
EXCLUDED_LAYER_PATTERNS = [
    'relu', 'gelu', 'silu', 'leakyrelu', 'sigmoid', 'tanh', 
    'dropout', 'identity', 'activation'
]

# Module types to exclude (for isinstance checks)
EXCLUDED_LAYER_TYPES = (
    nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,  # Activations
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,  # Dropout
    nn.Identity,  # Identity
)


def is_feature_layer(name: str, module: Optional[nn.Module] = None) -> bool:
    """
    Check if layer contains learned features (not activations/dropout).
    
    Args:
        name: Layer name (e.g., "features.0.relu" or "layer1.0.conv1")
        module: Optional PyTorch module instance for type checking
    
    Returns:
        True if layer should be included, False if it's an activation/dropout
    """
    name_lower = name.lower()
    
    # Check name patterns
    if any(pattern in name_lower for pattern in EXCLUDED_LAYER_PATTERNS):
        return False
    
    # Check module type if provided
    if module is not None:
        if isinstance(module, EXCLUDED_LAYER_TYPES):
            return False
        module_type = type(module).__name__.lower()
        if any(pattern in module_type for pattern in EXCLUDED_LAYER_PATTERNS):
            return False
    
    return True


def filter_feature_layers(
    layers: List[Dict[str, Any]], 
    model: Optional[nn.Module] = None
) -> List[Dict[str, Any]]:
    """
    Filter out activation and non-feature layers from discovered layers.
    
    Args:
        layers: List of layer dictionaries with at least a 'name' key
        model: Optional model to check module types (if layers have 'idx' key)
    
    Returns:
        Filtered list of feature-bearing layers
    """
    if not layers:
        return []

    # Build module index mapping if model is provided
    idx_to_module = {}
    if model is not None:
        modules = list(model.modules())
        idx_to_module = {i: m for i, m in enumerate(modules)}

    filtered = []
    for layer in layers:
        layer_name = layer.get('name', '')
        layer_idx = layer.get('idx')

        # Get module if available
        module = None
        if layer_idx is not None and layer_idx in idx_to_module:
            module = idx_to_module[layer_idx]

        # Check if this is a feature layer
        if is_feature_layer(layer_name, module):
            filtered.append(layer)

    excluded_count = len(layers) - len(filtered)
    if excluded_count > 0:
        logger.info(f"Filtered {excluded_count} non-feature layers (activations, dropout, etc.)")
        excluded_names = [l.get('name', 'unknown') for l in layers if l not in filtered]
        if len(excluded_names) <= 10:
            logger.debug(f"Excluded layers: {excluded_names}")
        else:
            logger.debug(f"Excluded layers: {excluded_names[:10]}... (and {len(excluded_names) - 10} more)")

    return filtered


# ==============================================================================
# Layer selection helpers for TULIP IC placement
# ==============================================================================
def select_tulip_layers(model, model_name: str, num_fallback: int = 7) -> List[str]:
    """
    Select strategic layers for TULIP internal classifiers.
    Strategy: Target the last BatchNorm of residual blocks.
    - ResNet-18/34 (BasicBlock): Target .bn2
    - ResNet-50+ (Bottleneck): Target .bn3
    """
    all_layers = []
    for name, module in model.named_modules():
        # Keep only leaf modules (no children) with a non-empty name
        if name and not list(module.children()):
            all_layers.append(name)

    model_name = model_name.lower()

    # Architecture-specific target patterns
    if "resnet18" in model_name:
        # 9 Layers: Stem + End of every block
        target_patterns = [
            "bn1",  # Stem
            "layer1.0.bn2",
            "layer1.1.bn2",
            "layer2.0.bn2",
            "layer2.1.bn2",
            "layer3.0.bn2",
            "layer3.1.bn2",
            "layer4.0.bn2",
            "layer4.1.bn2",
        ]
    elif "resnet34" in model_name:
        # ~9 Layers spaced out (ResNet34 has 3,4,6,3 blocks)
        target_patterns = [
            "bn1",  # Stem
            "layer1.2.bn2",  # L1 End
            "layer2.1.bn2",
            "layer2.3.bn2",  # L2 Mid/End
            "layer3.1.bn2",
            "layer3.3.bn2",
            "layer3.5.bn2",  # L3 Early/Mid/End
            "layer4.0.bn2",
            "layer4.2.bn2",  # L4 Start/End
        ]
    elif "resnet50" in model_name:
        # ~9 Layers spaced out (ResNet50 has 3,4,6,3 Bottleneck blocks)
        # Note: Bottleneck blocks end with .bn3, not .bn2
        target_patterns = [
            "bn1",  # Stem
            "layer1.2.bn3",  # L1 End
            "layer2.1.bn3",
            "layer2.3.bn3",  # L2 Mid/End
            "layer3.1.bn3",
            "layer3.3.bn3",
            "layer3.5.bn3",  # L3 Early/Mid/End
            "layer4.1.bn3",
            "layer4.2.bn3",  # L4 Mid/End
        ]
    elif "resnet101" in model_name:
        # Spaced out for depth (3,4,23,3 blocks)
        target_patterns = [
            "bn1",
            "layer1.2.bn3",
            "layer2.3.bn3",
            "layer3.3.bn3",
            "layer3.7.bn3",
            "layer3.11.bn3",
            "layer3.15.bn3",
            "layer3.19.bn3",
            "layer3.22.bn3",
            "layer4.2.bn3",
        ]
    elif "resnet152" in model_name:
        target_patterns = [
            "bn1",  # Stem
            "layer1.2.bn3",  # L1 End
            "layer2.3.bn3",
            "layer2.7.bn3",  # L2 Mid/End
            "layer3.7.bn3",
            "layer3.15.bn3",
            "layer3.23.bn3",
            "layer3.31.bn3",
            "layer3.35.bn3",  # L3 Distributed
            "layer4.0.bn3",
            "layer4.2.bn3",  # L4 Start/End
        ]

    else:
        # Generic evenly spaced fallback
        n_layers = len(all_layers)
        step = max(1, n_layers // num_fallback)
        return [all_layers[i] for i in range(0, n_layers, step)][:num_fallback]

    # Match target patterns to actual layer names
    selected: List[str] = []
    # Create a set for O(1) lookups to avoid partial matching issues
    all_layers_set = set(all_layers)

    for pattern in target_patterns:
        # Exact match check first (safest)
        if pattern in all_layers_set:
            selected.append(pattern)
            continue

        # Fallback to substring matching if exact name not found
        # (Useful if model wrappers add prefixes)
        for layer_name in all_layers:
            if pattern in layer_name:
                # Ensure we take the most specific match
                if not any(
                    pattern in other and len(other) > len(layer_name)
                    for other in all_layers
                ):
                    selected.append(layer_name)
                    break

    # Fallback if pattern matching fails
    if len(selected) < 5:
        logger.warning(
            f"Pattern matching found only {len(selected)} layers for {model_name}; using evenly spaced fallback"
        )
        n_layers = len(all_layers)
        step = max(1, n_layers // num_fallback)
        selected = [all_layers[i] for i in range(0, n_layers, step)][:num_fallback]

    return selected
