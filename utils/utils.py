import os
import sys
import json
import logging
import argparse
import gc
import tempfile
import errno
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

# Import calibrator
from Calibrators.geometric_calibrator import GeometricCalibrator

# Same utilities you used before
from utils.model_utils import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
)


from utils.logging_config import get_logger
from utils.layer_utils import filter_feature_layers

logger = get_logger(__name__)


def _safe_to_tensor(out):
    """Extract a tensor from a module's output, handling tuples/lists/dicts."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item):
                return item
        return None
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v):
                return v
        return None
    return None

def discover_model_layers(
    model,
    *,
    device="cuda",
    input_shape=(1, 3, 32, 32),
    keep_kinds=(
        "Conv",
        "BatchNorm",
        "ReLU",
        "SiLU",
        "GELU",
        "Bottleneck",
        "BasicBlock",
        "Linear",
    ),
    always_include_last=True,
) -> list:
    """
    Returns a list of dicts describing extractable layers in forward order:
      {'idx': int, 'name': str, 'type': str, 'shape': tuple|None, 'spatial': int|None}
    We consider a layer 'valid' if it yields a tensor output on a dummy forward hook.

    Uses single-pass forward with all hooks registered at once for efficiency.
    """
    model = model.to(device).eval()
    modules = list(model.modules())
    dummy = torch.randn(*input_shape, device=device)

    # Register all hooks at once
    acts = [None] * len(modules)
    hooks = []

    def make_hook(i):
        def _h(mod, inp, out):
            t = _safe_to_tensor(out)
            acts[i] = t.detach() if torch.is_tensor(t) else None

        return _h

    for i, m in enumerate(modules):
        hooks.append(m.register_forward_hook(make_hook(i)))

    # Single forward pass
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception as e:
            logger.warning(f"Forward pass failed during discovery: {e}")

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Build discovered list
    module_names = {}
    for name, mod in model.named_modules():
        module_names[id(mod)] = name

    discovered = []
    for idx, (m, t) in enumerate(zip(modules, acts)):
        mtype = m.__class__.__name__

        # Keep "semantic" modules by type hint
        keep = any(k in mtype for k in keep_kinds)
        if keep and t is not None:
            tshape = tuple(t.shape)
            spatial = int(t.shape[2]) if t.dim() >= 3 else None
            discovered.append(
                {
                    "idx": idx,
                    "name": module_names.get(id(m), mtype),
                    "type": mtype,
                    "shape": tshape,
                    "spatial": spatial,
                }
            )

    # Filter out activation and non-feature layers before returning
    filtered = filter_feature_layers(discovered, model)

    return filtered
