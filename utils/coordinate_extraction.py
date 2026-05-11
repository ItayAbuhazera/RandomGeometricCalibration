import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
from .layer_utils import is_feature_layer

# Configure local logger
logger = logging.getLogger(__name__)


def coerce_to_tensor(output: Any) -> torch.Tensor:
    """Safely coerce module output to a tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    return None


def find_layer_module(model: nn.Module, layer_name: str) -> nn.Module:
    try:
        return dict(model.named_modules())[layer_name]
    except KeyError:
        return None


def discover_coordinate_space(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Tuple[List[Dict], int]:
    """
    Map the coordinate space using (BaseName + CallCount) as the unique ID.
    Filters out activation and non-feature-bearing layers.
    """
    logger.info(f"Starting coordinate discovery on device {device} with input {input_shape}")
    model = model.to(device).eval()
    dummy_input = torch.zeros(input_shape, device=device)

    execution_trace = []

    def get_trace_hook(name):
        def hook(module, input, output):
            tensor = coerce_to_tensor(output)
            if tensor is not None:
                execution_trace.append(
                    {
                        "base_name": name,
                        "shape": tensor.shape,
                        "size": tensor[0].numel(),
                        "module": module,  # Store module reference for filtering
                    }
                )

        return hook

    hooks = []
    for name, module in model.named_modules():
        # Hook leaf modules only
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(get_trace_hook(name)))

    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception as e:
        logger.error(f"Discovery forward pass failed: {e}")
        raise e
    finally:
        for h in hooks:
            h.remove()

    # Filter out activation and non-feature layers using shared utility

    # Convert trace to unique layer map, filtering out non-feature layers
    layer_map = []
    global_idx = 0
    name_counters = defaultdict(int)
    filtered_count = 0

    for item in execution_trace:
        base_name = item["base_name"]
        module = item.get("module")
        
        # Filter out activation and non-feature layers
        if not is_feature_layer(base_name, module):
            filtered_count += 1
            continue
        
        call_idx = name_counters[base_name]
        name_counters[base_name] += 1

        unique_id = f"{base_name}#{call_idx}"
        size = item["size"]

        layer_map.append(
            {
                "name": unique_id,  # UNIQUE ID used for planning
                "base_name": base_name,  # PYTORCH ID used for hooking
                "call_idx": call_idx,  # EXECUTION ID used for filtering
                "start_idx": global_idx,
                "end_idx": global_idx + size,
                "size": size,
                "shape": item["shape"],
            }
        )
        global_idx += size

    logger.info(f"Coordinate space: {len(layer_map)} feature layers ({filtered_count} activation/dropout layers excluded)")
    logger.info(f"Discovery complete. Found {len(layer_map)} unique layer calls. Total size: {global_idx}")
    return layer_map, global_idx


def plan_coordinate_extraction(
    total_size: int,
    num_coordinates: int,
    layer_map: List[Dict],
    seed: int,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Standard independent sampling plan."""
    np.random.seed(seed)

    if num_coordinates > total_size:
        global_indices = np.arange(total_size)
    else:
        global_indices = np.random.choice(total_size, num_coordinates, replace=False)

    global_indices.sort()

    sampling_plan = {}
    current_k = 0
    total_k = len(global_indices)

    # Iterate through LAYERS (which are sorted by start_idx)
    for layer in layer_map:
        if current_k >= total_k:
            break

        start = layer["start_idx"]
        end = layer["end_idx"]

        indices_in_layer = []
        # Consume global indices that fit in this layer
        while current_k < total_k and global_indices[current_k] < end:
            idx = global_indices[current_k]
            if idx >= start:
                indices_in_layer.append(idx - start)
            current_k += 1

        if indices_in_layer:
            sampling_plan[layer["name"]] = indices_in_layer

    return sampling_plan, global_indices.tolist()


def plan_nested_coordinate_extraction(
    total_size: int,
    coordinate_counts: List[int],
    layer_map: List[Dict],
    seed: int,
) -> Tuple[Dict[int, Tuple[Dict, List]], List[int]]:
    """Nested sampling plan."""
    np.random.seed(seed)
    max_k = max(coordinate_counts)

    if max_k > total_size:
        master_indices = np.arange(total_size)
        np.random.shuffle(master_indices)
    else:
        master_indices = np.random.choice(total_size, max_k, replace=False)

    plans = {}

    for k in coordinate_counts:
        subset = np.sort(master_indices[:k])
        plan_k = {}
        curr = 0
        tot = len(subset)

        for layer in layer_map:
            if curr >= tot:
                break

            start = layer["start_idx"]
            end = layer["end_idx"]

            inds = []
            while curr < tot and subset[curr] < end:
                g_idx = subset[curr]
                if g_idx >= start:
                    inds.append(g_idx - start)
                curr += 1

            if inds:
                plan_k[layer["name"]] = inds

        plans[k] = (plan_k, subset.tolist())

    return plans, master_indices.tolist()


def extract_coordinate_features(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    sampling_plan: Dict[str, List[int]],
    layer_map: List[Dict],
    global_order: List[int],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract features using (BaseName + CallCount) logic with DEBUG logging.
    """
    model = model.to(device).eval()

    # 1. Organize plan by BaseName -> CallIdx -> Indices
    hooks_config = defaultdict(dict)

    # Validation check: ensure we have layer info for everything in the plan
    for unique_id, indices in sampling_plan.items():
        found = False
        for l in layer_map:
            if l["name"] == unique_id:
                base_name = l["base_name"]
                call_idx = l["call_idx"]
                start_global = l["start_idx"]
                hooks_config[base_name][call_idx] = {
                    "indices": indices,
                    "start_global": start_global,
                    "unique_id": unique_id,
                }
                found = True
                break
        if not found:
            logger.warning(f"Plan contains layer '{unique_id}' not found in layer_map!")

    global_to_col = {g: i for i, g in enumerate(global_order)}
    num_samples = len(data_loader.dataset)
    num_features = len(global_order)
    all_features = torch.zeros((num_samples, num_features), dtype=torch.float32)

    current_sample = 0

    # DEBUG: Track extraction stats
    stats = {"calls_seen": 0, "calls_matched": 0, "features_extracted": 0}

    def make_hook(base_name, config_map):
        state = {"call_counter": 0}

        def hook(module, input, output):
            tensor = coerce_to_tensor(output)
            if tensor is None:
                return

            current_call = state["call_counter"]
            state["call_counter"] += 1
            stats["calls_seen"] += 1

            if current_call in config_map:
                stats["calls_matched"] += 1
                cfg = config_map[current_call]

                B = tensor.shape[0]
                flat = tensor.reshape(B, -1)

                local_indices = cfg["indices"]
                max_valid = flat.shape[1]

                # Filter invalid indices
                valid_local = [idx for idx in local_indices if idx < max_valid]

                if not valid_local:
                    logger.debug(
                        f"Skipping {cfg['unique_id']}: all indices out of bounds (max {max_valid})"
                    )
                    return

                valid_tensor_indices = torch.tensor(valid_local, device=tensor.device)
                selected_vals = flat[:, valid_tensor_indices].cpu()

                stats["features_extracted"] += selected_vals.numel()

                # Assign to output
                for i, local_idx in enumerate(valid_local):
                    global_idx = cfg["start_global"] + local_idx
                    if global_idx in global_to_col:
                        col_idx = global_to_col[global_idx]
                        all_features[
                            current_sample : current_sample + B, col_idx
                        ] = selected_vals[:, i]

        hook.state = state
        return hook

    # Register hooks
    hooks = []
    registered_names = set()
    for name, module in model.named_modules():
        if name in hooks_config:
            h = make_hook(name, hooks_config[name])
            hooks.append((module.register_forward_hook(h), h))
            registered_names.add(name)

    logger.info(f"Registered extraction hooks on {len(registered_names)} unique modules.")

    # Inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)

            # Reset call counters
            for _, h_fn in hooks:
                h_fn.state["call_counter"] = 0

            model(inputs)
            current_sample += inputs.shape[0]

            if batch_idx == 0:
                logger.info(
                    f"Batch 0 stats: calls_seen={stats['calls_seen']}, "
                    f"matched={stats['calls_matched']}, features={stats['features_extracted']}"
                )

    for h, _ in hooks:
        h.remove()

    # Final check
    non_zeros = torch.count_nonzero(all_features).item()
    total_elements = all_features.numel()
    sparsity = 1.0 - (non_zeros / total_elements) if total_elements > 0 else 0

    logger.info(
        f"Extraction complete. Sparsity: {sparsity*100:.2f}% (Non-zero: {non_zeros}/{total_elements})"
    )

    if non_zeros == 0:
        logger.warning("!!! WARNING: Extracted feature matrix is ALL ZEROS. Check layer mapping !!!")

    return all_features

