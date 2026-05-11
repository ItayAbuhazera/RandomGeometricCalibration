#!/usr/bin/env python3
"""
Generate Figure 2 of the paper: Layer-wise Feature Norm Comparison.

Plots the mean L2 norm of pooled feature vectors across depth on CIFAR-100
for ResNet-152, DenseNet-121, and DINOv2-Large, illustrating that deeper
layers exhibit 5x-30x higher norms than early layers.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


TORCHVISION_DENSENET_LAYERS = [
    'features.pool0',
    'features.transition1',
    'features.transition2',
    'features.transition3',
    'features.norm5',
]

CUSTOM_DENSENET_LAYERS = ['conv1', 'trans1', 'trans2', 'trans3', 'bn']

DENSENET_LAYER_DISPLAY_MAP = {
    'features.pool0': 'Initial Pool',
    'features.transition1': 'Transition 1',
    'features.transition2': 'Transition 2',
    'features.transition3': 'Transition 3',
    'features.norm5': 'Final Norm',
    'conv1': 'Initial Conv',
    'trans1': 'Transition 1',
    'trans2': 'Transition 2',
    'trans3': 'Transition 3',
    'bn': 'Final Norm',
}


def load_resnet50(checkpoint_path=None, num_classes=100, device='cuda'):
    """
    Load ResNet-50 model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses torchvision pretrained.
        num_classes: Number of classes (100 for CIFAR-100).
        device: Device to load model on.
    
    Returns:
        Model in eval mode.
    """
    if checkpoint_path is None:
        # Use torchvision pretrained ResNet-50
        print("Loading torchvision pretrained ResNet-50...")
        model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        # Replace final layer for CIFAR-100
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Load from checkpoint
        print(f"Loading ResNet-50 from checkpoint: {checkpoint_path}")
        # Try to load using project's model loading logic
        try:
            # Import project's model loading function
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(project_root))
            
            from utils.model_utils import load_trained_model
            model = load_trained_model(
                checkpoint_path, 
                'resnet50', 
                num_classes, 
                device, 
                dataset='cifar100'
            )
        except Exception as e:
            print(f"Failed to load using project's loader: {e}")
            print("Attempting direct torch.load...")
            # Fallback: try direct loading
            from Net.resnet import resnet50
            model = resnet50(num_classes=num_classes)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def load_resnet101(checkpoint_path=None, num_classes=100, device='cuda'):
    """
    Load ResNet-101 model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses torchvision pretrained.
        num_classes: Number of classes (100 for CIFAR-100).
        device: Device to load model on.
    
    Returns:
        Model in eval mode.
    """
    if checkpoint_path is None:
        # Use torchvision pretrained ResNet-101
        print("Loading torchvision pretrained ResNet-101...")
        model = torchvision.models.resnet101(weights='IMAGENET1K_V2')
        # Replace final layer for CIFAR-100
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Load from checkpoint
        print(f"Loading ResNet-101 from checkpoint: {checkpoint_path}")
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(project_root))
            
            from utils.model_utils import load_trained_model
            model = load_trained_model(
                checkpoint_path, 
                'resnet101', 
                num_classes, 
                device, 
                dataset='cifar100'
            )
        except Exception as e:
            print(f"Failed to load using project's loader: {e}")
            print("Attempting direct torch.load...")
            from Net.resnet import resnet101
            model = resnet101(num_classes=num_classes)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def load_resnet152(checkpoint_path=None, num_classes=100, device='cuda'):
    """
    Load ResNet-152 model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses torchvision pretrained.
        num_classes: Number of classes (100 for CIFAR-100).
        device: Device to load model on.
    
    Returns:
        Model in eval mode.
    """
    if checkpoint_path is None:
        # Use torchvision pretrained ResNet-152
        print("Loading torchvision pretrained ResNet-152...")
        model = torchvision.models.resnet152(weights='IMAGENET1K_V2')
        # Replace final layer for CIFAR-100
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Load from checkpoint
        print(f"Loading ResNet-152 from checkpoint: {checkpoint_path}")
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(project_root))
            
            from utils.model_utils import load_trained_model
            model = load_trained_model(
                checkpoint_path, 
                'resnet152', 
                num_classes, 
                device, 
                dataset='cifar100'
            )
        except Exception as e:
            print(f"Failed to load using project's loader: {e}")
            print("Attempting direct torch.load...")
            from Net.resnet import resnet152
            model = resnet152(num_classes=num_classes)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def load_densenet121(checkpoint_path=None, num_classes=100, device='cuda'):
    """
    Load DenseNet-121 model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses torchvision pretrained.
        num_classes: Number of classes (100 for CIFAR-100).
        device: Device to load model on.
    
    Returns:
        Model in eval mode.
    """
    if checkpoint_path is None:
        # Use torchvision pretrained DenseNet-121
        print("Loading torchvision pretrained DenseNet-121...")
        model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        # Replace final layer for CIFAR-100
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        # Load from checkpoint
        print(f"Loading DenseNet-121 from checkpoint: {checkpoint_path}")
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(project_root))
            
            from utils.model_utils import load_trained_model
            model = load_trained_model(
                checkpoint_path, 
                'densenet121', 
                num_classes, 
                device, 
                dataset='cifar100'
            )
        except Exception as e:
            print(f"Failed to load using project's loader: {e}")
            print("Attempting direct torch.load...")
            from Net.densenet import densenet121
            model = densenet121(num_classes=num_classes)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def load_dinov2(model_size='large', device='cuda'):
    """
    Load DINOv2 model from torch hub.
    
    Args:
        model_size: 'small', 'base', 'large', or 'giant'
        device: Device to load model on.
    
    Returns:
        Model in eval mode.
    """
    model_name_map = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14',
        'large': 'dinov2_vitl14',
        'giant': 'dinov2_vitg14'
    }
    
    if model_size not in model_name_map:
        raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(model_name_map.keys())}")
    
    model_name = model_name_map[model_size]
    print(f"Loading {model_name} from torch hub...")
    model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    return model


def get_cifar100_test_loader(batch_size=256, num_workers=4):
    """
    Get CIFAR-100 test dataloader with standard transforms.
    
    Args:
        batch_size: Batch size for dataloader.
        num_workers: Number of worker processes.
    
    Returns:
        DataLoader for CIFAR-100 test set.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


def extract_layer_features(model, dataloader, device, layers_to_extract):
    """
    Extract features from specified layers and compute mean L2 norms.
    
    Args:
        model: ResNet-50 model.
        dataloader: DataLoader for test set.
        device: Device to run on.
        layers_to_extract: List of layer names to extract (e.g., ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']).
    
    Returns:
        Dictionary mapping layer names to mean L2 norms.
    """
    # Register hooks to capture layer outputs
    features = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # Special handling for custom ResNet: hook into layer1 input to capture initial conv output
    initial_conv_features = None
    if 'maxpool' not in layers_to_extract and 'layer1' in layers_to_extract:
        # Hook into layer1's input to get output after conv1+bn1+relu
        def get_initial_conv_hook():
            def hook(module, input, output):
                nonlocal initial_conv_features
                initial_conv_features = input[0].detach()  # input is a tuple
            return hook
        
        layer1_module = dict(model.named_modules())['layer1']
        hook = layer1_module.register_forward_pre_hook(get_initial_conv_hook())
        hooks.append(hook)
        print("Registered hook to capture initial conv block output (via layer1 input)")
    
    # Register hooks for each layer
    for layer_name in layers_to_extract:
        try:
            layer = dict(model.named_modules())[layer_name]
            hook = layer.register_forward_hook(get_activation(layer_name))
            hooks.append(hook)
        except KeyError:
            print(f"Warning: Layer '{layer_name}' not found in model.")
    
    # Accumulate norms for each layer
    # Add 'initial_conv' if we're capturing it separately (for custom ResNet without maxpool)
    all_layer_names = layers_to_extract.copy()
    capture_initial_conv = ('maxpool' not in layers_to_extract and 'layer1' in layers_to_extract)
    if capture_initial_conv:
        all_layer_names.insert(0, 'initial_conv')
    
    layer_norms = {layer: [] for layer in all_layer_names}
    
    print("Extracting features from test set...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Reset initial_conv_features for this batch
            if capture_initial_conv:
                initial_conv_features = None
            
            # Forward pass
            _ = model(images)
            
            # Process initial conv features (if captured separately)
            if capture_initial_conv and initial_conv_features is not None:
                layer_output = initial_conv_features
                
                # Apply Global Average Pooling if needed
                if len(layer_output.shape) == 4:
                    pooled = F.adaptive_avg_pool2d(layer_output, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                else:
                    pooled = layer_output.flatten(1)
                
                norms = pooled.norm(dim=1)
                layer_norms['initial_conv'].append(norms.cpu())
            
            # Process each layer's features
            for layer_name in layers_to_extract:
                if layer_name not in features:
                    continue
                
                layer_output = features[layer_name]
                
                # Apply Global Average Pooling if needed
                # Check if output is 4D (B, C, H, W) or already 2D (B, C)
                if len(layer_output.shape) == 4:
                    # Global Average Pooling: (B, C, H, W) -> (B, C)
                    pooled = F.adaptive_avg_pool2d(layer_output, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                else:
                    # Already flattened or 2D
                    pooled = layer_output.flatten(1)
                
                # Compute L2 norm for each image in batch
                # norms = [features[layer].flatten(1).norm(dim=1).mean() for layer in layers]
                norms = pooled.norm(dim=1)  # Shape: (batch_size,)
                layer_norms[layer_name].append(norms.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute mean norms across entire dataset
    mean_norms = {}
    for layer_name in all_layer_names:
        if layer_norms[layer_name]:
            all_norms = torch.cat(layer_norms[layer_name], dim=0)
            mean_norms[layer_name] = all_norms.mean().item()
        else:
            mean_norms[layer_name] = 0.0
    
    return mean_norms


def extract_densenet_layer_features(model, dataloader, device, layers_to_extract=None):
    """
    Extract features from specified layers in DenseNet and compute mean L2 norms.
    
    DenseNet structure:
    Torchvision DenseNet:
    - features.conv0: Initial convolution
    - features.norm0, features.relu0, features.pool0: Initial normalization/activation/pooling
    - features.denseblock1, features.transition1: First dense block and transition
    - features.denseblock2, features.transition2: Second dense block and transition
    - features.denseblock3, features.transition3: Third dense block and transition
    - features.denseblock4: Final dense block
    - features.norm5: Final normalization
    Custom Net/densenet.py DenseNet:
    - conv1: Initial convolution
    - trans1, trans2, trans3: Transition layers
    - bn: Final normalization
    
    Args:
        model: DenseNet model.
        dataloader: DataLoader for test set.
        device: Device to run on.
        layers_to_extract: List of layer names to extract. If None, uses default layers.
    
    Returns:
        Dictionary mapping layer names to mean L2 norms.
    """
    if layers_to_extract is None:
        if hasattr(model, 'features'):
            # Torchvision DenseNet
            layers_to_extract = TORCHVISION_DENSENET_LAYERS.copy()
            layer_display_map = {
                'features.pool0': 'Initial Pool',
                'features.transition1': 'Transition 1',
                'features.transition2': 'Transition 2',
                'features.transition3': 'Transition 3',
                'features.norm5': 'Final Norm',
            }
        elif hasattr(model, 'trans1') and hasattr(model, 'dense1'):
            # Custom DenseNet from Net/densenet.py
            layers_to_extract = CUSTOM_DENSENET_LAYERS.copy()
            layer_display_map = {
                'conv1': 'Initial Conv',
                'trans1': 'Transition 1',
                'trans2': 'Transition 2',
                'trans3': 'Transition 3',
                'bn': 'Final Norm',
            }
        else:
            raise ValueError(
                "Cannot auto-detect DenseNet variant. "
                "Pass layers_to_extract explicitly."
            )
        extract_densenet_layer_features.layer_display_map = layer_display_map
    else:
        extract_densenet_layer_features.layer_display_map = {
            layer: DENSENET_LAYER_DISPLAY_MAP.get(layer, layer.split('.')[-1].capitalize())
            for layer in layers_to_extract
        }
    
    # Register hooks to capture layer outputs
    features = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hooks for each layer
    for layer_name in layers_to_extract:
        try:
            # Navigate through nested modules
            parts = layer_name.split('.')
            layer = model
            for part in parts:
                layer = getattr(layer, part)
            hook = layer.register_forward_hook(get_activation(layer_name))
            hooks.append(hook)
        except (KeyError, AttributeError) as e:
            print(f"Warning: Layer '{layer_name}' not found in model: {e}")
    
    # Accumulate norms for each layer
    layer_norms = {layer: [] for layer in layers_to_extract}
    
    print("Extracting features from test set...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            _ = model(images)
            
            # Process each layer's features
            for layer_name in layers_to_extract:
                if layer_name not in features:
                    continue
                
                layer_output = features[layer_name]
                
                # Apply Global Average Pooling if needed
                if len(layer_output.shape) == 4:
                    pooled = F.adaptive_avg_pool2d(layer_output, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                else:
                    pooled = layer_output.flatten(1)
                
                # Compute L2 norm for each image in batch
                norms = pooled.norm(dim=1)
                layer_norms[layer_name].append(norms.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute mean norms across entire dataset
    mean_norms = {}
    for layer_name in layers_to_extract:
        if layer_norms[layer_name]:
            all_norms = torch.cat(layer_norms[layer_name], dim=0)
            mean_norms[layer_name] = all_norms.mean().item()
        else:
            mean_norms[layer_name] = 0.0
    
    return mean_norms


def extract_transformer_block_features(model, dataloader, device, num_blocks=None):
    """
    Extract features from each transformer block in DINOv2.
    
    DINOv2 structure:
    - patch_embed: Initial patch embedding
    - blocks[0..N-1]: Transformer blocks (N=12 for Small/Base, N=24 for Large/Giant)
    - norm: Final layer norm
    
    For each block, extract the output and compute mean L2 norm.
    
    Args:
        model: DINOv2 model
        dataloader: DataLoader for images
        device: Device
        num_blocks: Number of blocks to extract (None = all, or sample evenly if > 12)
    
    Returns:
        Dictionary mapping layer names to mean L2 norms
    """
    features = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            # Handle tuple outputs (some blocks return tuples)
            if isinstance(output, tuple):
                output = output[0]
            features[name] = output.detach()
        return hook
    
    # Hook into patch embedding
    patch_embed = None
    if hasattr(model, 'patch_embed'):
        patch_embed = model.patch_embed
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'embeddings'):
        patch_embed = model.backbone.embeddings
    elif hasattr(model, 'embeddings'):
        patch_embed = model.embeddings
    
    if patch_embed is not None:
        hook = patch_embed.register_forward_hook(get_activation('patch_embed'))
        hooks.append(hook)
    
    # Hook into transformer blocks
    # DINOv2 can have different structures:
    # 1. torch hub: model.blocks
    # 2. HuggingFace: model.backbone.encoder.layer or model.backbone.encoder.layers
    # 3. timm: model.blocks or model.backbone.blocks
    blocks = None
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        blocks = model.backbone.blocks
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
        encoder = model.backbone.encoder
        if hasattr(encoder, 'layer'):
            blocks = encoder.layer
        elif hasattr(encoder, 'layers'):
            blocks = encoder.layers
    elif hasattr(model, 'encoder'):
        encoder = model.encoder
        if hasattr(encoder, 'layer'):
            blocks = encoder.layer
        elif hasattr(encoder, 'layers'):
            blocks = encoder.layers
    
    if blocks is None or not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
        raise ValueError("Could not find transformer blocks in DINOv2 model. "
                        "Expected model.blocks, model.backbone.blocks, or model.backbone.encoder.layer")
    
    total_blocks = len(blocks)
    
    # Sample blocks evenly if there are many (e.g., for Large/Giant with 24 blocks)
    if total_blocks > 12:
        # Sample ~8 blocks evenly spaced
        block_indices = np.linspace(0, total_blocks - 1, 8, dtype=int).tolist()
        block_indices = sorted(set(block_indices))  # Remove duplicates and sort
    else:
        block_indices = list(range(total_blocks))
    
    # Limit to num_blocks if specified
    if num_blocks is not None:
        if total_blocks > num_blocks:
            block_indices = np.linspace(0, total_blocks - 1, num_blocks, dtype=int).tolist()
            block_indices = sorted(set(block_indices))
        else:
            block_indices = list(range(total_blocks))
    
    for i in block_indices:
        hook = blocks[i].register_forward_hook(get_activation(f'block_{i}'))
        hooks.append(hook)
    
    # Hook into final norm
    final_norm = None
    if hasattr(model, 'norm'):
        final_norm = model.norm
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layernorm'):
        final_norm = model.backbone.layernorm
    elif hasattr(model, 'layernorm'):
        final_norm = model.layernorm
    
    if final_norm is not None:
        hook = final_norm.register_forward_hook(get_activation('final_norm'))
        hooks.append(hook)
    
    # Process batches
    layer_norms = {name: [] for name in ['patch_embed'] + [f'block_{i}' for i in block_indices] + ['final_norm']}
    
    print(f"Extracting features from {len(block_indices)} transformer blocks...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            # DINOv2 expects 224x224 or 518x518, resize CIFAR if needed
            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Forward pass - hooks will capture intermediate outputs
            _ = model(images)
            
            for name, feat in features.items():
                # Transformer output: (B, num_patches+1, embed_dim)
                # For CLS token: feat[:, 0, :]
                # For all tokens: feat.mean(dim=1) or feat[:, 1:, :].mean(dim=1)
                
                if len(feat.shape) == 3:  # (B, N, D)
                    # Use CLS token for norm computation (first token)
                    cls_token = feat[:, 0, :]  # (B, D)
                    norms = cls_token.norm(dim=1)
                elif len(feat.shape) == 4:  # (B, C, H, W) - patch_embed output
                    # Global Average Pooling: (B, C, H, W) -> (B, C)
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                    norms = pooled.norm(dim=1)
                else:
                    # Already 2D or 1D
                    norms = feat.flatten(1).norm(dim=1)
                
                layer_norms[name].append(norms.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute mean norms across entire dataset
    mean_norms = {}
    for name, norm_list in layer_norms.items():
        if norm_list:
            all_norms = torch.cat(norm_list, dim=0)
            mean_norms[name] = all_norms.mean().item()
        else:
            mean_norms[name] = 0.0
    
    return mean_norms


def plot_layer_norms(mean_norms, save_path='figs/layer_norms.pdf', model_type='resnet', model_name='ResNet-50'):
    """
    Plot bar chart of mean L2 norms across layers.
    
    Args:
        mean_norms: Dictionary mapping layer names to mean L2 norms.
        save_path: Path to save the figure.
        model_type: 'resnet', 'transformer', or 'densenet'
        model_name: Display name for the model (e.g., 'ResNet-50', 'DINOv2-Large', 'DenseNet-121')
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if model_type == 'transformer':
        # For transformers, sort by block number
        def get_block_num(name):
            if name == 'patch_embed':
                return -1
            elif name == 'final_norm':
                return 1000
            elif name.startswith('block_'):
                return int(name.split('_')[1])
            return 0
        
        layer_names = sorted(mean_norms.keys(), key=get_block_num)
        norms = [mean_norms[layer] for layer in layer_names]
        
        # Create display names
        display_names = []
        for name in layer_names:
            if name == 'patch_embed':
                display_names.append('Patch Embed')
            elif name == 'final_norm':
                display_names.append('Final Norm')
            elif name.startswith('block_'):
                block_num = name.split('_')[1]
                display_names.append(f'Block {block_num}')
            else:
                display_names.append(name)
        
        color = 'coral'
        title = f'Average L2 Norm of CLS Token Features Across Layers\n{model_name} on CIFAR-100'
    elif model_type == 'densenet':
        # For DenseNet, sort by depth for both torchvision and custom CIFAR variants.
        layer_order = TORCHVISION_DENSENET_LAYERS + CUSTOM_DENSENET_LAYERS
        layer_names = [l for l in layer_order if l in mean_norms]
        # Add any remaining layers not in the standard order
        for layer in mean_norms.keys():
            if layer not in layer_names:
                layer_names.append(layer)
        
        norms = [mean_norms[layer] for layer in layer_names]
        
        # Create display names
        display_map = getattr(
            extract_densenet_layer_features,
            'layer_display_map',
            DENSENET_LAYER_DISPLAY_MAP,
        )
        display_names = []
        for name in layer_names:
            if name in display_map:
                display_names.append(display_map[name])
            else:
                # Extract the last part of the name
                display_names.append(name.split('.')[-1].capitalize())
        
        color = 'mediumseagreen'
        title = f'Average L2 Norm of Pooled Feature Vectors Across Layers\n{model_name} on CIFAR-100'
    else:
        # For ResNet, sort layers by depth (initial_conv/maxpool first, then layer1-4)
        layer_order = ['initial_conv', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        layer_names = [l for l in layer_order if l in mean_norms]
        # Add any remaining layers not in the standard order
        for layer in mean_norms.keys():
            if layer not in layer_names:
                layer_names.append(layer)
        
        norms = [mean_norms[layer] for layer in layer_names]
        
        # Create display names (replace initial_conv with "Initial Conv" and maxpool with "Initial Conv")
        display_names = []
        for name in layer_names:
            if name == 'initial_conv':
                display_names.append('Initial Conv')
            elif name == 'maxpool':
                display_names.append('Initial Conv')
            else:
                display_names.append(name.capitalize())
        
        color = 'steelblue'
        title = f'Average L2 Norm of Pooled Feature Vectors Across Layers\n{model_name} on CIFAR-100'
    
    # Create layer depth indices (0, 1, 2, ...)
    layer_indices = list(range(len(layer_names)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(layer_indices, norms, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Customize plot with larger fonts
    ax.set_xlabel('Layer Depth', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean L2 Norm', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=16)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, norm) in enumerate(zip(bars, norms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{norm:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Use log scale if there's large variation (10-30x as mentioned in paper)
    if max(norms) / min(norms) > 5:
        ax.set_yscale('log')
        ax.set_ylabel('Mean L2 Norm (log scale)', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    # Print statistics
    print("\nLayer Norm Statistics:")
    print("-" * 50)
    for layer_name, norm in mean_norms.items():
        print(f"{layer_name:15s}: {norm:10.4f}")
    print("-" * 50)
    ratio = max(norms) / min(norms) if min(norms) > 0 else 0
    print(f"Ratio (deepest/earliest): {ratio:.2f}x")


def plot_multi_model_norms(models_data, save_path='figs/multi_model_norms.pdf'):
    """
    Optimized plotting for paper: High contrast, large fonts, concise labels.
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    num_models = len(models_data)
    # WIDER figure to allow bars to breathe, but shorter height to fit paper flow
    fig, axes = plt.subplots(1, num_models, figsize=(18, 5))
    
    # Handle single model case
    if num_models == 1:
        axes = [axes]
    
    colors = ['steelblue', 'mediumseagreen', 'coral']
    
    # Global Font Sizes (Large for paper readability)
    TITLE_SIZE = 24
    AXIS_LABEL_SIZE = 22
    TICK_SIZE = 18
    ANNOTATION_SIZE = 18
    
    for idx, model_data in enumerate(models_data):
        mean_norms = model_data['norms']
        model_type = model_data['model_type']
        model_name = model_data['model_name']
        ax = axes[idx]
        
        # --- PREPARE DATA & LABELS (Concise versions) ---
        if model_type == 'transformer':
            # Sort logic
            def get_block_num(name):
                if name == 'patch_embed': return -1
                elif name == 'final_norm': return 1000
                elif name.startswith('block_'): return int(name.split('_')[1])
                return 0
            
            layer_names = sorted(mean_norms.keys(), key=get_block_num)
            
            # Concise Labels: "Patch", "0", "1", ..., "Norm"
            display_names = []
            for name in layer_names:
                if name == 'patch_embed': display_names.append('Patch')
                elif name == 'final_norm': display_names.append('Norm')
                elif name.startswith('block_'): 
                    display_names.append(name.split('_')[1]) # Just the number
                else: display_names.append(name)
                
            color = colors[2] # Coral
            # Simplified Title
            title = 'DINOv2-Large' if 'Large' in model_name else model_name
            
        elif model_type == 'densenet':
            layer_order = ['features.pool0', 'features.transition1', 'features.transition2', 
                          'features.transition3', 'features.norm5']
            layer_names = [l for l in layer_order if l in mean_norms]
            for layer in mean_norms.keys():
                if layer not in layer_names: layer_names.append(layer)
            
            # Concise Labels: "Pool", "T1", "T2", "T3", "Norm"
            display_names = []
            for name in layer_names:
                if 'pool0' in name: display_names.append('Pool')
                elif 'transition' in name: display_names.append(f"T{name[-1]}")
                elif 'norm5' in name: display_names.append('Norm')
                else: display_names.append(name)
                
            color = colors[1] # Green
            title = 'DenseNet-121'

        else: # ResNet
            layer_order = ['initial_conv', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
            layer_names = [l for l in layer_order if l in mean_norms]
            for layer in mean_norms.keys():
                if layer not in layer_names: layer_names.append(layer)
            
            # Concise Labels: "Conv", "L1", "L2", "L3", "L4"
            display_names = []
            for name in layer_names:
                if 'initial' in name or 'maxpool' in name: display_names.append('Conv')
                elif 'layer' in name: display_names.append(f"L{name[-1]}")
                else: display_names.append(name)
                
            color = colors[0] # Blue
            title = 'ResNet-152' if '152' in model_name else model_name

        norms = [mean_norms[layer] for layer in layer_names]
        layer_indices = list(range(len(layer_names)))
        
        # --- PLOTTING ---
        bars = ax.bar(layer_indices, norms, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Title
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        
        # X-Axis
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(display_names, rotation=0, fontsize=TICK_SIZE, fontweight='bold') # No rotation if possible!
        
        # Y-Axis
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        
        # Calculate Ratio
        ratio = max(norms) / min(norms) if min(norms) > 0 else 0
        
        # --- YELLOW BOX (Larger & Prominent) ---
        # Place it consistently in the top left or center-top to avoid blocking the rising bars
        ax.annotate(f'Ratio: {ratio:.1f}x', 
                    xy=(0.05, 0.92), xycoords='axes fraction',
                    ha='left', va='top', 
                    fontsize=ANNOTATION_SIZE, fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black'))
        
        # Log scale if needed
        if ratio > 5:
            ax.set_yscale('log')
            
    # --- COMMON LABELS ---
    # Single X-axis label at the bottom
    fig.text(0.5, 0.02, 'Layer Depth (Early $\\rightarrow$ Late)', ha='center', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    
    # Single Y-axis label at the left
    fig.text(0.08, 0.5, 'Mean $L_2$ Norm (log scale)', va='center', rotation='vertical', fontsize=AXIS_LABEL_SIZE, fontweight='bold')

    plt.tight_layout(rect=[0.09, 0.05, 1, 1]) # Adjust margins to make room for common labels
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Optimized figure saved to {save_path}")

def plot_cnn_vs_transformer_norms(cnn_norms, transformer_norms, save_path='figs/norm_comparison.pdf', 
                                   cnn_name='ResNet-50', transformer_name='DINOv2-Large'):
    """
    Create side-by-side comparison of CNN (ResNet) vs Transformer (DINOv2) norms.
    
    This directly addresses the reviewer concern about whether implicit weighting
    applies to transformers with constant embedding dimension.
    
    Args:
        cnn_norms: Dictionary mapping layer names to mean L2 norms for CNN
        transformer_norms: Dictionary mapping layer names to mean L2 norms for Transformer
        save_path: Path to save the figure
        cnn_name: Display name for CNN model
        transformer_name: Display name for Transformer model
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: CNN (ResNet or DenseNet) (expect 10-30x variation)
    ax1 = axes[0]
    
    # Sort CNN layers - check if it's DenseNet or ResNet
    if any('features.' in key for key in cnn_norms.keys()):
        # DenseNet
        layer_order = ['features.pool0', 'features.transition1', 'features.transition2', 
                      'features.transition3', 'features.norm5']
        layer_names_cnn = [l for l in layer_order if l in cnn_norms]
        for layer in cnn_norms.keys():
            if layer not in layer_names_cnn:
                layer_names_cnn.append(layer)
        
        # Create display names for DenseNet
        display_names_cnn = []
        for name in layer_names_cnn:
            if name == 'features.pool0':
                display_names_cnn.append('Initial Pool')
            elif name == 'features.transition1':
                display_names_cnn.append('Transition 1')
            elif name == 'features.transition2':
                display_names_cnn.append('Transition 2')
            elif name == 'features.transition3':
                display_names_cnn.append('Transition 3')
            elif name == 'features.norm5':
                display_names_cnn.append('Final Norm')
            else:
                display_names_cnn.append(name.split('.')[-1].capitalize())
    else:
        # ResNet
        layer_order = ['initial_conv', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        layer_names_cnn = [l for l in layer_order if l in cnn_norms]
        for layer in cnn_norms.keys():
            if layer not in layer_names_cnn:
                layer_names_cnn.append(layer)
        
        # Create display names for ResNet
        display_names_cnn = []
        for name in layer_names_cnn:
            if name == 'initial_conv':
                display_names_cnn.append('Initial Conv')
            elif name == 'maxpool':
                display_names_cnn.append('Initial Conv')
            else:
                display_names_cnn.append(name.capitalize())
    
    norms_cnn = [cnn_norms[layer] for layer in layer_names_cnn]
    
    ax1.bar(range(len(layer_names_cnn)), norms_cnn, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(layer_names_cnn)))
    ax1.set_xticklabels(display_names_cnn, rotation=45, ha='right', fontsize=14)
    ax1.set_ylabel('Mean L2 Norm', fontsize=16, fontweight='bold')
    ax1.set_title(f'{cnn_name} on CIFAR-100\n(Increasing channel count -> norm growth)', 
                  fontsize=17, fontweight='bold')
    
    # Add ratio annotation
    ratio_cnn = max(norms_cnn) / min(norms_cnn) if min(norms_cnn) > 0 else 0
    ax1.annotate(f'Ratio: {ratio_cnn:.1f}x', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=15, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Use log scale if needed
    if ratio_cnn > 5:
        ax1.set_yscale('log')
        ax1.set_ylabel('Mean L2 Norm (log scale)', fontsize=16, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_axisbelow(True)
    
    # Right: DINOv2 (expect ~flat or different pattern)
    ax2 = axes[1]
    
    # Sort transformer layers
    def get_block_num(name):
        if name == 'patch_embed':
            return -1
        elif name == 'final_norm':
            return 1000
        elif name.startswith('block_'):
            return int(name.split('_')[1])
        return 0
    
    layer_names_vit = sorted(transformer_norms.keys(), key=get_block_num)
    norms_vit = [transformer_norms[layer] for layer in layer_names_vit]
    
    # Create display names
    display_names_vit = []
    for name in layer_names_vit:
        if name == 'patch_embed':
            display_names_vit.append('Patch\nEmbed')
        elif name == 'final_norm':
            display_names_vit.append('Final\nNorm')
        elif name.startswith('block_'):
            block_num = name.split('_')[1]
            display_names_vit.append(f'B{block_num}')
        else:
            display_names_vit.append(name)
    
    ax2.bar(range(len(layer_names_vit)), norms_vit, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(layer_names_vit)))
    ax2.set_xticklabels(display_names_vit, rotation=45, ha='right', fontsize=13)
    ax2.set_ylabel('Mean L2 Norm', fontsize=16, fontweight='bold')
    ax2.set_title(f'{transformer_name} on CIFAR-100\n(Constant embed dim -> norm growth)', 
                  fontsize=17, fontweight='bold')
    
    # Add ratio annotation
    ratio_vit = max(norms_vit) / min(norms_vit) if min(norms_vit) > 0 else 0
    ax2.annotate(f'Ratio: {ratio_vit:.2f}x', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=15, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_axisbelow(True)
    
    plt.suptitle('Layer-wise Feature Norm Analysis: CNN vs Vision Transformer', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to {save_path}")
    
    # Print interpretation for paper
    print("\n" + "="*60)
    print("INTERPRETATION FOR PAPER:")
    print("="*60)
    if ratio_vit < 2.0:
        print(f"DINOv2 shows relatively FLAT norms (ratio: {ratio_vit:.2f}x)")
        print("-> The 'implicit weighting' mechanism does NOT explain transformer success")
        print("-> Need alternative explanation (e.g., attention refinement, residual accumulation)")
    else:
        print(f"DINOv2 shows INCREASING norms (ratio: {ratio_vit:.2f}x)")
        print("-> Despite constant embed dim, norms still grow (possibly due to residual connections)")
        print("-> Implicit weighting mechanism may still apply")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Plot average L2 norm of pooled feature vectors across layers for ResNet or DINOv2'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        choices=['resnet50', 'resnet18', 'resnet101', 'resnet152', 'densenet121', 
                 'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant'],
        help='Model architecture to analyze (default: resnet50)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint. If not provided, uses pretrained model.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for processing (default: 256)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of worker processes for data loading (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for figure (default: auto-generated based on model)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate comparison plot between ResNet and DINOv2 (requires both models)'
    )
    parser.add_argument(
        '--cnn-model',
        type=str,
        default='resnet50',
        choices=['resnet50', 'resnet18', 'resnet101', 'resnet152', 'densenet121'],
        help='CNN model for comparison (default: resnet50)'
    )
    parser.add_argument(
        '--transformer-model',
        type=str,
        default='dinov2_large',
        choices=['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant'],
        help='Transformer model for comparison (default: dinov2_large)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='List of models to compare (e.g., --models resnet152 densenet121 dinov2_large). Overrides --model.'
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        default=None,
        help='List of checkpoint paths corresponding to --models (optional, same order)'
    )
    
    args = parser.parse_args()
    
    # Handle multi-model comparison
    if args.models is not None:
        print("=" * 70)
        print("Multi-Model Layer Norm Comparison")
        print("=" * 70)
        
        # Load CIFAR-100 test set
        test_loader = get_cifar100_test_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"Test set size: {len(test_loader.dataset)} images")
        
        models_data = []
        checkpoints_list = args.checkpoints if args.checkpoints else [None] * len(args.models)
        
        for idx, model_name in enumerate(args.models):
            print(f"\n[{idx+1}/{len(args.models)}] Processing {model_name}...")
            checkpoint_path = checkpoints_list[idx] if idx < len(checkpoints_list) else None
            
            # Load model
            if 'dinov2' in model_name:
                model_size = model_name.split('_')[1]
                model = load_dinov2(model_size, args.device)
                model_type = 'transformer'
                model_display_name = model_name.upper().replace('DINOV2_', 'DINOv2-').replace('_', ' ').title()
                
                # Extract features
                mean_norms = extract_transformer_block_features(
                    model, test_loader, args.device
                )
            elif model_name == 'densenet121':
                model = load_densenet121(
                    checkpoint_path=checkpoint_path,
                    num_classes=100,
                    device=args.device
                )
                model_type = 'densenet'
                model_display_name = 'DenseNet-121'
                
                # Extract features
                mean_norms = extract_densenet_layer_features(
                    model, test_loader, args.device
                )
            elif 'resnet' in model_name:
                if model_name == 'resnet50':
                    model = load_resnet50(
                        checkpoint_path=checkpoint_path,
                        num_classes=100,
                        device=args.device
                    )
                elif model_name == 'resnet101':
                    model = load_resnet101(
                        checkpoint_path=checkpoint_path,
                        num_classes=100,
                        device=args.device
                    )
                elif model_name == 'resnet152':
                    model = load_resnet152(
                        checkpoint_path=checkpoint_path,
                        num_classes=100,
                        device=args.device
                    )
                else:
                    # Fallback
                    model = load_resnet50(
                        checkpoint_path=checkpoint_path,
                        num_classes=100,
                        device=args.device
                    )
                model_type = 'resnet'
                model_display_name = model_name.upper().replace('RESNET', 'ResNet-')
                
                # Extract features
                has_maxpool = hasattr(model, 'maxpool') and model.maxpool is not None
                available_layers = list(dict(model.named_modules()).keys())
                
                if has_maxpool:
                    initial_layers = ['maxpool']
                else:
                    initial_layers = []
                
                residual_layers = ['layer1', 'layer2', 'layer3', 'layer4']
                layers_to_extract = []
                for layer in initial_layers + residual_layers:
                    if layer in available_layers:
                        layers_to_extract.append(layer)
                
                mean_norms = extract_layer_features(
                    model, test_loader, args.device, layers_to_extract
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            models_data.append({
                'norms': mean_norms,
                'model_type': model_type,
                'model_name': model_display_name
            })
        
        # Generate multi-model plot
        output_path = args.output or 'figs/multi_model_norms.pdf'
        plot_multi_model_norms(models_data, save_path=output_path)
        
        print("\nDone!")
        return
    
    # Handle comparison mode
    if args.compare:
        print("=" * 70)
        print("Layer Norm Comparison: CNN vs Vision Transformer")
        print("=" * 70)
        
        # Load CIFAR-100 test set
        test_loader = get_cifar100_test_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"Test set size: {len(test_loader.dataset)} images")
        
        # Load CNN model
        print(f"\nLoading {args.cnn_model}...")
        if 'resnet' in args.cnn_model:
            if args.cnn_model == 'resnet50':
                model_cnn = load_resnet50(
                    checkpoint_path=args.checkpoint,
                    num_classes=100,
                    device=args.device
                )
            elif args.cnn_model == 'resnet101':
                model_cnn = load_resnet101(
                    checkpoint_path=args.checkpoint,
                    num_classes=100,
                    device=args.device
                )
            elif args.cnn_model == 'resnet152':
                model_cnn = load_resnet152(
                    checkpoint_path=args.checkpoint,
                    num_classes=100,
                    device=args.device
                )
            else:
                # For resnet18 or other variants, try to use resnet50 loader
                model_cnn = load_resnet50(
                    checkpoint_path=args.checkpoint if args.cnn_model == 'resnet50' else None,
                    num_classes=100,
                    device=args.device
                )
            
            # Extract CNN features
            has_maxpool = hasattr(model_cnn, 'maxpool') and model_cnn.maxpool is not None
            available_layers = list(dict(model_cnn.named_modules()).keys())
            
            if has_maxpool:
                initial_layers = ['maxpool']
            else:
                initial_layers = []
            
            residual_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            layers_to_extract = []
            for layer in initial_layers + residual_layers:
                if layer in available_layers:
                    layers_to_extract.append(layer)
            
            cnn_norms = extract_layer_features(
                model_cnn, test_loader, args.device, layers_to_extract
            )
            cnn_display_name = args.cnn_model.upper().replace('RESNET', 'ResNet-')
        elif args.cnn_model == 'densenet121':
            model_cnn = load_densenet121(
                checkpoint_path=args.checkpoint,
                num_classes=100,
                device=args.device
            )
            
            # Extract DenseNet features
            cnn_norms = extract_densenet_layer_features(
                model_cnn, test_loader, args.device
            )
            cnn_display_name = 'DenseNet-121'
        else:
            raise ValueError(f"Unsupported CNN model: {args.cnn_model}")
        
        # Load Transformer model
        print(f"\nLoading {args.transformer_model}...")
        model_size = args.transformer_model.split('_')[1]  # 'large', 'giant', etc.
        model_transformer = load_dinov2(model_size, args.device)
        
        # Extract Transformer features
        transformer_norms = extract_transformer_block_features(
            model_transformer, test_loader, args.device
        )
        transformer_display_name = args.transformer_model.upper().replace('DINOV2_', 'DINOv2-').replace('_', ' ').title()
        
        # Generate comparison plot
        output_path = args.output or 'figs/norm_comparison_cnn_vs_transformer.pdf'
        plot_cnn_vs_transformer_norms(
            cnn_norms, transformer_norms, 
            save_path=output_path,
            cnn_name=cnn_display_name,
            transformer_name=transformer_display_name
        )
        
        print("\nDone!")
        return
    
    # Single model analysis
    print("=" * 70)
    if 'dinov2' in args.model:
        model_size = args.model.split('_')[1]  # 'large', 'giant', etc.
        model_display_name = args.model.upper().replace('DINOV2_', 'DINOv2-').replace('_', ' ').title()
        print(f"Layer Norm Analysis: {model_display_name} on CIFAR-100")
    elif args.model == 'densenet121':
        model_display_name = 'DenseNet-121'
        print(f"Layer Norm Analysis: {model_display_name} on CIFAR-100")
    else:
        model_display_name = args.model.upper().replace('RESNET', 'ResNet-')
        print(f"Layer Norm Analysis: {model_display_name} on CIFAR-100")
    print("=" * 70)
    
    # Load model
    if 'dinov2' in args.model:
        model_size = args.model.split('_')[1]
        model = load_dinov2(model_size, args.device)
        model_type = 'transformer'
    elif args.model == 'resnet50':
        model = load_resnet50(
            checkpoint_path=args.checkpoint,
            num_classes=100,
            device=args.device
        )
        model_type = 'resnet'
    elif args.model == 'resnet101':
        model = load_resnet101(
            checkpoint_path=args.checkpoint,
            num_classes=100,
            device=args.device
        )
        model_type = 'resnet'
    elif args.model == 'resnet152':
        model = load_resnet152(
            checkpoint_path=args.checkpoint,
            num_classes=100,
            device=args.device
        )
        model_type = 'resnet'
    elif args.model == 'densenet121':
        model = load_densenet121(
            checkpoint_path=args.checkpoint,
            num_classes=100,
            device=args.device
        )
        model_type = 'densenet'
    elif 'resnet' in args.model:
        # Fallback for other resnet variants
        model = load_resnet50(
            checkpoint_path=args.checkpoint,
            num_classes=100,
            device=args.device
        )
        model_type = 'resnet'
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Load CIFAR-100 test set
    test_loader = get_cifar100_test_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Test set size: {len(test_loader.dataset)} images")
    
    # Extract features
    if model_type == 'transformer':
        mean_norms = extract_transformer_block_features(
            model, test_loader, args.device
        )
    elif model_type == 'densenet':
        # Extract DenseNet features
        mean_norms = extract_densenet_layer_features(
            model, test_loader, args.device
        )
    else:
        # Define layers to extract for ResNet
        has_maxpool = hasattr(model, 'maxpool') and model.maxpool is not None
        available_layers = list(dict(model.named_modules()).keys())
        
        if has_maxpool:
            initial_layers = ['maxpool']
            print("Detected torchvision-style ResNet (with maxpool)")
        else:
            initial_layers = []
            print("Detected custom ResNet (no maxpool)")
            print("  Note: Will extract from layer1 input to capture initial conv block output")
        
        residual_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        layers_to_extract = []
        for layer in initial_layers + residual_layers:
            if layer in available_layers:
                layers_to_extract.append(layer)
        
        if not layers_to_extract:
            print("Error: No valid layers found to extract!")
            print(f"Available layers: {available_layers[:20]}...")
            return
        
        print(f"Extracting features from layers: {layers_to_extract}")
        mean_norms = extract_layer_features(
            model, test_loader, args.device, layers_to_extract
        )
    
    # Plot results
    output_path = args.output
    if output_path is None:
        if model_type == 'transformer':
            output_path = f'figs/layer_norms_{args.model}.pdf'
        else:
            output_path = f'figs/layer_norms_{args.model}.pdf'
    
    plot_layer_norms(mean_norms, save_path=output_path, 
                     model_type=model_type, model_name=model_display_name)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
