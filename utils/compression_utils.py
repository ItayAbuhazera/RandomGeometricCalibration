"""
Compression utilities for high-dimensional features in geometric calibration.
Provides PCA compression that preserves spatial information while managing memory efficiently.
"""

import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Union

from utils.logging_config import get_logger
logger = get_logger(__name__)


class SmartCompression:
    """
    Smart compression system that preserves spatial information while managing memory.
    
    This class provides compression methods that work with the existing StabilitySpace
    infrastructure and are designed specifically for geometric calibration.
    """
    
    def __init__(self, method: str = 'pca', target_dims: int = 512, 
                 compression_ratio: float = None, preserve_spatial: bool = True, 
                 random_state: int = 42, **kwargs):
        """
        Initialize smart compression.
        
        Args:
            method: Compression method ('pca', 'random_projection', 'autoencoder')
            target_dims: Target number of dimensions after compression
            compression_ratio: Ratio to compress dimensions (target = input / ratio)
            preserve_spatial: Whether to preserve spatial structure
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.target_dims = target_dims
        self.compression_ratio = compression_ratio
        self.preserve_spatial = preserve_spatial
        self.random_state = random_state
        
        # Initialize compression components
        self.pca = None
        self.random_projection = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"SmartCompression initialized: {method} -> {target_dims} dims (ratio={compression_ratio})")
    
    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                 train: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply compression to the data.
        
        Args:
            X: Input data [n_samples, n_features]
            y: Labels (optional, for compatibility with StabilitySpace)
            train: Whether this is training data (for fitting)
            
        Returns:
            Compressed data, or (compressed_data, labels) if y is provided
        """
        if self.method == 'pca':
            return self._apply_pca(X, y, train)
        elif self.method == 'random_projection':
            return self._apply_random_projection(X, y, train)
        else:
            raise ValueError(f"Unsupported compression method: {self.method}")
    
    def _apply_pca(self, X: np.ndarray, y: Optional[np.ndarray], 
                   train: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply PCA compression."""
        
        # Ensure X is 2D
        original_shape = X.shape
        if len(X.shape) > 2:
            X_reshaped = X.reshape(X.shape[0], -1)
            logger.debug(f"Reshaping for PCA: {original_shape} -> {X_reshaped.shape}")
        else:
            X_reshaped = X
        
        # Fit PCA if training
        if train and not self.is_fitted:
            # Update target dims if compression ratio is provided
            if self.compression_ratio is not None:
                self.target_dims = max(1, int(X_reshaped.shape[1] / self.compression_ratio))
                logger.info(f"Calculated target dims from ratio {self.compression_ratio}: {self.target_dims}")
            
            logger.info(f"Fitting PCA: {X_reshaped.shape[1]} -> {self.target_dims} dimensions")
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X_reshaped)
            
            # Fit PCA
            self.pca = PCA(n_components=self.target_dims, random_state=self.random_state)
            self.pca.fit(X_scaled)
            
            # Log explained variance
            explained_variance_ratio = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_variance_ratio:.3f} "
                       f"({explained_variance_ratio*100:.1f}%)")
            
            self.is_fitted = True
        
        # Transform the data
        if self.is_fitted:
            X_scaled = self.scaler.transform(X_reshaped)
            X_compressed = self.pca.transform(X_scaled)
            
            logger.debug(f"PCA compression: {X_reshaped.shape} -> {X_compressed.shape}")
            
            # Return consistent format
            if y is not None:
                return X_compressed, y
            else:
                return X_compressed
        else:
            logger.warning("PCA not fitted, returning original data")
            # Return consistent format
            if y is not None:
                return X, y
            else:
                return X
    
    def _apply_random_projection(self, X: np.ndarray, y: Optional[np.ndarray], 
                                train: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply random projection compression."""
        try:
            from sklearn.random_projection import GaussianRandomProjection
        except ImportError:
            logger.error("Random projection requires scikit-learn")
            return X if y is None else (X, y)
        
        # Ensure X is 2D
        if len(X.shape) > 2:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        logger.debug(f"_apply_random_projection called: train={train}, is_fitted={self.is_fitted}, X_reshaped.shape={X_reshaped.shape}")
        
        # Fit random projection if training
        if train and not self.is_fitted:
            # Update target dims if compression ratio is provided
            if self.compression_ratio is not None:
                self.target_dims = max(1, int(X_reshaped.shape[1] / self.compression_ratio))
                logger.info(f"Calculated target dims from ratio {self.compression_ratio}: {self.target_dims}")

            logger.info(f"Fitting Random Projection: {X_reshaped.shape[1]} -> {self.target_dims} dimensions")
            
            try:
                self.random_projection = GaussianRandomProjection(
                    n_components=self.target_dims, 
                    random_state=self.random_state
                )
                self.random_projection.fit(X_reshaped)
                self.is_fitted = True
                logger.debug(f"Random projection fitted successfully")
            except Exception as e:
                logger.error(f"Failed to fit random projection: {e}")
                raise
        
        # Transform the data
        if self.is_fitted:
            X_compressed = self.random_projection.transform(X_reshaped)
            logger.debug(f"Random projection: {X_reshaped.shape} -> {X_compressed.shape}")
            
            if y is not None:
                return X_compressed, y
            else:
                return X_compressed
        else:
            logger.warning("Random projection not fitted, returning original data")
            if y is not None:
                return X, y
            else:
                return X
    
    def get_compression_info(self) -> dict:
        """Get information about the compression."""
        info = {
            'method': self.method,
            'target_dims': self.target_dims,
            'compression_ratio': self.compression_ratio,
            'is_fitted': self.is_fitted,
            'preserve_spatial': self.preserve_spatial
        }
        
        if self.is_fitted and self.pca is not None:
            info['explained_variance_ratio'] = self.pca.explained_variance_ratio_.sum()
            info['n_components'] = self.pca.n_components_
        
        return info


def create_compression_module(method: str, in_channels: int, final_output_dim: int,
                             pyramid_levels: list, mid_channels: int = None, seed: int = 42,
                             pooling_mode: str = 'max'):
    """
    Factory function to create compression module.
    
    Args:
        method: Compression method ('fixed_spp_jl', 'channel_first_spp_jl', 'multiscale_spp_jl')
        in_channels: Input channels
        final_output_dim: Output dimension
        pyramid_levels: SPP pyramid levels
        mid_channels: Intermediate channels (for channel_first_spp_jl)
        seed: Random seed
        pooling_mode: Pooling mode for SPP ('avg' or 'max'). Default 'max'.
    """
    method = method.lower()
    if method == 'fixed_spp_jl':
        return FixedSizeSPP_JL(in_channels, final_output_dim, pyramid_levels, seed, 
                              compression_ratio=None, pooling_mode=pooling_mode)
    elif method == 'channel_first_spp_jl':
        if mid_channels is None:
            mid_channels = min(256, in_channels // 2)
        return ChannelFirstSPP_JL(in_channels, mid_channels, final_output_dim, pyramid_levels, seed,
                                 compression_ratio=None, pooling_mode=pooling_mode)
    elif method == 'multiscale_spp_jl':
        return MultiScaleSPP_JL(in_channels, final_output_dim, pyramid_levels, seed,
                               compression_ratio=None, pooling_mode=pooling_mode)
    else:
        raise ValueError(f"Unknown compression method: {method}")



def estimate_optimal_compression_dims(original_dims: int, layer_name: str, 
                                    cal_method: str) -> int:
    """
    Estimate optimal compression dimensions based on layer and method.
    
    Args:
        original_dims: Original number of dimensions
        layer_name: Name of the layer
        cal_method: Calibration method
        
    Returns:
        Optimal target dimensions
    """
    # Early layers: more aggressive compression
    if layer_name in ['layer1', 'layer2']:
        if original_dims > 50000:
            return 512  # Very aggressive
        elif original_dims > 20000:
            return 1024  # Aggressive
        else:
            return 2048  # Moderate
    
    # Middle layers: moderate compression
    elif layer_name in ['layer3']:
        if original_dims > 20000:
            return 1024
        else:
            return 2048
    
    # Late layers: light compression
    elif layer_name in ['layer4', 'layer4.1']:
        if original_dims > 10000:
            return 2048
        else:
            return 4096
    
    # FAISS methods: more conservative
    if 'faiss' in cal_method:
        return min(2048, original_dims // 4)
    
    # Default: reduce by factor of 4
    return min(4096, original_dims // 4) 


import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedSizeSPP_JL(nn.Module):
    """
    Combines Spatial Pyramid Pooling (SPP) with a fixed Random Projection to
    produce a feature vector of a constant size, regardless of the input's
    channel count. This entire operation is parameter-free and deterministic.
    """
    def __init__(self, in_channels: int, final_output_dim: Optional[int] = None, 
                 pyramid_levels: list = [4, 2, 1], seed: int = 42, 
                 compression_ratio: Optional[float] = None, pooling_mode: str = 'max',
                 normalize_output: bool = True):
        """
        Args:
            in_channels (int): The number of channels (C) of the input feature map.
            final_output_dim (int): The desired fixed size of the output vector.
            pyramid_levels (list): Grid sizes for the SPP part.
            seed (int): A fixed seed to ensure the random matrix is always the same.
            compression_ratio (float): Ratio to compress dimensions (optional).
            pooling_mode (str): Pooling mode for SPP ('avg' or 'max'). Default 'max'.
            normalize_output (bool): If True, applies L2 normalization to the projected
                output features along the feature dimension (dim=1). If False, returns
                raw projected features without normalization. Default True to preserve
                legacy behavior.
        """
        super(FixedSizeSPP_JL, self).__init__()
        
        if pooling_mode not in ['avg', 'max']:
            raise ValueError(f"pooling_mode must be 'avg' or 'max', got '{pooling_mode}'")
        
        self.pooling_mode = pooling_mode
        self.normalize_output = normalize_output
        
        # Calculate the size of the feature vector after SPP
        num_spp_bins = sum([level**2 for level in pyramid_levels])
        spp_output_dim = in_channels * num_spp_bins
        
        # Determine final output dimension
        if final_output_dim is None:
            if compression_ratio is not None:
                final_output_dim = max(1, int(spp_output_dim / compression_ratio))
                logger.debug(f"Calculated target dims from ratio {compression_ratio}: {spp_output_dim} -> {final_output_dim}")
            else:
                raise ValueError("Either final_output_dim or compression_ratio must be provided")
        
        # Create the fixed random projection matrix
        # Use a generator with a fixed seed for reproducibility
        generator = torch.Generator().manual_seed(seed)
        
        # The projection matrix. Its weights are random but frozen.
        projection_matrix = torch.randn(
            final_output_dim, 
            spp_output_dim, 
            generator=generator
        )
        
        # Register as a buffer, not a parameter, so it's part of the state
        # but not considered trainable by PyTorch.
        self.register_buffer('projection_matrix', projection_matrix)
        
        self.pyramid_levels = pyramid_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A 4D input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: A 2D tensor of shape (N, final_output_dim).
                L2-normalized if self.normalize_output is True, otherwise unnormalized.
        """
        # Ensure contiguous layout for safe reshaping
        x = x.contiguous()
        batch_size = x.size(0)
        
        # 1. Perform Spatial Pyramid Pooling
        pooled_features = []
        for level in self.pyramid_levels:
            if self.pooling_mode == 'max':
                pooled = F.adaptive_max_pool2d(x, output_size=(level, level))
            else:  # 'avg'
                pooled = F.adaptive_avg_pool2d(x, output_size=(level, level))
            # Use reshape to handle non-contiguous tensors safely
            pooled = pooled.reshape(batch_size, -1)
            pooled_features.append(pooled)
        
        spp_output = torch.cat(pooled_features, dim=1)

        # 2. Apply the fixed random projection
        # We use F.linear which is a functional form that can accept a weight tensor
        projected_output = F.linear(spp_output, self.projection_matrix)

        # 3. Optional L2 normalization along the feature dimension
        if self.normalize_output:
            projected_output = F.normalize(projected_output, p=2, dim=1)
        
        return projected_output


class ChannelFirstSPP_JL(nn.Module):
    """
    First applies a fixed JL projection to reduce channels (C -> mid_channels),
    then applies SPP pooling on the reduced channels, and finally applies a 
    second JL projection to target dims.
    
    This is designed to be faster when C is very large, as SPP operates on 
    fewer channels.
    """
    def __init__(self, in_channels: int, mid_channels: int, final_output_dim: Optional[int] = None, 
                 pyramid_levels: list = [4, 2, 1], seed: int = 42,
                 compression_ratio: Optional[float] = None, pooling_mode: str = 'max'):
        """
        Args:
            in_channels (int): Input channels.
            mid_channels (int): Intermediate channels after first projection.
            final_output_dim (int): Final output dimension.
            pyramid_levels (list): SPP levels.
            seed (int): Random seed.
            compression_ratio (float): Ratio to compress dimensions (optional).
            pooling_mode (str): Pooling mode for SPP ('avg' or 'max'). Default 'max'.
        """
        super(ChannelFirstSPP_JL, self).__init__()
        
        if pooling_mode not in ['avg', 'max']:
            raise ValueError(f"pooling_mode must be 'avg' or 'max', got '{pooling_mode}'")
        
        self.pooling_mode = pooling_mode
        self.pyramid_levels = pyramid_levels
        
        # Generator for reproducibility
        # Use two different seeds/generators for the two projections
        gen1 = torch.Generator().manual_seed(seed)
        gen2 = torch.Generator().manual_seed(seed + 1)
        
        # 1. Channel Reduction Matrix (simulating 1x1 conv)
        # Shape: (mid_channels, in_channels, 1, 1) for Conv2d
        channel_proj = torch.randn(mid_channels, in_channels, 1, 1, generator=gen1)
        self.register_buffer('channel_proj', channel_proj)
        
        # 2. Final Projection Matrix
        num_spp_bins = sum([level**2 for level in pyramid_levels])
        # SPP is applied on mid_channels
        spp_output_dim = mid_channels * num_spp_bins
        
        # Determine final output dimension
        if final_output_dim is None:
            if compression_ratio is not None:
                # Base compression on original expanded size equivalent or just mid_channel expanded?
                # Usually we want target dim relative to "information content".
                # If we consider input information as in_channels * spatial, or just spp_output_dim?
                # For consistency with FixedSizeSPP_JL, let's use spp_output_dim (which is compressed from in_channels already by mid_channels)
                # Or maybe we should use the *full* theoretical SPP dimension?
                # Let's stick to the dimension immediately preceding the final projection.
                final_output_dim = max(1, int(spp_output_dim / compression_ratio))
            else:
                raise ValueError("Either final_output_dim or compression_ratio must be provided")

        final_proj = torch.randn(final_output_dim, spp_output_dim, generator=gen2)
        self.register_buffer('final_proj', final_proj)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, H, W).
            
        Returns:
            torch.Tensor: Output vector (N, final_output_dim).
        """
        # x: (N, C, H, W)
        x = x.contiguous()
        
        # 1. Channel Reduction
        # F.conv2d with 1x1 kernels is efficient for channel projection
        x_reduced = F.conv2d(x, self.channel_proj)
        
        # 2. SPP
        batch_size = x_reduced.size(0)
        pooled_features = []
        for level in self.pyramid_levels:
            if self.pooling_mode == 'max':
                pooled = F.adaptive_max_pool2d(x_reduced, output_size=(level, level))
            else:  # 'avg'
                pooled = F.adaptive_avg_pool2d(x_reduced, output_size=(level, level))
            pooled = pooled.reshape(batch_size, -1)
            pooled_features.append(pooled)
            
        spp_output = torch.cat(pooled_features, dim=1)
        
        # 3. Final Projection
        projected = F.linear(spp_output, self.final_proj)
        
        # 4. Normalize
        return F.normalize(projected, p=2, dim=1)


class MultiScaleSPP_JL(nn.Module):
    """
    Builds a feature pyramid (original, 0.5x, 0.25x) BEFORE JL projection.
    Applies SPP to each scale separately, concatenates, then projects.
    Captures multi-scale information while preserving geometry.
    """
    def __init__(self, in_channels: int, final_output_dim: Optional[int] = None, 
                 pyramid_levels: list = [4, 2, 1], seed: int = 42,
                 compression_ratio: Optional[float] = None, pooling_mode: str = 'max'):
        """
        Args:
            in_channels (int): Input channels.
            final_output_dim (int): Final output dimension.
            pyramid_levels (list): SPP levels.
            seed (int): Random seed.
            compression_ratio (float): Ratio to compress dimensions (optional).
            pooling_mode (str): Pooling mode for SPP ('avg' or 'max'). Default 'max'.
        """
        super(MultiScaleSPP_JL, self).__init__()
        
        if pooling_mode not in ['avg', 'max']:
            raise ValueError(f"pooling_mode must be 'avg' or 'max', got '{pooling_mode}'")
        
        self.pooling_mode = pooling_mode
        self.pyramid_levels = pyramid_levels
        
        # 3 scales: 1x, 0.5x, 0.25x
        num_scales = 3
        
        # Calculate total input size for projection
        # SPP output size for one scale = in_channels * num_bins
        num_spp_bins = sum([level**2 for level in pyramid_levels])
        total_spp_dim = in_channels * num_spp_bins * num_scales
        
        # Determine final output dimension
        if final_output_dim is None:
            if compression_ratio is not None:
                final_output_dim = max(1, int(total_spp_dim / compression_ratio))
            else:
                raise ValueError("Either final_output_dim or compression_ratio must be provided")
        
        generator = torch.Generator().manual_seed(seed)
        projection_matrix = torch.randn(final_output_dim, total_spp_dim, generator=generator)
        self.register_buffer('projection_matrix', projection_matrix)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, H, W).
            
        Returns:
            torch.Tensor: Output vector (N, final_output_dim).
        """
        x = x.contiguous()
        batch_size = x.size(0)
        
        # Create scales
        scales = [x]
        # 0.5x
        scales.append(F.interpolate(x, scale_factor=0.5, mode='area'))
        # 0.25x
        scales.append(F.interpolate(x, scale_factor=0.25, mode='area'))
        
        all_pooled = []
        
        for scale_tensor in scales:
            # Apply SPP to this scale
            scale_features = []
            for level in self.pyramid_levels:
                if self.pooling_mode == 'max':
                    pooled = F.adaptive_max_pool2d(scale_tensor, output_size=(level, level))
                else:  # 'avg'
                    pooled = F.adaptive_avg_pool2d(scale_tensor, output_size=(level, level))
                pooled = pooled.reshape(batch_size, -1)
                scale_features.append(pooled)
            # Concat SPP features for this scale
            all_pooled.extend(scale_features)
            
        # Concat everything
        concat_features = torch.cat(all_pooled, dim=1)
        
        # Projection
        projected = F.linear(concat_features, self.projection_matrix)
        
        # Normalize
        return F.normalize(projected, p=2, dim=1)


class TuckerCompression(nn.Module):
    """
    Compresses high-dimensional convolutional features using Tucker Decomposition 
    on the feature covariance tensor, as described in 'Probabilistic Skip Connections'.
    
    Paper methodology:
    1. Reshape (N, C, H, W) -> (N, C, HW)
    2. Center and Scale features channel-wise
    3. Project Spatial dim (HW -> d_proj) using factor matrix B
    4. Project Channel dim (C -> c_proj) using factor matrix A
    """
    
    def __init__(self, in_channels: int, 
                 c_proj: int, 
                 d_proj: int, 
                 device='cuda'):
        """
        Args:
            in_channels (int): Original number of channels (C)
            c_proj (int): Target channel dimension (paper parameter c_proj)
            d_proj (int): Target spatial dimension (paper parameter d_proj)
            device: Device for computation
        """
        super().__init__()
        self.in_channels = in_channels
        self.c_proj = c_proj
        self.d_proj = d_proj
        self.device = device
        self.is_fitted = False
        
        # Buffer registration ensures these are saved with the model state_dict
        # but are not updated by the optimizer.
        
        # Statistics for Standardization (Section A.2.1 "Scale")
        self.register_buffer('mu', torch.zeros(1, in_channels, 1))
        self.register_buffer('sigma_inv', torch.ones(1, in_channels, 1))
        
        # Factor Matrices for Projection (Section 4.2 "Projecting")
        # A: Projects Channels (C -> c_proj)
        self.register_buffer('A', torch.eye(c_proj, in_channels)) 
        # B: Projects Spatial (HW -> d_proj) - will be initialized in fit() with correct size
        self.register_buffer('B', None)
        
    def fit(self, dataloader, model_adapter, layer_idx, num_batches=50):
        """
        Fits the Tucker decomposition parameters (Means, Variances, and Factor Matrices A & B).
        Based on Algorithm 2 in Appendix A.2.2 of the paper.
        
        Args:
            dataloader: Training dataloader
            model_adapter: Adapter to run the model and get features
            layer_idx: The layer to hook
            num_batches: Limit fitting to N batches to save time (paper suggests subset is fine)
        """
        logger.info(f"Fitting TuckerCompression (Layer {layer_idx})...")
        model = model_adapter.model
        device = torch.device(self.device)
        
        # --- Pass 1: Estimate Mean (Section A.2.2) ---
        # "It was important that the estimate of the mean was as accurate as possible" [cite: 508]
        running_sum = None
        total_samples = 0
        
        # We need raw 4D features, so ensure your extractor returns (N, C, H, W)
        # Note: You might need to temporarily disable your SPP wrapper in extract_features_directly 
        # or access the raw hook output. Assuming here we get raw 4D tensors.
        
        accumulated_features = []
        
        # For simplicity in this implementation, we collect a subset of data first.
        # The paper mentions streaming updates, but collecting X batches is safer for stability.
        ctr = 0
        for batch in dataloader:
            if ctr >= num_batches: break
            
            # Extract raw features (N, C, H, W)
            # You will need a version of extract_features that returns the RAW 4D tensor
            # before your FixedSizeSPP_JL is applied.
            # Here we assume `extract_raw_features` exists or you modify your hook.
            with torch.no_grad():
                inputs = batch[0].to(device)
                # Hook the specific layer
                features = self._extract_raw_layer_output(model, inputs, layer_idx)
                
            accumulated_features.append(features.cpu())
            ctr += 1
            
        # Stack all collected features: (N_total, C, H, W)
        X_full = torch.cat(accumulated_features, dim=0)
        N, C, H, W = X_full.shape
        HW = H * W
        
        # Reshape to (N, C, HW) 
        X_reshaped = X_full.reshape(N, C, HW)
        
        # 1. Compute Channel-wise Mean & Std [cite: 492]
        # Mean across N and Spatial dims
        mu_c = X_reshaped.mean(dim=(0, 2), keepdim=True) # (1, C, 1)
        # Std across N and Spatial dims
        sigma_c = X_reshaped.std(dim=(0, 2), keepdim=True) + 1e-8 # (1, C, 1)
        
        self.mu.data = mu_c.to(self.device)
        self.sigma_inv.data = (1.0 / sigma_c).to(self.device)
        
        # 2. Center and Scale (Standardization) [cite: 491, 496]
        # (X - mu) * sigma_inv
        X_scaled = (X_reshaped - mu_c) / sigma_c
        
        # 3. Compute Channel-wise Covariance Tensor [cite: 193]
        # Sigma_tensor: (C, HW, HW)
        # We compute covariance for each channel independently
        # Since X is already centered/scaled, Cov = (X^T @ X) / (N-1)
        
        # Transpose to (C, N, HW) for easier batch matmul
        X_permuted = X_scaled.permute(1, 0, 2) # (C, N, HW)
        
        # Batch matrix multiplication: (C, HW, N) @ (C, N, HW) -> (C, HW, HW)
        # This computes the sum of outer products per channel
        cov_tensor = torch.bmm(X_permuted.transpose(1, 2), X_permuted) / (N - 1)
        
        # 4. Tucker Decomposition (Approximation) [cite: 196]
        # The paper simplifies this step.
        # "Simplified Tucker: SVD on average covariance" (commonly used approximation)
        # Alternatively, we follow the paper logic:
        # "Projecting the layers is done by first computing the channelwise covariance... 
        # results in a tensor... perform a Tucker decomposition"
        
        # Factor Matrix B (Spatial Projection):
        # We want the "principal components" of the spatial interactions.
        # A robust way is taking the SVD of the mean covariance across channels.
        avg_spatial_cov = cov_tensor.mean(dim=0) # (HW, HW)
        U_b, S_b, Vh_b = torch.linalg.svd(avg_spatial_cov)
        
        # B corresponds to the top d_proj eigenvectors
        # Paper Eq: X_2 B -> Projecting mode 2 (spatial) with B
        B_matrix = U_b[:, :self.d_proj] # (HW, d_proj)
        # Register B buffer if not already registered, otherwise update it
        if self.B is None:
            self.register_buffer('B', B_matrix.to(self.device))
        else:
            self.B.data = B_matrix.to(self.device)
        
        # Factor Matrix A (Channel Projection):
        # We project the data spatially first, then find channel correlations.
        # X_spatial = X_scaled @ B  -> (N, C, HW) @ (HW, d_proj) -> (N, C, d_proj)
        # Since this fits in memory (d_proj is small), we can do PCA on (N, C*d_proj) 
        # or simpler: PCA on the channel means/correlations.
        
        # The paper implies A comes from the channel-mode unfolding of the core tensor 
        # or simply decomposing the channel-covariance of the spatially-projected data.
        
        # Let's project spatially first as per paper flow:
        # Flatten spatial: (N, C, HW) -> project -> (N, C, d_proj)
        X_spatially_projected = torch.matmul(X_scaled, self.B)
        
        # Now find A to project C -> c_proj.
        # Reshape to (N, d_proj, C) to treat C as the features dim
        X_for_channel_pca = X_spatially_projected.permute(0, 2, 1).reshape(-1, C)
        
        # PCA on channels
        # Center the data (it's already roughly centered but PCA does it internally usually)
        # We use SVD on the covariance of the features
        _, _, Vh_a = torch.linalg.svd(X_for_channel_pca, full_matrices=False)
        
        # Vh_a is (C, C), we want top c_proj rows.
        # A should be (C, c_proj) for multiplication X @ A
        # But wait, paper says: "Project X onto factor matrices... A... B"
        # Usually X_new = X x1 A x2 B
        
        # Let's stick to: we want to reduce C -> c_proj.
        A_matrix = Vh_a[:self.c_proj, :].T # (C, c_proj)
        self.A.data = A_matrix.to(self.device)
        
        self.is_fitted = True
        logger.info(f"Fitted TuckerCompression: {C}x{HW} -> {self.c_proj}x{self.d_proj}")
    
    def forward(self, x):
        """
        Args:
            x: (N, C, H, W) Raw features
        Returns:
            z: (N, c_proj * d_proj) Projected, flattened features
        """
        if not self.is_fitted:
            raise RuntimeError("TuckerCompression not fitted! Call fit() before forward().")
        
        if self.B is None:
            raise RuntimeError("TuckerCompression factor matrix B not initialized. Call fit() first.")
        
        N, C, H, W = x.shape
        HW = H * W
        
        # 1. Reshape [cite: 485]
        x_reshaped = x.reshape(N, C, HW)
        
        # 2. Scale (Standardization) [cite: 489, 490]
        # Uses buffer statistics computed in fit()
        x_scaled = (x_reshaped - self.mu) * self.sigma_inv
        
        # 3. Project Spatial (Mode 2) [cite: 199]
        # x_scaled: (N, C, HW)
        # B: (HW, d_proj)
        # Result: (N, C, d_proj)
        x_spatial = torch.matmul(x_scaled, self.B)
        
        # 4. Project Channel (Mode 1) [cite: 199]
        # We need to contract the channel dimension C.
        # x_spatial: (N, C, d_proj)
        # A: (C, c_proj)
        # We want (N, c_proj, d_proj)
        
        # Permute to put C last: (N, d_proj, C)
        x_permuted = x_spatial.permute(0, 2, 1)
        
        # Matmul: (N, d_proj, C) @ (C, c_proj) -> (N, d_proj, c_proj)
        x_channel = torch.matmul(x_permuted, self.A)
        
        # 5. Final Flatten [cite: 200]
        # Result: (N, c_proj * d_proj)
        z = x_channel.reshape(N, -1)
        
        return z
    
    def _extract_raw_layer_output(self, model, x, layer_idx):
        """Helper to get raw 4D tensor for a specific layer index."""
        # Simple hook implementation
        output_holder = {}
        def hook(m, i, o): output_holder['out'] = o
        
        layers = list(model.modules())
        handle = layers[layer_idx].register_forward_hook(hook)
        _ = model(x)
        handle.remove()
        return output_holder['out']


class SPP_Only(nn.Module):
    """
    Spatial Pyramid Pooling without random projection.
    Returns concatenated multi-scale features preserving all channel information.
    """
    def __init__(self, pyramid_levels: list = [4, 2, 1], pooling_mode: str = 'max'):
        super(SPP_Only, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.pooling_mode = pooling_mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        batch_size = x.size(0)
        pooled_features = []
        
        for level in self.pyramid_levels:
            if self.pooling_mode == 'max':
                pooled = F.adaptive_max_pool2d(x, output_size=(level, level))
            else:
                pooled = F.adaptive_avg_pool2d(x, output_size=(level, level))
            pooled = pooled.reshape(batch_size, -1)
            pooled_features.append(pooled)
        
        spp_output = torch.cat(pooled_features, dim=1)
        # No JL projection, no normalization
        return spp_output