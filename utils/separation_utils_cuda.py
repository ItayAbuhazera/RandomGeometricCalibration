import numpy as np
import torch
import logging
from tqdm import tqdm

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)
def check_cuda_availability():
    """Check if CUDA is available and return appropriate device."""
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.info("CUDA not available, using CPU instead")
        return torch.device("cpu")

# Global device variable to be used across functions
DEVICE = check_cuda_availability()

def get_distance_gpu(x1, x2, metric='l2'):
    """
    Calculate distance between two points using PyTorch on GPU if available.
    
    Args:
        x1 (numpy.ndarray): First point
        x2 (numpy.ndarray): Second point
        metric (str): Distance metric to use ('l1', 'l2', 'linf', 'cosine')
        
    Returns:
        float: Distance between the points
    """
    # Convert numpy arrays to PyTorch tensors on appropriate device
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=DEVICE)
    x2_tensor = torch.tensor(x2, dtype=torch.float32, device=DEVICE)
    
    # Calculate distance based on metric
    if metric.lower() == 'l2':
        return torch.sqrt(torch.sum((x1_tensor - x2_tensor) ** 2)).item()
    elif metric.lower() == 'l1':
        return torch.sum(torch.abs(x1_tensor - x2_tensor)).item()
    elif metric.lower() in ['linf', 'chebyshev']:
        return torch.max(torch.abs(x1_tensor - x2_tensor)).item()
    elif metric.lower() == 'cosine':
        cos_sim = torch.nn.functional.cosine_similarity(x1_tensor.unsqueeze(0), x2_tensor.unsqueeze(0))
        return 1.0 - cos_sim.item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def batch_pairwise_distances_gpu(X, Y, metric='l2', batch_size=1024):
    """
    Calculate pairwise distances between two sets of points using GPU batching.
    
    Args:
        X (numpy.ndarray): First set of points, shape (n, d)
        Y (numpy.ndarray): Second set of points, shape (m, d)
        metric (str): Distance metric ('l1', 'l2', 'linf', 'cosine')
        batch_size (int): Batch size for GPU processing
        
    Returns:
        numpy.ndarray: Distance matrix of shape (n, m)
    """
    X_torch = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Y_torch = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    
    n = X_torch.shape[0]
    m = Y_torch.shape[0]
    distances = torch.zeros((n, m), device=DEVICE)
    
    # Process in batches to avoid GPU memory overflow
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        X_batch = X_torch[i:end_i]
        
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            Y_batch = Y_torch[j:end_j]
            
            if metric.lower() == 'l2':
                # Efficient L2 distance calculation using broadcasting
                X_squared = torch.sum(X_batch**2, dim=1, keepdim=True)
                Y_squared = torch.sum(Y_batch**2, dim=1)
                XY = torch.mm(X_batch, Y_batch.T)
                dist = torch.sqrt(X_squared - 2 * XY + Y_squared.unsqueeze(0))
                distances[i:end_i, j:end_j] = dist
            
            elif metric.lower() == 'l1':
                # L1 distance calculation
                for k in range(i, end_i):
                    distances[k, j:end_j] = torch.sum(torch.abs(X_torch[k].unsqueeze(0) - Y_batch), dim=1)
            
            elif metric.lower() in ['linf', 'chebyshev']:
                # L-infinity distance calculation
                for k in range(i, end_i):
                    distances[k, j:end_j] = torch.max(torch.abs(X_torch[k].unsqueeze(0) - Y_batch), dim=1)[0]
            
            elif metric.lower() == 'cosine':
                # Cosine distance calculation
                for k in range(i, end_i):
                    norm_x = torch.nn.functional.normalize(X_torch[k].unsqueeze(0), p=2, dim=1)
                    norm_y = torch.nn.functional.normalize(Y_batch, p=2, dim=1)
                    cos_sim = torch.mm(norm_x, norm_y.T)
                    distances[k, j:end_j] = 1 - cos_sim.squeeze()
            
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    
    return distances.cpu().numpy()

def two_point_sep_calc_gpu(x, x_s, x_o, metric='l2'):
    """
    GPU-accelerated calculation of the separation between a point and two other points.
    
    Args:
        x (numpy.ndarray): Test point
        x_s (numpy.ndarray): Same-class point
        x_o (numpy.ndarray): Other-class point
        metric (str): Distance metric to use
        
    Returns:
        float: Separation value
    """
    x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    x_s_tensor = torch.tensor(x_s, dtype=torch.float32, device=DEVICE)
    x_o_tensor = torch.tensor(x_o, dtype=torch.float32, device=DEVICE)
    
    if metric.lower() == 'l2':
        d_xs = torch.sqrt(torch.sum((x_tensor - x_s_tensor) ** 2))
        d_xo = torch.sqrt(torch.sum((x_tensor - x_o_tensor) ** 2))
        d_so = torch.sqrt(torch.sum((x_s_tensor - x_o_tensor) ** 2))
        
        separation = (d_xo**2 - d_xs**2) / (2 * d_so)
        return separation.item()
    
    # For other metrics, use a simpler calculation
    d_xs = get_distance_gpu(x, x_s, metric)
    d_xo = get_distance_gpu(x, x_o, metric)
    d_so = get_distance_gpu(x_s, x_o, metric)
    
    # Use the distance ratio as an approximation
    return (d_xo - d_xs) / (2 * d_so)

# Define universal metrics mapping for different libraries
UNIVERSAL_METRICS = {
    'l1': {'sklearn': 'manhattan', 'torch': 'l1', 'faiss': 'l1'},
    'l2': {'sklearn': 'euclidean', 'torch': 'l2', 'faiss': 'l2'},
    'linf': {'sklearn': 'chebyshev', 'torch': 'linf', 'faiss': 'linf'},
    'cosine': {'sklearn': 'cosine', 'torch': 'cosine', 'faiss': 'cosine'}
}


def sep_calc_batch_gpu(X_test, X_train, y_train, pred_y, metric='l2', batch_size=128):
    """
    Calculate separation for multiple test examples using GPU batching.
    
    Args:
        X_test (numpy.ndarray): Test data points
        X_train (numpy.ndarray): Training data points
        y_train (numpy.ndarray): Training labels
        pred_y (numpy.ndarray): Predicted labels for test data
        metric (str): Distance metric
        batch_size (int): Size of batches for GPU processing
        
    Returns:
        numpy.ndarray: Separation values for each test point
    """
    logger.info(f"Starting GPU-accelerated batch separation calculation with {metric} metric.")
    
    # Make sure data is properly shaped
    if len(X_test.shape) > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    
    # For large datasets, calculate all pairwise distances once
    logger.info(f"Moving data to {DEVICE} for batch processing")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=DEVICE)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
    y_train_tensor = torch.tensor(y_train, device=DEVICE)
    pred_y_tensor = torch.tensor(pred_y, device=DEVICE)
    
    # Pre-compute indices for each class to avoid repeated calculations
    class_indices = {}
    for label in torch.unique(y_train_tensor):
        class_indices[label.item()] = torch.where(y_train_tensor == label)[0]
    
    separation_values = []
    
    # Process test points in batches
    for i in tqdm(range(0, len(X_test), batch_size), desc="Calculating Separation (GPU Batched)", unit="batch"):
        end_idx = min(i + batch_size, len(X_test))
        batch_X = X_test_tensor[i:end_idx]
        batch_pred = pred_y_tensor[i:end_idx]
        
        batch_results = []
        
        for j in range(len(batch_X)):
            x = batch_X[j]
            pred = batch_pred[j].item()
            
            # Get indices of same-class and different-class points
            same_indices = class_indices.get(pred, torch.tensor([], device=DEVICE))
            other_indices_list = [indices for label, indices in class_indices.items() if label != pred]
            
            if len(same_indices) == 0 or len(other_indices_list) == 0:
                # Handle edge case where there are no points in a class
                batch_results.append(0.0)
                continue
                
            # Combine all other-class indices
            other_indices = torch.cat(other_indices_list)
            
            # Calculate distances to same-class and different-class points
            same_distances = torch.sqrt(torch.sum((x.unsqueeze(0) - X_train_tensor[same_indices])**2, dim=1))
            other_distances = torch.sqrt(torch.sum((x.unsqueeze(0) - X_train_tensor[other_indices])**2, dim=1))
            
            # Sort distances
            same_distances, same_indices_sorted = torch.sort(same_distances)
            other_distances, other_indices_sorted = torch.sort(other_distances)
            
            # Get closest points
            closest_same_dist = same_distances[0].item()
            closest_other_dist = other_distances[0].item()
            
            # Simple approximation of separation based on fast-separation formula
            separation = (closest_other_dist - closest_same_dist) / 2
            batch_results.append(separation)
        
        separation_values.extend(batch_results)
    
    return np.array(separation_values)

def sep_calc_point_gpu(x, X_train, y_train, y_pred, metric='l2'):
    """
    Calculate the separation for a single test instance using GPU.
    
    Args:
        x (numpy.ndarray): Test point
        X_train (numpy.ndarray): Training data points
        y_train (numpy.ndarray): Training labels
        y_pred (int or numpy.ndarray): Predicted label for the test point
        metric (str): Distance metric to use
        
    Returns:
        float: Geometric separation value for the point
    """
    # Ensure `y_pred` is a scalar
    if hasattr(y_pred, "__len__") and len(y_pred) > 1:
        y_pred = np.argmax(y_pred)  # Convert probability vector to a class label if needed

    # Flatten `x` if it has more than one dimension
    if x.ndim > 1:
        x = x.flatten()
    
    # Convert data to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float64, device=DEVICE)
    X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float64, device=DEVICE)
    y_train_tensor = torch.tensor(y_train, device=DEVICE)
    
    # Get indices for same-class and different-class points
    same_indices = torch.where(y_train_tensor == y_pred)[0]
    other_indices = torch.where(y_train_tensor != y_pred)[0]
    
    # Calculate distances to same-class and different-class points
    same_distances = []
    for idx in same_indices:
        dist = get_distance_gpu(x, X_train[idx].flatten(), metric)
        same_distances.append((dist, idx.item()))
    
    other_distances = []
    for idx in other_indices:
        dist = get_distance_gpu(x, X_train[idx].flatten(), metric)
        other_distances.append((dist, idx.item()))
    
    # Sort by distance
    same_distances.sort(key=lambda x: x[0])
    other_distances.sort(key=lambda x: x[0])
    
    if not same_distances or not other_distances:
        return 0.0  # Handle edge case where there are no points in a class
    
    # Calculate minimum separation radius
    min_r = same_distances[0][0] + 2 * other_distances[0][0]
    sep_other = min_r
    
    for o_dist, o_idx in other_distances:
        sep_same = float('-inf')
        if o_dist > min_r:
            break
        
        for s_dist, s_idx in same_distances:
            if s_dist > min(min_r, o_dist) and o_dist > same_distances[0][0]:
                break
                
            x_s = X_train[s_idx].flatten()
            x_o = X_train[o_idx].flatten()
            sep_same = max(two_point_sep_calc_gpu(x, x_s, x_o, metric), sep_same)
            
        sep_other = min(sep_same, sep_other)
        min_r = same_distances[0][0] + 2 * max(0, sep_other)
    
    return sep_other

def calculate_stability_separation_gpu(X_test, X_train, y_train, y_pred, metric='l2', use_batching=True, batch_size=128):
    """
    Calculate separation-based stability with GPU acceleration.
    
    Args:
        X_test (numpy.ndarray): Test data points
        X_train (numpy.ndarray): Training data points
        y_train (numpy.ndarray): Training labels
        y_pred (numpy.ndarray): Predicted labels for test data
        metric (str): Distance metric to use
        use_batching (bool): Whether to use batch processing (faster for large datasets)
        batch_size (int): Batch size for GPU processing
        
    Returns:
        numpy.ndarray: Stability values for each test point based on separation
    """
    logger.info(f"Calculating stability using GPU-accelerated separation method with {metric} metric.")
    
    # Determine the optimal approach based on dataset size
    if use_batching and len(X_test) > 10:  # Batch processing for larger datasets
        return sep_calc_batch_gpu(X_test, X_train, y_train, y_pred, metric, batch_size)
    else:
        # Process points individually (might be better for very small datasets)
        results = []
        for i, x in tqdm(enumerate(X_test), desc="Calculating Separation (GPU)", unit="sample", total=len(X_test)):
            result = sep_calc_point_gpu(x, X_train, y_train, y_pred[i], metric)
            results.append(result)
            
        return np.array(results)