import numpy as np
import concurrent.futures
import logging
from tqdm import tqdm

def get_norm_value(metric):
    """
    Get the numpy norm value for the given metric.
    
    Args:
        metric (str): The distance metric type ('l1', 'l2', 'linf', etc.)
        
    Returns:
        int or numpy.inf: The norm value to use with numpy.linalg.norm
    """
    norm_map = {
        'l1': 1,
        'l2': 2,
        'linf': np.inf
    }
    return norm_map.get(metric.lower(), 2)  # Default to L2 if metric not found

def get_distance(x, y, metric):
    """
    Compute distance between two points using specified metric.
    
    Args:
        x (numpy.ndarray): First point
        y (numpy.ndarray): Second point
        metric (str): Distance metric to use ('l1', 'l2', 'linf', 'cosine')
        
    Returns:
        float: Distance between points
    """
    if metric == 'cosine':
        dot_product = np.dot(x, y)
        norms = np.linalg.norm(x) * np.linalg.norm(y)
        return 1 - (dot_product / norms if norms != 0 else 0)
    else:
        norm_val = get_norm_value(metric)
        return np.linalg.norm(x - y, ord=norm_val)

def two_point_sep_calc(x, x1, x2, metric='l2'):
    """
    Calculate the separation parameter for a single test point and two nearest points.
    
    Args:
        x (numpy.ndarray): Test point
        x1 (numpy.ndarray): First reference point
        x2 (numpy.ndarray): Second reference point
        metric (str): Distance metric to use
        
    Returns:
        float: Geometric separation value
    """
    a = get_distance(x, x1, metric)
    b = get_distance(x, x2, metric)
    c = get_distance(x1, x2, metric)
    return ((b ** 2 - a ** 2) / (2 * c))

# Universal distance metric mapping
UNIVERSAL_METRICS = {
    'l2': {'faiss': 'l2', 'scann': 'squared_l2', 'annoy': 'euclidean', 'nmslib': 'l2', 'hnsw': 'l2'},
    'cosine': {'faiss': 'cosine', 'scann': 'dot_product', 'annoy': 'angular', 'nmslib': 'cosinesimil',
               'hnsw': 'cosine'},
    'inner_product': {'faiss': 'inner_product', 'scann': 'dot_product', 'annoy': 'dot', 'nmslib': 'ip', 'hnsw': None}
}


# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)

def sep_calc_parallel(X_test, X_train, y_train, pred_y, metric='l2'):
    """
    Calculate the separation of all test/val examples in parallel with progress tracking.
    
    Args:
        X_test (numpy.ndarray): Test data points
        X_train (numpy.ndarray): Training data points
        y_train (numpy.ndarray): Training labels
        pred_y (numpy.ndarray): Predicted labels for test data
        metric (str): Distance metric to use
        
    Returns:
        numpy.ndarray: Separation values for each test point
    """
    logger.info("Starting parallel separation calculation.")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each task to the executor and track progress with tqdm
        futures = [
            executor.submit(sep_calc_point, x, X_train, y_train, pred, metric)
            for x, pred in zip(X_test, pred_y)
        ]
        
        logger.info(f"Submitted {len(futures)} tasks to the executor.")
        
        # Use tqdm to show progress as futures complete
        separation = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                           desc="Calculating Separation (Parallel)", unit="sample"):
            try:
                result = future.result()  # Get the result of each future
                separation.append(result)
            except Exception as e:
                logger.error(f"Error in parallel separation calculation: {e}")
        
        logger.info("Completed gathering results from futures.")
    
    return np.array(separation)

def sep_calc(X_test, X_train, y_train, pred_y, metric='l2'):
    """
    Calculate the separation of all test/val examples without parallel processing, with progress tracking.
    
    Args:
        X_test (numpy.ndarray): Test data points
        X_train (numpy.ndarray): Training data points
        y_train (numpy.ndarray): Training labels
        pred_y (numpy.ndarray): Predicted labels for test data
        metric (str): Distance metric to use
        
    Returns:
        numpy.ndarray: Separation values for each test point
    """
    logger.info("Starting sequential separation calculation with tqdm progress bar.")
    
    # Use tqdm to track progress over X_test for the sequential calculation
    results = []
    for i, x in tqdm(enumerate(X_test), desc="Calculating Separation", unit="sample", total=len(X_test)):
        result = sep_calc_point(x, X_train, y_train, pred_y[i], metric)
        results.append(result)
        
    logger.info("Completed sequential separation calculation.")
    return np.array(results)

def sep_calc_point(x, X_train, y_train, y_pred, metric='l2'):
    """
    Calculate the separation for a single test instance.
    
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
    
    # Calculate distances to same-class and different-class points
    same = [(get_distance(x, train.flatten(), metric), index) 
            for index, train in enumerate(X_train) if y_train[index] == y_pred]
    
    others = [(get_distance(x, train.flatten(), metric), index) 
              for index, train in enumerate(X_train) if y_train[index] != y_pred]
    
    # Sort by distance
    same.sort(key=lambda x: x[0])
    others.sort(key=lambda x: x[0])
    
    # Calculate minimum separation radius
    min_r = same[0][0] + 2 * others[0][0]
    sep_other = min_r
    
    for o in others:
        sep_same = np.NINF
        if o[0] > min_r:
            break
        
        for s in same:
            if s[0] > min(min_r, o[0]) and o[0] > same[0][0]:
                break
                
            x_s = X_train[s[1]].flatten()
            x_o = X_train[o[1]].flatten()
            sep_same = max(two_point_sep_calc(x, x_s, x_o, metric), sep_same)
            
        sep_other = min(sep_same, sep_other)
        min_r = same[0][0] + 2 * max(0, sep_other)
    
    return sep_other

def calculate_stability_separation(X_test, X_train, y_train, y_pred, metric='l2', parallel=False):
    """
    Calculate separation-based stability with progress tracking.
    
    Args:
        X_test (numpy.ndarray): Test data points
        X_train (numpy.ndarray): Training data points
        y_train (numpy.ndarray): Training labels
        y_pred (numpy.ndarray): Predicted labels for test data
        metric (str): Distance metric to use
        parallel (bool): Whether to use parallel processing
        
    Returns:
        numpy.ndarray: Stability values for each test point based on separation
    """
    logger.info("Calculating stability using separation method.")
    
    if parallel:
        logger.info("Using parallel separation calculation with progress tracking.")
        return sep_calc_parallel(X_test, X_train, y_train, y_pred, metric)
    else:
        logger.info("Using sequential separation calculation with progress tracking.")
        return sep_calc(X_test, X_train, y_train, y_pred, metric)