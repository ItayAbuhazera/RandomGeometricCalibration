import numpy as np
import time
import logging
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from .separation_utils import calculate_stability_separation
from .separation_utils_cuda import calculate_stability_separation_gpu

try:
    import faiss
except ImportError:  # pragma: no cover - depends on optional local install
    faiss = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_faiss():
    """Raise a clear error when FAISS-only code paths are requested."""
    if faiss is None:
        raise ImportError(
            "FAISS is not installed. Install optional FAISS dependencies or use "
            "library='fast_separation' / library='kdtree' instead."
        )

class StabilitySpace:
    """
    GPU-accelerated class to compute stability and geometric values for the input X.
    """

    def __init__(self, X_train, y_train, compression=None, library='fast_separation', metric='l2', num_labels=None,
                 faiss_mode='exact', nlist=None, use_cuda=True):
        """
        Initialize the stability space with GPU acceleration when available.
        
        Args:
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Training labels
            compression (object, optional): Compression object to reduce dimensionality
            library (str): Library to use for similarity search ('faiss', 'fast_separation', 'kdtree', 'separation')
            metric (str): Distance metric to use ('l1', 'l2', 'linf', 'cosine')
            num_labels (int, optional): Number of unique class labels
            faiss_mode (str): FAISS indexing mode ('exact' or 'approximate')
            nlist (int, optional): Number of clusters for FAISS approximate indexing
            use_cuda (bool): Whether to use CUDA acceleration if available
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing StabilitySpace with {library} library and {metric} metric.")

        self.metric = metric.lower()
        self.library = library
        self.num_labels = num_labels or len(set(y_train))
        self.compression = compression
        self.faiss_mode = faiss_mode
        self.nlist = nlist or self.num_labels
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.logger.info(f"CUDA support enabled: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CUDA support disabled or unavailable")

        # Store original shapes for logging
        original_shape = X_train.shape

        # Apply compression if provided
        if self.compression:
            self.logger.info("Applying compression to training data.")
            X_train, y_train = self.compression(X_train, y_train)
            self.logger.info(
                f"Compression applied. Original shape: {original_shape}, Compressed shape: {X_train.shape}")
            # Store compression input/output shapes for validation
            self.input_shape = original_shape[1:]
            self.output_shape = X_train.shape[1:]
        else:
            self.input_shape = original_shape[1:]
            self.output_shape = original_shape[1:]

        # Ensure data is properly shaped for the chosen library
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            self.logger.info(f"Reshaped training data to 2D: {X_train.shape}")

        self.X_train = X_train
        self.y_train = y_train

        # Initialize appropriate model based on library choice
        if library == 'faiss':
            _require_faiss()
            self._initialize_faiss()
        elif library == 'fast_separation':
            self._initialize_fast_separation()
        elif library == 'kdtree':
            self._initialize_kdtree()
        elif library == 'separation':
            self.logger.info("Using separation-based stability calculation.")
        else:
            raise ValueError(f"Unsupported library: {library}")

    def _get_faiss_index(self, dim, metric='l2'):
        """
        Get appropriate FAISS index based on metric, using GPU if available.
        
        Args:
            dim: Dimension of the vectors
            metric: Distance metric to use
        Returns:
            FAISS index object
        """
        _require_faiss()
        # Create the base index based on the metric
        if metric == 'l2':
            index = faiss.IndexFlatL2(dim)
        elif metric == 'l1':
            index = faiss.IndexFlat(dim, faiss.METRIC_L1)
        elif metric == 'linf':
            index = faiss.IndexFlat(dim, faiss.METRIC_Linf)
        elif metric == 'cosine':
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        else:
            raise ValueError(f"Unsupported FAISS metric: {metric}")
        
        # Try to move index to GPU if CUDA is available and requested
        if self.use_cuda:
            try:
                # Check if FAISS was built with GPU support
                gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                self.logger.info("Successfully created GPU-accelerated FAISS index")
                return gpu_index
            except Exception as e:
                self.logger.warning(f"Failed to create GPU index: {e}. Falling back to CPU.")
                return index
        else:
            return index

    def _initialize_kdtree(self):
        """
        Initialize KDTrees for each label, with data preloaded to GPU if available.
        """
        self.logger.info("Initializing KDTree indices for each class.")
        
        # Flatten X_train if it has more than 2 dimensions
        if len(self.X_train.shape) > 2:
            self.logger.info("Flattening X_train for KDTree compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)

        # Initialize KDTrees for each class
        self.kdtrees = [None] * self.num_labels
        
        # If using CUDA, precompute class indices for faster GPU lookups
        if self.use_cuda:
            self.class_data = {}  # Store each class's data in GPU memory
            
        for label in range(self.num_labels):
            # Get points for current class
            idx_same = np.where(self.y_train == label)[0]
            if len(idx_same) > 0:
                X_label = self.X_train[idx_same]
                
                # Store data in GPU memory if using CUDA
                if self.use_cuda:
                    self.class_data[label] = torch.tensor(X_label, dtype=torch.float32, device=DEVICE)
                
                # We still need the KDTree for CPU fallback
                self.kdtrees[label] = KDTree(X_label, metric=self.metric)
                self.logger.info(f"KDTree initialized for class {label} with {len(idx_same)} points")
            else:
                self.logger.warning(f"No points found for class {label}")
    
    def _initialize_faiss(self):
        """
        Initialize FAISS indices for each label with GPU support when available.
        """
        _require_faiss()
        self.same_nbrs = {}
        self.other_nbrs = {}

        # Ensure X_train is 2D for FAISS
        if len(self.X_train.shape) > 2:
            self.logger.info("Flattening X_train for FAISS compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1).astype('float32')

        # Normalize vectors if using cosine similarity
        if self.metric == 'cosine':
            self.logger.info("Normalizing vectors for cosine similarity.")
            faiss.normalize_L2(self.X_train)

        dim = self.X_train.shape[1]
        
        # Check if we can use GPU for FAISS
        if self.use_cuda:
            try:
                self.logger.info("Attempting to initialize GPU resources for FAISS")
                self.gpu_resources = faiss.StandardGpuResources()
                self.logger.info("Successfully initialized GPU resources for FAISS")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU resources for FAISS: {e}. Using CPU instead.")
                self.use_cuda = False
        
        for label in range(self.num_labels):
            idx_same = np.where(self.y_train == label)[0]
            idx_other = np.where(self.y_train != label)[0]

            if self.faiss_mode == 'exact':
                # Exact FAISS indices
                self.same_nbrs[label] = self._get_faiss_index(dim, self.metric)
                if len(idx_same) > 0:
                    self.same_nbrs[label].add(self.X_train[idx_same])

                self.other_nbrs[label] = self._get_faiss_index(dim, self.metric)
                if len(idx_other) > 0:
                    self.other_nbrs[label].add(self.X_train[idx_other])

            elif self.faiss_mode == 'approximate':
                # Approximate FAISS indices with clustering
                base_index = self._get_faiss_index(dim, self.metric)

                # Ensure nlist is smaller than or equal to the number of training points
                adjusted_nlist_same = min(self.nlist, max(1, len(idx_same) // 39))
                adjusted_nlist_other = min(self.nlist, max(1, len(idx_other) // 39))

                # For same_nbrs
                if len(idx_same) > 0:
                    if self.use_cuda:
                        # Create CPU index first, train it, then move to GPU
                        cpu_index = faiss.IndexIVFFlat(
                            faiss.IndexFlatL2(dim), dim, adjusted_nlist_same
                        )
                        cpu_index.train(self.X_train[idx_same])
                        # Move to GPU
                        self.same_nbrs[label] = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
                        self.same_nbrs[label].add(self.X_train[idx_same])
                    else:
                        # CPU-only version
                        self.same_nbrs[label] = faiss.IndexIVFFlat(base_index, dim, adjusted_nlist_same)
                        self.same_nbrs[label].train(self.X_train[idx_same])
                        self.same_nbrs[label].add(self.X_train[idx_same])
                    
                    # Set nprobe for search effectiveness
                    self.same_nbrs[label].nprobe = min(adjusted_nlist_same // 4, adjusted_nlist_same)
                else:
                    self.logger.warning(f"No points for label {label} in same_nbrs.")
                
                # For other_nbrs
                if len(idx_other) > 0:
                    if self.use_cuda:
                        # Create CPU index first, train it, then move to GPU
                        cpu_index = faiss.IndexIVFFlat(
                            faiss.IndexFlatL2(dim), dim, adjusted_nlist_other
                        )
                        cpu_index.train(self.X_train[idx_other])
                        # Move to GPU
                        self.other_nbrs[label] = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
                        self.other_nbrs[label].add(self.X_train[idx_other])
                    else:
                        # CPU-only version
                        self.other_nbrs[label] = faiss.IndexIVFFlat(base_index, dim, adjusted_nlist_other)
                        self.other_nbrs[label].train(self.X_train[idx_other])
                        self.other_nbrs[label].add(self.X_train[idx_other])
                    
                    # Set nprobe for search effectiveness
                    self.other_nbrs[label].nprobe = min(adjusted_nlist_other // 4, adjusted_nlist_other)
                else:
                    self.logger.warning(f"No points for label {label} in other_nbrs.")
            else:
                raise ValueError(f"Unsupported FAISS mode: {self.faiss_mode}")

    def _initialize_fast_separation(self):
        """
        Initialize NearestNeighbors models for each label with GPU acceleration for distance calculations.
        """
        self.same_nbrs = []
        self.other_nbrs = []
        if self.metric == "linf":
            self.metric = "chebyshev"
        
        # Flatten X_train if needed
        if len(self.X_train.shape) > 2:
            self.logger.info("Flattening X_train for fast_separation compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1).astype('float32')
        
        # Check dataset size - for large datasets, skip precomputing class tensors
        self.precomputed_tensors = len(self.X_train) < 10000 and self.num_labels < 50
        
        if self.use_cuda and self.precomputed_tensors:
            # Only precompute tensors for smaller datasets
            self.logger.info("Precomputing class tensors (small dataset detected)")
            self.class_indices = {}
            self.class_tensors = {}
            X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32, device=DEVICE)
            y_train_tensor = torch.tensor(self.y_train, device=DEVICE)
            
            for label in range(self.num_labels):
                idx_same = torch.where(y_train_tensor == label)[0]
                idx_other = torch.where(y_train_tensor != label)[0]
                
                if len(idx_same) > 0:
                    self.class_indices[label] = {'same': idx_same, 'other': idx_other}
                    self.class_tensors[label] = {
                        'same': X_train_tensor[idx_same],
                        'other': X_train_tensor[idx_other]
                    }
        elif self.use_cuda:
            # For large datasets, just store indices without creating tensor copies
            self.logger.info("Skipping tensor precomputation for large dataset")
            self.use_gpu_indices_only = True
            # Store the full dataset on GPU just once
            self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32, device=DEVICE)
            self.y_train_tensor = torch.tensor(self.y_train, device=DEVICE)
        else:
            self.logger.info("CUDA not available, using CPU processing")
        
        # Always initialize CPU NearestNeighbors as fallback
        for label in range(self.num_labels):
            idx_same = np.where(self.y_train == label)[0]
            idx_other = np.where(self.y_train != label)[0]
            
            same_nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(self.X_train[idx_same])
            other_nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(self.X_train[idx_other])
            
            self.same_nbrs.append(same_nn)
            self.other_nbrs.append(other_nn)

    def _stability_kdtree_gpu(self, valX, val_y_pred):
        """
        Calculate stability using KDTrees with GPU-accelerated distance calculations.
        FIXED: Removed inner tqdm to avoid nested progress bars
        
        Args:
            valX (numpy.ndarray): Validation data
            val_y_pred (numpy.ndarray): Predicted labels for validation data
            
        Returns:
            numpy.ndarray: Stability scores for validation data
        """
        self.logger.info("Calculating stability using KDTree with GPU acceleration.")
        
        # Ensure input is 2D
        valX = valX.reshape(valX.shape[0], -1)
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred
        
        # Move validation data to GPU
        valX_tensor = torch.tensor(valX, dtype=torch.float32, device=DEVICE)
        
        # Initialize stability scores
        stability = np.zeros(len(valX))
        
        # Calculate stability scores with batch processing for efficiency
        batch_size = 128  # Adjust based on GPU memory
        total_batches = (len(valX) + batch_size - 1) // batch_size
        
        # FIXED: Remove tqdm here to avoid nested progress bars
        for batch_idx, start_idx in enumerate(range(0, len(valX), batch_size)):
            end_idx = min(start_idx + batch_size, len(valX))
            batch_X = valX_tensor[start_idx:end_idx]
            batch_pred = predicted_labels[start_idx:end_idx]
            
            # Log progress every 10 batches or at the end
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                self.logger.debug(f"   KDTree stability progress: {batch_idx + 1}/{total_batches} batches")
            
            # Calculate distances for each sample in the batch
            for i in range(len(batch_X)):
                x = batch_X[i]
                pred_label = batch_pred[i]
                
                if pred_label not in self.class_data:
                    # Skip if predicted class has no data
                    stability[start_idx + i] = 0
                    continue
                
                # Calculate pairwise distances between this point and all points in the predicted class
                class_data = self.class_data[pred_label]
                distances = torch.sqrt(torch.sum((x.unsqueeze(0) - class_data)**2, dim=1))
                min_same_dist = torch.min(distances).item()
                
                # Calculate distances to other classes
                min_other_dist = float('inf')
                for label in range(self.num_labels):
                    if label != pred_label and label in self.class_data:
                        other_data = self.class_data[label]
                        other_distances = torch.sqrt(torch.sum((x.unsqueeze(0) - other_data)**2, dim=1))
                        label_min_dist = torch.min(other_distances).item()
                        min_other_dist = min(min_other_dist, label_min_dist)
                
                # Calculate stability
                if min_other_dist < float('inf'):
                    stability[start_idx + i] = (min_other_dist - min_same_dist) / 2
        
        return stability

    def _stability_fast_separation_gpu(self, valX, val_y_pred):
        """
        Calculate stability using fast_separation with GPU-accelerated distance calculations.
        FIXED: Removed inner tqdm to avoid nested progress bars
        
        Args:
            valX (numpy.ndarray): Validation data
            val_y_pred (numpy.ndarray): Predicted labels for validation data
            
        Returns:
            numpy.ndarray: Stability scores for validation data
        """
        self.logger.info("Calculating stability using fast_separation with GPU acceleration.")
        stability = np.zeros(len(valX))
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred

        # Log shape information
        self.logger.info(f"Validation data shape: {valX.shape}, predicted labels shape: {predicted_labels.shape}")
        self.logger.info(f"Number of unique predicted labels: {len(np.unique(predicted_labels))}")

        # Verify dimensions match
        expected_features = self.X_train.shape[1]
        if valX.shape[1] != expected_features:
            self.logger.warning(f"Validation data shape mismatch: {valX.shape[1]} vs {expected_features}")
            return np.zeros(len(valX))

        # Move validation data to GPU
        valX_tensor = torch.tensor(valX.reshape(valX.shape[0], -1), dtype=torch.float32, device=DEVICE)
        
        # Process in batches
        # Use smaller batch sizes for CIFAR100
        if self.num_labels > 50:  # Assuming CIFAR100
            batch_size = 32  # Smaller batch size for CIFAR100
        else:
            batch_size = 64  # Original batch size
        
        # Track distance statistics
        total_same_distances = []
        total_other_distances = []
        empty_same_indices_count = 0
        empty_other_indices_count = 0
        
        # FIXED: Remove tqdm here to avoid nested progress bars
        # Instead, just log progress periodically
        total_batches = (len(valX) + batch_size - 1) // batch_size
        
        for batch_idx, start_idx in enumerate(range(0, len(valX), batch_size)):
            end_idx = min(start_idx + batch_size, len(valX))
            batch_X = valX_tensor[start_idx:end_idx]
            batch_pred = predicted_labels[start_idx:end_idx]
            
            # Log progress every 5 batches or at the end
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                self.logger.debug(f"   Stability calculation progress: {batch_idx + 1}/{total_batches} batches")
            
            # Ensure the batch sizes match
            batch_size_actual = min(len(batch_X), len(batch_pred))
            
            for i in range(batch_size_actual):
                x = batch_X[i]
                pred_label = int(batch_pred[i])
                
                # Log the predicted label occasionally to verify we have variety
                if i == 0 and start_idx == 0:
                    self.logger.info(f"Sample 0 predicted label: {pred_label}")
                
                try:
                    if self.use_cuda:
                        if hasattr(self, 'precomputed_tensors') and self.precomputed_tensors and pred_label in self.class_tensors:
                            # For small datasets - use precomputed tensors
                            same_data = self.class_tensors[pred_label]['same']
                            other_data = self.class_tensors[pred_label]['other']
                            
                            # Log the shapes of the data tensors
                            if i == 0 and start_idx == 0:
                                self.logger.info(f"Same class data shape: {same_data.shape}, Other class data shape: {other_data.shape}")
                            
                            same_distances = torch.sqrt(torch.sum((x.unsqueeze(0) - same_data)**2, dim=1))
                            other_distances = torch.sqrt(torch.sum((x.unsqueeze(0) - other_data)**2, dim=1))
                            
                            dist_same = torch.min(same_distances).item()
                            dist_other = torch.min(other_distances).item()
                        elif hasattr(self, 'use_gpu_indices_only') and self.use_gpu_indices_only:
                            # For large datasets - calculate on-the-fly
                            same_indices = torch.where(self.y_train_tensor == pred_label)[0]
                            
                            # Log when we have empty indices
                            if len(same_indices) == 0:
                                empty_same_indices_count += 1
                                if empty_same_indices_count <= 5:
                                    self.logger.warning(f"No same-class points found for label {pred_label}")
                                dist_same = float('inf')
                            else:
                                same_distances = torch.sqrt(torch.sum((x.unsqueeze(0) - self.X_train_tensor[same_indices])**2, dim=1))
                                dist_same = torch.min(same_distances).item()
                                del same_distances  # Free memory
                                    
                            other_indices = torch.where(self.y_train_tensor != pred_label)[0]
                            
                            if len(other_indices) == 0:
                                empty_other_indices_count += 1
                                if empty_other_indices_count <= 5:
                                    self.logger.warning(f"No other-class points found for label {pred_label}")
                                dist_other = float('inf')
                            else:
                                # Process in sub-batches if needed
                                if len(other_indices) > 10000:
                                    sub_batch_size = 10000
                                    min_dist = float('inf')
                                    for j in range(0, len(other_indices), sub_batch_size):
                                        sub_indices = other_indices[j:j+sub_batch_size]
                                        sub_distances = torch.sqrt(torch.sum(
                                            (x.unsqueeze(0) - self.X_train_tensor[sub_indices])**2, dim=1))
                                        sub_min = torch.min(sub_distances).item()
                                        min_dist = min(min_dist, sub_min)
                                        del sub_distances  # Free memory
                                    dist_other = min_dist
                                else:
                                    other_distances = torch.sqrt(torch.sum(
                                        (x.unsqueeze(0) - self.X_train_tensor[other_indices])**2, dim=1))
                                    dist_other = torch.min(other_distances).item()
                                    del other_distances  # Free memory
                            
                            # Log distance statistics periodically
                            if len(total_same_distances) < 100 and dist_same != float('inf'):
                                total_same_distances.append(dist_same)
                            if len(total_other_distances) < 100 and dist_other != float('inf'):
                                total_other_distances.append(dist_other)
                                
                            torch.cuda.empty_cache()  # Clear GPU cache
                        else:
                            # Fallback to CPU
                            dist_same, _ = self.same_nbrs[pred_label].kneighbors(valX[start_idx + i].reshape(1, -1))
                            dist_other, _ = self.other_nbrs[pred_label].kneighbors(valX[start_idx + i].reshape(1, -1))
                            dist_same = dist_same[0][0]
                            dist_other = dist_other[0][0]
                    else:
                        # CPU fallback
                        dist_same, _ = self.same_nbrs[pred_label].kneighbors(valX[start_idx + i].reshape(1, -1))
                        dist_other, _ = self.other_nbrs[pred_label].kneighbors(valX[start_idx + i].reshape(1, -1))
                        dist_same = dist_same[0][0]
                        dist_other = dist_other[0][0]
                    
                    # Log sample distances occasionally
                    if (start_idx + i) % 1000 == 0:
                        self.logger.info(f"Sample {start_idx + i}: dist_same={dist_same:.6f}, dist_other={dist_other:.6f}, stability={(dist_other - dist_same) / 2:.6f}")
                    
                    stability[start_idx + i] = (dist_other - dist_same) / 2
                except Exception as e:
                    self.logger.error(f"Error in calculation for sample {start_idx + i}: {e}")
                    stability[start_idx + i] = np.nan
            
            # Free batch memory
            del batch_X
            torch.cuda.empty_cache()
        
        # Log final distance statistics
        if total_same_distances:
            self.logger.info(f"Same-class distances [min, max, mean]: [{min(total_same_distances):.6f}, {max(total_same_distances):.6f}, {sum(total_same_distances)/len(total_same_distances):.6f}]")
        if total_other_distances:
            self.logger.info(f"Other-class distances [min, max, mean]: [{min(total_other_distances):.6f}, {max(total_other_distances):.6f}, {sum(total_other_distances)/len(total_other_distances):.6f}]")
        
        # Check for numerical issues
        stability_stats = {
            "zeros": np.sum(stability == 0),
            "neg": np.sum(stability < 0),
            "pos": np.sum(stability > 0),
            "nan": np.sum(np.isnan(stability)),
            "min": np.nanmin(stability) if not np.all(np.isnan(stability)) else "all nan",
            "max": np.nanmax(stability) if not np.all(np.isnan(stability)) else "all nan",
            "mean": np.nanmean(stability) if not np.all(np.isnan(stability)) else "all nan"
        }
        self.logger.info(f"Stability statistics: {stability_stats}")
        
        # If we have empty indices, log them
        if empty_same_indices_count > 0:
            self.logger.warning(f"Found {empty_same_indices_count} samples with no same-class points")
        if empty_other_indices_count > 0:
            self.logger.warning(f"Found {empty_other_indices_count} samples with no other-class points")

        return stability


    def _stability_faiss_gpu(self, valX, val_y_pred):
        """
        Calculate stability using FAISS with GPU acceleration.
        FIXED: Removed inner tqdm to avoid nested progress bars
        
        Args:
            valX (numpy.ndarray): Validation data
            val_y_pred (numpy.ndarray): Predicted labels for validation data
            
        Returns:
            numpy.ndarray: Stability scores for validation data
        """
        _require_faiss()
        self.logger.info("Calculating stability using FAISS with GPU acceleration.")
        stability = np.zeros(len(valX))
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred

        # Get the expected feature dimension from the training data
        expected_features = self.X_train.shape[1]

        # Check if validation data has the correct shape
        if valX.shape[1] != expected_features:
            self.logger.warning(
                f"Validation data shape {valX.shape[1]} does not match training shape {expected_features}. "
                "This might indicate a compression mismatch.")
            return np.zeros(len(valX))  # Return zeros instead of raising an error

        # Prepare validation data
        valX = valX.reshape(len(valX), -1).astype('float32')
        if self.metric == 'cosine':
            faiss.normalize_L2(valX)

        # Process in batches for efficiency
        batch_size = 128  # Adjust based on GPU memory and dataset size
        total_batches = (len(valX) + batch_size - 1) // batch_size
        
        # FIXED: Remove tqdm here to avoid nested progress bars
        for batch_idx, start_idx in enumerate(range(0, len(valX), batch_size)):
            end_idx = min(start_idx + batch_size, len(valX))
            batch_X = valX[start_idx:end_idx]
            batch_pred = predicted_labels[start_idx:end_idx]
            
            # Log progress every 10 batches or at the end
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                self.logger.debug(f"   FAISS stability progress: {batch_idx + 1}/{total_batches} batches")
            
            for i in range(len(batch_X)):
                x = batch_X[i:i+1]  # Keep as 2D array
                pred_label = int(batch_pred[i])

                try:
                    _, dist_same = self.same_nbrs[pred_label].search(x, 1)
                    _, dist_other = self.other_nbrs[pred_label].search(x, 1)

                    # For cosine similarity, convert similarity to distance
                    if self.metric == 'cosine':
                        dist_same = 1 - dist_same
                        dist_other = 1 - dist_other

                    stability[start_idx + i] = (dist_other[0][0] - dist_same[0][0]) / 2
                except Exception as e:
                    self.logger.error(f"Error in FAISS stability calculation for sample {start_idx + i}: {e}")
                    stability[start_idx + i] = np.nan

        return stability

    def calc_stab(self, X_val, y_val_pred, timeout=1800):
        """
        Calculate stability for the validation set with GPU acceleration when available.
        
        Args:
            X_val (numpy.ndarray): Validation data
            y_val_pred (numpy.ndarray): Predicted labels for validation data
            timeout (int): Maximum time in seconds to run the calculation
            
        Returns:
            numpy.ndarray: Stability scores for validation data
        """
        start_time = time.time()

        # Apply compression if provided
        if self.compression:
            self.logger.info("Applying compression before stability calculation")
            X_val_original_shape = X_val.shape
            X_val_compressed, _ = self.compression(X_val, None, train=False)
            self.logger.info(f"Compressed validation data from {X_val_original_shape} to {X_val_compressed.shape}")
            X_val = X_val_compressed  # Use compressed data for stability calculation

        # Ensure data is properly shaped
        if len(X_val.shape) > 2:
            X_val = X_val.reshape(X_val.shape[0], -1)
            self.logger.info(f"Reshaped validation data to 2D: {X_val.shape}")

        # Verify shapes match
        if X_val.shape[1] != self.X_train.shape[1]:
            self.logger.error(f"Validation data shape mismatch")
            return np.zeros(len(X_val))

        # Automatically switch to FAISS for large CIFAR100-like datasets
        is_large_dataset = len(self.X_train) > 10000 and self.num_labels > 50
        use_original_library = self.library
        
        # if is_large_dataset and self.library == 'fast_separation':
        #     self.logger.info(f"Switching from {self.library} to faiss for large dataset")
        #     original_lib = self.library
        #     self.library = 'faiss'
        #     # Initialize FAISS if it wasn't initialized before
        #     if not hasattr(self, 'same_nbrs') or not isinstance(self.same_nbrs, dict):
        #         self._initialize_faiss()
        
        # Calculate stability using the appropriate method
        if self.library == 'faiss':
            stability = self._stability_faiss_gpu(X_val, y_val_pred)
        elif self.library == 'fast_separation':
            stability = self._stability_fast_separation_gpu(X_val, y_val_pred)
        elif self.library == 'kdtree':
            if self.use_cuda and hasattr(self, 'class_data'):
                stability = self._stability_kdtree_gpu(X_val, y_val_pred)
            else:
                stability = self._stability_kdtree(X_val, y_val_pred)
        elif self.library == 'separation':
            stability = calculate_stability_separation(X_val, self.X_train, self.y_train, 
                                                    y_val_pred, self.metric, 
                                                    use_batching=self.use_cuda)
        else:
            raise ValueError(f"Unsupported library: {self.library}")
        
        # Reset to original library for future calls
        if is_large_dataset and use_original_library != self.library:
            self.library = use_original_library

        elapsed_time = time.time() - start_time
        self.logger.info(f"Time taken for stability calculation: {elapsed_time:.2f} seconds")

        return stability
        
    # Original CPU implementation methods kept for fallback
    def _stability_kdtree(self, valX, val_y_pred):
        """Original CPU implementation of KDTree stability calculation."""
        self.logger.info("Calculating stability using KDTree (CPU fallback).")
        
        # Ensure input is 2D
        valX = valX.reshape(valX.shape[0], -1)
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred
        
        # Initialize distance matrix
        distances = np.zeros((len(valX), self.num_labels))
        
        # Calculate distances to each class
        for label in range(self.num_labels):
            if self.kdtrees[label] is not None:
                # Get distance to 2nd nearest neighbor (k=2) as in TrustScore
                distances[:, label] = self.kdtrees[label].query(valX, k=2)[0][:, -1]
        
        # Calculate stability scores
        stability = np.zeros(len(valX))
        for i in range(len(valX)):
            pred_label = predicted_labels[i]
            
            # Get distance to predicted class
            d_to_pred = distances[i, pred_label]
            
            # Get distances to other classes
            other_distances = distances[i, :]
            other_distances[pred_label] = np.inf  # Exclude predicted class
            d_to_closest_not_pred = np.min(other_distances)
            
            # Calculate stability as in original implementation
            stability[i] = (d_to_closest_not_pred - d_to_pred) / 2
        
        return stability
        
    def calc_sep(self, X_val, y_val_pred, parallel=False):
        """
        Calculate separation for validation data with GPU acceleration if available.
        
        Args:
            X_val (numpy.ndarray): Validation data
            y_val_pred (numpy.ndarray): Predicted class labels
            parallel (bool): Whether to use parallel processing on CPU
            
        Returns:
            numpy.ndarray: Separation values
        """
        # Apply compression if provided
        if self.compression:
            self.logger.info("Applying compression before separation calculation")
            X_val_original_shape = X_val.shape
            X_val_compressed, _ = self.compression(X_val, None, train=False)
            self.logger.info(f"Compressed validation data from {X_val_original_shape} to {X_val_compressed.shape}")
            X_val = X_val_compressed
            
        # Use GPU-accelerated separation calculation
        if self.use_cuda:
            self.logger.info("Using GPU-accelerated separation calculation")
            return calculate_stability_separation_gpu(X_val, self.X_train, self.y_train, y_val_pred, 
                                                     self.metric, use_batching=True)
        else:
            # Fall back to CPU implementation
            self.logger.info("Using CPU separation calculation")
            from separation_utils import calculate_stability_separation
            return calculate_stability_separation(X_val, self.X_train, self.y_train, y_val_pred, 
                                                 self.metric, parallel)

