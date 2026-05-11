# Calibrators/geometric_calibrator.py
"""
Geometric Calibrator using Fast Separation + Isotonic Regression
"""

import numpy as np
import logging
import torch
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from .base_calibrator import BaseCalibrator
from utils.stability_space import StabilitySpace
import time
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset
from utils.compression_utils import SmartCompression

from utils.logging_config import get_logger
logger = get_logger(__name__)


# ============================================================================
class GeometricCalibrator(BaseCalibrator):
    """
    Geometric calibration using stability/separation scores and isotonic regression.
    """

    def __init__(self, model, X_train_embed=None, y_train=None, X_train_original=None,
                 compression_mode=None, compression_param=None,
                 metric='l2', stability_space=None,
                 library='fast_separation', use_binning=True, n_bins=200,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 fitting_method: str = 'isotonic',  # Options: 'isotonic', 'beta', 'platt', 'logistic'
                 **kwargs):
        """
        Initializes the GeometricCalibrator with isotonic regression.
        Uses the separation score (Eq. 11 in the paper) for confidence.
        """
        logging.info("\n=== GeometricCalibrator Initialization ===")
        if X_train_embed is not None:
            logging.info(f"Input X_train_embed shape: {X_train_embed.shape}")
        if X_train_original is not None:
            logging.info(f"Input X_train_original shape: {X_train_original.shape}")
        logging.info(f"Compression mode: {compression_mode}")
        logging.info(f"Compression param: {compression_param}")
        logging.info(f"Library: {library}")
        logging.info(f"Metric: {metric}")

        # Drop legacy kwargs from older experiments so callers that still pass
        # them keep working without spreading silent errors downstream.
        kwargs.pop('scoring_method', None)
        kwargs.pop('auto_select_layer', None)

        super().__init__(**kwargs)
        self.model = model
        self.isotonic_regressor = None
        self.regressor = None
        self.fitting_method = fitting_method
        self.is_fitted = False
        self.metric = metric.lower()
        self.use_binning = use_binning
        self.n_bins = n_bins
        self.device = device
        self.compression_mode = compression_mode
        self.compression_param = compression_param
        self.library = library

        if y_train is not None:
            self.num_labels = len(np.unique(y_train))
        else:
            self.num_labels = None

        if X_train_embed is None:
            raise ValueError("X_train_embed is required")
        if y_train is None:
            raise ValueError("y_train is required")
        self.y_train = y_train
        if stability_space:
            self.stab_space = stability_space
        else:
            compression = SmartCompression(method='fixed_spp_jl', target_dims=512, in_channels=X_train_embed.shape[1]) if compression_mode else None
            self.stab_space = StabilitySpace(X_train_embed, y_train, compression=compression, library=library, metric=self.metric)
        
        logger.info(f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__}")

    def _create_regressor(self):
        """
        Create the appropriate regressor based on fitting_method.
        
        Returns:
            A fitted regressor with .fit(X, y) and .predict(X) methods
        """
        if self.fitting_method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            logger.info("    Using Isotonic Regression (non-parametric)")
            return IsotonicRegression(out_of_bounds="clip")
        
        elif self.fitting_method == 'beta':
            from Calibrators.beta_calibration import _BetaCal
            logger.info("    Using Beta Calibration (3 parameters: a, b, m)")
            return _BetaCal()
        
        elif self.fitting_method == 'platt':
            from sklearn.linear_model import LogisticRegression
            logger.info("    Using Platt Scaling (logistic regression)")
            # Platt scaling: logit(p) = a*s + b
            return LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=42
            )
        
        elif self.fitting_method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            logger.info("    Using Logistic Regression with log transforms")
            # Similar to beta but simpler: logit(p) = a + b*log(s)
            return LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown fitting_method: {self.fitting_method}. "
                            f"Choose from: 'isotonic', 'beta', 'platt', 'logistic'")

    def _compute_score(self, X_val, y_val_pred):
        """
        Compute the geometric separation score (Eq. 11 in the paper).

        Args:
            X_val (numpy.ndarray): Validation data embeddings
            y_val_pred (numpy.ndarray): Predicted labels or probabilities

        Returns:
            numpy.ndarray: Separation score values
        """
        return self.stab_space.calc_stab(X_val, y_val_pred)

    def fit(self, X_val_embed: Optional[np.ndarray] = None, y_val=None, X_val_original: Optional[np.ndarray] = None, features=None, fit_batch_size=1000, model_probs: Optional[np.ndarray] = None):
        """
        Fit the geometric calibrator using validation data.
        
        Args:
            X_val_embed: Validation embeddings/features
            y_val: Validation labels
            X_val_original: Raw validation inputs (required for predict_proba)
            features: Unused, for API compatibility
            fit_batch_size: Batch size for processing
            model_probs: Optional pre-computed model probabilities
        """
        logger.info(f"{self.__class__.__name__}: Fitting calibrator.")
        
        # Require original images for predict_proba
        if X_val_original is None:
            raise ValueError(
                "X_val_original (raw inputs) is required for predict_proba during fitting."
            )
        
        prediction_data = X_val_original
        embedding_data = X_val_embed
        labels_data = y_val
        logger.info(f"    Using validation data: {len(labels_data)} samples")

        # =====================================================================
        #  BATCHED SEPARATION CALCULATION AND PREDICTION PROCESSING
        # =====================================================================

        score_name = "Stability"
        logger.info(f" Computing {score_name.lower()} values and predictions in batches...")
        all_stability_vals, all_predictions, all_labels = [], [], []
        n_samples = len(labels_data)
        n_batches = (n_samples + fit_batch_size - 1) // fit_batch_size
        
        logger.info(f"   Processing {n_samples} samples in {n_batches} batches of {fit_batch_size}")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * fit_batch_size
            end_idx = min(start_idx + fit_batch_size, n_samples)
            
            batch_pred_data = prediction_data[start_idx:end_idx]
            batch_embed_data = embedding_data[start_idx:end_idx]
            batch_labels = labels_data[start_idx:end_idx]
            
            # Get model predictions for this batch
            batch_predictions = self.model.predict_proba(batch_pred_data)
            
            # Calculate stability (separation) values using the stability space
            batch_stability = self._compute_score(batch_embed_data, batch_predictions)
            
            # Collect results
            all_stability_vals.extend(batch_stability)
            all_predictions.extend(np.argmax(batch_predictions, axis=1))
            all_labels.extend(batch_labels)
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                logger.info(f"   Processed batch {batch_idx + 1}/{n_batches}")
        
        # Convert to numpy arrays
        stability_val = np.array(all_stability_vals)
        y_pred_classes = np.array(all_predictions)
        y_val_processed = np.array(all_labels)
        
        logger.info(f" {score_name} calculation complete:")
        logger.info(f"    {score_name} range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        logger.info(f"    Mean {score_name.lower()}: {np.mean(stability_val):.6f}")
        
        # =====================================================================
        #  STABILITY NORMALIZATION (Rank-based)
        # =====================================================================
        
        logger.info(" Storing sorted validation scores for rank-based normalization...")
        
        # Store sorted validation scores for rank-based normalization
        # Isotonic regression only cares about ordering, not absolute values
        self.val_scores_sorted = np.sort(stability_val)
        n_val_samples = len(self.val_scores_sorted)
        
        logger.info(f"    Stored {n_val_samples} sorted validation scores")
        logger.info(f"    Score range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        logger.info(f"    Mean score: {np.mean(stability_val):.6f}")
        
        # For fitting, we still need normalized values, so normalize using ranks
        # This preserves the ordering information that isotonic regression needs
        from scipy.stats import rankdata
        stability_val = rankdata(stability_val, method='average') / n_val_samples
        
        logger.info(f"    Applied rank-based normalization")
        logger.info(f"    Normalized range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        
        # =====================================================================
        #  BINNING (if enabled)
        # =====================================================================
        
        # Only use binning for isotonic regression; parametric methods need raw data
        use_binning_effective = self.use_binning and (self.fitting_method == 'isotonic')
        
        # Log warning if binning was requested but disabled
        if self.use_binning and not use_binning_effective:
            logger.warning(f" Binning disabled for {self.fitting_method} calibration (requires raw data).")

        if use_binning_effective:
            logger.info(f" Applying binning with {self.n_bins} bins...")
            
            bin_edges = np.linspace(0, 1, self.n_bins + 1, dtype=np.float32)
            bin_indices = np.digitize(stability_val, bin_edges, right=True) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            
            binned_stability, binned_accuracy = [], []
            
            for bin_idx in range(self.n_bins):
                indices_in_bin = np.where(bin_indices == bin_idx)[0]
                
                if len(indices_in_bin) > 0:
                    # Calculate accuracy for this bin
                    accuracy = np.mean(y_val_processed[indices_in_bin] == y_pred_classes[indices_in_bin])
                    
                    # Use bin center as stability value
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    
                    binned_stability.append(bin_center)
                    binned_accuracy.append(accuracy)
            
            stability_vals = np.array(binned_stability)
            accuracies = np.array(binned_accuracy)
            
            logger.info(f"    Created {len(stability_vals)} bins with data")
            logger.info(f"    Accuracy range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
            
        else:
            logger.info(" Using individual samples (no binning)")
            stability_vals = stability_val
            accuracies = (y_val_processed == y_pred_classes).astype(float)
            
            logger.info(f"    Accuracy: {np.mean(accuracies):.4f}")
        
        # =====================================================================
        #  REGRESSOR FITTING
        # =====================================================================
        
        logger.info(f" Fitting {self.fitting_method} regressor...")
        
        # Create the appropriate regressor
        self.regressor = self._create_regressor()
        self.isotonic_regressor = self.regressor # Keep for backward compatibility if needed, but regressor is main
        
        # Prepare input data based on method
        if self.fitting_method in ['beta', 'logistic']:
            # These methods need log-transformed features
            eps = 1e-6
            stability_clipped = np.clip(stability_vals, eps, 1 - eps)
            
            if self.fitting_method == 'beta':
                # Beta calibration uses log(s) and log(1-s)
                X_fit = stability_clipped  # _BetaCal handles transform internally
            elif self.fitting_method == 'logistic':
                # Logistic with log transform
                X_fit = np.log(stability_clipped).reshape(-1, 1)
            
            y_fit = accuracies.astype(int)  # Binary labels for classification
            
        elif self.fitting_method in ['isotonic', 'platt']:
            # Direct mapping from stability to accuracy
            X_fit = stability_vals
            
            if self.fitting_method == 'platt':
                # Platt scaling needs binary labels
                y_fit = accuracies.astype(int)
                X_fit = X_fit.reshape(-1, 1)  # sklearn needs 2D input
            else:
                # Isotonic can use continuous targets
                y_fit = accuracies
        
        # Fit the regressor
        logger.info(f"    Fitting on {len(X_fit)} samples...")
        self.regressor.fit(X_fit, y_fit)
        
        # Log fitted parameters if available
        if self.fitting_method == 'beta' and hasattr(self.regressor, 'map_'):
            a, b, m = self.regressor.map_
            logger.info(f"    Beta parameters: a={a:.4f}, b={b:.4f}, m={m:.4f}")
        
        elif self.fitting_method in ['platt', 'logistic']:
            if hasattr(self.regressor, 'coef_') and hasattr(self.regressor, 'intercept_'):
                coef = self.regressor.coef_[0] if len(self.regressor.coef_.shape) > 1 else self.regressor.coef_
                intercept = self.regressor.intercept_[0] if hasattr(self.regressor.intercept_, '__len__') else self.regressor.intercept_
                logger.info(f"    Logistic parameters: coef={coef}, intercept={intercept:.4f}")
        
        elif self.fitting_method == 'isotonic':
            if hasattr(self.regressor, 'X_thresholds_'):
                n_thresholds = len(self.regressor.X_thresholds_)
                logger.info(f"    Isotonic fit: {n_thresholds} thresholds")
        
        # Note: val_scores_sorted is already stored above for rank-based normalization
        # No need to store percentile parameters anymore
        
        # Initialize storage for stability scores (for per-sample saving)
        self.last_stability_scores = None
        
        # Mark as fitted
        self.is_fitted = True
        
        logger.info(f" {self.__class__.__name__} fitting complete!")
        logger.info(f"    Trained on {len(stability_vals)} {'bins' if self.use_binning else 'samples'}")
        logger.info(f"    Method: {self.fitting_method}")
        logger.info(f"    Rank-based normalization: stored {len(self.val_scores_sorted)} sorted validation scores")

    def calibrate_batched(self, X_test_embed: Optional[np.ndarray] = None, X_test_original: Optional[np.ndarray] = None, features=None, batch_size=1000, return_probs=False):
        """
        Memory-efficient batched calibration to prevent GPU OOM.
        
        Args:
            X_test_embed: Test embeddings/features
            X_test_original: Raw test inputs (required for predict_proba)
            features: Unused, for API compatibility
            batch_size: Batch size for processing
            return_probs: If True, returns (calibrated_probs, model_probs) tuple.
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        # Require original images for predict_proba
        if X_test_original is None:
            raise ValueError(
                "X_test_original (raw inputs) is required for predict_proba."
            )
        
        logger.info(f"GeometricCalibrator: Calibrating test data in batches of {batch_size}.")
        n_samples = X_test_original.shape[0]
        
        # Guard against missing num_labels
        if self.num_labels is None:
            # lazily infer from model on a tiny batch
            tmp = self.model.predict_proba(X_test_original[:1])
            self.num_labels = tmp.shape[1]
            logger.info(f" Inferred num_labels = {self.num_labels} from model output")
        
        num_classes = self.num_labels
        all_calibrated_probs = np.zeros((n_samples, num_classes))
        all_model_probs = np.zeros((n_samples, num_classes)) if return_probs else None
        all_stability_scores = np.zeros(n_samples)  # Accumulate stability scores
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        start_time = time.time()
        with tqdm(total=n_batches, desc="Calibrating Test Batches", unit="batch") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_embed = X_test_embed[start_idx:end_idx]
                batch_original = X_test_original[start_idx:end_idx]
                logger.debug(f"   Batch {batch_idx + 1}: embed={batch_embed.shape}, original={batch_original.shape}")
                try:
                    # Removed torch.cuda.empty_cache() from tight loops for performance
                    if return_probs:
                        batch_calibrated, batch_model_probs, batch_stability = self._calibrate_single_batch(batch_embed, batch_original, return_probs=True, return_stability=True)
                        all_calibrated_probs[start_idx:end_idx] = batch_calibrated
                        all_model_probs[start_idx:end_idx] = batch_model_probs
                        all_stability_scores[start_idx:end_idx] = batch_stability
                    else:
                        batch_probs, batch_stability = self._calibrate_single_batch(batch_embed, batch_original, return_stability=True)
                        all_calibrated_probs[start_idx:end_idx] = batch_probs
                        all_stability_scores[start_idx:end_idx] = batch_stability
                    pbar.update(1)
                except Exception as e:
                    logger.error(f" Calibration batch {batch_idx + 1} failed: {e}")
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    all_stability_scores[start_idx:end_idx] = 0.5  # Default stability
                    pbar.update(1)
        total_time = time.time() - start_time
        logger.info(f" Batched calibration completed in {total_time:.1f}s")
        logger.info(f"GeometricCalibrator: Batched calibration complete.")
        
        # Store stability scores for per-sample saving
        self.last_stability_scores = all_stability_scores
        
        if return_probs:
            return all_calibrated_probs, all_model_probs
        return all_calibrated_probs

    def calibrate_batched_precomputed(self, X_test_embed: np.ndarray, X_test_original: np.ndarray, model_probs: np.ndarray, batch_size=1000):
        """
        Batched calibration using PRE-COMPUTED probabilities to measure calibration throughput 
        without inference overhead (Apples-to-Apples with DAC).
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")
            
        if len(model_probs) != len(X_test_original):
            raise ValueError(f"Length of model_probs ({len(model_probs)}) must match data ({len(X_test_original)})")
            
        logger.info(f"GeometricCalibrator: Calibrating test data (PRE-COMPUTED) in batches of {batch_size}.")
        n_samples = X_test_original.shape[0]
        num_classes = model_probs.shape[1]
        
        all_calibrated_probs = np.zeros((n_samples, num_classes))
        all_stability_scores = np.zeros(n_samples)  # Accumulate stability scores
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with tqdm(total=n_batches, desc="Calibrating (Pre-computed)", unit="batch") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_embed = X_test_embed[start_idx:end_idx]
                # Passed for shape checks/logging but not used for inference
                batch_original = X_test_original[start_idx:end_idx] 
                batch_probs = model_probs[start_idx:end_idx]
                
                try:
                    batch_calibrated, batch_stability = self._calibrate_single_batch_precomputed(batch_embed, batch_probs, return_stability=True)
                    all_calibrated_probs[start_idx:end_idx] = batch_calibrated
                    all_stability_scores[start_idx:end_idx] = batch_stability
                    pbar.update(1)
                except Exception as e:
                    logger.error(f" Calibration batch {batch_idx + 1} failed: {e}")
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    all_stability_scores[start_idx:end_idx] = 0.5  # Default stability
                    pbar.update(1)
        
        # Store stability scores for per-sample saving
        self.last_stability_scores = all_stability_scores
                    
        return all_calibrated_probs

    def calibrate(self, X_test_embed: Optional[np.ndarray] = None, X_test_original: Optional[np.ndarray] = None, features=None):
        """
        Calibrates the test data using the fitted isotonic regression model.
        
        Args:
            X_test_embed: Test embeddings/features
            X_test_original: Raw test inputs (required for predict_proba)
            features: Unused, for API compatibility
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        # Require original images for predict_proba
        if X_test_original is None:
            raise ValueError(
                "X_test_original (raw inputs) is required for predict_proba."
            )

        n_samples = len(X_test_embed)
        if n_samples > 1000:
            return self.calibrate_batched(X_test_embed, X_test_original, features, batch_size=1000)
        else:
            logger.info(f"GeometricCalibrator: Calibrating {n_samples} samples (small dataset)")
            # X_test_original is guaranteed to be available (checked above)
            logger.debug(f"Embed data shape: {X_test_embed.shape}")
            logger.debug(f"Original data shape: {X_test_original.shape}")
            try:
                # Removed torch.cuda.empty_cache() from tight loops for performance
                calibrated_probs, stability_scores = self._calibrate_single_batch(X_test_embed, X_test_original, return_stability=True)
                # Store stability scores for per-sample saving
                self.last_stability_scores = stability_scores
                logger.info(f"GeometricCalibrator: Small dataset calibration complete")
                return calibrated_probs
            except Exception as e:
                logger.error(f"GeometricCalibrator: Small dataset calibration failed: {e}")
                num_classes = self.num_labels
                uniform_probs = np.ones((n_samples, num_classes)) / num_classes
                self.last_stability_scores = np.ones(n_samples) * 0.5  # Default stability
                return uniform_probs

    def _calibrate_single_batch_precomputed(self, batch_embed, batch_probs, return_stability=False):
        """
        Process a single batch using PRE-COMPUTED probabilities.
        Skips self.model.predict_proba() to isolate calibration cost.
        
        Args:
            return_stability: If True, returns (calibrated_probs, normalized_stability_batch) tuple
        """
        # Validate that calibrator is properly fitted
        if self.stab_space is None:
            raise ValueError("stab_space is None - calibrator may not be fitted properly")
        if self.regressor is None:
            raise ValueError("regressor is None - calibrator may not be fitted properly")
        
        # Use provided probabilities directly
        y_batch_pred = batch_probs
        y_batch_labels = np.argmax(y_batch_pred, axis=1)

        batch_size = batch_probs.shape[0]
        num_classes = batch_probs.shape[1]
        calibrated_probs = np.zeros((batch_size, num_classes))

        # Compute separation score for batch (using pre-computed logits)
        stability_batch = self._compute_score(batch_embed, y_batch_pred)

        # Validate stability_batch
        if stability_batch is None:
            raise ValueError("stab_space.calc_stab returned None")
        if not isinstance(stability_batch, np.ndarray):
            raise TypeError(f"stab_space.calc_stab returned {type(stability_batch)}, expected np.ndarray")
        
        # Normalize stability values using rank-based normalization (store before isotonic transform)
        # Use searchsorted to find ranks efficiently
        normalized_stability_batch = np.searchsorted(self.val_scores_sorted, stability_batch) / len(self.val_scores_sorted)
        
        # Apply calibration using the fitted regressor
        if self.fitting_method == 'beta':
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        elif self.fitting_method == 'logistic':
            eps = 1e-6
            stability_clipped = np.clip(normalized_stability_batch, eps, 1 - eps)
            X_pred = np.log(stability_clipped).reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        elif self.fitting_method == 'platt':
            X_pred = normalized_stability_batch.reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        elif self.fitting_method == 'isotonic':
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        else:
            raise ValueError(f"Unknown fitting_method: {self.fitting_method}")
        
        # Validate calibrated_values
        if calibrated_values is None:
            raise ValueError(f"regressor.predict/predict_proba returned None for fitting_method={self.fitting_method}")
        if not isinstance(calibrated_values, np.ndarray):
            raise TypeError(f"regressor.predict/predict_proba returned {type(calibrated_values)}, expected np.ndarray")
        
        eps = 1e-6
        calibrated_values = np.clip(calibrated_values, eps, 1 - eps)

        # Distribute calibrated values
        for i in range(batch_size):
            c = y_batch_labels[i]
            if self.num_labels > 1:
                rem = (1.0 - calibrated_values[i]) / (self.num_labels - 1)
                calibrated_probs[i, :] = rem
                calibrated_probs[i, c] = calibrated_values[i]
            else:
                calibrated_probs[i, c] = 1.0

        calibrated_probs = np.clip(calibrated_probs, 0, 1)
        
        if return_stability:
            return calibrated_probs, normalized_stability_batch
        return calibrated_probs

    def _calibrate_single_batch(self, batch_embed, batch_original, return_probs=False, return_stability=False):
        """Process a single batch using the original calibration logic
        
        Args:
            return_stability: If True, returns normalized stability scores (after normalization, before isotonic transform)
        """
        
        #  DEBUG: Add detailed logging for data validation
        logger.debug(f" _calibrate_single_batch received:")
        logger.debug(f"   batch_embed shape: {batch_embed.shape}")
        logger.debug(f"   batch_embed dtype: {batch_embed.dtype}")
        logger.debug(f"   batch_original shape: {batch_original.shape}")
        logger.debug(f"   batch_original dtype: {batch_original.dtype}")
        
        #  CRITICAL FIX: Detect if logits are accidentally passed instead of raw images
        if len(batch_original.shape) == 2 and self.num_labels is not None and batch_original.shape[1] == self.num_labels and batch_original.dtype in (np.float32, np.float64):
            logger.error(f" CRITICAL ERROR: Logits detected in batch_original instead of raw images!")
            logger.error(f"   batch_original shape: {batch_original.shape} (looks like logits for {batch_original.shape[1]} classes)")
            logger.error(f"   Expected raw input data, not probabilities/logits")
            logger.error(f"   This indicates a data flow bug in the calibration pipeline")
            raise ValueError(f"Probabilities/logits detected in X_test_original; pass raw inputs instead.")
        
        # Validate data types for physical vs semantic methods
        if len(batch_embed.shape) == 4 and batch_embed.shape[1] in [1, 3]:
            logger.debug(f"    Physical method detected: batch_embed appears to be raw images")
        elif len(batch_embed.shape) == 2:
            logger.debug(f"    Semantic method detected: batch_embed appears to be features")
        else:
            logger.warning(f"    Unexpected batch_embed shape: {batch_embed.shape}")
        
        # Use model adapter to predict on batch
        y_batch_pred = self.model.predict_proba(batch_original)
        y_batch_labels = np.argmax(y_batch_pred, axis=1)

        batch_size = batch_original.shape[0]
        num_classes = y_batch_pred.shape[1]
        calibrated_probs = np.zeros((batch_size, num_classes))

        # Compute separation score for batch
        stability_batch = self._compute_score(batch_embed, y_batch_pred)

        # Log separation score statistics before normalization
        score_name = "Stability"
        logger.info(f"Batch {score_name.lower()} stats before normalization:")
        logger.info(f"  Min: {np.min(stability_batch):.6f}")
        logger.info(f"  Max: {np.max(stability_batch):.6f}")
        logger.info(f"  Mean: {np.mean(stability_batch):.6f}")
        logger.info(f"  Std: {np.std(stability_batch):.6f}")
        
        # Normalize stability values using rank-based normalization (store before isotonic transform)
        # Use searchsorted to find ranks efficiently
        normalized_stability_batch = np.searchsorted(self.val_scores_sorted, stability_batch) / len(self.val_scores_sorted)
        
        # Apply calibration using the fitted regressor
        if self.fitting_method == 'beta':
            # Beta calibration expects raw stability values
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        
        elif self.fitting_method == 'logistic':
            # Logistic with log transform
            eps = 1e-6
            stability_clipped = np.clip(normalized_stability_batch, eps, 1 - eps)
            X_pred = np.log(stability_clipped).reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        
        elif self.fitting_method == 'platt':
            # Platt scaling on raw stability
            X_pred = normalized_stability_batch.reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        
        elif self.fitting_method == 'isotonic':
            # Isotonic regression
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        
        eps = 1e-6
        calibrated_values = np.clip(calibrated_values, eps, 1 - eps)

        # Distribute calibrated values across predicted classes without renormalization
        for i in range(batch_size):
            c = y_batch_labels[i]
            if self.num_labels > 1:
                rem = (1.0 - calibrated_values[i]) / (self.num_labels - 1)
                calibrated_probs[i, :] = rem
                calibrated_probs[i, c] = calibrated_values[i]
            else:
                calibrated_probs[i, c] = 1.0

        # Ensure probabilities are valid
        calibrated_probs = np.clip(calibrated_probs, 0, 1)

        # Return based on flags
        if return_probs and return_stability:
            return calibrated_probs, y_batch_pred, normalized_stability_batch
        elif return_probs:
            return calibrated_probs, y_batch_pred
        elif return_stability:
            return calibrated_probs, normalized_stability_batch
        return calibrated_probs

    def get_params(self) -> dict:
        params = {
            'metric': self.metric,
            'use_binning': self.use_binning,
            'n_bins': self.n_bins,
            'num_labels': self.num_labels,
            'is_fitted': self.is_fitted,
            'fitting_method': self.fitting_method,
            'fitting_function': self.fitting_method,
            # Rank-based normalization: stored sorted validation scores
            'normalization_method': 'rank_based',
            'val_scores_count': int(len(getattr(self, 'val_scores_sorted', []))),
        }
        
        # Add method-specific parameters
        if self.is_fitted:
            if self.fitting_method == 'isotonic' and hasattr(self.regressor, 'X_thresholds_'):
                y_thresholds = self.regressor.y_thresholds_
                
                params['isotonic_params'] = {
                    'n_thresholds': int(len(self.regressor.X_thresholds_)),
                    'X_min': float(self.regressor.X_min_),
                    'X_max': float(self.regressor.X_max_),
                    'y_min': float(y_thresholds[0]),
                    'y_max': float(y_thresholds[-1])
                }
            elif self.fitting_method == 'beta' and hasattr(self.regressor, 'map_'):
                params['beta_params'] = {
                    'a': float(self.regressor.map_[0]),
                    'b': float(self.regressor.map_[1]),
                    'm': float(self.regressor.map_[2])
                }
            elif self.fitting_method in ['platt', 'logistic'] and hasattr(self.regressor, 'coef_'):
                params['logistic_params'] = {
                    'coef': self.regressor.coef_.tolist(),
                    'intercept': float(self.regressor.intercept_[0]) if hasattr(self.regressor.intercept_, '__len__') else float(self.regressor.intercept_)
                }

        return params
