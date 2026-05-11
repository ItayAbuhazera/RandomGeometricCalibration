import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
from .base_calibrator import BaseCalibrator

from utils.logging_config import get_logger
logger = get_logger(__name__)

class PlattScaling:
    """
    Platt Scaling using logistic regression.
    For multiclass: one-vs-rest approach.
    """
    
    def __init__(self, **kwargs):
        self.calibrators = []
        self.n_classes = None
        self.is_fitted = False
        
    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """Fit Platt scaling for each class."""
        logger.info(" Fitting Platt scaling...")
        
        self.n_classes = logits.shape[1] if logits.ndim > 1 else 2
        
        if self.n_classes == 2:
            # Binary case: fit on the positive class logit
            y_binary = labels.astype(float)
            # Use raw logits directly - for binary, use logits[:, 1] if 2D, else use logits
            if logits.ndim > 1:
                X = logits[:, 1].reshape(-1, 1)
            else:
                X = logits.reshape(-1, 1)
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X, y_binary)
            self.calibrators = [lr]
        else:
            # Multiclass case - one calibrator per class
            self.calibrators = []
            
            for i in range(self.n_classes):
                try:
                    # Create binary labels for class i
                    y_binary = (labels == i).astype(float)
                    
                    # Use raw logits directly for class i
                    X = logits[:, i].reshape(-1, 1)
                    
                    # Check if we have both classes
                    if len(np.unique(y_binary)) < 2:
                        logger.warning(f"Class {i} has only one label, skipping calibration")
                        self.calibrators.append(None)
                        continue
                    
                    # Fit logistic regression on raw logits
                    lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
                    lr.fit(X, y_binary)
                    self.calibrators.append(lr)
                    
                except Exception as e:
                    logger.warning(f"Failed to fit calibrator for class {i}: {e}")
                    self.calibrators.append(None)
        
        self.is_fitted = True
        logger.info(" Platt scaling fitting completed")
    
    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Platt scaling calibration."""
        if not self.is_fitted:
            raise RuntimeError("Must fit before calibrating")
        
        if self.n_classes == 2:
            # Binary case: use raw logits directly
            if self.calibrators[0] is not None:
                # Use raw logits directly - for binary, use logits[:, 1] if 2D, else use logits
                if logits.ndim > 1:
                    X = logits[:, 1].reshape(-1, 1)
                else:
                    X = logits.reshape(-1, 1)
                
                calibrated_prob = self.calibrators[0].predict_proba(X)[:, 1]
                return np.column_stack([1-calibrated_prob, calibrated_prob])
            else:
                # Fallback to softmax/sigmoid if calibrator not available
                probs = self._softmax(logits) if logits.ndim > 1 else self._sigmoid(logits)
                return probs
        else:
            # Multiclass case: use raw logits directly
            calibrated_probs = np.zeros((logits.shape[0], self.n_classes))
            
            # Compute fallback probabilities once, outside the loop
            fallback_probs = self._softmax(logits)
            
            for i in range(self.n_classes):
                if self.calibrators[i] is not None:
                    try:
                        # Use raw logits directly for class i
                        X = logits[:, i].reshape(-1, 1)
                        calibrated_probs[:, i] = self.calibrators[i].predict_proba(X)[:, 1]
                    except Exception as e:
                        logger.warning(f"Calibration failed for class {i}: {e}")
                        calibrated_probs[:, i] = fallback_probs[:, i]
                else:
                    calibrated_probs[:, i] = fallback_probs[:, i]
            
            # Normalize to ensure valid probabilities
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            calibrated_probs = calibrated_probs / row_sums
            
            return calibrated_probs
    
    def _softmax(self, x):
        """Numerically stable softmax."""
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))