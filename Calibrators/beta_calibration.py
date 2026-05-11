import numpy as np
import logging
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from typing import Dict, Any, Optional
from .base_calibrator import BaseCalibrator

from utils.logging_config import get_logger
logger = get_logger(__name__)

def _beta_calibration(df, y, sample_weight=None):
    """
    Internal function for fitting beta calibration for binary classification.
    
    Args:
        df: Probability estimates for one class
        y: Binary labels (0 or 1)
        sample_weight: Sample weights (optional)
        
    Returns:
        map: List containing [a, b, m] parameters
        lr: Fitted logistic regression model
    """
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(x)
    x[:, 1] *= -1

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]

    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]
    inter = lr.intercept_[0]

    m = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
                        bounds=[0, 1], method='Bounded').x
    map = [a, b, m]
    return map, lr


class _BetaCal(BaseEstimator, RegressorMixin):
    """
    Beta regression model with three parameters introduced in
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary
    classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m]

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data."""
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_calibration(X, y, sample_weight)
        return self

    def predict(self, S):
        """Predict new values."""
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]


class BetaCalibration(BaseCalibrator):
    """
    Beta Calibration for multiclass problems using one-vs-rest approach.
    
    Implements the method from "Beta calibration: a well-founded and easily implemented 
    improvement on logistic calibration for binary classifiers" by Kull et al. (2017).
    
    For multiclass problems, uses a one-vs-rest approach where each class gets its own
    binary beta calibrator trained on (class probability, is_true_class) pairs.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibrators = []
        self.n_classes = None
        
    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """
        Fit the beta calibration model for each class.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            labels: True labels of shape (n_samples,)
            features: Not used for beta calibration (optional parameter for interface compatibility)
        """
        logger.info(" Fitting beta calibration...")
        
        self.n_classes = logits.shape[1]
        self.calibrators = []
        
        # Convert logits to probabilities
        probs = self._softmax(logits)
        
        # Train a binary beta calibrator for each class
        for i in range(self.n_classes):
            logger.debug(f"   Training beta calibrator for class {i}")
            
            # Prepare binary classification data
            # Feature: probability of class i
            # Target: whether true label is class i
            X = probs[:, i]
            y = (labels == i).astype(float)
            
            # Create and fit binary beta calibrator
            calibrator = _BetaCal()
            
            try:
                calibrator.fit(X, y)
                self.calibrators.append(calibrator)
                
                # Log the fitted parameters
                if hasattr(calibrator, 'map_'):
                    a, b, m = calibrator.map_
                    logger.debug(f"   Class {i}: a={a:.4f}, b={b:.4f}, m={m:.4f}")
                
            except Exception as e:
                logger.warning(f"   Failed to fit beta calibrator for class {i}: {e}")
                # Fallback to None (will use identity mapping)
                self.calibrators.append(None)
        
        self.is_fitted = True
        logger.info(" Beta calibration fitting completed")
    
    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calibrate the logits using the fitted beta calibration models.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            features: Not used for beta calibration (optional parameter for interface compatibility)
            
        Returns:
            Calibrated probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        
        probs = self._softmax(logits)
        calibrated_probs = np.zeros_like(probs)
        
        for i in range(self.n_classes):
            if self.calibrators[i] is not None:
                try:
                    # Apply beta calibration for this class
                    calibrated_probs[:, i] = self.calibrators[i].predict(probs[:, i])
                    
                except Exception as e:
                    logger.warning(f"Beta calibration failed for class {i}: {e}, using original probabilities")
                    calibrated_probs[:, i] = probs[:, i]
            else:
                # Use original probabilities if calibrator failed to fit
                calibrated_probs[:, i] = probs[:, i]
        
        # Normalize to ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
        calibrated_probs = calibrated_probs / row_sums
        
        return calibrated_probs
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the beta calibration models.
        
        Returns:
            Dictionary containing the parameters of each calibrator
        """
        params = {
            "n_classes": self.n_classes,
            "calibrator_params": []
        }
        
        for i, calibrator in enumerate(self.calibrators):
            if calibrator is not None and hasattr(calibrator, 'map_'):
                a, b, m = calibrator.map_
                params["calibrator_params"].append({
                    "class": i,
                    "a": float(a),
                    "b": float(b),
                    "m": float(m),
                    "fitted": True
                })
            else:
                params["calibrator_params"].append({
                    "class": i,
                    "fitted": False
                })
        
        return params
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each set of scores in x.
        
        Args:
            x: Input array of shape (n_samples, n_classes)
            
        Returns:
            Softmax probabilities of shape (n_samples, n_classes)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def save(self, path: str) -> None:
        """
        Save the calibration models.
        
        Args:
            path: Path to save the model
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'n_classes': self.n_classes,
                'calibrators': self.calibrators
            }, f)
    
    def load(self, path: str) -> None:
        """
        Load the calibration models.
        
        Args:
            path: Path to load the model from
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.n_classes = data['n_classes']
            self.calibrators = data['calibrators']
            self.is_fitted = True 