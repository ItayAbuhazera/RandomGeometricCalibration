import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Any, Optional
from .base_calibrator import BaseCalibrator

class IsotonicRegressionCalibrator(BaseCalibrator):
    """
    Isotonic Regression calibrator.
    Implements the method from "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
    by Zadrozny and Elkan.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibrators = []
        self.n_classes = None
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the isotonic regression model for each class.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            labels: True labels of shape (n_samples,)
        """
        self.n_classes = logits.shape[1]
        self.calibrators = []
        
        # Convert logits to probabilities
        probs = self._softmax(logits)
        
        # Train a calibrator for each class
        for i in range(self.n_classes):
            calibrator = IsotonicRegression(out_of_bounds='clip')
            # Use the probability of the current class as the feature
            # and whether the true label is this class as the target
            calibrator.fit(probs[:, i], (labels == i).astype(float))
            self.calibrators.append(calibrator)
        
        self.is_fitted = True
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Calibrate the logits using the fitted isotonic regression models.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        
        probs = self._softmax(logits)
        calibrated_probs = np.zeros_like(probs)
        
        for i in range(self.n_classes):
            calibrated_probs[:, i] = self.calibrators[i].predict(probs[:, i])
        
        # Normalize to ensure probabilities sum to 1
        denom = calibrated_probs.sum(axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        calibrated_probs = calibrated_probs / denom
        return calibrated_probs

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return self.calibrate(logits)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the calibration models.
        
        Returns:
            Dictionary containing the parameters of each calibrator
        """
        return {
            "n_classes": self.n_classes,
            "calibrator_params": [cal.get_params() for cal in self.calibrators]
        }
    
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


class TopLabelIsotonicCalibrator(BaseCalibrator):
    """
    Isotonic on (max prob -> correctness), then adjust only the top-class prob and
    renormalize the rest. Keeps ranking among non-top classes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.iso = IsotonicRegression(out_of_bounds="clip")
        except Exception as e:
            raise RuntimeError("scikit-learn is required for Top-Label Isotonic") from e
        self.n_classes = None

    # --- NEW: add a local softmax helper (numerically stable) ---
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # x: (n_samples, n_classes) logits
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self.n_classes = logits.shape[1]
        probs = self._softmax(logits)
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == labels).astype(float)
        self.iso.fit(conf, correct)
        self.is_fitted = True

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        probs = self._softmax(logits).copy()
        idx = probs.argmax(axis=1)

        p_top_old = probs[np.arange(probs.shape[0]), idx]
        p_top_new = self.iso.predict(p_top_old)
        p_top_new = np.clip(p_top_new, 1e-6, 1-1e-6)

        scale = (1.0 - p_top_new) / np.clip(1.0 - p_top_old, 1e-6, 1.0)
        probs *= scale[:, None]
        probs[np.arange(probs.shape[0]), idx] = p_top_new

        probs = np.clip(probs, 1e-8, 1-1e-8)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return self.calibrate(logits)

    def get_params(self) -> Dict[str, Any]:
        try:
            iso_params = self.iso.get_params()
        except Exception:
            iso_params = {}
        return {
            "n_classes": self.n_classes,
            "is_fitted": bool(getattr(self, "is_fitted", False)),
            "iso_params": iso_params,
        }

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "n_classes": self.n_classes,
                "iso": self.iso,
                "is_fitted": self.is_fitted
            }, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_classes = data.get("n_classes")
        self.iso = data.get("iso", IsotonicRegression(out_of_bounds="clip"))
        self.is_fitted = data.get("is_fitted", False)
