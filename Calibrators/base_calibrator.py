from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseCalibrator(ABC):
    """Abstract base class for all calibration methods."""
    
    def __init__(self, **kwargs):
        """Initialize the calibrator with any necessary parameters."""
        self.is_fitted = False
        self.params = kwargs
    
    @abstractmethod
    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """
        Fit the calibration model using the provided logits and labels.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            labels: True labels of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calibrate the provided logits using the fitted calibration model.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities of shape (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the calibration model.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the calibration model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Save method not implemented")
    
    def load(self, path: str) -> None:
        """
        Load the calibration model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError("Load method not implemented") 