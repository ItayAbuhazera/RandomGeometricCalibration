import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
import sys
import os

# Add project root to path to import ECELoss
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Metrics.metrics import ECELoss
from .base_calibrator import BaseCalibrator

from utils.logging_config import get_logger
logger = get_logger(__name__)

class TemperatureScaling(BaseCalibrator):
    """
    Temperature Scaling calibrator.
    Code adapted from https://github.com/gpleiss/temperature_scaling
    
    Tunes the temperature of the model using validation set with cross-validation on ECE or NLL.
    """
    
    def __init__(self, cross_validate='ece', **kwargs):
        """
        Initialize temperature scaling.
        
        Args:
            cross_validate: 'ece' or 'nll' - which metric to optimize for
        """
        super().__init__(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize temperature to 1.0 (neutral scaling)
        self.temperature = 1.0
        self.cross_validate = cross_validate
        self.before_temperature_nll = None
        self.before_temperature_ece = None
        self.after_temperature_nll = None
        self.after_temperature_ece = None
    
    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """
        Tune the temperature of the model using validation set with cross-validation on ECE or NLL.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            labels: True labels of shape (n_samples,)
            features: Optional features (not used for temperature scaling)
        """
        logger.info(" Fitting temperature scaling...")
        
        # Convert numpy arrays to tensors and move to device
        logits_tensor = torch.FloatTensor(logits).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)
        
        # Initialize criteria
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)
        
        # Calculate NLL and ECE before temperature scaling
        self.before_temperature_nll = nll_criterion(logits_tensor, labels_tensor).item()
        self.before_temperature_ece = ece_criterion(logits_tensor, labels_tensor).item()
        logger.info('Before temperature - NLL: %.3f, ECE: %.3f' % (self.before_temperature_nll, self.before_temperature_ece))
        
        # Grid search for optimal temperature
        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        
        for i in range(100):
            temp_tensor = torch.tensor(T, device=self.device)
            scaled_logits = logits_tensor / temp_tensor
            
            after_temperature_nll = nll_criterion(scaled_logits, labels_tensor).item()
            after_temperature_ece = ece_criterion(scaled_logits, labels_tensor).item()
            
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll
            
            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            
            T += 0.1
        
        # Select optimal temperature based on cross_validate parameter
        if self.cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        
        # Calculate NLL and ECE after temperature scaling
        temp_tensor = torch.tensor(self.temperature, device=self.device)
        scaled_logits = logits_tensor / temp_tensor
        self.after_temperature_nll = nll_criterion(scaled_logits, labels_tensor).item()
        self.after_temperature_ece = ece_criterion(scaled_logits, labels_tensor).item()
        
        logger.info('Optimal temperature: %.3f' % self.temperature)
        logger.info('After temperature - NLL: %.3f, ECE: %.3f' % (self.after_temperature_nll, self.after_temperature_ece))
        
        self.is_fitted = True
    
    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calibrate the logits using the learned temperature.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            features: Optional features (not used for temperature scaling)
            
        Returns:
            Calibrated probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        
        with torch.no_grad():
            logits_tensor = torch.FloatTensor(logits).to(self.device)
            temperature_tensor = torch.tensor(self.temperature, device=self.device)
            
            # Apply temperature scaling
            scaled_logits = logits_tensor / temperature_tensor
            calibrated_probs = F.softmax(scaled_logits, dim=1)
            return calibrated_probs.cpu().numpy()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the temperature parameter and fitting info.
        
        Returns:
            Dictionary containing the temperature value and fitting info
        """
        return {
            "temperature": float(self.temperature),
            "cross_validate": self.cross_validate,
            "before_temperature_nll": self.before_temperature_nll,
            "before_temperature_ece": self.before_temperature_ece,
            "after_temperature_nll": self.after_temperature_nll,
            "after_temperature_ece": self.after_temperature_ece,
        }
    
    def save(self, path: str) -> None:
        """Save the temperature parameter and metadata."""
        torch.save({
            'temperature': self.temperature,
            'cross_validate': self.cross_validate,
            'before_temperature_nll': self.before_temperature_nll,
            'before_temperature_ece': self.before_temperature_ece,
            'after_temperature_nll': self.after_temperature_nll,
            'after_temperature_ece': self.after_temperature_ece,
        }, path)
    
    def load(self, path: str) -> None:
        """Load the temperature parameter and metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        self.temperature = float(checkpoint['temperature'])
        self.cross_validate = checkpoint.get('cross_validate', 'ece')
        self.before_temperature_nll = checkpoint.get('before_temperature_nll', None)
        self.before_temperature_ece = checkpoint.get('before_temperature_ece', None)
        self.after_temperature_nll = checkpoint.get('after_temperature_nll', None)
        self.after_temperature_ece = checkpoint.get('after_temperature_ece', None)
        self.is_fitted = True