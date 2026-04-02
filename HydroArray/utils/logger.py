"""
Logging System for HydroArray

Provides structured logging for training, evaluation, and experiment tracking.
Supports both console output and file logging, with optional TensorBoard integration.

Example:
    >>> from HydroArray.utils.logger import ExperimentLogger
    >>> 
    >>> logger = ExperimentLogger('runs/my_experiment')
    >>> logger.log_hyperparams({'lr': 0.001, 'batch_size': 32})
    >>> 
    >>> for epoch in range(10):
    ...     logger.log_metrics({'train_loss': 0.5, 'val_loss': 0.4}, step=epoch)
    
    >>> logger.close()
"""

import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import warnings


class ExperimentLogger:
    """
    Experiment logger for tracking training metrics and hyperparameters.
    
    This logger provides:
    - Console output with formatted messages
    - File logging to CSV and JSON
    - Optional TensorBoard integration
    - Metric history tracking
    
    Parameters
    ----------
    log_dir : Union[str, Path]
        Directory to save log files
    experiment_name : Optional[str]
        Name of the experiment. If None, uses timestamp.
    use_tensorboard : bool
        Whether to use TensorBoard logging
    verbose : bool
        Whether to print to console
    
    Attributes
    ----------
    log_dir : Path
        Path to log directory
    metrics_file : Path
        Path to CSV metrics file
    hyperparams_file : Path
        Path to JSON hyperparameters file
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        verbose: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.verbose = verbose
        
        # Create experiment subdirectory
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.metrics_file = self.exp_dir / 'metrics.csv'
        self.hyperparams_file = self.exp_dir / 'hyperparameters.json'
        self.log_file = self.exp_dir / 'training.log'
        
        # Initialize CSV file
        self._init_csv()
        
        # Metric history
        self.metrics_history: Dict[str, list] = {}
        self.step_count = 0
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            self._init_tensorboard()
        
        # Log initialization
        self.info(f"Experiment: {self.experiment_name}")
        self.info(f"Log directory: {self.exp_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'timestamp', 'metric_name', 'value'])
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.exp_dir / 'tensorboard')
            self.info("TensorBoard logging enabled")
        except ImportError:
            warnings.warn("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of hyperparameters
        """
        # Save to JSON
        with open(self.hyperparams_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Log to console
        self.info("Hyperparameters:")
        for key, value in params.items():
            self.info(f"  {key}: {value}")
        
        # Log to TensorBoard
        if self.writer is not None:
            # Convert nested dict to flat format for TensorBoard
            flat_params = self._flatten_dict(params)
            self.writer.add_hparams(flat_params, {})
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics for current step.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metric names and values
        step : Optional[int]
            Step number. If None, uses internal counter.
        prefix : str
            Prefix for metric names (e.g., 'train/', 'val/')
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        timestamp = datetime.now().isoformat()
        
        # Write to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for name, value in metrics.items():
                full_name = f"{prefix}{name}" if prefix else name
                writer.writerow([step, timestamp, full_name, value])
        
        # Update history
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append((step, value))
        
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                full_name = f"{prefix}{name}" if prefix else name
                self.writer.add_scalar(full_name, value, step)
    
    def log_epoch(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None
    ):
        """
        Log metrics for a complete epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        num_epochs : int
            Total number of epochs
        train_metrics : Dict[str, float]
            Training metrics
        val_metrics : Optional[Dict[str, float]]
            Validation metrics
        lr : Optional[float]
            Current learning rate
        """
        # Log metrics
        self.log_metrics(train_metrics, step=epoch, prefix='train/')
        if val_metrics:
            self.log_metrics(val_metrics, step=epoch, prefix='val/')
        
        # Format message
        msg = f"Epoch [{epoch}/{num_epochs}]"
        
        # Training metrics
        train_msg = " | ".join([f"{k}: {v:.6f}" for k, v in train_metrics.items()])
        msg += f" Train: {train_msg}"
        
        # Validation metrics
        if val_metrics:
            val_msg = " | ".join([f"{k}: {v:.6f}" for k, v in val_metrics.items()])
            msg += f" | Val: {val_msg}"
        
        # Learning rate
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        
        self.info(msg)
    
    def log_model_info(self, model: Any):
        """
        Log model architecture information.
        
        Parameters
        ----------
        model : Any
            PyTorch model
        """
        import torch.nn as nn
        
        if isinstance(model, nn.Module):
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info = {
                'model_class': model.__class__.__name__,
                'total_parameters': num_params,
                'trainable_parameters': trainable_params,
                'non_trainable_parameters': num_params - trainable_params
            }
            
            self.info("Model Information:")
            for key, value in info.items():
                self.info(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
            
            # Save to file
            with open(self.exp_dir / 'model_info.json', 'w') as f:
                json.dump(info, f, indent=2)
    
    def info(self, message: str):
        """
        Log info message.
        
        Parameters
        ----------
        message : str
            Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        
        # Console output
        if self.verbose:
            print(log_line)
        
        # File output
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def warning(self, message: str):
        """Log warning message."""
        self.info(f"WARNING: {message}")
    
    def error(self, message: str):
        """Log error message."""
        self.info(f"ERROR: {message}")
    
    def get_metrics_history(self, metric_name: str) -> list:
        """
        Get history of a specific metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
        
        Returns
        -------
        list
            List of (step, value) tuples
        """
        return self.metrics_history.get(metric_name, [])
    
    def save_summary(self, summary: Dict[str, Any]):
        """
        Save training summary.
        
        Parameters
        ----------
        summary : Dict[str, Any]
            Summary dictionary
        """
        summary_file = self.exp_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.info("Training Summary:")
        for key, value in summary.items():
            self.info(f"  {key}: {value}")
    
    def close(self):
        """Close logger and cleanup."""
        if self.writer is not None:
            self.writer.close()
        
        self.info("Logger closed")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ConsoleLogger:
    """
    Simple console-only logger for quick experiments.
    
    Example:
        >>> logger = ConsoleLogger()
        >>> logger.log_epoch(1, 10, {'loss': 0.5}, {'loss': 0.4})
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def log_epoch(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None
    ):
        """Log epoch metrics to console."""
        if not self.verbose:
            return
        
        msg = f"Epoch [{epoch}/{num_epochs}]"
        
        train_msg = " | ".join([f"{k}: {v:.6f}" for k, v in train_metrics.items()])
        msg += f" Train: {train_msg}"
        
        if val_metrics:
            val_msg = " | ".join([f"{k}: {v:.6f}" for k, v in val_metrics.items()])
            msg += f" | Val: {val_msg}"
        
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        
        print(msg)
    
    def info(self, message: str):
        """Print info message."""
        if self.verbose:
            print(message)
    
    def close(self):
        """No-op for console logger."""
        pass
