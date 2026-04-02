"""
Simple LSTM Model for Sequence Prediction

A basic LSTM implementation for time series forecasting tasks.
This model can be used for rainfall-runoff prediction, water level
forecasting, and other hydrological sequence prediction tasks.

Example:
    >>> from HydroArray.ml.models import create_model
    >>> config = {'model': {'type': 'lstm', 'input_dim': 10, 'hidden_dim': 64}}
    >>> model = create_model(config)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for sequence-to-sequence prediction.
    
    Architecture:
        - LSTM encoder for processing input sequences
        - Fully connected output layer for prediction
    
    Args:
        config: Configuration object containing model parameters
    
    Example:
        >>> config = Config({
        ...     'model': {
        ...         'input_dim': 5,
        ...         'hidden_dim': 64,
        ...         'num_layers': 2,
        ...         'output_dim': 1,
        ...         'dropout': 0.1
        ...     }
        ... })
        >>> model = SimpleLSTM(config)
        >>> x = torch.randn(32, 10, 5)  # (batch, seq_len, features)
        >>> y = model(x, future_steps=5)
        >>> print(y.shape)  # (32, 5, 1)
    """
    
    def __init__(self, config: Any):
        """Initialize the LSTM model."""
        super(SimpleLSTM, self).__init__()
        
        # Extract parameters from config
        self.input_dim = getattr(config.model, 'input_dim', 1)
        self.hidden_dim = getattr(config.model, 'hidden_dim', 64)
        self.num_layers = getattr(config.model, 'num_layers', 2)
        self.output_dim = getattr(config.model, 'output_dim', self.input_dim)
        self.dropout = getattr(config.model, 'dropout', 0.0)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )
        
        # Output projection layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor, future_steps: int = 1) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            future_steps: Number of future steps to predict
        
        Returns:
            Predictions of shape (batch_size, future_steps, output_dim)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode input sequence
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for prediction
        predictions = []
        current_input = x[:, -1:, :]  # Last time step
        
        for _ in range(future_steps):
            # LSTM step
            lstm_out, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            
            # Predict next step
            pred = self.fc(self.dropout_layer(lstm_out))  # (batch, 1, output_dim)
            predictions.append(pred)
            
            # Use prediction as next input (auto-regressive)
            current_input = pred
        
        # Concatenate predictions
        output = torch.cat(predictions, dim=1)

        # Squeeze output dim if single step
        if future_steps == 1:
            output = output.squeeze(-1)  # (batch, future_steps)

        return output
    
    def predict(self, x: torch.Tensor, future_steps: int = 1) -> torch.Tensor:
        """
        Make predictions (inference mode).
        
        Args:
            x: Input tensor
            future_steps: Number of steps to predict
        
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, future_steps)


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence prediction.
    
    Processes sequences in both forward and backward directions,
    useful for capturing patterns from past and future context.
    
    Args:
        config: Configuration object
    """
    
    def __init__(self, config: Any):
        super(BidirectionalLSTM, self).__init__()
        
        self.input_dim = getattr(config.model, 'input_dim', 1)
        self.hidden_dim = getattr(config.model, 'hidden_dim', 64)
        self.num_layers = getattr(config.model, 'num_layers', 2)
        self.output_dim = getattr(config.model, 'output_dim', self.input_dim)
        self.dropout = getattr(config.model, 'dropout', 0.0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )
        
        # Output layer (2 * hidden_dim for bidirectional)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor, future_steps: int = 1) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        device = x.device
        
        # Encode
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Predict future steps
        predictions = []
        current_input = x[:, -1:, :]
        
        for _ in range(future_steps):
            lstm_out, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            pred = self.fc(self.dropout_layer(lstm_out))
            predictions.append(pred)
            current_input = pred
        
        return torch.cat(predictions, dim=1)


# =============================================================================
# Training Functions
# =============================================================================

def train_lstm(
    dataset: Any = None,
    config: Optional[str] = None,
    model: str = "lstm",
    hidden_dim: int = 64,
    num_layers: int = 2,
    output_dim: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    patience: int = 10,
    val_ratio: float = 0.1,
    save_dir: Optional[str] = None,
    device: str = "auto",
    verbose: bool = True,
    log_interval: int = 1,
    **kwargs
) -> tuple:
    """
    Train LSTM model with minimal configuration.

    This is the simplest way to train an LSTM model in HydroArray.

    Args:
        dataset: PyTorch Dataset. If None, must provide via config.
        config: Path to YAML config file or dict with parameters.
        model: Model type ('lstm' or 'bilstm')
        hidden_dim: Hidden dimension for LSTM
        num_layers: Number of LSTM layers
        output_dim: Output dimension
        dropout: Dropout rate
        epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        grad_clip: Gradient clipping threshold
        patience: Early stopping patience
        val_ratio: Validation split ratio
        save_dir: Directory to save results
        device: Device to use ('auto', 'cpu', 'cuda')
        verbose: Whether to print progress
        log_interval: Print every N epochs
        **kwargs: Additional parameters

    Returns:
        Tuple of (model, results_dict)

    Example:
        >>> from HydroArray.datasets import CAMELSUSDataset
        >>> from HydroArray.ml.models.sequence.lstm import train_lstm
        >>>
        >>> dataset = CAMELSUSDataset("path/to/camels", basins=["01013500"])
        >>> model, results = train_lstm(dataset, epochs=50)
    """
    import shutil
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    import numpy as np
    import torch

    from HydroArray.utils.config import Config
    from HydroArray.utils.logger import ExperimentLogger
    from HydroArray.analysis.statistics import nse, rmse, mae
    from HydroArray.plotting import (
        plot_loss_curve,
        plot_predictions_vs_observations,
        plot_time_series_comparison,
        plot_metrics_summary,
    )

    # Load config from file
    experiment_name = None
    output_dir = None
    config_path = None
    visualize = True
    plot_style = 'hess'
    plots_to_generate = ['loss', 'scatter', 'timeseries', 'metrics']

    if config is not None:
        cfg = Config(config) if isinstance(config, str) else Config(config) if isinstance(config, dict) else config
        config_path = Path(config) if isinstance(config, str) else None

        model = cfg.get('model.type', model)
        hidden_dim = cfg.get('model.hidden_dim', hidden_dim)
        num_layers = cfg.get('model.num_layers', num_layers)
        output_dim = cfg.get('model.output_dim', output_dim)
        dropout = cfg.get('model.dropout', dropout)

        epochs = cfg.get('training.num_epochs', epochs)
        batch_size = cfg.get('data.batch_size', batch_size)
        learning_rate = cfg.get('training.learning_rate', learning_rate)
        weight_decay = cfg.get('training.weight_decay', weight_decay)
        grad_clip = cfg.get('training.grad_clip', grad_clip)
        patience = cfg.get('training.early_stopping.patience', patience)

        val_ratio = cfg.get('data.val_ratio', val_ratio)
        device = cfg.get('experiment.device', device)
        log_interval = cfg.get('experiment.log_interval', log_interval)
        experiment_name = cfg.get('experiment.experiment_name', experiment_name)
        output_dir = cfg.get('experiment.output_dir', output_dir)

        visualize = cfg.get('visualization.enabled', visualize)
        plot_style = cfg.get('visualization.style', plot_style)
        plots_to_generate = cfg.get('visualization.plots', plots_to_generate)

        # Create dataset from config if not provided
        if dataset is None:
            data_cfg = cfg.data if hasattr(cfg, 'data') else {}
            if isinstance(data_cfg, dict):
                dataset_name = data_cfg.get('dataset')
                data_dir = data_cfg.get('data_dir')
            else:
                dataset_name = getattr(data_cfg, 'dataset', None)
                data_dir = getattr(data_cfg, 'data_dir', None)

            if dataset_name and data_dir:
                data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

                if dataset_name.lower() == 'camels_us':
                    from HydroArray.datasets import CAMELSUSDataset
                    basins = data_cfg.get('basins') if isinstance(data_cfg, dict) else getattr(data_cfg, 'basins', None)
                    seq_len = data_cfg.get('seq_len', 365) if isinstance(data_cfg, dict) else getattr(data_cfg, 'seq_len', 365)
                    pred_len = data_cfg.get('pred_len', 1) if isinstance(data_cfg, dict) else getattr(data_cfg, 'pred_len', 1)

                    dataset = CAMELSUSDataset(
                        data_dir=data_dir,
                        basins=basins,
                        seq_len=seq_len,
                        pred_len=pred_len,
                    )

    if dataset is None:
        raise ValueError("dataset must be provided either directly or via config")

    # Device setup
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Setup experiment folder
    if save_dir is not None:
        save_dir = Path(save_dir)
    elif experiment_name and output_dir:
        save_dir = Path(output_dir) / experiment_name
    else:
        save_dir = None

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = save_dir / 'logs'
        checkpoints_dir = save_dir / 'checkpoints'
        figs_dir = save_dir / 'Figs'
        logs_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        figs_dir.mkdir(exist_ok=True)

        if config_path and config_path.exists():
            shutil.copy(config_path, save_dir / 'config.yml')

        logger = ExperimentLogger(log_dir=str(logs_dir), experiment_name='train', verbose=False)
    else:
        logs_dir = checkpoints_dir = figs_dir = logger = None

    # Get input dimension
    if hasattr(dataset, 'input_vars'):
        input_dim = len(dataset.input_vars)
    elif hasattr(dataset, 'forcing_vars'):
        input_dim = len(dataset.forcing_vars)
    else:
        try:
            sample_x, _ = dataset[0]
            input_dim = sample_x.shape[-1]
        except:
            input_dim = 1

    # Create model
    class _ModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    if model.lower() in ('lstm', 'simple_lstm', 'bilstm'):
        model_config = _ModelConfig(
            model=_ModelConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout
            )
        )
        lstm_model = SimpleLSTM(model_config)
    else:
        raise ValueError(f"Unknown model: {model}")

    lstm_model = lstm_model.to(device)

    # Split dataset
    n = len(dataset)
    n_val = int(n * val_ratio)
    indices = list(range(n))
    np.random.shuffle(indices)
    val_idx = set(indices[:n_val])
    train_idx = indices[n_val:]

    train_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(list(val_idx)),
        num_workers=0
    )

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    if verbose:
        print(f"Training on {n} samples ({n - n_val} train, {n_val} val)...")

    for epoch in range(epochs):
        lstm_model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = lstm_model(x)
            loss = criterion(out, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        lstm_model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = lstm_model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            print(msg)

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    lstm_model.load_state_dict(best_state)

    # Final evaluation
    lstm_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = lstm_model(x)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    metrics = {
        'nse': nse(predictions, targets),
        'rmse': rmse(predictions, targets),
        'mae': mae(predictions, targets),
    }

    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'metrics': metrics,
        'epochs': epoch + 1,
    }

    # Save results
    if save_dir is not None:
        torch.save(best_state, save_dir / 'model.pth')
        with open(save_dir / 'results.json', 'w') as f:
            json.dump({
                'best_val_loss': float(best_val_loss),
                'metrics': {k: float(v) for k, v in metrics.items()},
                'epochs': epoch + 1,
            }, f, indent=2)

        if logger:
            logger.save_summary({
                'best_val_loss': float(best_val_loss),
                'test_rmse': float(metrics['rmse']),
                'test_nse': float(metrics['nse']),
                'epochs_trained': epoch + 1,
            })
            logger.close()

        # Generate plots
        if visualize and figs_dir is not None:
            figs_dir.mkdir(exist_ok=True)
            if 'loss' in plots_to_generate:
                plot_loss_curve(train_losses, val_losses, save_path=figs_dir / 'loss_curve.png', style=plot_style)
            if 'scatter' in plots_to_generate:
                plot_predictions_vs_observations(predictions, targets, save_path=figs_dir / 'predictions_vs_observations.png', style=plot_style)
            if 'timeseries' in plots_to_generate:
                plot_time_series_comparison(predictions, targets, n_samples=1000, save_path=figs_dir / 'time_series_comparison.png', style=plot_style)
            if 'metrics' in plots_to_generate:
                plot_metrics_summary(metrics, save_path=figs_dir / 'metrics_summary.png', style=plot_style)

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test NSE: {metrics['nse']:.4f}")

    return lstm_model, results
