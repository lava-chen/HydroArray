"""
Simple Training Interface for HydroArray

基于 trainer.py 的简单训练接口，提供 3-5 行代码的训练体验。

Example:
    >>> from HydroArray.datasets import CAMELSUSDataset
    >>> from HydroArray.ml import train

    >>> # 3 lines to train
    >>> dataset = CAMELSUSDataset("path/to/camels", basins=["01013500"])
    >>> model, results = train(dataset, model="lstm")

    >>> # Or with config file
    >>> model, results = train(config="config.yml")
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from HydroArray.ml.trainer import SequenceTrainer
from HydroArray.ml.models import create_model, list_available_models
from HydroArray.utils.config import Config
from HydroArray.utils.logger import ExperimentLogger
from HydroArray.analysis.statistics import nse, rmse, mae
from HydroArray.plotting import (
    plot_loss_curve,
    plot_predictions_vs_observations,
    plot_time_series_comparison,
    plot_metrics_summary,
)


class _ModelConfig:
    """用于创建模型的配置类。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _create_model(model_name: str, input_dim: int, hidden_dim: int = 64,
                  num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1,
                  **kwargs):
    """根据模型名称使用注册表创建模型。"""
    # 构建模型配置
    if model_name.lower() in ('lstm', 'simple_lstm', 'bilstm'):
        model_config = _ModelConfig(
            model=_ModelConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout
            )
        )
        return create_model(model_config, model_type=model_name.lower())

    elif model_name.lower() in ('convlstm',):
        from HydroArray.ml.models.spatial import EncoderForecaster
        return EncoderForecaster(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [hidden_dim * 2, hidden_dim, hidden_dim]),
            kernel_size=kwargs.get('kernel_size', 5),
            num_layers=num_layers,
            bias=True
        )

    else:
        available = list_available_models()
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")


def train(
    dataset: Any = None,
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
    save_dir: Optional[Union[str, Path]] = None,
    device: str = "auto",
    verbose: bool = True,
    log_interval: int = 1,
    visualize: bool = True,
    plot_style: str = "hess",
    plots: Optional[List[str]] = None,
    config: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model on a dataset with minimal configuration.

    This is the simplest way to train a model in HydroArray.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset, optional
        A PyTorch Dataset containing (x, y) samples. Required if no config.
        x shape: (batch, seq_len, features)
        y shape: (batch, pred_len, targets)
    model : str
        Model type: 'lstm' (default)
    hidden_dim : int
        Hidden dimension for LSTM (default: 64)
    num_layers : int
        Number of LSTM layers (default: 2)
    output_dim : int
        Output dimension (default: 1)
    dropout : float
        Dropout rate (default: 0.1)
    epochs : int
        Maximum number of epochs (default: 50)
    batch_size : int
        Batch size (default: 32)
    learning_rate : float
        Learning rate (default: 0.001)
    weight_decay : float
        Weight decay (default: 1e-5)
    grad_clip : float
        Gradient clipping threshold (default: 1.0)
    patience : int
        Early stopping patience (default: 10)
    val_ratio : float
        Validation split ratio (default: 0.1)
    save_dir : str or Path, optional
        Directory to save results. If None, results are not saved.
    device : str
        Device to use: 'auto', 'cpu', 'cuda' (default: 'auto')
    verbose : bool
        Whether to print progress (default: True)
    log_interval : int
        Print every N epochs (default: 1, i.e. every epoch)
    config : str or Path, optional
        Path to YAML config file. If provided, other parameters are
        ignored and loaded from config.

    Returns
    -------
    Tuple[nn.Module, Dict]
        Trained model and results dictionary containing:
        - train_losses: List of training losses
        - val_losses: List of validation losses
        - metrics: Dictionary of evaluation metrics

    Example:
        >>> from HydroArray.datasets import CAMELSUSDataset
        >>> from HydroArray.ml import train

        >>> dataset = CAMELSUSDataset("path/to/camels", basins=["01013500"])
        >>> model, results = train(dataset, model="lstm", epochs=50)

        >>> # Or with config file
        >>> model, results = train(config="config.yml")
    """
    experiment_name = None
    output_dir = None
    config_path = None

    # Load from config if provided - 使用 Config 类
    if config is not None:
        cfg = Config(config)
        config_path = Path(config) if isinstance(config, (str, Path)) else None

        # Extract parameters from config
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

        # Visualization options
        visualize = cfg.get('visualization.enabled', visualize)
        plot_style = cfg.get('visualization.style', plot_style)
        plots_to_generate = cfg.get('visualization.plots', plots or ['loss', 'scatter', 'timeseries', 'metrics'])

        # Dataset config storage for later use
        _dataset_info = None

        # Try to create dataset from config if not provided
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
                    if isinstance(data_cfg, dict):
                        basins = data_cfg.get('basins')
                        seq_len = data_cfg.get('seq_len', 365)
                        pred_len = data_cfg.get('pred_len', 1)
                    else:
                        basins = getattr(data_cfg, 'basins', None)
                        seq_len = getattr(data_cfg, 'seq_len', 365)
                        pred_len = getattr(data_cfg, 'pred_len', 1)

                    dataset = CAMELSUSDataset(
                        data_dir=data_dir,
                        basins=basins,
                        seq_len=seq_len,
                        pred_len=pred_len,
                    )

                elif dataset_name.lower() in ('moving_mnist', 'mnist'):
                    from HydroArray.datasets import MovingMNISTDataset
                    if isinstance(data_cfg, dict):
                        data_path = data_cfg.get('data_path')
                        seq_len = data_cfg.get('seq_len', 10)
                        pred_len = data_cfg.get('pred_len', 10)
                        train_ratio = data_cfg.get('train_ratio', 0.9)
                    else:
                        data_path = getattr(data_cfg, 'data_path', None)
                        seq_len = getattr(data_cfg, 'seq_len', 10)
                        pred_len = getattr(data_cfg, 'pred_len', 10)
                        train_ratio = getattr(data_cfg, 'train_ratio', 0.9)

                    if data_path is None:
                        data_path = data_dir / 'mnist_test_seq.npy'
                    else:
                        data_path = Path(data_path) if isinstance(data_path, str) else data_path

                    # Store dataset info for val dataset creation
                    _dataset_info = {
                        'type': 'moving_mnist',
                        'data_path': data_path,
                        'seq_len': seq_len,
                        'pred_len': pred_len,
                        'train_ratio': train_ratio,
                        'future_steps': pred_len  # ConvLSTM needs this
                    }

                    # 创建配置对象
                    class _DataConfig:
                        def __init__(self, data_path, seq_len, pred_len, train_ratio):
                            self.data = type('obj', (object,), {
                                'data_path': data_path,
                                'seq_len': seq_len,
                                'future_len': pred_len,
                                'train_ratio': train_ratio
                            })()
                            self._cfg = {'data': {'data_path': data_path, 'seq_len': seq_len, 'future_len': pred_len, 'train_ratio': train_ratio}}
                        def get(self, key, default=None):
                            keys = key.split('.')
                            val = self._cfg
                            for k in keys:
                                if isinstance(val, dict) and k in val:
                                    val = val[k]
                                else:
                                    return default
                            return val

                    data_config = _DataConfig(data_path, seq_len, pred_len, train_ratio)
                    dataset = MovingMNISTDataset(data_config, period='train')

    # Validate dataset
    if dataset is None:
        raise ValueError(
            "dataset must be provided either directly or via config with data_dir and dataset type"
        )

    # Device setup
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Setup experiment folder structure
    if save_dir is not None:
        save_dir = Path(save_dir)
    elif experiment_name and output_dir:
        save_dir = Path(output_dir) / experiment_name
    else:
        save_dir = None

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        logs_dir = save_dir / 'logs'
        checkpoints_dir = save_dir / 'checkpoints'
        figs_dir = save_dir / 'Figs'

        logs_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        figs_dir.mkdir(exist_ok=True)

        # Copy config file to experiment folder
        if config_path is not None and config_path.exists():
            shutil.copy(config_path, save_dir / 'config.yml')

        # Setup logger using ExperimentLogger
        logger = ExperimentLogger(
            log_dir=str(logs_dir),
            experiment_name='train',
            verbose=False
        )
    else:
        logs_dir = None
        checkpoints_dir = None
        logger = None

    # Get input dimension from dataset
    if hasattr(dataset, 'input_vars'):
        input_dim = len(dataset.input_vars)
    elif hasattr(dataset, 'forcing_vars'):
        input_dim = len(dataset.forcing_vars)
    else:
        try:
            sample_x, _ = dataset[0]
            # Handle different data formats:
            # - Sequence data: (T, features) -> input_dim = features (1D)
            # - Spatial data: (T, C, H, W) -> input_dim = C (1 channel)
            # - Batched spatial data: (B, T, C, H, W) -> input_dim = C
            if sample_x.ndim == 5:
                input_dim = sample_x.shape[2]  # Channels for ConvLSTM (B, T, C, H, W)
            elif sample_x.ndim == 4:
                input_dim = sample_x.shape[1]  # Channels for 4D spatial (T, C, H, W)
            else:
                input_dim = sample_x.shape[-1]
        except:
            input_dim = 1

    if logger:
        logger.info(f"Creating {model} model (input_dim={input_dim}, hidden_dim={hidden_dim})")

    # Create model
    net = _create_model(
        model_name=model,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)

    # Check if dataset has internal train/val split (like MovingMNIST)
    if _dataset_info is not None and _dataset_info['type'] == 'moving_mnist':
        from HydroArray.datasets import MovingMNISTDataset

        # Create val dataset with same config
        class _DataConfig:
            def __init__(self, cfg_dict):
                self._cfg = cfg_dict
                self.data = type('obj', (object,), {
                    'data_path': cfg_dict.get('data_path'),
                    'seq_len': cfg_dict.get('seq_len', 10),
                    'future_len': cfg_dict.get('pred_len', 10),
                    'train_ratio': cfg_dict.get('train_ratio', 0.9)
                })()
            def get(self, key, default=None):
                keys = key.split('.')
                val = self._cfg
                for k in keys:
                    if isinstance(val, dict) and k in val:
                        val = val[k]
                    else:
                        return default
                return val

        val_data_cfg = _DataConfig(_dataset_info)
        val_dataset = MovingMNISTDataset(val_data_cfg, period='val')

        train_loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        n = len(dataset)
        n_val = len(val_dataset)
    else:
        # Split dataset using SubsetRandomSampler
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

    # Use SequenceTrainer for training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Create trainer using SequenceTrainer
    trainer = SequenceTrainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        future_steps=1,
        num_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=grad_clip,
        save_dir=str(checkpoints_dir) if checkpoints_dir else None,
        verbose=False
    )

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    if logger:
        logger.info(f"Training on {n} samples ({n - n_val} train, {n_val} val)...")
    elif verbose:
        print(f"Training on {n} samples ({n - n_val} train, {n_val} val)...")

    # Get future_steps if model needs it
    future_steps = 1
    if _dataset_info is not None and _dataset_info.get('future_steps'):
        future_steps = _dataset_info['future_steps']

    # Training loop
    for epoch in range(epochs):
        # Train epoch
        net.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Handle models that need future_steps (like ConvLSTM)
            import inspect
            sig = inspect.signature(net.forward)
            if 'future_steps' in sig.parameters:
                out = net(x, future_steps=future_steps)
            else:
                out = net(x)

            loss = criterion(out, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # Handle models that need future_steps
                if 'future_steps' in inspect.signature(net.forward).parameters:
                    out = net(x, future_steps=future_steps)
                else:
                    out = net(x)

                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

        if patience_counter >= patience:
            if logger:
                logger.info(f"Early stopping at epoch {epoch+1}")
            elif verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    net.load_state_dict(best_state)

    # Final evaluation using statistics module
    import inspect
    net.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            # Handle models that need future_steps
            if 'future_steps' in inspect.signature(net.forward).parameters:
                out = net(x, future_steps=future_steps)
            else:
                out = net(x)

            all_preds.append(out.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Compute metrics using statistics module
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

    # Save experiment results
    if save_dir is not None:
        torch.save(best_state, save_dir / 'model.pth')

        with open(save_dir / 'results.json', 'w') as f:
            json.dump({
                'best_val_loss': float(best_val_loss),
                'metrics': {k: float(v) for k, v in metrics.items()},
                'epochs': epoch + 1,
            }, f, indent=2)

        # Save summary
        summary = {
            'best_val_loss': float(best_val_loss),
            'test_rmse': float(metrics['rmse']),
            'test_nse': float(metrics['nse']),
            'epochs_trained': epoch + 1,
        }
        if logger:
            logger.save_summary(summary)
            logger.close()

        # Generate plots based on config
        if visualize and figs_dir is not None:
            if logger:
                logger.info("Generating plots...")

            plots_dir = figs_dir
            plots_dir.mkdir(exist_ok=True)

            if 'loss' in plots_to_generate:
                plot_loss_curve(
                    train_losses, val_losses,
                    save_path=plots_dir / 'loss_curve.png',
                    style=plot_style
                )

            if 'scatter' in plots_to_generate:
                plot_predictions_vs_observations(
                    predictions, targets,
                    save_path=plots_dir / 'predictions_vs_observations.png',
                    style=plot_style
                )

            if 'timeseries' in plots_to_generate:
                plot_time_series_comparison(
                    predictions, targets,
                    n_samples=1000,
                    save_path=plots_dir / 'time_series_comparison.png',
                    style=plot_style
                )

            if 'metrics' in plots_to_generate:
                plot_metrics_summary(
                    metrics,
                    save_path=plots_dir / 'metrics_summary.png',
                    style=plot_style
                )

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test NSE: {metrics['nse']:.4f}")
        if save_dir:
            print(f"  Results saved to: {save_dir}")

    return net, results
