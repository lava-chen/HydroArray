"""
ConvLSTM: Convolutional LSTM Network
基于论文 "Convolutional LSTM Network: A Machine Learning Approach 
for Precipitation Nowcasting" (Xingjian et al., 2015)

论文地址: https://arxiv.org/abs/1506.04214
"""

import torch
import torch.nn as nn
from typing import Optional, Any


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元
    
    论文公式(3)：
        i = σ(W_xi * x + W_hi * h + W_ci * c + b_i)  # 输入门
        f = σ(W_xf * x + W_hf * h + W_cf * c + b_f)  # 遗忘门
        c_next = f * c_cur + i * g                    # 单元状态
        g = tanh(W_xg * x + W_hg * h + b_g)           # 候选值
        o = σ(W_xo * x + W_ho * h + W_co * c_next + b_o)  # 输出门
        h_next = o * tanh(c_next)
    
    Args:
        input_dim: 输入通道数
        hidden_dim: 隐藏状态通道数
        kernel_size: 卷积核大小
        bias: 是否使用偏置
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # 保持空间尺寸
        self.bias = bias

        # 论文公式(3)中，所有的门控计算可以合并成一个卷积操作
        # 输出通道为 4 * hidden_dim，分别对应 i, f, g, o 四个门
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor: (B, C_in, H, W)
            cur_state: (h, c) 各自 (B, hidden_dim, H, W)
        Returns:
            (h_next, c_next)
        """
        h_cur, c_cur = cur_state

        # 在通道(channel)维度拼接输入和上一时刻隐藏状态
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # 执行卷积
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 对应论文公式(3)的门控逻辑
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新单元状态 C 和隐藏状态 H
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM 网络
    
    论文中使用的3层架构：
    - 第1层: hidden_dim=128
    - 第2层: hidden_dim=64  
    - 第3层: hidden_dim=64
    """
    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64, 64],
        kernel_size=5,
        num_layers=3,
        bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # 构建多层ConvLSTM
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.cells.append(
                ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size, bias)
            )
    
    def forward(self, input_seq, hidden_states=None):
        """
        Args:
            input_seq: (B, T, C, H, W)
            hidden_states: 初始隐藏状态列表
        Returns:
            layer_outputs: 每层的输出列表
            last_states: 最后的隐藏状态列表
        """
        batch_size, seq_len, _, h, w = input_seq.size()
        device = input_seq.device
        
        # 初始化隐藏状态
        if hidden_states is None:
            hidden_states = self._init_hidden(batch_size, h, w, device)
        
        layer_outputs = []
        current_input = input_seq
        
        # 逐层处理
        for layer_idx, cell in enumerate(self.cells):
            h_state, c_state = hidden_states[layer_idx]
            layer_output = []
            
            # 时间步展开
            for t in range(seq_len):
                h_state, c_state = cell(current_input[:, t], (h_state, c_state))
                layer_output.append(h_state)
            
            # 堆叠时间维度
            layer_output = torch.stack(layer_output, dim=1)
            layer_outputs.append(layer_output)
            
            # 下一层的输入是这一层的隐藏状态
            current_input = layer_output
            hidden_states[layer_idx] = (h_state, c_state)
        
        return layer_outputs, hidden_states
    
    def _init_hidden(self, batch_size, h, w, device):
        """初始化所有层的隐藏状态"""
        hidden_states = []
        for hidden_dim in self.hidden_dims:
            h_state = torch.zeros(batch_size, hidden_dim, h, w, device=device)
            c_state = torch.zeros(batch_size, hidden_dim, h, w, device=device)
            hidden_states.append((h_state, c_state))
        return hidden_states


class EncoderForecaster(nn.Module):
    """
    编码器-预测器架构 (论文中的 Seq2Seq 结构)
    
    使用3层 ConvLSTM 进行编码，然后循环预测未来帧
    """
    def __init__(
        self,
        input_dim=1,
        hidden_dims=[128, 64, 64],
        kernel_size=5,
        num_layers=3,
        bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        
        # 编码器: 3层 ConvLSTM
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias
        )
        
        # 预测器: 与编码器结构相同
        self.forecaster = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias
        )
        
        # 输出层: 将最后一层的隐藏状态映射回输入维度
        self.output_conv = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    def forward(self, input_seq, future_steps):
        """
        Args:
            input_seq: (B, T, C, H, W) 历史序列
            future_steps: 预测步数
        Returns:
            (B, future_steps, input_dim, H, W)
        """
        batch_size, seq_len, _, h, w = input_seq.size()
        device = input_seq.device
        
        # --- Encoding 阶段 ---
        # 逐层编码历史序列
        _, encoder_states = self.encoder(input_seq)
        
        # --- Forecasting 阶段 ---
        # 使用编码器的最终状态作为预测器的初始状态
        outputs = []
        curr_input = input_seq[:, -1]  # 最后一帧作为起始
        forecaster_states = encoder_states  # 传递隐藏状态
        
        for _ in range(future_steps):
            # 单步预测
            layer_outputs, forecaster_states = self.forecaster(
                curr_input.unsqueeze(1),  # (B, 1, C, H, W)
                forecaster_states
            )
            
            # 取最后一层的输出，映射回输入维度
            last_layer_output = layer_outputs[-1].squeeze(1)  # (B, hidden_dim, H, W)
            output = self.output_conv(last_layer_output)  # (B, input_dim, H, W)
            outputs.append(output)
            
            # 循环输入
            curr_input = output
        
        return torch.stack(outputs, dim=1)


# 保持向后兼容的单层版本
class SingleLayerEncoderForecaster(nn.Module):
    """单层 ConvLSTM (用于向后兼容)"""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
        self.output_conv = nn.Conv2d(cell.hidden_dim, cell.input_dim, kernel_size=1)

    def forward(self, input_seq, future_steps):
        batch_size, seq_len, _, h, w = input_seq.size()
        device = input_seq.device
        
        h_state = torch.zeros(batch_size, self.cell.hidden_dim, h, w, device=device)
        c_state = torch.zeros(batch_size, self.cell.hidden_dim, h, w, device=device)

        # Encoding
        for t in range(seq_len):
            h_state, c_state = self.cell(input_seq[:, t], (h_state, c_state))

        # Forecasting
        outputs = []
        curr_input = input_seq[:, -1]
        
        for _ in range(future_steps):
            h_state, c_state = self.cell(curr_input, (h_state, c_state))
            output = self.output_conv(h_state)
            outputs.append(output)
            curr_input = output

        return torch.stack(outputs, dim=1)


# =============================================================================
# Training Functions
# =============================================================================

def train_convlstm(
    dataset: Any = None,
    config: Optional[str] = None,
    hidden_dim: int = 64,
    num_layers: int = 3,
    kernel_size: int = 5,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    patience: int = 10,
    val_ratio: float = 0.1,
    save_dir: Optional[str] = None,
    device: str = "auto",
    verbose: bool = True,
    log_interval: int = 5,
    **kwargs
) -> tuple:
    """
    Train ConvLSTM model for video prediction.

    Args:
        dataset: PyTorch Dataset (MovingMNIST). If None, must provide via config.
        config: Path to YAML config file.
        hidden_dim: ConvLSTM hidden dimension
        num_layers: Number of ConvLSTM layers
        kernel_size: Convolutional kernel size
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

    Returns:
        Tuple of (model, results_dict)

    Example:
        >>> from HydroArray.ml.models.spatial.conv_lstm import train_convlstm
        >>> model, results = train_convlstm(dataset, epochs=20)
    """
    import shutil
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    import numpy as np
    import torch
    import torch.nn as nn

    from HydroArray.utils.config import Config
    from HydroArray.utils.logger import ExperimentLogger
    from HydroArray.plotting import (
        plot_loss_curve,
        plot_metrics_summary,
    )

    # Load config from file
    experiment_name = None
    output_dir = None
    config_path = None
    visualize = True
    plot_style = 'hess'
    plots_to_generate = ['loss', 'metrics']

    if config is not None:
        cfg = Config(config) if isinstance(config, str) else Config(config) if isinstance(config, dict) else config
        config_path = Path(config) if isinstance(config, str) else None

        hidden_dim = cfg.get('model.hidden_dim', hidden_dim)
        num_layers = cfg.get('model.num_layers', num_layers)
        kernel_size = cfg.get('model.kernel_size', kernel_size)

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

                if dataset_name.lower() in ('moving_mnist', 'mnist'):
                    from HydroArray.datasets import MovingMNISTDataset

                    data_path = data_cfg.get('data_path') if isinstance(data_cfg, dict) else getattr(data_cfg, 'data_path', None)
                    seq_len = data_cfg.get('seq_len', 10) if isinstance(data_cfg, dict) else getattr(data_cfg, 'seq_len', 10)
                    pred_len = data_cfg.get('pred_len', 10) if isinstance(data_cfg, dict) else getattr(data_cfg, 'pred_len', 10)
                    train_ratio = data_cfg.get('train_ratio', 0.9) if isinstance(data_cfg, dict) else getattr(data_cfg, 'train_ratio', 0.9)

                    if data_path is None:
                        data_path = data_dir / 'mnist_test_seq.npy'
                    else:
                        data_path = Path(data_path) if isinstance(data_path, str) else data_path

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

    # Get input dimension from dataset
    try:
        sample_x, _ = dataset[0]
        if sample_x.ndim == 4:
            input_dim = sample_x.shape[1]  # (T, C, H, W) -> C
        else:
            input_dim = 1
    except:
        input_dim = 1

    # Get future_steps from dataset config
    future_steps = 10  # default for MNIST
    if hasattr(dataset, '__class__') and dataset.__class__.__name__ == 'MovingMNISTDataset':
        future_steps = getattr(dataset, 'future_len', 10)

    # Create model
    model = EncoderForecaster(
        input_dim=input_dim,
        hidden_dims=[hidden_dim * 2, hidden_dim, hidden_dim],
        kernel_size=kernel_size,
        num_layers=num_layers,
        bias=True
    ).to(device)

    # Create val dataset
    if hasattr(dataset, '__class__') and dataset.__class__.__name__ == 'MovingMNISTDataset':
        # Create a simple config object that MovingMNISTDataset expects
        class _SimpleDataConfig:
            def __init__(self, data_path, seq_len, future_len, train_ratio):
                self.data = type('obj', (object,), {
                    'data_path': data_path,
                    'seq_len': seq_len,
                    'future_len': future_len,
                    'train_ratio': train_ratio
                })()
                self._cfg = {
                    'data': {
                        'data_path': data_path,
                        'seq_len': seq_len,
                        'future_len': future_len,
                        'train_ratio': train_ratio
                    }
                }
            def get(self, key, default=None):
                keys = key.split('.')
                val = self._cfg
                for k in keys:
                    if isinstance(val, dict) and k in val:
                        val = val[k]
                    else:
                        return default
                return val

        val_data_cfg = _SimpleDataConfig(
            data_path=dataset.data_path,
            seq_len=dataset.seq_len,
            future_len=dataset.future_len,
            train_ratio=0.9  # Same as train
        )
        val_dataset = MovingMNISTDataset(val_data_cfg, period='val')
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        n = len(dataset)
        n_val = len(val_dataset)
    else:
        n = len(dataset)
        n_val = int(n * val_ratio)
        indices = list(range(n))
        np.random.shuffle(indices)
        val_idx = set(indices[:n_val])
        train_idx = indices[n_val:]
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), num_workers=0)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(list(val_idx)), num_workers=0)

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    if verbose:
        print(f"Training ConvLSTM on {n} samples ({n - n_val} train, {n_val} val)...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, future_steps=future_steps)
            loss = criterion(out, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, future_steps=future_steps)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x, future_steps=future_steps)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Metrics for video prediction
    metrics = {
        'mse': float(np.mean((predictions - targets) ** 2)),
        'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
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
                'epochs_trained': epoch + 1,
            })
            logger.close()

        # Generate plots
        if visualize and figs_dir is not None:
            figs_dir.mkdir(exist_ok=True)
            if 'loss' in plots_to_generate:
                plot_loss_curve(train_losses, val_losses, save_path=figs_dir / 'loss_curve.png', style=plot_style)
            if 'metrics' in plots_to_generate:
                plot_metrics_summary(metrics, save_path=figs_dir / 'metrics_summary.png', style=plot_style)

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Best val loss: {best_val_loss:.6f}")

    return model, results
