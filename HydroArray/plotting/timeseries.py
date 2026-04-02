"""
时间序列绘图模块

提供训练监控和模型评估所需的绘图函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    show: bool = False
):
    """
    绘制训练损失曲线。

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
        style: 绘图风格
        show: 是否显示
    """
    from HydroArray.plotting.styles import use_hydro_style
    use_hydro_style(style)

    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train', linewidth=1.5)
    ax.plot(epochs, val_losses, label='Validation', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_predictions_vs_observations(
    predictions: np.ndarray,
    observations: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    show: bool = False
):
    """
    绘制预测值 vs 观测值散点图。

    Args:
        predictions: 预测值
        observations: 观测值
        save_path: 保存路径
        style: 绘图风格
        show: 是否显示
    """
    from HydroArray.plotting.styles import use_hydro_style
    use_hydro_style(style)

    fig, ax = plt.subplots()

    # 展平数组
    pred = predictions.flatten()
    obs = observations.flatten()

    # 散点图
    ax.scatter(obs, pred, alpha=0.3, s=10)

    # 1:1 线
    min_val = min(obs.min(), pred.min())
    max_val = max(obs.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='1:1 Line')

    ax.set_xlabel('Observations')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions vs Observations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 计算 NSE
    from HydroArray.analysis.statistics import nse
    nse_val = nse(pred, obs)
    ax.text(0.05, 0.95, f'NSE = {nse_val:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_time_series_comparison(
    predictions: np.ndarray,
    observations: np.ndarray,
    time_index: Optional[np.ndarray] = None,
    n_samples: int = 500,
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    show: bool = False
):
    """
    绘制时间序列对比图。

    Args:
        predictions: 预测值
        observations: 观测值
        time_index: 时间索引
        n_samples: 显示样本数
        save_path: 保存路径
        style: 绘图风格
        show: 是否显示
    """
    from HydroArray.plotting.styles import use_hydro_style
    use_hydro_style(style)

    fig, ax = plt.subplots()

    # 展平
    pred = predictions.flatten()
    obs = observations.flatten()

    # 采样显示
    if len(pred) > n_samples:
        idx = np.linspace(0, len(pred) - 1, n_samples, dtype=int)
        pred = pred[idx]
        obs = obs[idx]
        if time_index is not None:
            time_index = time_index[idx]

    x = time_index if time_index is not None else range(len(pred))

    ax.plot(x, obs, label='Observations', linewidth=1, alpha=0.8)
    ax.plot(x, pred, label='Predictions', linewidth=1, alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_metrics_summary(
    metrics: dict,
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    show: bool = False
):
    """
    绘制指标汇总柱状图。

    Args:
        metrics: 指标字典
        save_path: 保存路径
        style: 绘图风格
        show: 是否显示
    """
    from HydroArray.plotting.styles import use_hydro_style
    use_hydro_style(style)

    fig, ax = plt.subplots()

    names = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values)
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Metrics')
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱状图上显示数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)