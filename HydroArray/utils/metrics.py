"""
水文模型评估指标

提供常用的水文模型评价指标计算函数。
"""

from typing import Union, Tuple
import numpy as np


def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算 Nash-Sutcliffe Efficiency (NSE)

    NSE = 1 - Σ(Q_sim - Q_obs)² / Σ(Q_obs - Q_mean)²

    取值范围: -∞ 到 1
    - NSE = 1: 完美模拟
    - NSE = 0: 模拟与均值一样好
    - NSE < 0: 模型比均值更差

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        NSE 值
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # 移除 NaN
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    numerator = np.sum((sim - obs) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator


def calculate_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算 Root Mean Square Error (RMSE)

    RMSE = sqrt(Σ(Q_sim - Q_obs)² / n)

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        RMSE 值
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    return np.sqrt(np.mean((sim - obs) ** 2))


def calculate_kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算 Kling-Gupta Efficiency (KGE)

    KGE = 1 - sqrt((r - 1)² + (α - 1)² + (β - 1)²)

    其中:
    - r: 相关系数
    - α = σ_sim / σ_obs (相对变异系数)
    - β = μ_sim / μ_obs (相对偏差)

    取值范围: -∞ 到 1

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        KGE 值
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    # 计算统计量
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    # KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge


def calculate_pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算 Percent Bias (PBias)

    PBias = 100 * Σ(Q_sim - Q_obs) / Σ(Q_obs)

    取值范围: -∞ 到 +∞
    - PBias = 0: 无偏差
    - PBias > 0: 模拟值偏高（高估）
    - PBias < 0: 模拟值偏低（低估）

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        PBias 百分比
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    return 100 * np.sum(sim - obs) / np.sum(obs)


def calculate_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算 Mean Bias (BIAS)

    BIAS = Σ(Q_sim - Q_obs) / n

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        平均偏差
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    return np.mean(sim - obs)


def calculate_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算 Mean Absolute Error (MAE)

    MAE = Σ|Q_sim - Q_obs| / n

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        MAE 值
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    return np.mean(np.abs(sim - obs))


def calculate_r_squared(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算决定系数 R²

    R² = (Σ(Q_obs - Q_mean)(Q_sim - Q_mean))² / (Σ(Q_obs - Q_mean)² * Σ(Q_sim - Q_mean)²)

    取值范围: 0 到 1

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        R² 值
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    correlation = np.corrcoef(obs, sim)[0, 1]
    return correlation ** 2


def calculate_nse_log(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算对数 Nash-Sutcliffe Efficiency (NSE)

    用于评估低流量模拟效果。

    NSE_log = 1 - Σ(log(Q_sim+1) - log(Q_obs+1))² / Σ(log(Q_obs+1) - log(Q_mean+1))²

    Args:
        observed: 观测值
        simulated: 模拟值

    Returns:
        NSE_log 值
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    mask = ~(np.isnan(observed) | np.isnan(simulated) | (observed <= 0) | (simulated <= 0))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    log_obs = np.log(obs + 1)
    log_sim = np.log(sim + 1)
    log_mean = np.mean(log_obs)

    numerator = np.sum((log_sim - log_obs) ** 2)
    denominator = np.sum((log_obs - log_mean) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator


def mm_to_cms(mm: Union[float, np.ndarray], area: float, dt_hours: float) -> Union[float, np.ndarray]:
    """将毫米转换为立方米/秒

    Q (m³/s) = P (mm) * A (km²) / (3.6 * dt)

    Args:
        mm: 毫米单位的量
        area: 流域面积 (km²)
        dt_hours: 时间步长 (小时)

    Returns:
        流量 (m³/s)
    """
    return mm * area / (3.6 * dt_hours)


def cms_to_mm(cms: Union[float, np.ndarray], area: float, dt_hours: float) -> Union[float, np.ndarray]:
    """将立方米/秒转换为毫米

    P (mm) = Q (m³/s) * 3.6 * dt / A (km²)

    Args:
        cms: 流量 (m³/s)
        area: 流域面积 (km²)
        dt_hours: 时间步长 (小时)

    Returns:
        毫米单位的量
    """
    return cms * 3.6 * dt_hours / area


def evaluate_model(observed: np.ndarray, simulated: np.ndarray,
                   metrics: Tuple[str, ...] = ('nse', 'rmse', 'kge', 'pbias')) -> dict:
    """批量计算多个评估指标

    Args:
        observed: 观测值
        simulated: 模拟值
        metrics: 要计算的指标列表

    Returns:
        指标字典
    """
    results = {}

    metric_funcs = {
        'nse': calculate_nse,
        'nse_log': calculate_nse_log,
        'rmse': calculate_rmse,
        'mae': calculate_mae,
        'kge': calculate_kge,
        'pbias': calculate_pbias,
        'bias': calculate_bias,
        'r2': calculate_r_squared,
    }

    for m in metrics:
        if m in metric_funcs:
            results[m] = metric_funcs[m](observed, simulated)
        else:
            results[m] = np.nan

    return results
