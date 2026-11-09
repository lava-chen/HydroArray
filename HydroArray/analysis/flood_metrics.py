"""
通用水文评价指标。

不依赖具体模型 / 具体尺度 / 具体流域:
所有函数只接收两个等长的一维数组 (simulated, observed)。
"""
from __future__ import annotations
import numpy as np


__all__ = [
    "volume_error_pct",
    "peak_error_pct",
    "peak_time_lag",
    "nse",
    "rmse",
    "kge",
    "pbias",
    "r_squared",
    "regression_summary",
]


# ─────────────────── 水文常用:水量 / 洪峰 / 时差 ───────────────────
def volume_error_pct(simulated, observed) -> float:
    """水量相对误差 (%) = (sum(sim) - sum(obs)) / sum(obs) * 100"""
    s_obs = float(np.sum(observed))
    if s_obs == 0:
        return float("inf")
    return float((np.sum(simulated) - s_obs) / s_obs * 100)


def peak_error_pct(simulated, observed) -> float:
    """洪峰相对误差 (%) = (max(sim) - max(obs)) / max(obs) * 100"""
    p_obs = float(np.max(observed))
    if p_obs == 0:
        return float("inf")
    return float((np.max(simulated) - p_obs) / p_obs * 100)


def peak_time_lag(simulated, observed) -> int:
    """峰现时差 (步数), 正=模拟滞后, 负=模拟超前"""
    return int(np.argmax(simulated) - np.argmax(observed))


# ─────────────────── 经典统计指标 ───────────────────
def nse(simulated, observed) -> float:
    """Nash-Sutcliffe Efficiency"""
    o = np.asarray(observed, dtype=float)
    s = np.asarray(simulated, dtype=float)
    denom = float(np.sum((o - o.mean()) ** 2))
    if denom == 0:
        return float("-inf")
    return 1.0 - float(np.sum((s - o) ** 2) / denom)


def rmse(simulated, observed) -> float:
    """Root Mean Squared Error"""
    o = np.asarray(observed, dtype=float)
    s = np.asarray(simulated, dtype=float)
    return float(np.sqrt(np.mean((s - o) ** 2)))


def pbias(simulated, observed) -> float:
    """Percent Bias (%) = sum(sim-obs)/sum(obs)*100

    Returns NaN when the sum of observations is zero (PBIAS is undefined).
    Matches the behavior of analysis.statistics.pbiass and
    utils.metrics.calculate_pbias to keep downstream training loops
    from silently propagating inf.
    """
    o = np.asarray(observed, dtype=float)
    s = np.asarray(simulated, dtype=float)
    s_obs = float(o.sum())
    if s_obs == 0:
        return float("nan")
    return float((s.sum() - s_obs) / s_obs * 100)


def r_squared(simulated, observed) -> float:
    """Coefficient of Determination R²"""
    o = np.asarray(observed, dtype=float)
    s = np.asarray(simulated, dtype=float)
    if o.std() == 0 or s.std() == 0:
        return float("-inf")
    return float(np.corrcoef(o, s)[0, 1] ** 2)


def kge(simulated, observed) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009)"""
    o = np.asarray(observed, dtype=float)
    s = np.asarray(simulated, dtype=float)
    if o.std() == 0 or s.std() == 0:
        return float("-inf")
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = float(s.std() / o.std())
    beta = float(s.mean() / o.mean())
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ─────────────────── 一键汇总 ───────────────────
def regression_summary(simulated, observed) -> dict:
    """8 指标一并返回。任意两列等长数据都能调。"""
    return {
        "nse":            nse(simulated, observed),
        "rmse":           rmse(simulated, observed),
        "kge":            kge(simulated, observed),
        "r_squared":      r_squared(simulated, observed),
        "pbias_pct":      pbias(simulated, observed),
        "volume_err_pct": volume_error_pct(simulated, observed),
        "peak_err_pct":   peak_error_pct(simulated, observed),
        "peak_time_lag":  peak_time_lag(simulated, observed),
    }
