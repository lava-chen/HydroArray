"""
水文统计指标
NSE / KGE / RMSE / PBIAS / MAE
"""

import numpy as np


def nse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    纳什效率系数（Nash-Sutcliffe Coefficient of Efficiency）

    NSE = 1 - Σ(Q_obs - Q_sim)² / Σ(Q_obs - mean(Q_obs))²

    Args:
        simulated: 模型模拟值/卫星估算值 (n,)
        observed:  实际观测值 (n,)

    Returns:
        NSE ∈ (-∞, 1]，完美为1
    """
    simulated = np.asarray(simulated, dtype=np.float64)
    observed  = np.asarray(observed,  dtype=np.float64)

    if simulated.shape != observed.shape:
        raise ValueError("simulated 和 observed 维度必须一致")
    if simulated.size == 0:
        raise ValueError("输入不能为空")

    # 去除 NaN
    mask = ~(np.isnan(simulated) | np.isnan(observed))
    sim = simulated[mask]
    obs = observed[mask]

    if obs.size == 0:
        return np.nan

    num   = np.sum((obs - sim) ** 2)
    denom = np.sum((obs - np.mean(obs)) ** 2)

    if denom == 0:
        return np.nan

    return 1.0 - num / denom


def kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    克林-古普塔效率系数（Kling-Gupta Efficiency）

    KGE = 1 - √[(r-1)² + (β-1)² + (γ-1)²]

    其中:
        r   = 相关系数
        β   = μ_sim / μ_obs （均值比，偏差校正）
        γ   = (σ_sim / μ_sim) / (σ_obs / μ_obs) （变异比）

    Args:
        simulated: 模型模拟值 (n,)
        observed:  实际观测值 (n,)

    Returns:
        KGE ∈ (-∞, 1]，完美为1
    """
    simulated = np.asarray(simulated, dtype=np.float64)
    observed  = np.asarray(observed,  dtype=np.float64)

    if simulated.shape != observed.shape:
        raise ValueError("simulated 和 observed 维度必须一致")
    if simulated.size == 0:
        raise ValueError("输入不能为空")

    mask = ~(np.isnan(simulated) | np.isnan(observed))
    sim = simulated[mask]
    obs = observed[mask]

    if obs.size < 2:
        return np.nan

    # 相关系数
    r = np.corrcoef(obs, sim)[0, 1]

    # 均值比
    beta = np.mean(sim) / np.mean(obs)

    # 变异系数比
    cv_sim = np.std(sim, ddof=1) / np.mean(sim) if np.mean(sim) != 0 else np.nan
    cv_obs = np.std(obs, ddof=1) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    gamma = cv_sim / cv_obs if not (np.isnan(cv_sim) or np.isnan(cv_obs)) else np.nan

    if np.isnan(r) or np.isnan(beta) or np.isnan(gamma):
        return np.nan

    return 1.0 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)


def rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    均方根误差（Root Mean Square Error）

    RMSE = √[1/n · Σ(S_i - G_i)²]

    Args:
        simulated: 估算值 (n,)
        observed:  观测值 (n,)

    Returns:
        RMSE ≥ 0，越小越好
    """
    simulated = np.asarray(simulated, dtype=np.float64)
    observed  = np.asarray(observed,  dtype=np.float64)

    if simulated.shape != observed.shape:
        raise ValueError("simulated 和 observed 维度必须一致")

    mask = ~(np.isnan(simulated) | np.isnan(observed))
    sim = simulated[mask]
    obs = observed[mask]

    if obs.size == 0:
        return np.nan

    return np.sqrt(np.mean((sim - obs) ** 2))


def pbiass(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    相对偏差（Percent Bias，百分比表示）

    PBIAS = Σ(S_i - G_i) / Σ(G_i) × 100%

    Args:
        simulated: 估算值 (n,)
        observed:  观测值 (n,)

    Returns:
        PBIAS ∈ (-∞, +∞)%，完美为0
        负值表示低估，正值表示高估
    """
    simulated = np.asarray(simulated, dtype=np.float64)
    observed  = np.asarray(observed,  dtype=np.float64)

    if simulated.shape != observed.shape:
        raise ValueError("simulated 和 observed 维度必须一致")

    mask = ~(np.isnan(simulated) | np.isnan(observed))
    sim = simulated[mask]
    obs = observed[mask]

    if obs.size == 0:
        return np.nan

    if np.sum(obs) == 0:
        return np.nan

    return (np.sum(sim - obs) / np.sum(obs)) * 100.0


def mae(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    平均绝对误差（Mean Absolute Error）

    MAE = (1/n) · Σ|S_i - G_i|

    Args:
        simulated: 估算值 (n,)
        observed:  观测值 (n,)

    Returns:
        MAE ≥ 0，越小越好
    """
    simulated = np.asarray(simulated, dtype=np.float64)
    observed  = np.asarray(observed,  dtype=np.float64)

    if simulated.shape != observed.shape:
        raise ValueError("simulated 和 observed 维度必须一致")

    mask = ~(np.isnan(simulated) | np.isnan(observed))
    sim = simulated[mask]
    obs = observed[mask]

    if obs.size == 0:
        return np.nan

    return np.mean(np.abs(sim - obs))




def evaluate(simulated: np.ndarray, observed: np.ndarray) -> dict:
    """
    同时计算全部5个指标，返回 dict

    Returns:
        {
            "NSE":   float,
            "KGE":   float,
            "RMSE":  float,
            "PBIAS": float,
            "MAE":   float,
        }
    """
    return {
        "NSE":   round(nse(simulated, observed), 4),
        "KGE":   round(kge(simulated, observed), 4),
        "RMSE":  round(rmse(simulated, observed), 4),
        "PBIAS": round(pbiass(simulated, observed), 2),
        "MAE":   round(mae(simulated, observed), 4),
    }
