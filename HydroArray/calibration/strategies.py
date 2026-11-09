"""
5 个 strategy 的具体实现。

每个 strategy 接受: 归一化后的 objective、单位空间 bounds、配置参数
返回: List[(x_physical, score, n_evals)] — 各次重启的最优
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import optimize as scipy_optimize

from HydroArray.calibration.normalizer import Normalizer, lhs_starts
from HydroArray.domain.calibration import ParameterBounds


# 类型: 返回 (x_physical, score, n_evals) 三元组
StrategyResult = Tuple[np.ndarray, float, int]


# ─────────────────── 内部:objective 桥接 ───────────────────


def _make_unit_objective(
    obj_metric: Callable[[dict], dict],
    score_func: Callable[[dict], float],
    norm: Normalizer,
) -> Callable[[np.ndarray], float]:
    """把 '物理参数 dict -> metrics dict -> score' 包装成 '单位数组 -> 分数'。"""
    def _obj(u_arr: np.ndarray) -> float:
        x = norm.to_physical(u_arr)
        phys = norm.physical_to_dict(x)
        metrics = obj_metric(phys)
        return score_func(metrics)
    return _obj


def _make_physical_objective(
    obj_metric: Callable[[dict], dict],
    score_func: Callable[[dict], float],
    norm: Normalizer,
) -> Callable[[np.ndarray], float]:
    """'物理参数数组 -> 分数'。用于在物理空间跑 scipy 优化。"""
    def _obj(x_arr: np.ndarray) -> float:
        phys = norm.physical_to_dict(x_arr)
        metrics = obj_metric(phys)
        return score_func(metrics)
    return _obj


def _minimize_lbfgs(
    obj_physical: Callable[[np.ndarray], float],
    x0: np.ndarray,
    norm: Normalizer,
    max_evals: int,
) -> Tuple[np.ndarray, int]:
    """在物理空间直接调 scipy L-BFGS-B(用 norm 的真实 bounds)。"""
    n_calls = [0]
    def _counted(x):
        n_calls[0] += 1
        return obj_physical(x)
    bounds = list(zip(norm.lo.tolist(), norm.hi.tolist()))
    res = scipy_optimize.minimize(
        _counted, x0, method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_evals, "maxfun": max_evals,
                 "ftol": 1e-9, "gtol": 1e-7},
    )
    return np.clip(res.x, norm.lo, norm.hi), n_calls[0]


def _minimize_nelder(
    obj_physical: Callable[[np.ndarray], float],
    x0: np.ndarray,
    norm: Normalizer,
    max_evals: int,
) -> Tuple[np.ndarray, int]:
    n_calls = [0]
    def _counted(x):
        n_calls[0] += 1
        return obj_physical(x)
    res = scipy_optimize.minimize(
        _counted, x0, method="Nelder-Mead",
        options={"maxiter": max_evals, "maxfev": max_evals},
    )
    return np.clip(res.x, norm.lo, norm.hi), n_calls[0]


def _ga_search(
    obj_physical: Callable[[np.ndarray], float],
    norm: Normalizer,
    *,
    pop_size: int,
    generations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """简单 GA:均匀交叉 + 高斯扰动,在物理空间采样。"""
    n_calls = [0]
    def _counted(x):
        n_calls[0] += 1
        return obj_physical(x)

    dim = norm.n_params
    pop = rng.uniform(norm.lo, norm.hi, size=(pop_size, dim))
    fitness = np.array([_counted(ind) for ind in pop])
    for _ in range(generations):
        order = np.argsort(-fitness)  # 越大越好
        elite = pop[order[:pop_size // 2]]
        elite_f = fitness[order[:pop_size // 2]]
        kids = np.empty_like(pop)
        kids[:pop_size // 2] = elite
        for i in range(pop_size // 2, pop_size):
            a, b = rng.integers(0, pop_size // 2, size=2)
            alpha = rng.uniform(0.0, 1.0, size=dim)
            child = alpha * elite[a] + (1 - alpha) * elite[b]
            child = np.clip(
                child + rng.normal(0, 0.1 * (norm.hi - norm.lo)),
                norm.lo, norm.hi,
            )
            kids[i] = child
        pop = kids
        fitness = np.array([_counted(ind) for ind in pop])
        for k, idx in enumerate(order[:pop_size // 2]):
            if elite_f[k] > fitness[idx]:
                pop[idx] = elite[k]
                fitness[idx] = elite_f[k]
    best_idx = int(np.argmax(fitness))
    return pop[best_idx], n_calls[0]


def _safe_extract(
    x_best: np.ndarray,
    norm: Normalizer,
    score_func: Callable[[dict], float],
    obj_metric: Callable[[dict], dict],
) -> Tuple[np.ndarray, float]:
    x_best = np.clip(np.asarray(x_best, dtype=float), norm.lo, norm.hi)
    phys = norm.physical_to_dict(x_best)
    metrics = obj_metric(phys)
    try:
        return x_best, float(score_func(metrics))
    except TypeError:
        return x_best, float(score_func(phys))


# ─────────────────── 1. lbfgsb: 单点 L-BFGS-B(从中点起步) ───────────────────


def run_lbfgsb(
    obj_metric: Callable,
    score_func: Callable,
    norm: Normalizer,
    *,
    objective: str = "nse",
    max_evaluations: int = 200,
    verbose: bool = False,
    **_unused,
) -> List[StrategyResult]:
    """单点 L-BFGS-B,从物理空间的中点起步。"""
    phys_obj = _make_physical_objective(obj_metric, score_func, norm)
    x0 = (norm.lo + norm.hi) / 2
    x_best, n = _minimize_lbfgs(phys_obj, x0, norm, max_evaluations)
    x, score = _safe_extract(x_best, norm, score_func, obj_metric)
    if verbose:
        print(f"  [lbfgsb] score={score:.4f}  evals={n}")
    return [(x, score, n)]


# ─────────────────── 2. multi_start_lbfgsb: 多起点 L-BFGS-B ───────────────────


def run_multi_start_lbfgsb(
    obj_metric: Callable,
    score_func: Callable,
    norm: Normalizer,
    *,
    objective: str = "nse",
    max_evaluations: int = 200,
    n_starts: int = 4,
    first_budget_share: float = 0.4,
    seed: int = 42,
    verbose: bool = False,
    **_unused,
) -> List[StrategyResult]:
    """多起点 L-BFGS-B,物理空间起步。

    预算分配: 第一个起点(中点)吃 ``first_budget_share``;其余均分给 (n_starts-1) 个 LHS 扰动起点。
    """
    rng = np.random.default_rng(seed)
    n_starts = max(1, int(n_starts))
    evals_first = max(20, int(max_evaluations * first_budget_share))
    evals_rest_total = max(20 * (n_starts - 1), max_evaluations - evals_first)
    evals_per_rest = max(15, evals_rest_total // max(n_starts - 1, 1))

    # 在归一化空间生成 LHS,转回物理空间作为起点
    u_starts = np.vstack([
        norm.unit_center()[None, :],
        lhs_starts(n_starts - 1, norm.n_params, rng),
    ])
    x_starts = np.array([norm.to_physical(u) for u in u_starts])
    evals_list = [evals_first] + [evals_per_rest] * (n_starts - 1)

    phys_obj = _make_physical_objective(obj_metric, score_func, norm)
    results: List[StrategyResult] = []
    for i, (x0, ev) in enumerate(zip(x_starts, evals_list)):
        x_best, n = _minimize_lbfgs(phys_obj, x0, norm, ev)
        x, score = _safe_extract(x_best, norm, score_func, obj_metric)
        if verbose:
            print(f"  [multi start {i}] score={score:.4f}  evals={n}")
        results.append((x, score, n))
    return results


# ─────────────────── 3. ga: Genetic Algorithm 全局 ───────────────────


def run_ga(
    obj_metric: Callable,
    score_func: Callable,
    norm: Normalizer,
    *,
    objective: str = "nse",
    max_evaluations: int = 600,
    pop_size: int = 30,
    seed: int = 42,
    verbose: bool = False,
    **_unused,
) -> List[StrategyResult]:
    """GA 全局搜索,物理空间采样。返回 [(best_x, best_score, total_evals)]。"""
    rng = np.random.default_rng(seed)
    generations = max(1, max_evaluations // max(pop_size, 1))
    phys_obj = _make_physical_objective(obj_metric, score_func, norm)
    x_best, n = _ga_search(phys_obj, norm,
                            pop_size=pop_size, generations=generations, rng=rng)
    x, score = _safe_extract(x_best, norm, score_func, obj_metric)
    if verbose:
        print(f"  [ga] score={score:.4f}  evals={n}")
    return [(x, score, n)]


# ─────────────────── 4. ga_then_lbfgsb: GA 粗搜 + L-BFGS-B 精修 ───────────────────


def run_ga_then_lbfgsb(
    obj_metric: Callable,
    score_func: Callable,
    norm: Normalizer,
    *,
    objective: str = "nse",
    max_evaluations: int = 600,
    pop_size: int = 30,
    seed: int = 42,
    verbose: bool = False,
    **_unused,
) -> List[StrategyResult]:
    """GA 粗搜 → 拿最优 → L-BFGS-B 在该点附近精修。"""
    ga_evals = min(max_evaluations, pop_size * 40)
    lbfgsb_evals = max(20, max_evaluations - ga_evals)

    ga_results = run_ga(
        obj_metric, score_func, norm,
        objective=objective, max_evaluations=ga_evals,
        pop_size=pop_size, seed=seed, verbose=verbose,
    )
    ga_x, ga_score, ga_n = ga_results[0]

    phys_obj = _make_physical_objective(obj_metric, score_func, norm)
    x_refined, lbfgs_n = _minimize_lbfgs(phys_obj, ga_x, norm, lbfgsb_evals)
    x, score_refined = _safe_extract(x_refined, norm, score_func, obj_metric)
    if verbose:
        print(f"  [ga → lbfgsb] ga={ga_score:.4f}  refined={score_refined:.4f}")
    return [
        (ga_x, ga_score, ga_n),
        (x, score_refined, lbfgs_n),
    ]


# ─────────────────── 5. nelder: Nelder-Mead 无梯度 ───────────────────


def run_nelder(
    obj_metric: Callable,
    score_func: Callable,
    norm: Normalizer,
    *,
    objective: str = "nse",
    max_evaluations: int = 200,
    verbose: bool = False,
    **_unused,
) -> List[StrategyResult]:
    """Nelder-Mead 无梯度,适合目标不可微/有噪声的场景。"""
    phys_obj = _make_physical_objective(obj_metric, score_func, norm)
    x0 = (norm.lo + norm.hi) / 2
    x_best, n = _minimize_nelder(phys_obj, x0, norm, max_evaluations)
    x, score = _safe_extract(x_best, norm, score_func, obj_metric)
    if verbose:
        print(f"  [nelder] score={score:.4f}  evals={n}")
    return [(x, score, n)]


# ─────────────────── 调度器 ───────────────────


STRATEGY_REGISTRY = {
    "lbfgsb": run_lbfgsb,
    "multi_start_lbfgsb": run_multi_start_lbfgsb,
    "ga": run_ga,
    "ga_then_lbfgsb": run_ga_then_lbfgsb,
    "nelder": run_nelder,
}


def run_strategy(
    name: str,
    obj_metric: Callable,
    score_func: Callable,
    norm: Normalizer,
    *,
    objective: str = "nse",
    max_evaluations: int = 200,
    n_starts: int = 4,
    first_budget_share: float = 0.4,
    pop_size: int = 30,
    seed: int = 42,
    verbose: bool = False,
) -> List[StrategyResult]:
    """根据 name 调对应的 strategy 函数。"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name!r}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](
        obj_metric, score_func, norm,
        max_evaluations=max_evaluations,
        n_starts=n_starts,
        first_budget_share=first_budget_share,
        pop_size=pop_size,
        seed=seed,
        verbose=verbose,
    )
