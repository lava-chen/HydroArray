"""
Objective function spec 规范化。

支持:
- 字符串: "nse" / "kge" / "rmse" / "pbias" / "multi"
- 字典:   {"nse": 1.0, "pbias_pct": -0.02, "peak_err_pct": -0.01}
          正权重 = 越大越好,负权重 = 反号(让 |err| 越接近 0 越好)
- 可调用对象: 接受 metrics dict,返回单一分数

返回统一签名的 objective:  ``(params: dict) -> float``(越大越好)
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Union, Any, Optional

from HydroArray.analysis import regression_summary


ObjectiveSpec = Union[str, Mapping[str, float], Callable[[Dict[str, float]], float]]


# ─────────────────── 预置权重 ───────────────────


# 这些是校准器"nse"目标时单选的目标;
# "multi" 是一组常用的组合权重,可由用户覆盖
OBJECTIVE_PRESETS: Dict[str, Dict[str, float]] = {
    "nse":   {"nse": 1.0},
    "kge":   {"kge": 1.0},
    # rmse/pbias 越小越好, 用负号让统一目标"越大越好"
    "rmse":  {"rmse": -1.0},
    "pbias": {"pbias_pct": -1.0},
    "multi": {
        "nse":               1.0,
        "volume_err_pct":   -0.010,   # 鼓励水量接近
        "peak_err_pct":     -0.006,   # 鼓励洪峰接近
        "peak_time_lag":    -0.010,   # 鼓励峰现时差小
    },
}


# ─────────────────── 工具:把指标 dict 转成"越大越好"的分数 ───────────────────


def _score_from_metrics(
    metrics: Dict[str, float], weights: Dict[str, float]
) -> float:
    """加权汇总: sum(weight * metric_value)。weight 为负则鼓励 metric 接近 0。"""
    total = 0.0
    for name, w in weights.items():
        v = metrics.get(name)
        if v is None or v != v:  # NaN 检查
            return -1e18  # 失败评估
        total += float(w) * float(v)
    return total


# ─────────────────── 构造器 ───────────────────


def build_objective(
    spec: ObjectiveSpec,
    *,
    metric_func: Optional[Callable[[Any], Dict[str, float]]] = None,
) -> Callable[[Any], float]:
    """从 spec 构造统一签名的 objective: ``(params) -> float``(越大越好)。

    Args:
        spec: 见模块顶部
        metric_func: ``params -> metrics_dict``。默认是::

            def metric_func(params):
                # 调用方需提供 sim/obs
                return regression_summary(sim, obs)

            用法: ``build_objective("nse", metric_func=lambda p: regression_summary(run(p), obs))``

            如果 metric_func 内部已经跑了模拟并计算了 metrics,直接传它即可。

    Returns:
        callable: ``(params) -> float``(越大越好)
    """
    # 字符串 spec -> 权重
    if isinstance(spec, str):
        if spec not in OBJECTIVE_PRESETS:
            raise ValueError(
                f"Unknown objective preset: {spec!r}. "
                f"Available: {list(OBJECTIVE_PRESETS.keys())}"
            )
        weights = dict(OBJECTIVE_PRESETS[spec])
    elif isinstance(spec, Mapping):
        weights = dict(spec)  # 拷贝
    elif callable(spec):
        # 可调用 spec: 直接当 objective 用,但包一层让"越大越好"语义成立
        raw_obj = spec
        def _from_callable(params):
            score = raw_obj(params)
            # 如果可调用直接返回 metrics 字典,转成分数
            if isinstance(score, Mapping):
                # 简单求和(保持向后兼容,允许返回值是 metrics dict)
                return sum(score.values())
            return float(score)
        return _from_callable
    else:
        raise TypeError(f"objective spec must be str / Mapping / callable, "
                        f"got {type(spec).__name__}")

    if metric_func is None:
        raise ValueError(
            "build_objective requires metric_func= when spec is str / Mapping"
        )

    def _obj(params: Any) -> float:
        metrics = metric_func(params)
        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"metric_func must return a dict of metrics, got {type(metrics).__name__}"
            )
        return _score_from_metrics(dict(metrics), weights)

    # 把权重挂在函数上,方便 Calibrator 提取
    _obj.__dict__["weights"] = weights
    _obj.__dict__["__name__"] = (
        f"objective({list(weights.items())[:2]}...)"
        if len(weights) > 2 else f"objective({weights})"
    )
    return _obj


def get_weights_from_objective(obj: Callable) -> Dict[str, float]:
    """提取 build_objective 包装的权重;否则返回空 dict。"""
    return dict(obj.__dict__.get("weights", {}))


def _is_maximize(spec: ObjectiveSpec) -> bool:
    """判断 spec 形式是否"越大越好"。所有 internal 形式都是 max 化。"""
    return True


# ─────────────────── 旧 API 兼容:create_objective_function ───────────────────


def create_objective_function(
    simulate_func: Callable[[Dict[str, float]], Any],
    observations: Any,
    objective: str = "nse",
    weights: Optional[Dict[str, float]] = None,
) -> Callable[[Dict[str, float]], Dict[str, float]]:
    """兼容老 ``HydroArray.domain.calibration.create_objective_function``。

    返回 ``(params) -> metrics_dict``(满足 Calibrator 内部 / 旧 optimizer 调用)。
    """
    if objective == "nse":
        def f(params):
            sim = simulate_func(params)
            m = regression_summary(sim, observations)
            return m
    elif objective == "rmse":
        def f(params):
            sim = simulate_func(params)
            m = regression_summary(sim, observations)
            return m
    elif objective == "kge":
        def f(params):
            sim = simulate_func(params)
            m = regression_summary(sim, observations)
            return m
    elif objective == "multi":
        if weights is None:
            weights = OBJECTIVE_PRESETS["multi"]
        def f(params):
            sim = simulate_func(params)
            m = regression_summary(sim, observations)
            m["multi"] = _score_from_metrics(m, weights)
            return m
    else:
        def f(params):
            sim = simulate_func(params)
            m = regression_summary(sim, observations)
            return m
    return f
