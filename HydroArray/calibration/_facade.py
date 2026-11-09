"""
calibrate(): 一行率定的门面函数。

如果调用方已经有 HydroArray 模型(model.initialize / water_balance 协议),
直接::

    result = ha.calibrate(
        model,
        forcing=(precip, pet),
        observations=q_obs_mm,
        ref=ref_table,
        strategy="ga_then_lbfgsb",
    )

否则,调用方自己写 run_func,然后::

    cal = Calibrator(run_func=..., ref=ref_table, ...)
    res = cal.run()
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from HydroArray.calibration.calibrator import Calibrator, CalibratorConfig
from HydroArray.calibration.result import CalibrationResult
from HydroArray.config.parameters import bounds_from_ref_table
from HydroArray.analysis import regression_summary


def calibrate(
    model=None,
    forcing: Optional[Tuple[Any, Any]] = None,
    observations: Any = None,
    *,
    run_func: Optional[Callable] = None,
    ref: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
    bounds=None,
    name_alias: Optional[Mapping[str, str]] = None,
    objective: Union[str, Mapping[str, float], Callable] = "nse",
    strategy: str = "ga_then_lbfgsb",
    max_evaluations: int = 600,
    n_starts: int = 4,
    pop_size: int = 30,
    seed: int = 42,
    n_jobs: int = 1,
    step_hours: float = 24.0,
    verbose: bool = False,
) -> CalibrationResult:
    """一行率定。

    Args:
        model: HydroArray 模型(支持 .initialize / .water_balance)。如果不传,必须传 run_func。
        forcing: ``(precip, pet)``,两者都是 array-like,长度相同
        observations: 实测流量数组(与 model 输出同单位)
        run_func: 高级用户自己写的 ``(params dict) -> metrics dict`` 闭包
        ref / overrides / bounds: 参数约束(必传其一,优先 ref)
        objective: "nse" / "kge" / "rmse" / "pbias" / "multi" / 权重 dict / 可调用
        strategy: 优化策略
        max_evaluations / n_starts / pop_size: optimizer 预算
        seed: 随机种子
        n_jobs: 并行核数
        step_hours: 模型步长(小时)
        verbose: 是否打印进度

    Returns:
        CalibrationResult

    Example::

        import HydroArray as ha

        result = ha.calibrate(
            ha.XinAnjiangModel(),
            forcing=(precip, pet),
            observations=q_obs_mm,
            ref={"K": "0.8~1.2", "B": 0.3, "WM": 120},
            overrides={"KE": (1.0, 72.0)},
            strategy="ga_then_lbfgsb",
            max_evaluations=600,
        )
    """
    if run_func is None:
        if model is None or forcing is None or observations is None:
            raise ValueError(
                "Pass either run_func=, or all of (model, forcing, observations)."
            )
        run_func = _build_model_run_func(
            model, forcing, observations,
            step_hours=step_hours,
        )

    cfg = CalibratorConfig(
        strategy=strategy,
        objective=objective,
        max_evaluations=max_evaluations,
        n_starts=n_starts,
        pop_size=pop_size,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    cal = Calibrator(
        run_func=run_func,
        ref=ref,
        overrides=overrides,
        bounds=bounds,
        name_alias=name_alias,
        config=cfg,
    )
    return cal.run()


# ─────────────────── 模型 → run_func 桥接 ───────────────────


def _build_model_run_func(
    model,
    forcing: Tuple[Any, Any],
    observations: Any,
    *,
    step_hours: float = 24.0,
) -> Callable[[Dict[str, float]], Dict[str, float]]:
    """把 (model, forcing, obs) 拼成 (params) -> metrics dict。"""
    precip, pet = forcing
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    obs = np.asarray(observations, dtype=float)
    n = len(precip)

    # 闭包变量:model + 强迫数据
    state = {"p": precip, "e": pet, "n": n}

    def _run_func(params: Dict[str, float]) -> Dict[str, float]:
        # 把 params 同步到 model._params(支持 XinAnjiangParameters 风格)
        m = model
        if hasattr(m, "_params") and m._params is not None:
            for k, v in params.items():
                if hasattr(m._params, k):
                    setattr(m._params, k, float(v))
        # 单节点
        if hasattr(m, "initialize") and not getattr(m, "_initialized", False):
            from HydroArray.domain import GridNode
            try:
                m.initialize(
                    [GridNode(node_id=0, x=0.0, y=0.0, area=1.0)],
                    m._params,
                )
            except Exception:
                pass  # 已初始化,继续

        # 逐时步
        sim = np.empty(state["n"], dtype=float)
        p_arr = state["p"]
        e_arr = state["e"]
        for t in range(state["n"]):
            p_t = np.array([float(p_arr[t])])
            e_t = np.array([float(e_arr[t])])
            try:
                _, _, _, states = m.water_balance(step_hours, p_t, e_t)
                runoff = float(states["runoff_generated"][0])
            except Exception:
                _, _, _, _ = m.water_balance(step_hours, p_t, e_t)
                # fallback: 旧 3 元组
                runoff = 0.0
            sim[t] = max(runoff, 0.0)
        return regression_summary(sim, obs)

    return _run_func
