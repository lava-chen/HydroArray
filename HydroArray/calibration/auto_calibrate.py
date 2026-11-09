"""
旧 auto_calibrate 入口:保留为 Calibrator 的薄包装,签名为 run_func/obs。

新代码请用::

    from HydroArray.calibration import Calibrator, CalibratorConfig, calibrate

老 API::

    from HydroArray.calibration import auto_calibrate
    result = auto_calibrate(
        run_func=..., obs=q_obs,
        ref=ref, overrides={...},
        max_evaluations=200, strategy="multi_start_lbfgsb",
    )
    # result 是 AutoCalibrateResult(旧类型),但内部委托给 Calibrator
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from HydroArray.calibration.calibrator import Calibrator, CalibratorConfig
from HydroArray.calibration.result import AutoCalibrateResult, CalibrationResult


def auto_calibrate(
    run_func: Callable,
    obs=None,
    ref: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
    *,
    strategy: str = "multi_start_lbfgsb",
    max_evaluations: int = 200,
    n_starts: int = 4,
    first_budget_share: float = 0.4,
    pop_size: int = 30,
    objective: Union[str, Mapping[str, float], Callable] = "nse",
    name_alias: Optional[Mapping[str, str]] = None,
    seed: int = 42,
    verbose: bool = False,
    **kwargs,
) -> AutoCalibrateResult:
    """兼容旧 API 的 shim。

    行为变化:
    - obs= 仍然是可选的。如果传了,会包成 (params -> metrics) 闭包
      内部调 HydroArray.analysis.regression_summary
    - 不传 obs 时,假设 run_func 自己返回 metrics dict

    Args:
        run_func: ``(params) -> sim`` 或 ``(params) -> metrics``
        obs: 实测值数组(可选;传了则把 run_func 包成 metrics 返回)
        ref / overrides / strategy / max_evaluations / n_starts / pop_size:
            透传给 Calibrator
        objective: 透传给 Calibrator
        name_alias / seed / verbose: 透传

    Returns:
        AutoCalibrateResult(旧类型,字段照旧)
    """
    warnings.warn(
        "auto_calibrate is deprecated; use Calibrator() or ha.calibrate() instead.",
        DeprecationWarning, stacklevel=2,
    )

    # 包装 run_func
    if obs is not None:
        from HydroArray.analysis import regression_summary
        base_func = run_func

        def wrapped_run_func(params: dict) -> dict:
            sim = base_func(params)
            return regression_summary(sim, obs)
        actual_run_func = wrapped_run_func
    else:
        actual_run_func = run_func

    cfg = CalibratorConfig(
        strategy=strategy,
        objective=objective,
        max_evaluations=max_evaluations,
        n_starts=n_starts,
        first_budget_share=first_budget_share,
        pop_size=pop_size,
        seed=seed,
        verbose=verbose,
    )
    cal = Calibrator(
        run_func=actual_run_func,
        ref=ref,
        overrides=overrides,
        name_alias=name_alias,
        config=cfg,
    )
    res: CalibrationResult = cal.run()
    return AutoCalibrateResult.from_calibration_result(res)
