"""
续算工具:从 yaml checkpoint 恢复到 Calibrator 状态,然后继续 run。

Calibrator 自身已经有 save_checkpoint / load_checkpoint,这里补一些
处理历史 CalibrationResult 的便利函数。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from HydroArray.calibration.calibrator import Calibrator
from HydroArray.calibration.result import CalibrationResult


def resume_from_result(
    run_func,
    result_path: Union[str, Path],
    *,
    ref=None,
    overrides=None,
    name_alias=None,
    config=None,
) -> Calibrator:
    """从 CalibrationResult 的 yaml 文件读出最优参数,作为 warm_start。

    典型场景:跑了一半崩了,或者想基于已有 best 继续 refine。
    """
    res = CalibrationResult.from_yaml(result_path)
    return _build_calibrator_with_warm_start(
        run_func, res.parameters, ref, overrides, name_alias, config,
    )


def resume_from_calibrator_checkpoint(
    run_func,
    checkpoint_path: Union[str, Path],
    *,
    ref=None,
    overrides=None,
    name_alias=None,
    config=None,
) -> Calibrator:
    """从 Calibrator.save_checkpoint 的 yaml 读出 warm_x。"""
    import yaml
    p = Path(checkpoint_path)
    with open(p, "r", encoding="utf-8") as f:
        state = yaml.safe_load(f)

    warm_x = state.get("warm_x")
    if warm_x is None:
        return _build_calibrator_with_warm_start(
            run_func, None, ref, overrides, name_alias, config,
        )
    # 把 warm_x(归一化?物理?)转回物理参数
    # 旧格式:warm_x 是物理值列表
    # 这里简单按物理值处理
    if ref is not None and overrides is not None:
        cal = Calibrator(
            run_func=run_func, ref=ref, overrides=overrides,
            name_alias=name_alias, config=config,
        )
    else:
        # 至少一个
        cal = Calibrator(
            run_func=run_func, bounds=[],
            name_alias=name_alias, config=config,
        )
    return cal.warm_start(dict(zip(cal.parameter_names, warm_x)))


def _build_calibrator_with_warm_start(
    run_func,
    warm_params: Optional[Dict[str, float]],
    ref, overrides, name_alias, config,
) -> Calibrator:
    cal = Calibrator(
        run_func=run_func, ref=ref, overrides=overrides,
        name_alias=name_alias, config=config,
    )
    if warm_params:
        cal.warm_start(warm_params)
    return cal
