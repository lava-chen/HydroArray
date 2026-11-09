"""
HydroArray.calibration — 高层水文参数率定。

主要 API:

1. ``calibrate(model, forcing, observations, ...)`` — 一行率定(最常用)
2. ``Calibrator`` — 主类,需要更细控制时用
3. ``CalibrationPipeline`` — 多阶段(calibrate → refine → validate)
4. ``CalibratorConfig`` / ``CalibratorPreset`` — 配置与预置
5. ``CalibrationResult`` — 标准结果(可 YAML 序列化)
6. ``auto_calibrate(...)`` — 旧 API shim(已 deprecated)

典型用法::

    import HydroArray as ha

    result = ha.calibrate(
        model,
        forcing=(precip, pet),
        observations=q_obs_mm,
        ref={"K": "0.8~1.2", "B": 0.3, "WM": 120},
        overrides={"KE": (1.0, 72.0)},
        strategy="ga_then_lbfgsb",
        max_evaluations=600,
    )
    print(result.nse, result.kge)
    result.to_yaml("calib.yaml")
"""

from HydroArray.calibration.result import (
    CalibrationResult,
    AutoCalibrateResult,
)
from HydroArray.calibration.objective import (
    OBJECTIVE_PRESETS,
    build_objective,
    create_objective_function,
)
from HydroArray.calibration.normalizer import Normalizer, lhs_starts
from HydroArray.calibration.strategies import (
    STRATEGY_REGISTRY,
    run_strategy,
    run_lbfgsb,
    run_multi_start_lbfgsb,
    run_ga,
    run_ga_then_lbfgsb,
    run_nelder,
)
from HydroArray.calibration.calibrator import Calibrator, CalibratorConfig
from HydroArray.calibration.pipeline import CalibrationPipeline
from HydroArray.calibration.presets import (
    PRESETS,
    CalibratorPreset,
    get_preset,
    list_presets,
    make_calibrator_from_preset,
)
from HydroArray.calibration.checkpoints import (
    resume_from_result,
    resume_from_calibrator_checkpoint,
)
from HydroArray.calibration.auto_calibrate import auto_calibrate

# 顶层 facade
from HydroArray.calibration._facade import calibrate

__all__ = [
    # 核心数据
    "CalibrationResult",
    "AutoCalibrateResult",
    # 工具
    "OBJECTIVE_PRESETS",
    "build_objective",
    "create_objective_function",
    "Normalizer",
    "lhs_starts",
    # strategies
    "STRATEGY_REGISTRY",
    "run_strategy",
    "run_lbfgsb",
    "run_multi_start_lbfgsb",
    "run_ga",
    "run_ga_then_lbfgsb",
    "run_nelder",
    # 主类
    "Calibrator",
    "CalibratorConfig",
    "CalibrationPipeline",
    # 预置
    "PRESETS",
    "CalibratorPreset",
    "get_preset",
    "list_presets",
    "make_calibrator_from_preset",
    # 续算
    "resume_from_result",
    "resume_from_calibrator_checkpoint",
    # 兼容
    "auto_calibrate",
    # 顶层 facade
    "calibrate",
]
