"""
预置配置:专家经验封装的 CalibratorConfig。

用 ``Calibrator.from_preset("xinanjiang_daily", run_func, ref)`` 即可。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional

from HydroArray.calibration.calibrator import Calibrator, CalibratorConfig


@dataclass
class CalibratorPreset:
    """命名预置。"""
    name: str
    description: str
    config: CalibratorConfig


PRESETS: Dict[str, CalibratorPreset] = {
    "xinanjiang_daily": CalibratorPreset(
        name="xinanjiang_daily",
        description=(
            "日尺度新安江模型:9 个产流+分水源参数,多起点 L-BFGS-B,"
            "中点 40% 预算起步"
        ),
        config=CalibratorConfig(
            strategy="multi_start_lbfgsb",
            objective="nse",
            max_evaluations=400,
            n_starts=4,
            first_budget_share=0.4,
            pop_size=30,
            seed=42,
        ),
    ),
    "xinanjiang_event": CalibratorPreset(
        name="xinanjiang_event",
        description=(
            "次洪尺度新安江:10 个参数(含 EX),GA 粗搜 + L-BFGS-B 精修,"
            "适配多场次池化"
        ),
        config=CalibratorConfig(
            strategy="ga_then_lbfgsb",
            objective="multi",
            max_evaluations=600,
            n_starts=4,
            first_budget_share=0.4,
            pop_size=40,
            seed=42,
        ),
    ),
    "quick_demo": CalibratorPreset(
        name="quick_demo",
        description=(
            "快速 demo:单点 L-BFGS-B,100 评估,适合教学 / 单元测试"
        ),
        config=CalibratorConfig(
            strategy="lbfgsb",
            objective="nse",
            max_evaluations=100,
            n_starts=1,
            first_budget_share=1.0,
            pop_size=20,
            seed=42,
        ),
    ),
    "production": CalibratorPreset(
        name="production",
        description=(
            "生产环境:GA 粗搜 + L-BFGS-B 精修,1000 评估,适合正式报告"
        ),
        config=CalibratorConfig(
            strategy="ga_then_lbfgsb",
            objective="kge",
            max_evaluations=1000,
            n_starts=4,
            first_budget_share=0.3,
            pop_size=50,
            seed=42,
        ),
    ),
}


def list_presets() -> Dict[str, str]:
    """返回 {name: description}。"""
    return {p.name: p.description for p in PRESETS.values()}


def get_preset(name: str) -> CalibratorPreset:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset: {name!r}. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]


def make_calibrator_from_preset(
    name: str,
    run_func: Callable,
    ref,
    overrides=None,
    *,
    name_alias=None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Calibrator:
    """从预置名构造 Calibrator,允许 config 局部覆盖。"""
    preset = get_preset(name)
    cfg = preset.config
    if config_overrides:
        cfg = replace(cfg, **config_overrides)
    return Calibrator(
        run_func=run_func,
        ref=ref,
        overrides=overrides,
        name_alias=name_alias,
        config=cfg,
    )
