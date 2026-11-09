"""
CalibrationResult: 参数率定的标准输出。

- 必填字段:parameters, objective_value, metrics (dict)
- 可选字段:n_evaluations, elapsed_seconds, converged, message, strategy
- 多起点时:per_start_score, best_start_index
- 边界信息:bounds_used (List[ParameterBounds])
- 序列化:to_dict / to_yaml / from_yaml / from_dict
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    _YAML = True
except ImportError:  # yaml 不是硬依赖
    _YAML = False

from HydroArray.domain.calibration import ParameterBounds


# ─────────────────── 核心结果 ───────────────────


@dataclass
class CalibrationResult:
    """参数率定结果。

    Attributes:
        parameters: 最优参数字典 {name: value}
        objective_value: 主目标函数值(越大越好;rmse/pbias 已反号)
        metrics: 完整 8 指标 {nse, rmse, kge, r_squared, pbias_pct,
                               volume_err_pct, peak_err_pct, peak_time_lag}
        objective_name: 用了哪个 objective spec, "nse"/"kge"/"multi"/"{custom}"
        strategy: 用的 strategy
        n_evaluations: 总评估次数
        elapsed_seconds: 耗时(秒)
        converged: 是否收敛
        message: 详细说明
        bounds_used: 实际用到的 ParameterBounds 列表
        per_start_score: 多起点时各起点得分(同 objective_value 量纲)
        best_start_index: 多起点时最优起点索引
        extras: 额外信息(子模块自填,例如 best 附近分布)
    """

    parameters: Dict[str, float]
    objective_value: float
    metrics: Dict[str, float]
    objective_name: str = "nse"
    strategy: str = "multi_start_lbfgsb"
    n_evaluations: int = 0
    elapsed_seconds: float = 0.0
    converged: bool = True
    message: str = ""
    bounds_used: List[ParameterBounds] = field(default_factory=list)
    per_start_score: List[float] = field(default_factory=list)
    best_start_index: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    # ────────── 便捷属性 ──────────

    @property
    def nse(self) -> float:
        return float(self.metrics.get("nse", float("nan")))

    @property
    def kge(self) -> float:
        return float(self.metrics.get("kge", float("nan")))

    @property
    def rmse(self) -> float:
        return float(self.metrics.get("rmse", float("nan")))

    @property
    def pbias_pct(self) -> float:
        return float(self.metrics.get("pbias_pct", float("nan")))

    # ────────── 序列化 ──────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ParameterBounds 需要单独转换
        d["bounds_used"] = [
            {"name": pb.name, "min_value": pb.min_value,
             "max_value": pb.max_value, "default_value": pb.default_value}
            for pb in self.bounds_used
        ]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationResult":
        d = dict(d)  # 避免修改原 dict
        bounds_raw = d.pop("bounds_used", [])
        bounds = [
            ParameterBounds(
                name=b["name"],
                min_value=b["min_value"],
                max_value=b["max_value"],
                default_value=b.get("default_value"),
            )
            for b in bounds_raw
        ]
        return cls(bounds_used=bounds, **d)

    def to_yaml(self, path: Union[str, Path]) -> None:
        if not _YAML:
            raise ImportError("PyYAML is required for to_yaml(). "
                              "Install with: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f,
                           allow_unicode=True, sort_keys=False, indent=2)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CalibrationResult":
        if not _YAML:
            raise ImportError("PyYAML is required for from_yaml(). "
                              "Install with: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # ────────── 漂亮打印 ──────────

    def summary(self) -> str:
        """返回一段人类可读的多行摘要,带颜色 (ANSI)。"""
        GREEN = "\033[92m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        lines = [
            f"{BOLD}CalibrationResult{RESET}  strategy={self.strategy}",
            f"  objective ({self.objective_name}) = {self.objective_value:+.4f}",
            f"  metrics:",
        ]
        for k, v in self.metrics.items():
            color = GREEN if k in ("nse", "kge", "r_squared") and v > 0.5 else ""
            end = RESET if color else ""
            lines.append(f"    {k:18s} = {v:+10.4f}{color}{end}")
        lines.append(f"  n_evaluations = {self.n_evaluations}")
        lines.append(f"  elapsed       = {self.elapsed_seconds:.2f}s")
        if self.per_start_score:
            lines.append(f"  per_start     = "
                         f"{[f'{s:+.4f}' for s in self.per_start_score]}")
            lines.append(f"  best_start    = {self.best_start_index}")
        lines.append(f"  parameters:")
        for k, v in self.parameters.items():
            lines.append(f"    {k:8s} = {v:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"CalibrationResult(strategy={self.strategy!r}, "
                f"objective_name={self.objective_name!r}, "
                f"objective_value={self.objective_value:+.4f}, "
                f"n_params={len(self.parameters)}, "
                f"n_evals={self.n_evaluations})")


# ─────────────────── 旧 API 兼容视图 ───────────────────


@dataclass
class AutoCalibrateResult:
    """旧 auto_calibrate 的返回类型,内部委托给 CalibrationResult。

    保留这个类是为不破旧调用方代码;新代码请用 CalibrationResult。
    """
    parameters: Dict[str, float]
    objective_value: float
    nse: float
    rmse: float
    kge: float
    n_evaluations: int
    converged: bool
    message: str
    strategy: str
    elapsed_seconds: float
    best_start_index: int
    per_start_nse: List[float] = field(default_factory=list)

    @classmethod
    def from_calibration_result(
        cls, r: CalibrationResult
    ) -> "AutoCalibrateResult":
        return cls(
            parameters=r.parameters,
            objective_value=r.objective_value,
            nse=r.nse,
            rmse=r.rmse,
            kge=r.kge,
            n_evaluations=r.n_evaluations,
            converged=r.converged,
            message=r.message,
            strategy=r.strategy,
            elapsed_seconds=r.elapsed_seconds,
            best_start_index=r.best_start_index,
            per_start_nse=r.per_start_score,
        )
