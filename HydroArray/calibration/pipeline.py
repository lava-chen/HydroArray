"""
CalibrationPipeline: 多阶段率定(calibrate → refine → validate → report)。

Example::

    pipe = CalibrationPipeline(run_func, ref=ref)
    pipe.calibrate(strategy="ga_then_lbfgsb", max_evaluations=600)
    pipe.refine(shrink=0.5, max_evaluations=200)  # 在 best 附近缩窗口
    pipe.validate()                                # 拿最终 best 重新算全指标
    report = pipe.report()                         # 汇总成 dict
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import time

import numpy as np

from HydroArray.calibration.calibrator import Calibrator, CalibratorConfig
from HydroArray.calibration.result import CalibrationResult
from HydroArray.config.parameters import bounds_from_ref_table
from HydroArray.domain.calibration import ParameterBounds


@dataclass
class PipelineStage:
    """一次阶段率定的记录。"""
    name: str
    strategy: str
    max_evaluations: int
    result: Optional[CalibrationResult] = None
    elapsed_seconds: float = 0.0
    started_at: str = ""


class CalibrationPipeline:
    """多阶段率定流程。

    与单次 Calibrator.run() 的区别:
    - 可以串行多次(粗搜 + 精修 + 验证)
    - refine 阶段会自动把 bounds 收缩到 best 附近
    - 每个阶段都有独立时间戳 + result,可单独取
    - report() 把所有阶段汇总
    """

    def __init__(
        self,
        run_func: Callable[[Dict[str, float]], Dict[str, float]],
        ref: Optional[Mapping[str, Any]] = None,
        overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
        *,
        bounds: Optional[List[ParameterBounds]] = None,
        name_alias: Optional[Mapping[str, str]] = None,
        objective: Union[str, Mapping[str, float], Callable] = "nse",
        seed: int = 42,
        verbose: bool = False,
    ):
        self._run_func = run_func
        self._ref = ref
        self._overrides = overrides
        self._name_alias = name_alias
        self._objective = objective
        self._seed = seed
        self._verbose = verbose

        if bounds is not None:
            self._initial_bounds = list(bounds)
        else:
            self._initial_bounds = bounds_from_ref_table(
                dict(ref) if ref else {},
                overrides=overrides,
                name_alias=name_alias,
            )
        self._current_bounds: List[ParameterBounds] = list(self._initial_bounds)
        self._stages: List[PipelineStage] = []
        self._last_result: Optional[CalibrationResult] = None
        self._t_start: float = time.time()

    # ────────── 构造当前 Calibrator(用 self._current_bounds) ──────────

    def _make_calibrator(
        self, strategy: str, max_evaluations: int, n_starts: int,
        n_jobs: int = 1, first_budget_share: float = 0.4,
    ) -> Calibrator:
        cfg = CalibratorConfig(
            strategy=strategy,
            objective=self._objective,
            max_evaluations=max_evaluations,
            n_starts=n_starts,
            first_budget_share=first_budget_share,
            seed=self._seed,
            n_jobs=n_jobs,
            verbose=self._verbose,
        )
        # 用 bounds= 路径,避免重新跑 bounds_from_ref_table
        cal = Calibrator(
            run_func=self._run_func,
            bounds=self._current_bounds,
            config=cfg,
        )
        if self._last_result is not None:
            cal.warm_start(self._last_result.parameters)
        return cal

    # ────────── 阶段 1: calibrate ──────────

    def calibrate(
        self,
        strategy: str = "ga_then_lbfgsb",
        max_evaluations: int = 600,
        n_starts: int = 4,
        n_jobs: int = 1,
    ) -> CalibrationResult:
        """第一阶段:全局率定,找初始 best。"""
        t0 = time.time()
        cal = self._make_calibrator(strategy, max_evaluations, n_starts, n_jobs)
        res = cal.run()
        elapsed = time.time() - t0

        stage = PipelineStage(
            name="calibrate", strategy=strategy,
            max_evaluations=max_evaluations, result=res,
            elapsed_seconds=elapsed, started_at=_now_iso(),
        )
        self._stages.append(stage)
        self._last_result = res
        if self._verbose:
            print(f"[pipeline.calibrate] {res.summary()}")
        return res

    # ────────── 阶段 2: refine ──────────

    def refine(
        self,
        shrink: float = 0.5,
        strategy: str = "lbfgsb",
        max_evaluations: int = 200,
        n_starts: int = 4,
    ) -> CalibrationResult:
        """第二阶段:在 best 附近缩窗口,再率定。

        Args:
            shrink: 收缩比例, 0=不动, 1=全部塌到 best 点(单点 L-BFGS-B)
                   0.5=当前 bounds 半宽,以 best 为中心
            strategy: refine 阶段用的 strategy
        """
        if self._last_result is None:
            raise RuntimeError("Call calibrate() first, then refine()")
        if not (0.0 <= shrink <= 1.0):
            raise ValueError(f"shrink must be in [0, 1], got {shrink}")

        # 收缩 bounds
        best_x = np.array(
            [self._last_result.parameters.get(pb.name,
                                               0.5 * (pb.min_value + pb.max_value))
             for pb in self._current_bounds],
            dtype=float,
        )
        new_bounds: List[ParameterBounds] = []
        for pb, x_best in zip(self._current_bounds, best_x):
            half = (pb.max_value - pb.min_value) * 0.5 * shrink
            lo = max(pb.min_value, x_best - half)
            hi = min(pb.max_value, x_best + half)
            new_bounds.append(
                ParameterBounds(pb.name, lo, hi, default_value=float(x_best))
            )
        if self._verbose:
            print(f"[pipeline.refine] bounds shrunk by {shrink} around best")
        self._current_bounds = new_bounds

        t0 = time.time()
        cal = self._make_calibrator(strategy, max_evaluations, n_starts)
        # warm_start 用最新 best
        cal.warm_start(self._last_result.parameters)
        res = cal.run()
        elapsed = time.time() - t0

        stage = PipelineStage(
            name="refine", strategy=strategy,
            max_evaluations=max_evaluations, result=res,
            elapsed_seconds=elapsed, started_at=_now_iso(),
        )
        self._stages.append(stage)

        # 只在新的更好时更新 last_result
        if res.objective_value > self._last_result.objective_value:
            self._last_result = res
        if self._verbose:
            print(f"[pipeline.refine] {res.summary()}")
        return res

    # ────────── 阶段 3: validate ──────────

    def validate(
        self,
        validation_run_func: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
    ) -> CalibrationResult:
        """第三阶段:用 best 重新跑一遍,得最终 metrics。

        Args:
            validation_run_func: 验证期用 run_func(默认同 calibrate 用的)
        """
        if self._last_result is None:
            raise RuntimeError("Call calibrate() / refine() first")
        func = validation_run_func or self._run_func
        metrics = func(self._last_result.parameters)
        # 重新生成一份带新 metrics 的 result(其它字段照旧)
        validated = CalibrationResult(
            parameters=dict(self._last_result.parameters),
            objective_value=self._last_result.objective_value,
            metrics=metrics,
            objective_name=self._last_result.objective_name,
            strategy=self._last_result.strategy,
            n_evaluations=self._last_result.n_evaluations,
            elapsed_seconds=self._last_result.elapsed_seconds,
            converged=self._last_result.converged,
            message="validated" if validation_run_func is None
                    else "validated (custom run_func)",
            bounds_used=self._last_result.bounds_used,
            per_start_score=self._last_result.per_start_score,
            best_start_index=self._last_result.best_start_index,
            extras={**self._last_result.extras,
                    "validation_metrics": metrics},
        )
        stage = PipelineStage(
            name="validate", strategy="-", max_evaluations=1,
            result=validated, elapsed_seconds=0.0, started_at=_now_iso(),
        )
        self._stages.append(stage)
        if self._verbose:
            print(f"[pipeline.validate] {validated.summary()}")
        return validated

    # ────────── 报告 ──────────

    def report(self) -> Dict[str, Any]:
        """汇总所有阶段成 dict。"""
        return {
            "stages": [
                {
                    "name": s.name,
                    "strategy": s.strategy,
                    "max_evaluations": s.max_evaluations,
                    "elapsed_seconds": s.elapsed_seconds,
                    "started_at": s.started_at,
                    "result": s.result.to_dict() if s.result else None,
                }
                for s in self._stages
            ],
            "final": self._last_result.to_dict() if self._last_result else None,
            "total_elapsed_seconds": time.time() - self._t_start,
        }

    def report_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.report(), f, allow_unicode=True,
                           sort_keys=False, indent=2)

    @property
    def last_result(self) -> Optional[CalibrationResult]:
        return self._last_result

    @property
    def stages(self) -> List[PipelineStage]:
        return list(self._stages)


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
