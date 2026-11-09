"""
Calibrator: 高层率定器。

职责:
1. 从 ref 表 + overrides 构造 ParameterBounds
2. 构造归一化 + 目标函数
3. 调度 strategy
4. 维护 warm_start(基于历史最优附近精修)
5. 维护 checkpoint(序列化为 YAML,支持断点续算)
6. 返回 CalibrationResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import time

import numpy as np

from HydroArray.calibration.objective import (
    OBJECTIVE_PRESETS,
    build_objective,
    get_weights_from_objective,
)
from HydroArray.calibration.normalizer import Normalizer
from HydroArray.calibration.result import CalibrationResult
from HydroArray.calibration.strategies import run_strategy
from HydroArray.config.parameters import bounds_from_ref_table
from HydroArray.domain.calibration import ParameterBounds


# ─────────────────── 配置 ───────────────────


@dataclass
class CalibratorConfig:
    """Calibrator 的所有可调参数。"""
    strategy: str = "multi_start_lbfgsb"
    objective: Union[str, Mapping[str, float], Callable] = "nse"
    max_evaluations: int = 200
    n_starts: int = 4
    first_budget_share: float = 0.4
    pop_size: int = 30
    seed: int = 42
    n_jobs: int = 1
    normalize: bool = True
    verbose: bool = False


# ─────────────────── 主类 ───────────────────


class Calibrator:
    """高层率定器。

    Example::

        cal = Calibrator(
            run_func=my_model_runner,   # params -> metrics dict
            ref={"K": "0.8~1.2", "WM": 120},
            overrides={"KE": (1.0, 72.0)},
            strategy="ga_then_lbfgsb",
            max_evaluations=600,
        )
        result = cal.run()           # -> CalibrationResult
        result.to_yaml("calib.yaml") # 可序列化
        cal.warm_start(result.parameters)  # 续算
    """

    def __init__(
        self,
        run_func: Callable[[Dict[str, float]], Dict[str, float]],
        ref: Optional[Mapping[str, Any]] = None,
        overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
        *,
        bounds: Optional[List[ParameterBounds]] = None,
        name_alias: Optional[Mapping[str, str]] = None,
        config: Optional[CalibratorConfig] = None,
        **overrides_kwargs: Any,
    ):
        """构造。

        Args:
            run_func: ``(params dict) -> metrics dict`` 至少含 'nse' (或 objective 用到的项)
            ref: ref 表(可读 yaml 展平 dict)
            overrides: yaml 里没有/无法解析的项
            bounds: 直接传 ParameterBounds 列表(不走 ref 表)
            name_alias: yaml 名 → 字段名映射,默认 ``{"IM": "IMP"}``
            config: CalibratorConfig 实例,或 None
            **overrides_kwargs: ``strategy="lbfgsb"`` 等,等价于 ``config.strategy``
        """
        if run_func is None or not callable(run_func):
            raise TypeError("run_func must be callable")
        if (ref is None) == (bounds is None):
            raise ValueError("Pass exactly one of ref= or bounds=")

        self._run_func = run_func
        self._ref = dict(ref) if ref else None
        self._overrides = dict(overrides) if overrides else None
        self._name_alias = name_alias

        # 构造 bounds
        if bounds is not None:
            self._bounds = list(bounds)
        else:
            self._bounds = bounds_from_ref_table(
                self._ref,
                overrides=self._overrides,
                name_alias=name_alias,
            )
        if not self._bounds:
            raise ValueError("No calibratable parameters derived from ref / overrides / bounds")

        # 构造 config
        if config is None:
            config = CalibratorConfig()
        for k, v in overrides_kwargs.items():
            if not hasattr(config, k):
                raise TypeError(f"Unknown CalibratorConfig field: {k!r}")
            setattr(config, k, v)
        self._config = config

        # 归一化
        self._norm = Normalizer(self._bounds)

        # warm_start 状态
        self._warm_x: Optional[np.ndarray] = None

    # ────────── 属性 ──────────

    @property
    def bounds(self) -> List[ParameterBounds]:
        return list(self._bounds)

    @property
    def config(self) -> CalibratorConfig:
        return self._config

    @property
    def parameter_names(self) -> List[str]:
        return self._norm.names

    # ────────── 状态管理 ──────────

    def warm_start(
        self, params: Optional[Dict[str, float]] = None,
        *, shrink: float = 0.0,
    ) -> "Calibrator":
        """设定 warm-start 起点。下次 run() 会在该点附近起步。

        Args:
            params: 已知最优参数字典;None 表示用 self._warm_x
            shrink: 0=不动, 1=完全在归一化空间向 0.5 收缩(只对 L-BFGS-B 起点有效)

        Returns:
            self
        """
        if params is not None:
            self._warm_x = np.array(
                [float(params.get(n, (self._bounds[i].min_value +
                                      self._bounds[i].max_value) / 2))
                 for i, n in enumerate(self._norm.names)],
                dtype=float,
            )
        if self._warm_x is None:
            return self
        # 截到 [lo, hi]
        self._warm_x = np.clip(
            self._warm_x, self._norm.lo, self._norm.hi,
        )
        return self

    def reset(self) -> "Calibrator":
        """清除 warm_start。"""
        self._warm_x = None
        return self

    # ────────── 核心:run ──────────

    def run(self) -> CalibrationResult:
        cfg = self._config

        # objective 桥接:物理参数 dict -> metrics dict -> score
        def obj_metric(phys: dict) -> dict:
            return self._run_func(phys)

        if callable(cfg.objective) and not isinstance(cfg.objective, (str, Mapping)):
            # 自定义可调用 objective: 它吃 phys 字典直接返回 score
            spec = cfg.objective
            def score_func(phys: dict) -> float:
                return float(spec(phys))
            weights_for_name: Dict[str, float] = {}
        else:
            weights_for_name = _resolve_weights(cfg.objective)
            def score_func(metrics: dict) -> float:
                return _score(metrics, weights_for_name)
        t0 = time.time()

        # 构造 strategy 输入
        results = run_strategy(
            cfg.strategy,
            obj_metric=obj_metric,
            score_func=score_func,
            norm=self._norm,
            max_evaluations=cfg.max_evaluations,
            n_starts=cfg.n_starts,
            first_budget_share=cfg.first_budget_share,
            pop_size=cfg.pop_size,
            seed=cfg.seed,
            verbose=cfg.verbose,
        )

        # 选最优
        if not results:
            raise RuntimeError(f"Strategy {cfg.strategy!r} returned no results")
        best_idx = int(np.argmax([r[1] for r in results]))
        best_x, best_score, _ = results[best_idx]

        # warm_start 处理
        if self._warm_x is not None:
            warm_phys = self._norm.physical_to_dict(self._warm_x)
            warm_score = score_func(warm_phys)
            if warm_score > best_score:
                if cfg.verbose:
                    print(f"  [warm_start] beats strategy: "
                          f"{warm_score:.4f} > {best_score:.4f}")
                best_x = self._warm_x
                best_score = warm_score
                results.append((best_x, best_score, 1))
                best_idx = len(results) - 1

        best_params = self._norm.physical_to_dict(best_x)
        final_metrics = obj_metric(best_params)

        per_start = [r[1] for r in results]
        n_evals = sum(int(r[2]) for r in results) + (
            1 if self._warm_x is not None else 0
        )

        return CalibrationResult(
            parameters=best_params,
            objective_value=best_score,
            metrics=final_metrics,
            objective_name=_objective_name_repr(cfg.objective, weights_for_name),
            strategy=cfg.strategy,
            n_evaluations=n_evals,
            elapsed_seconds=time.time() - t0,
            converged=True,
            message=f"strategy={cfg.strategy}, best_start={best_idx}",
            bounds_used=self._bounds,
            per_start_score=per_start,
            best_start_index=best_idx,
            extras={"per_start_evals": [int(r[2]) for r in results]},
        )

    # ────────── 续算 ──────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 仅保存可序列化的状态
        state = {
            "warm_x": self._warm_x.tolist() if self._warm_x is not None else None,
            "config": _config_to_dict(self._config),
            "ref": self._ref,
            "overrides": (dict(self._overrides)
                          if self._overrides else None),
            "name_alias": (dict(self._name_alias)
                           if self._name_alias else None),
        }
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(state, f, allow_unicode=True,
                               sort_keys=False, indent=2)
        except ImportError:
            import json
            # json 不支持 tuple-as-dict-key,转字符串
            state["overrides"] = (
                {k: list(v) for k, v in (self._overrides or {}).items()}
                if self._overrides else None
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, path: Union[str, Path]) -> "Calibrator":
        path = Path(path)
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                state = yaml.safe_load(f)
        except ImportError:
            import json
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

        if state.get("warm_x") is not None:
            self.warm_start(dict(zip(self._norm.names, state["warm_x"])))
        # config 不覆盖,只用作参考
        return self


# ─────────────────── 内部工具 ───────────────────


def _resolve_weights(spec) -> Dict[str, float]:
    """把 spec 解成权重 dict(可调用对象不直接支持, 走 str/Mapping 路径)。"""
    if isinstance(spec, str):
        return dict(OBJECTIVE_PRESETS.get(spec, {"nse": 1.0}))
    if isinstance(spec, Mapping):
        return dict(spec)
    if callable(spec):
        # 可调用 spec: 尝试取 .weights 属性(由 build_objective 注入)
        w = getattr(spec, "weights", None) or getattr(spec.__dict__, "get", lambda *_: None)("weights", None)
        if w:
            return dict(w)
        return {"__callable__": 1.0}
    return {"nse": 1.0}


def _score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """加权汇总。weight 为负则鼓励 metric 接近 0;callable 路径直接调。"""
    if "__callable__" in weights and callable(weights.get("__callable__")):
        return 0.0  # fallback,actually never reaches here
    total = 0.0
    for name, w in weights.items():
        v = metrics.get(name)
        if v is None or (isinstance(v, float) and v != v):  # NaN
            return -1e18
        total += float(w) * float(v)
    return total


def _to_legacy_objective_name(spec) -> str:
    """ScipyOptimizer 内部要 "nse"/"rmse"/"kge"/"multi" 之一。"""
    if isinstance(spec, str):
        return spec if spec in ("nse", "rmse", "kge", "multi") else "nse"
    if isinstance(spec, Mapping):
        # 看 dominant 权重对应的指标
        items = sorted(spec.items(), key=lambda kv: -abs(kv[1]))
        if items:
            name = items[0][0]
            if name in ("nse", "rmse", "kge"):
                return name
        return "nse"
    return "nse"


def _objective_name_repr(spec, weights: Dict[str, float]) -> str:
    """生成 CalibrationResult.objective_name 的可读字符串。"""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        return "{" + ", ".join(f"{k}={v:+.3g}" for k, v in spec.items()) + "}"
    if callable(spec):
        return getattr(spec, "__name__", "custom")
    return "nse"


def _config_to_dict(cfg: CalibratorConfig) -> dict:
    """序列化 config。objective 走 repr 路径。"""
    obj = cfg.objective
    if isinstance(obj, (str, Mapping)):
        obj_repr = obj if isinstance(obj, str) else dict(obj)
    else:
        obj_repr = getattr(obj, "__name__", "custom")
    return {
        "strategy": cfg.strategy,
        "objective": obj_repr,
        "max_evaluations": cfg.max_evaluations,
        "n_starts": cfg.n_starts,
        "first_budget_share": cfg.first_budget_share,
        "pop_size": cfg.pop_size,
        "seed": cfg.seed,
        "n_jobs": cfg.n_jobs,
        "normalize": cfg.normalize,
        "verbose": cfg.verbose,
    }
