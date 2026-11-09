"""
Tests for HydroArray.calibration module.

覆盖:
- Calibrator 5 个 strategy 收敛
- Normalizer 互转正确
- bounds_from_ref_table 解析 ref + overrides
- CalibrationResult YAML 序列化往返
- Pipeline calibrate → refine → validate
- 预置 / 权重 / 异常路径
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import HydroArray as ha
from HydroArray.calibration import (
    Calibrator,
    CalibratorConfig,
    CalibratorPreset,
    CalibrationPipeline,
    CalibrationResult,
    Normalizer,
    OBJECTIVE_PRESETS,
    build_objective,
    lhs_starts,
    make_calibrator_from_preset,
)
from HydroArray.domain.calibration import ParameterBounds


# ────────── 测试用目标 ──────────


def sphere_obj(params):
    """Sphere function: min at (0, 0)."""
    x, y = params["x"], params["y"]
    f = x ** 2 + y ** 2
    return {"nse": -f, "rmse": float(f), "kge": 1.0 / (1.0 + f)}


def test_sphere_obj_is_dict():
    out = sphere_obj({"x": 0.0, "y": 0.0})
    assert isinstance(out, dict)
    assert "nse" in out
    assert out["nse"] == 0.0


# ────────── Normalizer ──────────


class TestNormalizer:
    def test_roundtrip(self):
        norm = Normalizer([
            ParameterBounds("K", 0.5, 1.2),
            ParameterBounds("WM", 80, 200),
        ])
        x = np.array([0.85, 120.0])
        u = norm.to_unit(x)
        # K=0.85 → (0.85-0.5)/(1.2-0.5) = 0.5
        # WM=120 → (120-80)/(200-80) = 1/3
        np.testing.assert_allclose(u, [0.5, 1.0 / 3.0])
        x_back = norm.to_physical(u)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_clipping(self):
        norm = Normalizer([ParameterBounds("K", 0.5, 1.2)])
        x = norm.to_physical(np.array([-1.0]))  # 出界
        assert 0.5 <= x[0] <= 1.2

    def test_physical_to_dict(self):
        norm = Normalizer([ParameterBounds("K", 0.5, 1.2)])
        d = norm.physical_to_dict(np.array([0.85]))
        assert d == {"K": 0.85}

    def test_zero_span_fallback(self):
        # lo==hi 时 span=0 不会除零
        norm = Normalizer([ParameterBounds("K", 1.0, 1.0)])
        u = norm.to_unit(np.array([1.0]))
        assert np.isfinite(u[0])


class TestLhsStarts:
    def test_shape(self):
        rng = np.random.default_rng(42)
        starts = lhs_starts(4, 3, rng)
        assert starts.shape == (4, 3)
        assert ((starts >= 0) & (starts <= 1)).all()


# ────────── Calibrator 5 strategy ──────────


class TestCalibratorStrategies:
    REF = {"x": "-2~2", "y": "-2~2"}

    def test_lbfgsb(self):
        cal = Calibrator(
            run_func=sphere_obj, ref=self.REF,
            config=CalibratorConfig(
                strategy="lbfgsb", max_evaluations=50, n_starts=1,
            ),
        )
        res = cal.run()
        assert res.nse >= -0.1
        assert abs(res.parameters["x"]) < 0.1
        assert abs(res.parameters["y"]) < 0.1

    def test_multi_start_lbfgsb(self):
        cal = Calibrator(
            run_func=sphere_obj, ref=self.REF,
            config=CalibratorConfig(
                strategy="multi_start_lbfgsb", max_evaluations=120, n_starts=3,
            ),
        )
        res = cal.run()
        assert res.nse >= -0.1

    def test_ga(self):
        cal = Calibrator(
            run_func=sphere_obj, ref=self.REF,
            config=CalibratorConfig(
                strategy="ga", max_evaluations=120, pop_size=20,
            ),
        )
        res = cal.run()
        # GA 收敛稍慢
        assert res.nse >= -0.2

    def test_ga_then_lbfgsb(self):
        cal = Calibrator(
            run_func=sphere_obj, ref=self.REF,
            config=CalibratorConfig(
                strategy="ga_then_lbfgsb", max_evaluations=120, pop_size=15,
            ),
        )
        res = cal.run()
        assert res.nse >= -0.1

    def test_unknown_strategy_raises(self):
        cal = Calibrator(
            run_func=sphere_obj, ref=self.REF,
            config=CalibratorConfig(strategy="bogus", max_evaluations=10),
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            cal.run()


# ────────── Objective ──────────


class TestObjective:
    def test_presets_exist(self):
        assert set(OBJECTIVE_PRESETS.keys()) >= {
            "nse", "kge", "rmse", "pbias", "multi",
        }

    def test_build_objective_nse(self):
        def mf(p):
            return {"nse": 0.5, "rmse": 3.0, "kge": 0.6}
        obj = build_objective("nse", metric_func=mf)
        score = obj({"x": 1.0})
        assert abs(score - 0.5) < 1e-9

    def test_build_objective_dict(self):
        def mf(p):
            return {"nse": 0.5, "volume_err_pct": 10.0}
        obj = build_objective(
            {"nse": 1.0, "volume_err_pct": -0.1}, metric_func=mf,
        )
        # 0.5 - 0.1*10 = -0.5
        assert abs(obj({"x": 1.0}) - (-0.5)) < 1e-9

    def test_build_objective_callable(self):
        def custom(p):
            return p["x"] ** 2
        obj = build_objective(custom)
        assert abs(obj({"x": 3.0}) - 9.0) < 1e-9

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            build_objective("nope", metric_func=lambda p: {})


# ────────── CalibrationResult 序列化 ──────────


class TestCalibrationResult:
    def _make(self):
        return CalibrationResult(
            parameters={"K": 0.85, "WM": 120.0},
            objective_value=0.85,
            metrics={"nse": 0.85, "rmse": 3.0, "kge": 0.86,
                     "r_squared": 0.85, "pbias_pct": -1.5,
                     "volume_err_pct": -1.5, "peak_err_pct": -10.0,
                     "peak_time_lag": 0},
            objective_name="nse",
            strategy="multi_start_lbfgsb",
            n_evaluations=200,
            elapsed_seconds=12.3,
            converged=True,
            message="ok",
            bounds_used=[ParameterBounds("K", 0.5, 1.2),
                          ParameterBounds("WM", 80, 200)],
            per_start_score=[0.85, 0.83],
            best_start_index=0,
        )

    def test_to_dict(self):
        d = self._make().to_dict()
        assert d["parameters"]["K"] == 0.85
        assert d["objective_name"] == "nse"
        assert len(d["bounds_used"]) == 2

    def test_yaml_roundtrip(self):
        res = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.yaml"
            res.to_yaml(p)
            loaded = CalibrationResult.from_yaml(p)
        assert loaded.parameters == res.parameters
        assert loaded.objective_name == res.objective_name
        assert loaded.metrics == res.metrics
        assert len(loaded.bounds_used) == len(res.bounds_used)
        assert loaded.bounds_used[0].name == "K"

    def test_summary(self):
        s = self._make().summary()
        assert "CalibrationResult" in s
        assert "nse" in s
        assert "0.8500" in s

    def test_repr(self):
        r = repr(self._make())
        assert "CalibrationResult" in r
        assert "multi_start_lbfgsb" in r

    def test_yaml_without_pyyaml_raises(self):
        # 若 yaml 不存在,from_yaml 应抛 ImportError
        from HydroArray.calibration import result as res_mod
        if res_mod._YAML:
            pytest.skip("PyYAML available, can't test missing-yaml path")
        with pytest.raises(ImportError):
            CalibrationResult.from_yaml("dummy.yaml")


# ────────── Pipeline ──────────


class TestPipeline:
    REF = {"x": "-2~2", "y": "-2~2"}

    def test_calibrate_then_refine(self):
        pipe = CalibrationPipeline(
            run_func=sphere_obj, ref=self.REF, verbose=False,
        )
        pipe.calibrate(strategy="multi_start_lbfgsb",
                        max_evaluations=80, n_starts=3)
        pipe.refine(shrink=0.5, max_evaluations=60)
        assert pipe.last_result.nse >= -0.1
        assert len(pipe.stages) == 2

    def test_validate(self):
        pipe = CalibrationPipeline(
            run_func=sphere_obj, ref=self.REF, verbose=False,
        )
        pipe.calibrate(strategy="lbfgsb", max_evaluations=30)
        v = pipe.validate()
        assert "nse" in v.metrics

    def test_report(self):
        pipe = CalibrationPipeline(
            run_func=sphere_obj, ref=self.REF, verbose=False,
        )
        pipe.calibrate(strategy="lbfgsb", max_evaluations=20)
        r = pipe.report()
        assert "stages" in r
        assert "final" in r
        assert r["final"]["parameters"]["x"] == pytest.approx(0.0, abs=0.1)

    def test_refine_before_calibrate_raises(self):
        pipe = CalibrationPipeline(run_func=sphere_obj, ref=self.REF)
        with pytest.raises(RuntimeError, match="calibrate"):
            pipe.refine()

    def test_invalid_shrink_raises(self):
        pipe = CalibrationPipeline(run_func=sphere_obj, ref=self.REF)
        pipe.calibrate(strategy="lbfgsb", max_evaluations=20)
        with pytest.raises(ValueError, match="shrink"):
            pipe.refine(shrink=1.5)


# ────────── Presets ──────────


class TestPresets:
    def test_known_presets_exist(self):
        assert "xinanjiang_daily" in ha.list_presets()
        assert "quick_demo" in ha.list_presets()

    def test_make_calibrator_from_preset(self):
        cal = make_calibrator_from_preset(
            "quick_demo", run_func=sphere_obj,
            ref={"x": "-2~2", "y": "-2~2"},
        )
        assert isinstance(cal, Calibrator)
        assert cal.config.strategy == "lbfgsb"

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            make_calibrator_from_preset(
                "nope", run_func=sphere_obj, ref={},
            )


# ────────── Calibrator 错误路径 ──────────


class TestCalibratorErrors:
    def test_no_ref_no_bounds_raises(self):
        with pytest.raises(ValueError, match="ref= or bounds="):
            Calibrator(run_func=sphere_obj)

    def test_both_ref_and_bounds_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            Calibrator(
                run_func=sphere_obj,
                ref={"x": "0~1"},
                bounds=[ParameterBounds("y", 0, 1)],
            )

    def test_non_callable_run_func_raises(self):
        with pytest.raises(TypeError):
            Calibrator(run_func=42, ref={"x": "0~1"})

    def test_warm_start_then_run(self):
        cal = Calibrator(
            run_func=sphere_obj, ref={"x": "-2~2", "y": "-2~2"},
            config=CalibratorConfig(strategy="lbfgsb", max_evaluations=30),
        )
        cal.warm_start({"x": 0.1, "y": 0.1})
        res = cal.run()
        assert res.nse >= -0.1

    def test_reset(self):
        cal = Calibrator(
            run_func=sphere_obj, ref={"x": "-2~2", "y": "-2~2"},
        )
        cal.warm_start({"x": 0.0, "y": 0.0})
        cal.reset()
        # reset 之后内部 _warm_x 是 None,不应再影响 run
        res = cal.run()
        assert res is not None


# ────────── 旧 API 兼容 ──────────


class TestLegacyCompat:
    def test_auto_calibrate_warns(self):
        with pytest.warns(DeprecationWarning):
            # 跑一个最简单的 obj,不传 obs
            res = ha.auto_calibrate(
                run_func=sphere_obj, ref={"x": "-2~2", "y": "-2~2"},
                max_evaluations=30,
            )
        assert isinstance(res, ha.AutoCalibrateResult)
        assert "x" in res.parameters

    def test_auto_calibrate_with_obs(self):
        # 假 obs: 全 0
        obs = np.zeros(10)
        def sim(p):
            return np.zeros(10)
        with pytest.warns(DeprecationWarning):
            res = ha.auto_calibrate(
                run_func=sim, obs=obs,
                ref={"x": "-2~2", "y": "-2~2"},
                max_evaluations=20,
            )
        # q_sim=0 与 obs=0 比,NSE 是 -inf 或 nan,但容许
        assert "x" in res.parameters


# ────────── 顶层 calibrate() facade ──────────


class TestFacade:
    def test_calibrate_with_run_func(self):
        res = ha.calibrate(
            run_func=sphere_obj,
            ref={"x": "-2~2", "y": "-2~2"},
            strategy="lbfgsb",
            max_evaluations=30,
        )
        assert isinstance(res, CalibrationResult)
        assert res.nse >= -0.1

    def test_calibrate_without_run_func_raises(self):
        with pytest.raises(ValueError, match="run_func="):
            ha.calibrate()
