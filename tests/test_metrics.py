"""Tests for evaluation metrics."""
import numpy as np
import pytest

from HydroArray.utils.metrics import (
    calculate_nse,
    calculate_rmse,
    calculate_kge,
    calculate_pbias,
    calculate_r_squared,
    evaluate_model,
)
from HydroArray.analysis import nse, kge, rmse, pbias, r_squared


class TestNSE:
    def test_perfect_match(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert calculate_nse(x, x) == pytest.approx(1.0)

    def test_mean_prediction(self):
        """Predicting the mean gives NSE = 0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_pred = np.full_like(obs, obs.mean())
        assert calculate_nse(obs, mean_pred) == pytest.approx(0.0, abs=1e-10)

    def test_constant_observation_returns_nan(self):
        obs = np.full(10, 5.0)
        sim = np.linspace(0, 10, 10)
        # denominator = 0 -> NaN
        assert np.isnan(calculate_nse(obs, sim))

    def test_handles_nan(self):
        obs = np.array([1.0, np.nan, 3.0, 4.0])
        sim = np.array([1.1, 2.0, 2.9, 4.1])
        # Should ignore NaN, not crash
        result = calculate_nse(obs, sim)
        assert not np.isnan(result)

    def test_empty_input(self):
        assert np.isnan(calculate_nse(np.array([]), np.array([])))


class TestRMSE:
    def test_perfect_match(self):
        x = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([2.0, 3.0, 4.0])
        # Each error is 1, so RMSE = 1
        assert calculate_rmse(obs, sim) == pytest.approx(1.0)


class TestKGE:
    def test_perfect_match(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_kge(x, x) == pytest.approx(1.0)

    def test_constant_observation(self):
        """KGE on constant obs should not crash; returns NaN gracefully."""
        obs = np.full(10, 5.0)
        sim = np.array([4.0, 4.5, 5.5, 6.0, 5.0, 4.0, 4.5, 5.5, 6.0, 5.0])
        result = calculate_kge(obs, sim)
        # r is NaN, so the final KGE should also be NaN
        assert np.isnan(result)


class TestPBIAS:
    def test_no_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        assert calculate_pbias(x, x) == pytest.approx(0.0)

    def test_positive_bias(self):
        """Overestimating gives positive PBIAS."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([2.0, 4.0, 6.0])  # doubled
        # PBIAS = 100 * (sum(sim-obs)/sum(obs)) = 100 * 6/6 = 100
        assert calculate_pbias(obs, sim) == pytest.approx(100.0)

    def test_zero_observation_returns_nan(self):
        obs = np.zeros(10)
        sim = np.ones(10)
        # PBIAS denominator is 0 -> NaN
        assert np.isnan(calculate_pbias(obs, sim))


class TestEvaluateModel:
    def test_returns_all_metrics(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        sim = np.array([1.1, 1.9, 3.2, 3.8])
        result = evaluate_model(obs, sim, metrics=('nse', 'rmse', 'kge', 'pbias'))
        assert set(result.keys()) == {'nse', 'rmse', 'kge', 'pbias'}
        for v in result.values():
            assert not np.isnan(v)


class TestAnalysisModuleConsistency:
    """Verify analysis.statistics and utils.metrics give equivalent results."""

    def test_nse_agreement(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.5, 1.8, 3.2, 3.9, 5.1])
        # Both should give the same value
        from HydroArray.utils.metrics import calculate_nse
        assert nse(sim, obs) == pytest.approx(calculate_nse(obs, sim))

    def test_pbias_agreement(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        from HydroArray.utils.metrics import calculate_pbias
        assert pbias(sim, obs) == pytest.approx(calculate_pbias(obs, sim))

    def test_analysis_module_zero_observation_behavior(self):
        """Both modules should agree on NaN for zero-observation case."""
        obs = np.zeros(5)
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        from HydroArray.utils.metrics import calculate_pbias
        # Both should return NaN, not inf
        assert np.isnan(pbias(sim, obs))
        assert np.isnan(calculate_pbias(obs, sim))
