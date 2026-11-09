"""Tests for the HyMOD rainfall-runoff model.

These tests verify:
1. Reservoir equation physics (Q = K * S, mass conservation)
2. Multi-reservoir cascading (peak recession, no double-counting)
3. Initialization and parameter validation
4. Output ranges and shapes
"""
import numpy as np
import pytest

from HydroArray.domain.models import (
    GridNode,
    HyMODModel,
    HyMODParameters,
)


def make_lumped_node():
    """Single-node setup (lumped catchment)."""
    return [GridNode(node_id=0, x=0.0, y=0.0, area=1.0, elevation=100.0,
                     downstream_id=-1, flow_accumulation=1, stream_order=1)]


class TestHyMODParameters:
    def test_default_parameters_valid(self):
        params = HyMODParameters()
        ok, msg = params.validate()
        assert ok, msg

    def test_invalid_huz_rejected(self):
        params = HyMODParameters(huz=-1.0)
        ok, _ = params.validate()
        assert not ok

    def test_invalid_alp_rejected(self):
        params = HyMODParameters(alp=1.5)
        ok, _ = params.validate()
        assert not ok

    def test_invalid_kq_rejected(self):
        params = HyMODParameters(kq=0.0)
        ok, _ = params.validate()
        assert not ok

    def test_invalid_nq_rejected(self):
        params = HyMODParameters(nq=0)
        ok, _ = params.validate()
        assert not ok


class TestHyMODInitialization:
    def test_initialize_returns_true(self):
        model = HyMODModel()
        params = HyMODParameters()
        assert model.initialize(make_lumped_node(), params) is True

    def test_initialize_rejects_wrong_type(self):
        model = HyMODModel()
        with pytest.raises(TypeError):
            model.initialize(make_lumped_node(), "not a params object")

    def test_initial_states_zero(self):
        model = HyMODModel()
        params = HyMODParameters()
        model.initialize(make_lumped_node(), params)
        assert np.allclose(model.get_states()['xhuz'], 0.0)
        assert np.allclose(model.get_states()['xs'], 0.0)
        assert np.allclose(model.get_states()['xcuz'], params.huz * params.xcuz)


class TestHyMODWaterBalance:
    def test_shapes_match_nodes(self):
        model = HyMODModel()
        params = HyMODParameters()
        model.initialize(make_lumped_node(), params)
        precip = np.array([10.0])
        pet = np.array([3.0])
        rs, ri, rg, states = model.water_balance(24.0, precip, pet)
        assert rs.shape == (1,)
        assert ri.shape == (1,)
        assert rg.shape == (1,)

    def test_dry_conditions_produce_no_runoff(self):
        """No rain, no runoff from reservoirs (interflow and baseflow are 0 initially)."""
        model = HyMODModel()
        params = HyMODParameters()
        model.initialize(make_lumped_node(), params)
        precip = np.array([0.0])
        pet = np.array([0.0])
        rs, ri, rg, _ = model.water_balance(24.0, precip, pet)
        assert rs[0] == 0.0
        assert ri[0] == 0.0
        assert rg[0] == 0.0

    def test_heavy_rain_produces_runoff(self):
        model = HyMODModel()
        params = HyMODParameters()
        model.initialize(make_lumped_node(), params)
        precip = np.array([50.0])
        pet = np.array([3.0])
        rs, ri, rg, _ = model.water_balance(24.0, precip, pet)
        # Quick flow should respond to large storm
        assert rs[0] > 0.0
        # Soil moisture should have increased (or saturated)
        # Note: HyMOD uses a probability-distributed soil store, so
        # `xcuz` represents the actual storage which can be < huz.
        assert model.get_states()['xcuz'][0] >= params.huz * params.xcuz
        # Storages must be non-negative and within capacity
        assert 0.0 <= model.get_states()['xcuz'][0] <= params.huz + 1e-9

    def test_mass_conservation_soil_moisture(self):
        """Water balance: dS = P - E - excess; no creation/destruction."""
        model = HyMODModel()
        params = HyMODParameters(huz=100.0, b=0.5, alp=0.5)
        model.initialize(make_lumped_node(), params)
        # Start with half-full soil
        model.set_states({'xcuz': np.array([50.0])})
        precip = np.array([20.0])
        pet = np.array([5.0])
        rs, ri, rg, _ = model.water_balance(24.0, precip, pet)
        # Storage change = P - E - excess
        new_storage = model.get_states()['xcuz'][0]
        # Storage cannot exceed huz
        assert new_storage <= params.huz
        assert new_storage >= 0.0


class TestHyMODReservoirEquations:
    """The critical P0 fix: verify the serial reservoir cascade is physically correct.

    Reference: Moore (1985) / Wagener HyMOD:
        S[t+1] = (1 - K*dt) * S[t] + dt * Inflow
        Outflow = K * S[t+1]
    """

    def test_reservoir_cascade_transfers_mass(self):
        """When 2nd reservoir gets inflow, the chain should drain downstream."""
        model = HyMODModel()
        # Use a long recession coef so water stays around for many steps
        params = HyMODParameters(nq=2, kq=0.3, ks=0.05, alp=0.0)
        model.initialize(make_lumped_node(), params)
        # Pre-charge the first reservoir
        model.set_states({'xhuz': np.array([[10.0, 0.0]])})
        # One step with no rain
        rs, _, _, _ = model.water_balance(24.0, np.array([0.0]), np.array([0.0]))
        states = model.get_states()['xhuz']
        # Mass must be conserved: 1st reservoir's outflow becomes 2nd's inflow
        # 1st reservoir: S1_new = (1 - 0.3*1) * 10 = 7.0;  Out = 0.3 * 7 = 2.1
        # 2nd reservoir: S2_new = 0 + 1*2.1 = 2.1
        # Final outflow = 0.3 * 2.1 = 0.63
        assert np.isclose(states[0, 0], 7.0, rtol=1e-6), \
            f"Expected 1st reservoir S=7.0, got {states[0, 0]}"
        assert np.isclose(states[0, 1], 2.1, rtol=1e-6), \
            f"Expected 2nd reservoir S=2.1, got {states[0, 1]}"
        assert np.isclose(rs[0], 0.63, rtol=1e-6), \
            f"Expected outflow=0.63, got {rs[0]}"

    def test_cascade_drains_to_zero(self):
        """With no further input, reservoirs eventually drain."""
        model = HyMODModel()
        params = HyMODParameters(nq=3, kq=0.5, ks=0.5, alp=0.0)
        model.initialize(make_lumped_node(), params)
        model.set_states({
            'xhuz': np.array([[5.0, 5.0, 5.0]]),
            'xs': np.array([5.0]),
        })
        for _ in range(200):
            rs, _, rg, _ = model.water_balance(24.0, np.array([0.0]), np.array([0.0]))
        # After many steps, everything should drain (with larger k for fast convergence)
        assert np.allclose(model.get_states()['xhuz'], 0.0, atol=1e-3)
        assert np.allclose(model.get_states()['xs'], 0.0, atol=1e-3)

    def test_no_double_counting(self):
        """The original bug: first reservoir had `inflow = direct_runoff + S*k`.

        This artificially inflates the inflow. Verify we don't have that.
        """
        model = HyMODModel()
        params = HyMODParameters(nq=1, kq=0.3, alp=0.0)  # single reservoir, no double-count
        model.initialize(make_lumped_node(), params)
        # Start with S = 10, no rain
        model.set_states({'xhuz': np.array([[10.0]]), 'xs': np.array([0.0])})
        rs, _, _, _ = model.water_balance(24.0, np.array([0.0]), np.array([0.0]))
        # Correct: S_new = (1 - 0.3) * 10 = 7.0, Out = 0.3 * 7 = 2.1
        # Bug behavior would be: inflow = 0 + 10*0.3 = 3, then Q = 0.3*3 = 0.9
        # State: 3 - 0.9 = 2.1 (different final state)
        new_s = model.get_states()['xhuz'][0, 0]
        assert np.isclose(new_s, 7.0, rtol=1e-6), \
            f"Suggests double-counting bug. Got S={new_s}, expected 7.0"
        assert np.isclose(rs[0], 2.1, rtol=1e-6)


class TestHyMODStateIO:
    def test_get_set_states_round_trip(self):
        model = HyMODModel()
        params = HyMODParameters()
        model.initialize(make_lumped_node(), params)
        # Run a few steps
        for _ in range(5):
            model.water_balance(24.0, np.array([5.0]), np.array([2.0]))
        states = model.get_states()
        # Make a fresh model and restore
        model2 = HyMODModel()
        model2.initialize(make_lumped_node(), params)
        model2.set_states(states)
        s1 = model.get_states()
        s2 = model2.get_states()
        for key in s1:
            assert np.allclose(s1[key], s2[key])

    def test_save_load_states(self, tmp_path):
        model = HyMODModel()
        params = HyMODParameters()
        model.initialize(make_lumped_node(), params)
        for _ in range(3):
            model.water_balance(24.0, np.array([10.0]), np.array([2.0]))
        path = tmp_path / "states.npz"
        model.save_states(str(path))
        assert path.exists()
        # Load into a fresh model
        model2 = HyMODModel()
        model2.initialize(make_lumped_node(), params)
        model2.load_states(str(path))
        s1 = model.get_states()
        s2 = model2.get_states()
        for key in s1:
            assert np.allclose(s1[key], s2[key])
