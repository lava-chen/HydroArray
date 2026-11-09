"""Tests for the SAC-SMA model — focused on validate() and mass balance."""
import numpy as np
import pytest

from HydroArray.domain.models import (
    GridNode,
    SACModel,
    SACParameters,
)


def make_lumped_node():
    return [GridNode(node_id=0, x=0.0, y=0.0, area=1.0, elevation=100.0,
                     downstream_id=-1, flow_accumulation=1, stream_order=1)]


class TestSACParameters:
    def test_defaults_valid(self):
        ok, _ = SACParameters().validate()
        assert ok

    def test_pctim_gt_adimp_rejected(self):
        """P0 fix: ADIMP must be >= PCTIM at the parameter level."""
        ok, _ = SACParameters(PCTIM=0.5, ADIMP=0.3).validate()
        assert not ok

    def test_invalid_pctim_rejected(self):
        ok, _ = SACParameters(PCTIM=-0.1).validate()
        assert not ok

    def test_invalid_uztwm_rejected(self):
        ok, _ = SACParameters(UZTWM=-1.0).validate()
        assert not ok


class TestSACValidateAtInit:
    def test_invalid_params_raise_value_error(self):
        """P0 fix: validation must fire at initialize(), not silently."""
        model = SACModel()
        bad = SACParameters(PCTIM=0.5, ADIMP=0.3)  # violates ADIMP >= PCTIM
        with pytest.raises(ValueError, match="validation failed"):
            model.initialize(make_lumped_node(), bad)


class TestSACWaterBalance:
    def test_shapes_match(self):
        model = SACModel()
        model.initialize(make_lumped_node(), SACParameters())
        rs, ri, rg, _ = model.water_balance(24.0, np.array([10.0]), np.array([2.0]))
        assert rs.shape == ri.shape == rg.shape == (1,)

    def test_dry_conditions(self):
        model = SACModel()
        model.initialize(make_lumped_node(), SACParameters())
        rs, ri, rg, _ = model.water_balance(24.0, np.array([0.0]), np.array([0.0]))
        # No rain, no ET, all flows should be small/zero
        assert rs[0] >= 0.0
        assert ri[0] >= 0.0
        assert rg[0] >= 0.0

    def test_storage_stays_in_bounds(self):
        model = SACModel()
        params = SACParameters()
        model.initialize(make_lumped_node(), params)
        for _ in range(20):
            model.water_balance(24.0, np.array([20.0]), np.array([3.0]))
        s = model.get_states()
        # All storages must be within their bounds
        assert (s['uztwc'] >= 0).all() and (s['uztwc'] <= params.UZTWM).all()
        assert (s['uzfwc'] >= 0).all() and (s['uzfwc'] <= params.UZFWM).all()
        assert (s['lztwc'] >= 0).all() and (s['lztwc'] <= params.LZTWM).all()
        assert (s['lzfpc'] >= 0).all() and (s['lzfpc'] <= params.LZFPM).all()
        assert (s['lzfsc'] >= 0).all() and (s['lzfsc'] <= params.LZFSM).all()
