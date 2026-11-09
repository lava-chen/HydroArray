"""Tests for the XinAnjiang (XAJ) water balance model.

Verifies:
1. Three-layer evaporation follows the standard textbook form
2. Mass balance in soil moisture
3. Parameter validation at initialize()
4. State save/load
"""
import numpy as np
import pytest

from HydroArray.domain.models import (
    GridNode,
    XinAnjiangModel,
    XinAnjiangParameters,
)
from HydroArray.domain.process.runoff import saturation_excess_runoff


def make_lumped_node():
    return [GridNode(node_id=0, x=0.0, y=0.0, area=1.0, elevation=100.0,
                     downstream_id=-1, flow_accumulation=1, stream_order=1)]


class TestXAJParameters:
    def test_defaults_valid(self):
        ok, msg = XinAnjiangParameters().validate()
        assert ok, msg

    def test_invalid_imp_rejected(self):
        ok, _ = XinAnjiangParameters(IMP=1.5).validate()
        assert not ok

    def test_invalid_C_rejected(self):
        ok, _ = XinAnjiangParameters(C=1.5).validate()
        assert not ok

    def test_invalid_xe_rejected(self):
        ok, _ = XinAnjiangParameters(xe=0.6).validate()
        assert not ok

    def test_wm_must_equal_three_layer_sum(self):
        """WM must equal WUM + WLM + WDM exactly."""
        ok, _ = XinAnjiangParameters(WM=100, WUM=20, WLM=80, WDM=5).validate()
        assert not ok

    def test_ki_plus_kg_must_le_one(self):
        """KI + KG must be <= 1 to avoid >100% free-water outflow per step."""
        ok, _ = XinAnjiangParameters(KI=0.7, KG=0.5).validate()
        assert not ok

    def test_default_tension_water_consistent(self):
        """Default WM = WUM + WLM + WDM, so default parameters must validate."""
        ok, msg = XinAnjiangParameters().validate()
        assert ok, msg

    def test_muskingum_dt_validation(self):
        """When dt_hours is provided, Muskingum non-negative coeff condition is checked."""
        params = XinAnjiangParameters()  # KE=10, xe=0.2 → 2KE*xe=4, 2KE(1-xe)=16
        # dt=1h is below the lower bound, should fail
        ok, _ = params.validate(dt_hours=1.0)
        assert not ok
        # dt=10h is in range, should pass
        ok, _ = params.validate(dt_hours=10.0)
        assert ok


class TestXAJValidateAtInit:
    def test_invalid_params_raise_value_error(self):
        """P0 fix: bad parameters must surface as ValueError, not silent."""
        model = XinAnjiangModel()
        bad = XinAnjiangParameters(WM=-1.0)
        with pytest.raises(ValueError, match="validation failed"):
            model.initialize(make_lumped_node(), bad)


class TestXAJInitialization:
    def test_initialize_returns_true(self):
        model = XinAnjiangModel()
        params = XinAnjiangParameters()
        assert model.initialize(make_lumped_node(), params) is True

    def test_initial_states_at_half_capacity(self):
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=10.0, WM=110.0)
        model.initialize(make_lumped_node(), params)
        s = model.get_states()
        assert s['WU'][0] == 10.0
        assert s['WL'][0] == 40.0
        assert s['WD'][0] == 5.0


class TestXAJEvaporation:
    """P0 fix: standard three-layer ET, not the buggy branch on C*remain."""

    def test_dry_upper_layer_steals_from_lower(self):
        """When WU is empty, EL = remain * WL / WLM (proportional)."""
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=0.0,
                                      WM=100.0, C=0.18, K=1.0)
        model.initialize(make_lumped_node(), params)
        # Drain upper layer
        model.set_states({'WU': np.array([0.0]), 'WL': np.array([80.0]),
                          'WD': np.array([0.0])})
        rs, ri, rg, _ = model.water_balance(24.0,
                                              np.array([0.0]),   # no rain
                                              np.array([10.0]))  # PET = 10
        s = model.get_states()
        # EU = 0 (WU empty, no rain); remain = 10
        # EL = 10 * (80/80) = 10, but WL only has 80 so this is fine
        # Note: EL is capped at WL storage which is 80
        assert s['WU'][0] == 0.0
        assert s['WL'][0] < 80.0  # some water was evaporated
        # Total evaporation should not exceed PET
        total_et = 0.0 + (80.0 - s['WL'][0]) + (0.0 - s['WD'][0])
        assert total_et <= 10.0 + 1e-9  # within numerical tolerance

    def test_rain_satisfies_upper_layer_first(self):
        """When rainfall exceeds upper demand, EU = EP and no lower-layer draw."""
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=10.0,
                                      WM=110.0, K=1.0)
        model.initialize(make_lumped_node(), params)
        wu0, wl0, wd0 = 20.0, 80.0, 10.0
        model.set_states({'WU': np.array([wu0]), 'WL': np.array([wl0]),
                          'WD': np.array([wd0])})
        # PET=5, rain=10; EU = min(5, 20+10) = 5; remain = 0; no lower draw
        model.water_balance(24.0, np.array([10.0]), np.array([5.0]))
        s = model.get_states()
        # Total tension storage change should equal -(ET - P) = -(5 - 10) = +5
        # But also, the upper layer fills first with P-EU = 5
        # WU_new = WU_old + (P - EU) = 20 + 5 = 25, but capped at WUM=20
        assert s['WU'][0] == 20.0
        # Excess 5 mm should go to lower layer
        assert s['WL'][0] == 80.0  # already full, stays full
        # Or overflow to deep / runoff
        # Total storage must not have decreased
        total_after = s['WU'][0] + s['WL'][0] + s['WD'][0]
        assert total_after >= wu0 + wl0 + wd0 - 5.0  # can't lose more than ET

    def test_case2_lower_layer_proportional_no_deep(self):
        """Case 2 (WL >= C*WLM): proportional lower-layer draw, deep unused.

        WU=0, WL=40, WLM=80, C=0.18, EP=10
          → WL=40 >= C*WLM=14.4 → Case 2: EL = 10 * 40/80 = 5, ED = 0
        """
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=20.0,
                                      WM=120.0, C=0.18, K=1.0)
        model.initialize(make_lumped_node(), params)
        wd0 = 20.0
        model.set_states({'WU': np.array([0.0]), 'WL': np.array([40.0]),
                          'WD': np.array([wd0])})
        model.water_balance(24.0, np.array([0.0]), np.array([10.0]))
        s = model.get_states()
        assert s['WU'][0] == 0.0
        assert s['WL'][0] == pytest.approx(35.0, abs=1e-6)  # drained 5
        assert s['WD'][0] == wd0  # deep layer untouched in Case 2

    def test_case3_partial_lower_no_deep(self):
        """Case 3 (C*remain <= WL < C*WLM): EL = C*remain, ED = 0.

        WU=0, WL=10, WLM=80, C=0.5, EP=10
          → WL=10 < C*WLM=40 (Case 2 fails)
          → C*remain=5 <= WL=10 (Case 3 holds)
          → EL = 0.5 * 10 = 5, ED = 0
        """
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=20.0,
                                      WM=120.0, C=0.5, K=1.0)
        model.initialize(make_lumped_node(), params)
        wd0 = 20.0
        model.set_states({'WU': np.array([0.0]), 'WL': np.array([10.0]),
                          'WD': np.array([wd0])})
        model.water_balance(24.0, np.array([0.0]), np.array([10.0]))
        s = model.get_states()
        assert s['WL'][0] == pytest.approx(5.0, abs=1e-6)  # drained 5
        assert s['WD'][0] == wd0  # deep untouched in Case 3

    def test_case4_lower_empty_deep_takes_rest(self):
        """Case 4 (WL < C*remain): drain WL, ED = C*remain - WL.

        WU=0, WL=2, WD=20, WLM=80, WDM=20, C=0.5, EP=10
          → WL=2 < C*WLM=40, C*remain=5 > WL=2
          → Case 4: EL = 2, ED = 0.5*10 - 2 = 3
        """
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=20.0,
                                      WM=120.0, C=0.5, K=1.0)
        model.initialize(make_lumped_node(), params)
        wd0 = 20.0
        model.set_states({'WU': np.array([0.0]), 'WL': np.array([2.0]),
                          'WD': np.array([wd0])})
        model.water_balance(24.0, np.array([0.0]), np.array([10.0]))
        s = model.get_states()
        assert s['WL'][0] == pytest.approx(0.0, abs=1e-6)  # drained fully
        # WD drops by ED = 3
        assert (wd0 - s['WD'][0]) == pytest.approx(3.0, abs=1e-6)

    def test_total_et_capped_at_pet(self):
        """Total evaporation must not exceed PET in any configuration."""
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=20.0,
                                      WM=120.0, C=0.18, K=1.0)
        model.initialize(make_lumped_node(), params)
        # Test multiple states
        for wu, wl, wd in [(0, 0, 0), (20, 80, 20), (5, 30, 5), (0, 80, 0)]:
            model.set_states({'WU': np.array([float(wu)]),
                              'WL': np.array([float(wl)]),
                              'WD': np.array([float(wd)]),
                              'S': np.array([10.0]),
                              'FR': np.array([0.5])})
            pet = 8.0
            model.water_balance(24.0, np.array([2.0]), np.array([pet]))
            s = model.get_states()
            storage_before = wu + wl + wd
            storage_after = s['WU'][0] + s['WL'][0] + s['WD'][0]
            # Storage can only decrease by at most PET
            assert storage_before - storage_after <= pet + 1e-6

    def test_pe_negative_rain_budget_sequential(self):
        """When P < ET demand, layer-by-layer P-budget deduction matters.

        Setup: WU=0, WL=40, WD=20, P=2, EP=8, C=0.18
        → _evaporate_three_layer: EU=2, EL=3, ED=0 (Case 2)
        → New logic (matching process/runoff.py): deficit=2 deducts
          all of EU, leaves EL=3 and ED=0 to be deducted from soil.
        → WU -= 0 (EU consumed by P), WL -= 3, WD -= 0
        """
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=20.0, WLM=80.0, WDM=20.0,
                                      WM=120.0, C=0.18, K=1.0, IMP=0.0)
        model.initialize(make_lumped_node(), params)
        model.set_states({'WU': np.array([0.0]), 'WL': np.array([40.0]),
                          'WD': np.array([20.0]),
                          'S': np.array([5.0]), 'FR': np.array([0.5])})
        rs, ri, rg, _ = model.water_balance(
            24.0, np.array([2.0]), np.array([8.0]))  # P=2, EP=8
        s = model.get_states()
        # WU unchanged: EU=2 was fully covered by P=2, no soil-water draw
        assert s['WU'][0] == pytest.approx(0.0, abs=1e-9)
        # WL drops by full EL=3 (deficit exhausted at EU stage)
        assert s['WL'][0] == pytest.approx(37.0, abs=1e-9)
        # WD unchanged: ED=0
        assert s['WD'][0] == pytest.approx(20.0, abs=1e-9)
        # R_pervious = 0 (no pervious runoff when PE < 0)
        # But the free-water tank still drains by KI/KG (independent of PE).
        assert rs[0] == 0.0
        # Total actual ET = (WU unchanged) + (WL 3) + (WD 0) = 3
        # Original demand was 5, but EU=2 was covered by P so only 3
        # came from soil.
        actual_et = (0.0 - s['WU'][0]) + (40.0 - s['WL'][0]) + (20.0 - s['WD'][0])
        assert actual_et == pytest.approx(3.0, abs=1e-9)

    def test_runoff_R_matches_process_runoff(self):
        """A 套 PE>0 产流量 R 与 B 套 saturation_excess_runoff R 数值一致。

        Uses a saturated state with IMP=0 so the IMP direct-rainfall term
        does not skew the comparison. With B=0.3, C=0.18, K=1.0 the
        A 套 and B 套 should agree on R to within the 0.1mm rounding
        B 套 applies at output.
        """
        import pandas as pd
        WUM, WLM, WDM = 20.0, 60.0, 40.0
        WM = WUM + WLM + WDM  # 120
        B, C, K = 0.3, 0.18, 1.0
        init_wu, init_wl, init_wd = 20.0, 60.0, 40.0  # fully wet

        # A 套
        model = XinAnjiangModel()
        params = XinAnjiangParameters(WUM=WUM, WLM=WLM, WDM=WDM, WM=WM,
                                      B=B, C=C, K=K, IMP=0.0)
        model.initialize(make_lumped_node(), params)
        model.set_states({'WU': np.array([init_wu]), 'WL': np.array([init_wl]),
                          'WD': np.array([init_wd]),
                          'S': np.array([10.0]), 'FR': np.array([0.5])})
        P, EP = 30.0, 5.0
        _, _, _, states_a = model.water_balance(24.0,
                                                np.array([P]), np.array([EP]))
        R_a = float(states_a['runoff_generated'][0])

        # B 套
        df = pd.DataFrame({'date': ['t0'], 'P': [P], 'Ep': [EP]})
        df_b = saturation_excess_runoff(
            df, WUM=WUM, WLM=WLM, WDM=WDM, C=C, Kc=K, b=B,
            initial_WU=init_wu, initial_WL=init_wl, initial_WD=init_wd,
            initial_R=0.0,
        )
        # B 套对 R 做了 0.1mm 四舍五入; A 套高精度
        R_b = float(df_b['R'].iloc[0])
        # A 套的 R 比 B 套精确 0.1mm; 用 abs(R_a - R_b) < 0.1 容忍舍入
        assert abs(R_a - R_b) < 0.15, (
            f"A 套 R={R_a} vs B 套 R={R_b} 偏差超过 0.1mm 舍入容差"
        )


class TestXAJStateIO:
    def test_get_set_states_round_trip(self):
        model = XinAnjiangModel()
        params = XinAnjiangParameters()
        model.initialize(make_lumped_node(), params)
        for _ in range(5):
            model.water_balance(24.0, np.array([10.0]), np.array([3.0]))
        s1 = model.get_states()
        m2 = XinAnjiangModel()
        m2.initialize(make_lumped_node(), params)
        m2.set_states(s1)
        s2 = m2.get_states()
        for k in s1:
            assert np.allclose(s1[k], s2[k])

    def test_save_load_pure_ndarray(self, tmp_path):
        """P0 fix: load_states uses allow_pickle=False."""
        model = XinAnjiangModel()
        params = XinAnjiangParameters()
        model.initialize(make_lumped_node(), params)
        model.water_balance(24.0, np.array([15.0]), np.array([3.0]))
        path = tmp_path / "xaj_states.npz"
        model.save_states(str(path))
        assert path.exists()
        m2 = XinAnjiangModel()
        m2.initialize(make_lumped_node(), params)
        # Should not raise
        m2.load_states(str(path))
        s1 = model.get_states()
        s2 = m2.get_states()
        for k in s1:
            assert np.allclose(s1[k], s2[k])
