"""
XinAnjiang (新安江) Model Implementation

Three-source water balance model commonly used in China for flood simulation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from HydroArray.domain.models.base import (
    GridNode,
    ModelParameters,
    WaterBalanceModel,
)
from HydroArray.config.parameters import register_parameters


@register_parameters("xinanjiang")
@dataclass
class XinAnjiangParameters(ModelParameters):
    """XinAnjiang model parameters.

    Attributes:
        K: Ratio of potential evapotranspiration to pan evaporation (EPC).
        B: Exponential parameter for soil moisture storage distribution curve.
        IMP: Impermeable area fraction [0-1].
        WM: Maximum tension water storage capacity (mm). Must equal WUM+WLM+WDM.
        WUM: Upper zone tension water capacity (mm).
        WLM: Lower zone tension water capacity (mm).
        WDM: Deep zone tension water capacity (mm).
        SM: Maximum free water storage capacity (mm).
        EX: Free water storage capacity distribution exponent (0, 3].
        KI: Interflow outflow coefficient from free water storage [0, 1].
        KG: Groundwater outflow coefficient from free water storage [0, 1].
            Must satisfy KI + KG <= 1.
        C: Evapotranspiration diffusion coefficient [0, 1]. In Zhao Renjun
            (1984) the same C plays all three roles: the lower-layer critical
            storage C*WLM, the lower-layer draw EL = C*(EP-EU) when WL is
            partially sufficient, and the deep-layer supply ED = C*(EP-EU) - EL
            when WL is insufficient.
        CS: Channel network storage recession coefficient (0, 1).
        L: Lag time (number of time steps) [0, 10].
        KE: Muskingum routing parameter K (hours); must be positive.
        xe: Muskingum routing parameter x [0, 0.5].
        CI: Interflow recession coefficient [0, 1). Applied to RI after
            free-water separation: QI[t] = CI*QI[t-1] + (1-CI)*RI[t].
        CG: Groundwater recession coefficient [0, 1). Applied to RG after
            free-water separation: QG[t] = CG*QG[t-1] + (1-CG)*RG[t].
            Standard XAJ uses CG close to 1 (e.g. 0.98-0.998) for slow
            groundwater recession.
    """
    K: float = 0.8
    B: float = 0.3
    IMP: float = 0.0
    WM: float = 120.0
    WUM: float = 20.0
    WLM: float = 60.0
    WDM: float = 40.0
    SM: float = 15.0
    KI: float = 0.4
    KG: float = 0.05
    EX: float = 1.5     # 流域自由水蓄水容量分布曲线指数
    C:  float = 0.18    # 蒸散发扩散系数(教材中同时担纲 EL 临界值、EL 供水、ED 供水)
    CS: float = 0.5     # 河网蓄水消退系数
    L:  float = 1.0     # 滞后时段数
    KE: float = 10.0    # Muskingum K (hours)
    xe: float = 0.2     # Muskingum x
    CI: float = 0.5     # 壤中流消退系数
    CG: float = 0.99    # 地下水消退系数

    def validate(self, dt_hours: Optional[float] = None) -> Tuple[bool, str]:
        """Validate XinAnjiang parameters.

        Args:
            dt_hours: Optional simulation time step in hours. When provided,
                Muskingum parameters KE and xe are additionally checked against
                the non-negative coefficient condition
                ``2*KE*xe <= dt_hours <= 2*KE*(1-xe)``. Without it, only
                positivity/range of KE/xe is verified.

        Returns:
            (is_valid, message).
        """
        eps = 1e-6

        if not 0 <= self.IMP <= 1:
            return False, f"IMP must be in [0, 1], got {self.IMP}"

        if self.WUM <= 0:
            return False, f"WUM must be positive, got {self.WUM}"
        if self.WLM < 0:
            return False, f"WLM must be non-negative, got {self.WLM}"
        if self.WDM < 0:
            return False, f"WDM must be non-negative, got {self.WDM}"
        if self.WM <= 0:
            return False, f"WM must be positive, got {self.WM}"

        # Three-layer consistency: WM = WUM + WLM + WDM must hold.
        three_layer_sum = self.WUM + self.WLM + self.WDM
        if abs(self.WM - three_layer_sum) > eps:
            return False, (
                f"WM must equal WUM + WLM + WDM "
                f"(WM={self.WM}, sum={three_layer_sum:.6f})"
            )

        if not 0 < self.K <= 2:
            return False, f"K must be in (0, 2], got {self.K}"
        if not 0 < self.B <= 2:
            return False, f"B must be in (0, 2], got {self.B}"

        if self.SM <= 0:
            return False, f"SM must be positive, got {self.SM}"
        if not 0 < self.EX <= 3:
            return False, f"EX must be in (0, 3], got {self.EX}"

        if not 0 <= self.KI <= 1:
            return False, f"KI must be in [0, 1], got {self.KI}"
        if not 0 <= self.KG <= 1:
            return False, f"KG must be in [0, 1], got {self.KG}"
        if self.KI + self.KG > 1:
            return False, (
                f"KI + KG must be <= 1 "
                f"(KI={self.KI}, KG={self.KG}, sum={self.KI + self.KG})"
            )

        if not 0 <= self.C <= 1:
            return False, f"C must be in [0, 1], got {self.C}"
        if not 0 < self.CS < 1:
            return False, f"CS must be in (0, 1), got {self.CS}"
        if not 0 <= self.L <= 10:
            return False, f"L must be in [0, 10], got {self.L}"

        if self.KE <= 0:
            return False, f"KE must be positive, got {self.KE}"
        if not 0 <= self.xe <= 0.5:
            return False, f"xe must be in [0, 0.5], got {self.xe}"

        if not 0 <= self.CI < 1:
            return False, f"CI must be in [0, 1), got {self.CI}"
        if not 0 <= self.CG < 1:
            return False, f"CG must be in [0, 1), got {self.CG}"

        # Muskingum non-negative coefficient condition (only if dt is known).
        if dt_hours is not None:
            lo = 2.0 * self.KE * self.xe
            hi = 2.0 * self.KE * (1.0 - self.xe)
            if not (lo - eps <= dt_hours <= hi + eps):
                return False, (
                    f"Muskingum parameters may violate non-negative coefficient "
                    f"condition: 2*KE*xe={lo:.4f} <= dt_hours={dt_hours} "
                    f"<= 2*KE*(1-xe)={hi:.4f}"
                )

        return True, "OK"


class XinAnjiangModel(WaterBalanceModel):
    """Three-Source XinAnjiang Model.

    The model consists of:
    1. Evapotranspiration (three-layer)
    2. Runoff generation (saturation excess)
    3. Three-source separation (surface, interflow, groundwater)
    """

    def __init__(self):
        self._nodes: Optional[list] = None
        self._params: Optional[XinAnjiangParameters] = None
        self._n_nodes: int = 0

        # State variables
        self._WU: Optional[np.ndarray] = None  # Upper tension water storage
        self._WL: Optional[np.ndarray] = None  # Lower tension water storage
        self._WD: Optional[np.ndarray] = None  # Deep tension water storage
        self._S: Optional[np.ndarray] = None   # Free water storage
        self._FR: Optional[np.ndarray] = None  # Previous runoff-producing area

    def initialize(self, nodes: list, parameters: ModelParameters) -> bool:
        if not isinstance(parameters, XinAnjiangParameters):
            raise TypeError(f"Expected XinAnjiangParameters, got {type(parameters)}")

        # Reject invalid parameters (e.g. WM<=0, IMP>1) at init time.
        self._validate_parameters(parameters)

        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)

        WUM, WLM, WDM, WM = self._effective_tension_capacities()
        SM = self._params.SM

        self._WU = np.full(self._n_nodes, WUM * 0.5)
        self._WL = np.full(self._n_nodes, WLM * 0.5)
        self._WD = np.full(self._n_nodes, WDM * 0.5)
        self._S = np.full(self._n_nodes, SM * 0.5)
        self._FR = np.ones(self._n_nodes)

        return True

    def _effective_tension_capacities(self) -> Tuple[float, float, float, float]:
        """Return internally consistent three-layer tension water capacities."""
        p = self._params
        WUM = max(float(p.WUM), 0.0)
        WLM = max(float(p.WLM), 0.0)
        WDM = max(float(p.WDM), 0.0)
        WM = max(float(p.WM), WUM + WLM + WDM)

        if WDM == 0.0 and WM > WUM + WLM:
            WDM = WM - WUM - WLM
        elif WUM + WLM + WDM > 0 and abs((WUM + WLM + WDM) - WM) > 1e-9:
            WM = WUM + WLM + WDM

        return WUM, WLM, WDM, WM

    @staticmethod
    def _fill_tension_storage(
        surplus: float,
        wu: float,
        wl: float,
        wd: float,
        wum: float,
        wlm: float,
        wdm: float,
    ) -> Tuple[float, float, float]:
        """Fill upper, lower, then deep tension storages with non-runoff rainfall."""
        add = min(max(wum - wu, 0.0), surplus)
        wu += add
        surplus -= add
        add = min(max(wlm - wl, 0.0), surplus)
        wl += add
        surplus -= add
        add = min(max(wdm - wd, 0.0), surplus)
        wd += add
        return wu, wl, wd

    @staticmethod
    def _evaporate_three_layer(wu, wl, wd, ep, p, C, WUM, WLM, WDM):
        """Three-layer evapotranspiration (Zhao Renjun 1984, four cases).

        Following the standard textbook four-case formulation:

        Case 1 — upper layer (WU + rainfall) sufficient:
            EU = EP,    EL = 0,      ED = 0

        Case 2 — lower layer full enough (WL >= C*WLM):
            EU = WU + P, EL = (EP - EU) * WL / WLM,  ED = 0

        Case 3 — lower layer partially sufficient (C*(EP-EU) <= WL < C*WLM):
            EU = WU + P, EL = C * (EP - EU),         ED = 0

        Case 4 — lower layer insufficient (WL < C*(EP-EU)):
            EU = WU + P, EL = WL,                    ED = C * (EP - EU) - EL

        Args:
            wu, wl, wd: current tension storage in upper/lower/deep layers.
            ep:         potential ET (already multiplied by K).
            p:          precipitation.
            C:          evapotranspiration diffusion coefficient [0, 1].
            WUM, WLM, WDM: max tension storage capacities.

        Returns:
            (eu, el, ed): evaporation from each layer.
        """
        if ep <= 0.0:
            return 0.0, 0.0, 0.0

        # wd, WDM are not read here: the four-case formula bounds ED
        # by C*remain (Case 4) and the caller clips WD when applying.
        _ = (wd, WDM)

        # Step 1: upper layer (WU + rainfall satisfies EP first)
        eu = min(ep, wu + p)
        remain = ep - eu
        if remain <= 0.0:
            return eu, 0.0, 0.0

        # Steps 2-4: lower + deep layer by textbook four cases.
        # When WLM == 0 there is no lower layer; fall through to Case 4 to
        # avoid division by zero (the user must still set wl = 0 in that
        # configuration, but we don't crash on WLM == 0 alone).
        if WLM > 0.0 and wl >= C * WLM:
            # Case 2: WL full enough — proportional draw, deep unused
            el = remain * (wl / WLM)
            el = min(el, wl)  # numerical safety; should not trigger here
            ed = 0.0
        elif WLM > 0.0 and C * remain <= wl:
            # Case 3: WL partially sufficient — C-controlled draw, deep unused
            el = C * remain
            el = min(el, wl)
            ed = 0.0
        else:
            # Case 4: WL insufficient (or WLM == 0) — drain WL fully,
            # deep takes the rest
            el = wl
            ed = C * remain - el
            if ed < 0.0:
                ed = 0.0  # numerical guard (e.g. C tiny and remain small)

        return eu, el, ed

    def water_balance(self, step_hours: float, precip: np.ndarray,
                     pet: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Compute water balance and runoff components.

        Args:
            step_hours: Time step in hours.
            precip: Precipitation array (mm), shape (n_nodes,).
            pet: Potential evapotranspiration array (mm), shape (n_nodes,).

        Returns:
            Tuple of (surface_runoff, interflow, baseflow, states_dict).
        """
        params = self._params
        n = self._n_nodes

        P = np.asarray(precip)
        E = np.asarray(pet)

        if P.shape != (n,):
            P = np.broadcast_to(P.mean(), n)
        if E.shape != (n,):
            E = np.broadcast_to(E.mean(), n)

        WUM, WLM, WDM, WM = self._effective_tension_capacities()
        WMM = WM * (1.0 + params.B)
        SM = float(params.SM)
        SMM = SM * (1.0 + params.EX)
        ki = max(float(params.KI), 0.0)
        kg = max(float(params.KG), 0.0)
        if ki + kg > 0.95:
            scale = 0.95 / (ki + kg)
            ki *= scale
            kg *= scale

        RS = np.zeros(n, dtype=float)
        RI = np.zeros(n, dtype=float)
        RG = np.zeros(n, dtype=float)
        PE = np.zeros(n, dtype=float)
        E_actual = np.zeros(n, dtype=float)
        R_total = np.zeros(n, dtype=float)

        for i in range(n):
            p_i = max(float(P[i]), 0.0)
            ep_i = max(float(E[i]) * float(params.K), 0.0)
            wu, wl, wd = float(self._WU[i]), float(self._WL[i]), float(self._WD[i])
            w_before = wu + wl + wd

            # Three-layer evaporation (Zhao Renjun 1984 formulation)
            eu, el, ed = self._evaporate_three_layer(
                wu, wl, wd, ep_i, p_i, params.C, WUM, WLM, WDM
            )

            E_actual[i] = eu + el + ed
            PE[i] = p_i - E_actual[i]

            if PE[i] <= 0.0:
                # Rainfall was insufficient to meet ET demand.
                # Layer-by-layer deduction limited by the available P
                # (deficit), following process/runoff.py's
                # saturation_excess_runoff(): each layer's *remaining*
                # demand after the P budget is satisfied is what is
                # actually deducted from the soil layer.
                deficit = p_i
                deduct = min(eu, deficit)
                eu -= deduct
                deficit -= deduct
                deduct = min(el, deficit)
                el -= deduct
                deficit -= deduct
                ed = max(0.0, ed - deficit)
                wu -= eu
                wl -= el
                wd = max(wd - ed, 0.0)
                R_pervious = 0.0
            else:
                w_ratio = np.clip(w_before / WM, 0.0, 1.0)
                a = WMM * (1.0 - (1.0 - w_ratio) ** (1.0 / (1.0 + params.B)))
                if a + PE[i] <= WMM:
                    R_pervious = (
                        PE[i] + w_before - WM
                        + WM * (1.0 - (PE[i] + a) / WMM) ** (1.0 + params.B)
                    )
                else:
                    R_pervious = PE[i] - (WM - w_before)
                R_pervious = float(np.clip(R_pervious, 0.0, PE[i]))
                wu, wl, wd = self._fill_tension_storage(
                    PE[i] - R_pervious, wu, wl, wd, WUM, WLM, WDM
                )

            R_i = R_pervious * (1.0 - params.IMP) + p_i * params.IMP
            R_total[i] = R_i

            s_prev = float(np.clip(self._S[i], 0.0, SM))
            if SM > 0.0:
                available = s_prev + R_i
                RS[i] = max(available - SM, 0.0)
                free_storage = min(available, SM)
                RI[i] = ki * free_storage
                RG[i] = kg * free_storage
                self._S[i] = np.clip(free_storage - RI[i] - RG[i], 0.0, SM)
                self._FR[i] = float(
                    np.clip(1.0 - (1.0 - self._S[i] / SMM) ** params.EX, 1e-6, 1.0)
                )
            else:
                RS[i] = R_i
                self._S[i] = 0.0
                self._FR[i] = 1.0

            self._WU[i], self._WL[i], self._WD[i] = wu, wl, wd

        extra_states = {
            'upper_tension_water': self._WU.copy(),
            'lower_tension_water': self._WL.copy(),
            'deep_tension_water': self._WD.copy(),
            'tension_water': self._WU + self._WL + self._WD,
            'free_water': self._S.copy(),
            'runoff_area_fraction': self._FR.copy(),
            'PE': PE.copy(),
            'evaporation': E_actual.copy(),
            'runoff_generated': R_total.copy(),
        }

        return RS, RI, RG, extra_states

    def get_states(self) -> Dict[str, np.ndarray]:
        return {
            'WU': self._WU.copy() if self._WU is not None else None,
            'WL': self._WL.copy() if self._WL is not None else None,
            'WD': self._WD.copy() if self._WD is not None else None,
            'W': (self._WU + self._WL + self._WD).copy() if self._WU is not None else None,
            'S': self._S.copy() if self._S is not None else None,
            'FR': self._FR.copy() if self._FR is not None else None,
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        if 'WU' in states and states['WU'] is not None:
            self._WU = states['WU'].copy()
        if 'WL' in states and states['WL'] is not None:
            self._WL = states['WL'].copy()
        if 'WD' in states and states['WD'] is not None:
            self._WD = states['WD'].copy()
        if 'S' in states and states['S'] is not None:
            self._S = states['S'].copy()
        if 'FR' in states and states['FR'] is not None:
            self._FR = states['FR'].copy()

    def run_step(self, step_hours: float, precip: np.ndarray,
                 pet: np.ndarray) -> tuple:
        """Single step simulation.

        Args:
            step_hours: Time step in hours.
            precip: Precipitation array (mm).
            pet: Potential evapotranspiration array (mm).

        Returns:
            Tuple of (discharge, states_dict).
        """
        RS, RI, RG, states = self.water_balance(step_hours, precip, pet)
        total_runoff = RS + RI + RG
        return total_runoff, states
