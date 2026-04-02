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
        WM: Maximum tension water storage capacity (mm).
        WUM: Upper zone tension water capacity (mm).
        WLM: Lower zone tension water capacity (mm).
        WDM: Deep zone tension water capacity (mm).
        SM: Maximum free water storage capacity (mm).
        KI: Outflow coefficient for interflow from free water storage.
        KG: Outflow coefficient for groundwater from free water storage.
        KKX: Downward percolation coefficient for free water storage.
        KE: Muskingum routing parameter K (hours).
        xe: Muskingum routing parameter x [0-0.5].
    """
    K: float = 0.5
    B: float = 0.1
    IMP: float = 0.0
    WM: float = 100.0
    WUM: float = 20.0
    WLM: float = 80.0
    WDM: float = 0.0
    SM: float = 15.0
    KI: float = 0.4
    KG: float = 0.05
    KKX: float = 0.7
    KE: float = 10.0
    xe: float = 0.2

    def validate(self) -> Tuple[bool, str]:
        if not 0 <= self.IMP <= 1:
            return False, f"IMP must be in [0, 1], got {self.IMP}"
        if self.WM <= 0 or self.WUM <= 0 or self.WLM < 0:
            return False, "WM, WUM, WLM must be positive"
        if self.SM <= 0:
            return False, f"SM must be positive, got {self.SM}"
        if not 0 <= self.KI <= 1 or not 0 <= self.KG <= 1:
            return False, "KI and KG must be in [0, 1]"
        if not 0 <= self.xe <= 0.5:
            return False, f"xe must be in [0, 0.5], got {self.xe}"
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
        self._W: Optional[np.ndarray] = None  # Tension water storage
        self._S: Optional[np.ndarray] = None  # Free water storage
        self._a: Optional[np.ndarray] = None  # Antecedent moisture index

    def initialize(self, nodes: list, parameters: ModelParameters) -> bool:
        if not isinstance(parameters, XinAnjiangParameters):
            raise TypeError(f"Expected XinAnjiangParameters, got {type(parameters)}")

        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)

        WM = self._params.WM
        SM = self._params.SM

        self._W = np.full(self._n_nodes, WM * 0.5)
        self._S = np.full(self._n_nodes, SM * 0.5)
        self._a = np.full(self._n_nodes, 0.0)

        return True

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

        # Effective precipitation after evaporation
        PE = np.maximum(P - E * params.K, 0.0)

        # Tension water content before precipitation
        W_before = self._W.copy()

        # Update tension water storage with precipitation
        W_after = np.minimum(W_before + PE, params.WM)

        # Runoff from saturation excess
        WMM = params.WM * (1 + params.B)
        a_new = WMM * (1 - (1 - W_before / params.WM) ** (1 + params.B))
        a_new = np.clip(a_new, 0, WMM)

        R = np.zeros(n)
        for i in range(n):
            PE_i = PE[i]
            a_i = a_new[i]

            if PE_i > 0:
                if a_i + PE_i <= WMM:
                    R[i] = (PE_i + W_before[i] - params.WM +
                            params.WM * (1 - (PE_i + a_i) / WMM) ** (params.B + 1))
                else:
                    R[i] = max(PE_i - params.WM + W_before[i], 0)

                R[i] = max(R[i], 0)
                a_new[i] = min(a_i + PE_i, WMM)
            else:
                R[i] = 0

        # Impermeable area contribution
        R = R * (1 - params.IMP) + P * params.IMP

        # Free water storage and separation
        S_before = self._S.copy()
        SMM = params.SM * (1 + params.EX) if hasattr(params, 'EX') else params.SM * 1.5

        RI = params.KI * S_before  # Interflow
        RG = params.KG * S_before  # Groundwater
        RS = R - RI - RG  # Surface runoff

        RS = np.maximum(RS, 0)
        RI = np.maximum(RI, 0)
        RG = np.maximum(RG, 0)

        # Update state variables
        self._W = W_after
        self._S = S_before * (1 - params.KI - params.KG) + R
        self._S = np.clip(self._S, 0, params.SM)
        self._a = a_new

        extra_states = {
            'tension_water': self._W.copy(),
            'free_water': self._S.copy(),
            'a': self._a.copy(),
            'PE': PE.copy(),
        }

        return RS, RI, RG, extra_states

    def get_states(self) -> Dict[str, np.ndarray]:
        return {
            'W': self._W.copy() if self._W is not None else None,
            'S': self._S.copy() if self._S is not None else None,
            'a': self._a.copy() if self._a is not None else None,
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        if 'W' in states and states['W'] is not None:
            self._W = states['W'].copy()
        if 'S' in states and states['S'] is not None:
            self._S = states['S'].copy()
        if 'a' in states and states['a'] is not None:
            self._a = states['a'].copy()

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
