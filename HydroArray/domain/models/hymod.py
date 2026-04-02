"""
HyMOD (HYdrological MODel) Implementation

A simple conceptual rainfall-runoff model using probability-distributed soil moisture
and linear routing through quick/slow flow reservoirs.
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


@register_parameters("hymod")
@dataclass
class HyMODParameters(ModelParameters):
    """HyMOD model parameters.

    Attributes:
        huz: Maximum soil moisture storage in the upper zone (mm).
        b: Distribution parameter for soil moisture capacity [-].
        alp: Quick/slow flow distribution coefficient [0-1].
        nq: Number of quick flow reservoirs [-].
        kq: Quick flow reservoir recession coefficient [1/day].
        ks: Slow flow reservoir recession coefficient [1/day].
        xcuz: Initial soil moisture as fraction of huz [0-1].
    """
    huz: float = 100.0
    b: float = 0.5
    alp: float = 0.5
    nq: int = 3
    kq: float = 0.7
    ks: float = 0.05
    xcuz: float = 0.3

    def validate(self) -> Tuple[bool, str]:
        if self.huz <= 0:
            return False, f"huz must be positive, got {self.huz}"
        if self.b < 0:
            return False, f"b must be non-negative, got {self.b}"
        if not 0 <= self.alp <= 1:
            return False, f"alp must be in [0, 1], got {self.alp}"
        if self.nq < 1:
            return False, f"nq must be at least 1, got {self.nq}"
        if not 0 < self.kq <= 1 or not 0 < self.ks <= 1:
            return False, "kq and ks must be in (0, 1]"
        if not 0 <= self.xcuz <= 1:
            return False, f"xcuz must be in [0, 1], got {self.xcuz}"
        return True, "OK"


class HyMODModel(WaterBalanceModel):
    """HyMOD - Probability-Distributed Rainfall-Runoff Model.

    The model structure:
    1. Probability-distributed soil moisture storage (single bucket)
    2. Direct runoff from saturation excess
    3. Quick flow through N serial linear reservoirs
    4. Slow flow through single linear reservoir
    """

    def __init__(self):
        self._nodes: Optional[list] = None
        self._params: Optional[HyMODParameters] = None
        self._n_nodes: int = 0

        # State variables
        self._xcuz: Optional[np.ndarray] = None  # Upper zone soil moisture
        self._xhuz: Optional[np.ndarray] = None  # Quick flow reservoir states
        self._xs: Optional[np.ndarray] = None    # Slow flow reservoir state

    def initialize(self, nodes: list, parameters: ModelParameters) -> bool:
        if not isinstance(parameters, HyMODParameters):
            raise TypeError(f"Expected HyMODParameters, got {type(parameters)}")

        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)
        nq = parameters.nq

        self._xcuz = np.full(self._n_nodes, parameters.huz * parameters.xcuz)
        self._xhuz = np.zeros((self._n_nodes, nq))
        self._xs = np.zeros(self._n_nodes)

        return True

    def water_balance(self, step_hours: float, precip: np.ndarray,
                     pet: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Compute water balance for HyMOD.

        Args:
            step_hours: Time step in hours.
            precip: Precipitation array (mm), shape (n_nodes,).
            pet: Potential evapotranspiration array (mm), shape (n_nodes,).

        Returns:
            Tuple of (surface_runoff, interflow, baseflow, states_dict).
        """
        params = self._params
        n = self._n_nodes
        dt = step_hours / 24.0  # Convert to days

        P = np.asarray(precip)
        E = np.asarray(pet)

        if P.shape != (n,):
            P = np.broadcast_to(P.mean(), n)
        if E.shape != (n,):
            E = np.broadcast_to(E.mean(), n)

        PE = np.maximum(P - E, 0.0)

        # Soil moisture accounting
        xcuz_new = np.minimum(self._xcuz + PE, params.huz)
        actual_storage = (params.huz / (1 + params.b)) * (
            1 - (1 - xcuz_new / params.huz) ** (1 + params.b)
        )
        previous_storage = (params.huz / (1 + params.b)) * (
            1 - (1 - self._xcuz / params.huz) ** (1 + params.b)
        )
        excess = actual_storage - previous_storage
        excess = np.maximum(excess, 0.0)

        # Runoff separation
        # Direct runoff: fraction of excess going to quick flow
        direct_runoff = excess * params.alp

        # Groundwater recharge: fraction going to slow flow
        slow_recharge = excess * (1 - params.alp)

        # Update soil moisture
        self._xcuz = xcuz_new

        # Route through quick flow reservoirs (N serial linear reservoirs)
        qf = np.zeros(n)
        xhuz_new = np.zeros_like(self._xhuz)

        for i in range(params.nq):
            if i == 0:
                inflow = direct_runoff + self._xhuz[:, 0] * params.kq
            else:
                inflow = self._xhuz[:, i]

            outflow = params.kq * inflow
            xhuz_new[:, i] = inflow - outflow * dt
            xhuz_new[:, i] = np.maximum(xhuz_new[:, i], 0)

        qf = params.kq * self._xhuz[:, -1]
        self._xhuz = xhuz_new

        # Route through slow flow reservoir
        inflow_slow = slow_recharge + self._xs * params.ks
        outflow_slow = params.ks * inflow_slow
        self._xs = inflow_slow - outflow_slow * dt
        self._xs = np.maximum(self._xs, 0)

        # Outputs
        surface_runoff = qf  # Quick flow from serial reservoirs
        interflow = np.zeros(n)  # HyMOD doesn't explicitly have interflow
        baseflow = outflow_slow  # Slow flow reservoir outflow

        extra_states = {
            'soil_moisture': self._xcuz.copy(),
            'quick_flow_storage': self._xhuz.copy(),
            'slow_flow_storage': self._xs.copy(),
            'excess': excess.copy(),
        }

        return surface_runoff, interflow, baseflow, extra_states

    def get_states(self) -> Dict[str, np.ndarray]:
        return {
            'xcuz': self._xcuz.copy() if self._xcuz is not None else None,
            'xhuz': self._xhuz.copy() if self._xhuz is not None else None,
            'xs': self._xs.copy() if self._xs is not None else None,
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        if 'xcuz' in states and states['xcuz'] is not None:
            self._xcuz = states['xcuz'].copy()
        if 'xhuz' in states and states['xhuz'] is not None:
            self._xhuz = states['xhuz'].copy()
        if 'xs' in states and states['xs'] is not None:
            self._xs = states['xs'].copy()
