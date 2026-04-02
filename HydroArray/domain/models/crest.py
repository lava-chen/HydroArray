"""
CREST (Coupled Routing and Excess Storage) Model Implementation

A distributed hydrological model that couples runoff generation and routing.
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


@register_parameters("crest")
@dataclass
class CRESTParameters(ModelParameters):
    """CREST model parameters.

    Attributes:
        WM: Soil water storage capacity (mm).
        B: Exponential parameter for soil moisture distribution [-].
        IM: Impermeable fraction [-].
        KE: Ratio of potential evapotranspiration to pan evaporation [-].
        FC: Maximum infiltration capacity (mm/day).
        K: Recession baseflow coefficient [-].
        TIME: Time delay for unit hydrograph (hours).
    """
    WM: float = 120.0
    B: float = 1.5
    IM: float = 0.05
    KE: float = 1.0
    FC: float = 20.0
    K: float = 0.1
    TIME: float = 1.0

    def validate(self) -> Tuple[bool, str]:
        if self.WM <= 0:
            return False, f"WM must be positive, got {self.WM}"
        if not 0 <= self.IM <= 1:
            return False, f"IM must be in [0, 1], got {self.IM}"
        if self.FC <= 0:
            return False, f"FC must be positive, got {self.FC}"
        return True, "OK"


class CRESTModel(WaterBalanceModel):
    """CREST - Coupled Routing and Excess Storage Model.

    The model computes:
    1. Surface runoff using XinAnjiang-style excess storage
    2. Interflow from soil layer
    3. Baseflow recession
    4. Linear routing through watershed
    """

    def __init__(self):
        self._nodes: Optional[list] = None
        self._params: Optional[CRESTParameters] = None
        self._n_nodes: int = 0

        self._SM: Optional[np.ndarray] = None  # Soil moisture storage
        self._excess_overland: Optional[np.ndarray] = None
        self._excess_interflow: Optional[np.ndarray] = None

    def initialize(self, nodes: list, parameters: ModelParameters) -> bool:
        if not isinstance(parameters, CRESTParameters):
            raise TypeError(f"Expected CRESTParameters, got {type(parameters)}")

        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)

        self._SM = np.full(self._n_nodes, parameters.WM * 0.5)
        self._excess_overland = np.zeros(self._n_nodes)
        self._excess_interflow = np.zeros(self._n_nodes)

        return True

    def water_balance(self, step_hours: float, precip: np.ndarray,
                     pet: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Compute water balance for CREST.

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

        E *= params.KE

        # Initialize outputs
        surface_runoff = np.zeros(n)
        interflow = np.zeros(n)
        baseflow = np.zeros(n)

        for i in range(n):
            p_i = P[i]
            e_i = E[i]
            sm_i = self._SM[i]

            # Evapotranspiration
            et = min(e_i, sm_i)
            sm_i -= et
            sm_i = max(sm_i, 0)

            # Surface excess (runoff generation)
            wmm = params.WM * (1 + params.B)
            c = wmm * (1 - (1 - sm_i / params.WM) ** (1 + params.B))
            c = max(c, 0)

            excess = 0.0
            if p_i > 0:
                # Add precipitation
                sm_test = sm_i + p_i
                if sm_test >= wmm:
                    excess = sm_test - params.WM
                    sm_i = params.WM
                else:
                    # Compute new c
                    c_new = wmm * (1 - (1 - sm_test / params.WM) ** (1 + params.B))
                    c_new = max(c_new, 0)
                    excess = max(p_i - (c_new - c), 0)
                    sm_i = sm_test

            # Split excess into overland and interflow
            # Based on soil moisture state
            soil_fraction = sm_i / params.WM if params.WM > 0 else 0

            overland_fraction = params.IM + (1 - params.IM) * (1 - soil_fraction)
            interflow_fraction = (1 - params.IM) * soil_fraction

            surface_runoff[i] = excess * overland_fraction
            interflow[i] = excess * interflow_fraction

            # Update soil moisture
            self._SM[i] = sm_i

            # Baseflow from recession
            baseflow[i] = params.K * sm_i * (step_hours / 24.0)
            self._SM[i] -= baseflow[i]
            self._SM[i] = max(self._SM[i], 0)

        extra_states = {
            'soil_moisture': self._SM.copy(),
            'excess_overland': self._excess_overland.copy(),
            'excess_interflow': self._excess_interflow.copy(),
        }

        return surface_runoff, interflow, baseflow, extra_states

    def get_states(self) -> Dict[str, np.ndarray]:
        return {
            'SM': self._SM.copy() if self._SM is not None else None,
            'excess_overland': self._excess_overland.copy() if self._excess_overland is not None else None,
            'excess_interflow': self._excess_interflow.copy() if self._excess_interflow is not None else None,
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        if 'SM' in states and states['SM'] is not None:
            self._SM = states['SM'].copy()
        if 'excess_overland' in states and states['excess_overland'] is not None:
            self._excess_overland = states['excess_overland'].copy()
        if 'excess_interflow' in states and states['excess_interflow'] is not None:
            self._excess_interflow = states['excess_interflow'].copy()
