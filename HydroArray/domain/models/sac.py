"""
SAC-SMA (Sacramento Soil Moisture Accounting) Model Implementation

A two-layer soil moisture accounting model widely used in operational hydrology.
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


@register_parameters("sac")
@dataclass
class SACParameters(ModelParameters):
    """SAC-SMA model parameters.

    Upper Zone Parameters:
        UZTWM: Upper zone tension water maximum capacity (mm).
        UZFWM: Upper zone free water maximum capacity (mm).
        UZK: Upper zone free water lateral outflow rate [1/day].
        PCTIM: Fraction of impervious area [-].

    Lower Zone Parameters:
        LZTWM: Lower zone tension water maximum capacity (mm).
        LZFPM: Lower zone free water primary maximum capacity (mm).
        LZFPM: Lower zone free water supplementary maximum capacity (mm).
        LZSK: Lower zone free water primary outflow rate [1/day].
        LZPK: Lower zone free water supplementary outflow rate [1/day].
        LZFSM: Fraction of lower zone free water going to supplementary [-].

    Direct Runoff:
        ADIMP: Additional impervious area fraction [-].

    ZPerc:
        ZPERC: Percolation parameter controlling lower zone filling rate [-].
        REXP: Percolation exponent controlling lower zone tension water fill [-].
    """
    UZTWM: float = 50.0
    UZFWM: float = 25.0
    UZK: float = 0.30
    PCTIM: float = 0.0

    LZTWM: float = 100.0
    LZFPM: float = 100.0
    LZFSM: float = 100.0
    LZSK: float = 0.05
    LZPK: float = 0.10
    LZFAC: float = 1.0

    ADIMP: float = 0.0
    ZPERC: float = 50.0
    REXP: float = 1.5

    def validate(self) -> Tuple[bool, str]:
        for name, val in [('UZTWM', self.UZTWM), ('UZFWM', self.UZFWM),
                          ('LZTWM', self.LZTWM), ('LZFPM', self.LZFPM)]:
            if val <= 0:
                return False, f"{name} must be positive, got {val}"
        if not 0 <= self.PCTIM <= 1 or not 0 <= self.ADIMP <= 1:
            return False, "PCTIM and ADIMP must be in [0, 1]"
        if self.ADIMP < self.PCTIM:
            return False, "ADIMP must be >= PCTIM"
        return True, "OK"


class SACModel(WaterBalanceModel):
    """SAC-SMA - Sacramento Soil Moisture Accounting Model.

    Soil Structure:
    - Upper Zone:
      - Tension Water (UZTW): bound water, released only to evapotranspiration
      - Free Water (UZFW): can outflow laterally as interflow or percolate down

    - Lower Zone:
      - Tension Water (LZTW): similar to upper tension water
      - Free Primary Water (LZFP): primary groundwater
      - Free Supplementary Water (LZFS): supplementary groundwater
    """

    def __init__(self):
        self._nodes: Optional[list] = None
        self._params: Optional[SACParameters] = None
        self._n_nodes: int = 0

        # State variables
        self._uztwc: Optional[np.ndarray] = None  # Upper zone tension water content
        self._uzfwc: Optional[np.ndarray] = None  # Upper zone free water content
        self._lztwc: Optional[np.ndarray] = None  # Lower zone tension water content
        self._lzfsc: Optional[np.ndarray] = None  # Lower zone free supplementary content
        self._lzfpc: Optional[np.ndarray] = None  # Lower zone free primary content

    def initialize(self, nodes: list, parameters: ModelParameters) -> bool:
        if not isinstance(parameters, SACParameters):
            raise TypeError(f"Expected SACParameters, got {type(parameters)}")

        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)

        p = parameters

        # Initialize at 50% capacity
        self._uztwc = np.full(self._n_nodes, p.UZTWM * 0.5)
        self._uzfwc = np.full(self._n_nodes, p.UZFWM * 0.5)
        self._lztwc = np.full(self._n_nodes, p.LZTWM * 0.5)
        self._lzfsc = np.full(self._n_nodes, p.LZFSM * 0.5)
        self._lzfpc = np.full(self._n_nodes, p.LZFPM * 0.5)

        return True

    def water_balance(self, step_hours: float, precip: np.ndarray,
                     pet: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Compute water balance for SAC-SMA.

        Args:
            step_hours: Time step in hours.
            precip: Precipitation array (mm), shape (n_nodes,).
            pet: Potential evapotranspiration array (mm), shape (n_nodes,).

        Returns:
            Tuple of (surface_runoff, interflow, baseflow, states_dict).
        """
        params = self._params
        n = self._n_nodes
        dt = step_hours / 24.0

        P = np.asarray(precip)
        E = np.asarray(pet)

        if P.shape != (n,):
            P = np.broadcast_to(P.mean(), n)
        if E.shape != (n,):
            E = np.broadcast_to(E.mean(), n)

        # Initialize output
        surface_runoff = np.zeros(n)
        interflow = np.zeros(n)
        baseflow = np.zeros(n)

        for i in range(n):
            p_i = P[i]
            e_i = E[i]

            # Evapotranspiration from upper zone tension water
            e_used = min(e_i, self._uztwc[i])
            self._uztwc[i] -= e_used
            e_remain = e_i - e_used

            # If more ET needed, take from lower zone tension water
            if e_remain > 0:
                e_take = min(e_remain, self._lztwc[i])
                self._lztwc[i] -= e_take
                e_remain -= e_take

            # Add precipitation to upper zone
            p_add = p_i

            # Compute runoff from impervious area
            direct_runoff = p_add * params.PCTIM
            p_add *= (1 - params.PCTIM)

            # Tension water filling
            uz_td = params.UZTWM - self._uztwc[i]
            if p_add >= uz_td:
                self._uztwc[i] = params.UZTWM
                p_add -= uz_td
            else:
                self._uztwc[i] += p_add
                p_add = 0

            # Free water overflow to interflow
            uz_free_cap = params.UZFWM - self._uzfwc[i]
            interflow[i] = self._uzfwc[i] * params.UZK * dt
            self._uzfwc[i] -= interflow[i]
            self._uzfwc[i] = max(self._uzfwc[i], 0)

            # Percolation from upper to lower zone
            perc_rate = params.ZPERC * (1 - self._lztwc[i] / params.LZTWM) ** params.REXP
            perc_rate *= (self._uzfwc[i] / params.UZFWM)
            perc = min(perc_rate * dt, self._uzfwc[i])
            self._uzfwc[i] -= perc

            # Lower zone tension water filling
            lz_td = params.LZTWM - self._lztwc[i]
            if perc >= lz_td:
                self._lztwc[i] = params.LZTWM
                perc -= lz_td
            else:
                self._lztwc[i] += perc
                perc = 0

            # Lower zone free water outflow
            lzfp_out = self._lzfpc[i] * params.LZPK * dt
            lzfs_out = self._lzfsc[i] * params.LZSK * dt

            self._lzfpc[i] -= lzfp_out
            self._lzfpc[i] = max(self._lzfpc[i], 0)
            self._lzfsc[i] -= lzfs_out
            self._lzfsc[i] = max(self._lzfsc[i], 0)

            baseflow[i] = lzfp_out + lzfs_out

            # Surface runoff from remaining precipitation
            surface_runoff[i] = direct_runoff + p_add

        # Add ADIMP contribution
        adimp_fraction = params.ADIMP - params.PCTIM
        if adimp_fraction > 0:
            for i in range(n):
                # Additional impervious area generates runoff
                surface_runoff[i] += P[i] * adimp_fraction

        extra_states = {
            'uztwc': self._uztwc.copy(),
            'uzfwc': self._uzfwc.copy(),
            'lztwc': self._lztwc.copy(),
            'lzfsc': self._lzfsc.copy(),
            'lzfpc': self._lzfpc.copy(),
        }

        return surface_runoff, interflow, baseflow, extra_states

    def get_states(self) -> Dict[str, np.ndarray]:
        return {
            'uztwc': self._uztwc.copy() if self._uztwc is not None else None,
            'uzfwc': self._uzfwc.copy() if self._uzfwc is not None else None,
            'lztwc': self._lztwc.copy() if self._lztwc is not None else None,
            'lzfsc': self._lzfsc.copy() if self._lzfsc is not None else None,
            'lzfpc': self._lzfpc.copy() if self._lzfpc is not None else None,
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        for key in ['uztwc', 'uzfwc', 'lztwc', 'lzfsc', 'lzfpc']:
            if key in states and states[key] is not None:
                setattr(self, f'_{key}', states[key].copy())
