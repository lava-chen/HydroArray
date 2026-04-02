"""
Data Assimilation Module

Provides ensemble-based data assimilation methods for updating hydrological model states
with real-time or delayed observations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


@dataclass
class AssimilationResult:
    """Result from a data assimilation step."""
    state: np.ndarray
    uncertainty: np.ndarray
    innovation: Optional[np.ndarray] = None
    kalman_gain: Optional[np.ndarray] = None
    assimilation_success: bool = True
    message: str = ""


class DataAssimilator(ABC):
    """Abstract base class for data assimilation methods."""

    def __init__(self, state_dim: int, obs_dim: int):
        """Initialize assimilator.

        Args:
            state_dim: Dimension of the model state vector.
            obs_dim: Dimension of the observation vector.
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    @abstractmethod
    def assimilate(self, state: np.ndarray, observation: np.ndarray,
                   obs_error_std: float, model_error_std: float = 0.1) -> AssimilationResult:
        """Assimilate an observation into the model state.

        Args:
            state: Current model state vector.
            observation: Observation vector.
            obs_error_std: Standard deviation of observation error.
            model_error_std: Standard deviation of model error.

        Returns:
            AssimilationResult with updated state and uncertainty.
        """
        pass


class EnsembleKalmanFilter(DataAssimilator):
    """Ensemble Kalman Filter (EnKF) for hydrological model state updating.

    The EnKF uses an ensemble of model states to estimate the state covariance,
    avoiding the need to compute the full model Jacobian.
    """

    def __init__(self, state_dim: int, obs_dim: int, n_ensemble: int = 50):
        super().__init__(state_dim, obs_dim)
        self.n_ensemble = n_ensemble
        self.ensemble: Optional[np.ndarray] = None  # (n_ensemble, state_dim)

    def initialize_ensemble(self, state_mean: np.ndarray, state_std: float = 1.0):
        """Initialize ensemble around a mean state.

        Args:
            state_mean: Mean state vector.
            state_std: Standard deviation for ensemble spread.
        """
        self.ensemble = np.zeros((self.n_ensemble, self.state_dim))
        for i in range(self.n_ensemble):
            self.ensemble[i] = state_mean + np.random.normal(0, state_std, self.state_dim)

    def assimilate(self, state: np.ndarray, observation: np.ndarray,
                   obs_error_std: float, model_error_std: float = 0.1) -> AssimilationResult:
        """Perform EnKF assimilation step.

        Args:
            state: Current model state (used to update ensemble).
            observation: Observation vector.
            obs_error_std: Standard deviation of observation error.
            model_error_std: Fraction of state to add as model error.

        Returns:
            AssimilationResult with updated ensemble mean and spread.
        """
        if self.ensemble is None:
            self.initialize_ensemble(state)

        # Add model error to ensemble
        for i in range(self.n_ensemble):
            self.ensemble[i] += np.random.normal(0, model_error_std * np.abs(self.ensemble[i]))

        # Compute ensemble mean and covariance
        state_mean = np.mean(self.ensemble, axis=0)
        state_cov = np.cov(self.ensemble, rowvar=False)

        # Ensure minimum covariance (numerical stability)
        state_cov = np.maximum(state_cov, np.eye(self.state_dim) * 1e-6)

        # Observation operator (identity for direct state observation)
        H = np.eye(self.obs_dim, self.state_dim)

        # Observation error covariance
        R = np.eye(self.obs_dim) * (obs_error_std ** 2)

        # Kalman gain: G = P * H^T * (H * P * H^T + R)^-1
        PH = state_cov @ H.T
        HPH_R = H @ PH + R

        try:
            K = PH @ np.linalg.inv(HPH_R)
        except np.linalg.LinAlgError:
            K = PH @ np.linalg.pinv(HPH_R)

        # Update ensemble
        for i in range(self.n_ensemble):
            # Perturb observation for each ensemble member
            obs_perturbed = observation + np.random.normal(0, obs_error_std, self.obs_dim)
            innovation = obs_perturbed - H @ self.ensemble[i]
            self.ensemble[i] = self.ensemble[i] + K @ innovation

        # Updated state statistics
        updated_mean = np.mean(self.ensemble, axis=0)
        updated_std = np.std(self.ensemble, axis=0)

        return AssimilationResult(
            state=updated_mean,
            uncertainty=updated_std,
            innovation=observation - H @ state_mean,
            kalman_gain=K,
            assimilation_success=True,
            message="EnKF assimilation completed"
        )

    def get_ensemble(self) -> np.ndarray:
        """Get the current ensemble."""
        return self.ensemble


class ParticleFilter(DataAssimilator):
    """Sequential Importance Resampling (SIR) Particle Filter.

    A particle-based data assimilation method that can handle non-linear
    and non-Gaussian state-space models.
    """

    def __init__(self, state_dim: int, obs_dim: int, n_particles: int = 100):
        super().__init__(state_dim, obs_dim)
        self.n_particles = n_particles
        self.particles: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def initialize_particles(self, state_mean: np.ndarray, state_std: float = 1.0):
        """Initialize particle set around a mean state.

        Args:
            state_mean: Mean state vector.
            state_std: Standard deviation for particle spread.
        """
        self.particles = np.zeros((self.n_particles, self.state_dim))
        for i in range(self.n_particles):
            self.particles[i] = state_mean + np.random.normal(0, state_std, self.state_dim)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def assimilate(self, state: np.ndarray, observation: np.ndarray,
                   obs_error_std: float, model_error_std: float = 0.1) -> AssimilationResult:
        """Perform SIR particle filter assimilation step.

        Args:
            state: Current model state (used to initialize if needed).
            observation: Observation vector.
            obs_error_std: Standard deviation of observation error.
            model_error_std: Fraction of state to add as model error.

        Returns:
            AssimilationResult with updated particle mean and spread.
        """
        if self.particles is None:
            self.initialize_particles(state)

        # Predict: add model error to particles
        for i in range(self.n_particles):
            self.particles[i] += np.random.normal(0, model_error_std * np.abs(self.particles[i]))

        # Update: compute likelihood of observation for each particle
        likelihood = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            # Assume observation is first element of state (simplified)
            predicted_obs = self.particles[i, :self.obs_dim]
            likelihood[i] = np.exp(-0.5 * ((observation - predicted_obs) / obs_error_std) ** 2)

        # Normalize weights
        self.weights = self.weights * likelihood
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
        else:
            # Reset to uniform if all weights zero
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Compute effective sample size
        n_eff = 1.0 / np.sum(self.weights ** 2)

        # Resample if effective sample size is too small
        if n_eff < self.n_particles / 2:
            self._resample()

        # Updated state statistics
        updated_mean = np.average(self.particles, axis=0, weights=self.weights)
        updated_std = np.sqrt(np.average((self.particles - updated_mean) ** 2, axis=0, weights=self.weights))

        return AssimilationResult(
            state=updated_mean,
            uncertainty=updated_std,
            assimilation_success=True,
            message=f"PF assimilation completed, N_eff = {n_eff:.1f}"
        )

    def _resample(self):
        """Perform systematic resampling."""
        cumsum = np.cumsum(self.weights)
        u0 = np.random.uniform(0, 1.0 / self.n_particles)

        new_indices = np.zeros(self.n_particles, dtype=int)
        j = 0
        for i in range(self.n_particles):
            u = u0 + i / self.n_particles
            while cumsum[j] < u and j < self.n_particles - 1:
                j += 1
            new_indices[i] = j

        self.particles = self.particles[new_indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def get_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current particles and weights."""
        return self.particles, self.weights


class OfflineDA:
    """Framework for offline (batch) data assimilation.

    Supports running a hydrological model with intermittent state updates
    from observations.
    """

    def __init__(self, model, assimilator: DataAssimilator,
                 state_mapper: Optional[Callable] = None):
        """Initialize offline DA system.

        Args:
            model: HydrologyModel to run.
            assimilator: DataAssimilator instance.
            state_mapper: Function to extract DA state from model (optional).
        """
        self.model = model
        self.assimilator = assimilator
        self.state_mapper = state_mapper or (lambda m: m.get_states()['W'])

        self.state_history: List[np.ndarray] = []
        self.obs_history: List[np.ndarray] = []
        self.assimilation_times: List[int] = []

    def run(self, inputs, observations, obs_times, obs_error_std=1.0,
            model_error_std=0.1) -> Dict[str, np.ndarray]:
        """Run model with data assimilation.

        Args:
            inputs: Model inputs (precip, PET, etc.).
            observations: Array of observation values.
            obs_times: Time indices where observations are available.
            obs_error_std: Observation error standard deviation.
            model_error_std: Model error standard deviation.

        Returns:
            Dictionary with 'states', 'simulated', 'assimilation_times'.
        """
        n_steps = len(inputs)
        simulated = []
        current_state = None

        obs_idx = 0

        for t in range(n_steps):
            # Run model step
            if current_state is not None:
                self.model.set_states({'W': current_state})

            precip_t = inputs[t]['precip'] if isinstance(inputs[t], dict) else inputs[t]
            pet_t = inputs[t]['pet'] if isinstance(inputs[t], dict) else np.zeros_like(precip_t)

            RS, RI, RG, states = self.model.water_balance(1.0, precip_t, pet_t)
            current_state = states.get('tension_water', states.get('soil_moisture', states.get('W')))

            q_sim = RS.sum() + RI.sum() + RG.sum()
            simulated.append(q_sim)

            # Check if observation available at this time
            if obs_idx < len(obs_times) and t == obs_times[obs_idx]:
                obs = observations[obs_idx]
                da_state = self.state_mapper(self.model)

                result = self.assimilator.assimilate(
                    da_state, np.array([obs]), obs_error_std, model_error_std
                )

                # Update model state
                new_state = result.state
                if len(new_state) == len(current_state):
                    current_state = new_state

                self.assimilation_times.append(t)
                self.obs_history.append(obs)
                obs_idx += 1

            self.state_history.append(current_state.copy() if current_state is not None else None)

        return {
            'states': np.array([s for s in self.state_history if s is not None]),
            'simulated': np.array(simulated),
            'assimilation_times': self.assimilation_times,
        }


def create_assimilator(method: str, state_dim: int, obs_dim: int,
                       **kwargs) -> DataAssimilator:
    """Factory function to create data assimilators.

    Args:
        method: Assimilation method ('enkf' or 'pf').
        state_dim: State dimension.
        obs_dim: Observation dimension.
        **kwargs: Additional arguments for specific assimilators.

    Returns:
        DataAssimilator instance.
    """
    if method.lower() == 'enkf':
        return EnsembleKalmanFilter(state_dim, obs_dim, kwargs.get('n_ensemble', 50))
    elif method.lower() == 'pf':
        return ParticleFilter(state_dim, obs_dim, kwargs.get('n_particles', 100))
    else:
        raise ValueError(f"Unknown assimilation method: {method}")
