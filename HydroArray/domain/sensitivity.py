"""
Sensitivity Analysis Module

Provides global and local sensitivity analysis methods for hydrological model parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""
    parameter: str
    first_order_index: float  # S1 (main effect)
    total_order_index: float  # ST (including interactions)
    interaction_index: float  # S2 = ST - S1
    confidence_first: Optional[float] = None
    confidence_total: Optional[float] = None


class SensitivityAnalyzer:
    """Base class for sensitivity analysis."""

    def __init__(self, model: Callable, parameters: List[str]):
        """Initialize sensitivity analyzer.

        Args:
            model: Function that takes parameter dict and returns scalar.
            parameters: List of parameter names to analyze.
        """
        self.model = model
        self.parameters = parameters
        self.n_params = len(parameters)

    def analyze(self, n_samples: int = 1000) -> List[SensitivityResult]:
        """Run sensitivity analysis.

        Args:
            n_samples: Number of model evaluations.

        Returns:
            List of SensitivityResult for each parameter.
        """
        raise NotImplementedError


class SobolAnalyzer(SensitivityAnalyzer):
    """Sobol' indices for global sensitivity analysis.

    Computes first-order (S1) and total-order (ST) sensitivity indices
    using Saltelli's sampling scheme.
    """

    def __init__(self, model: Callable, parameters: List[str],
                 bounds: Dict[str, Tuple[float, float]]):
        """Initialize Sobol analyzer.

        Args:
            model: Function that takes parameter dict and returns scalar.
            parameters: List of parameter names.
            bounds: Dictionary mapping parameter names to (min, max) bounds.
        """
        super().__init__(model, parameters)
        self.bounds = bounds

    def analyze(self, n_samples: int = 1000) -> List[SensitivityResult]:
        """Compute Sobol' indices.

        Args:
            n_samples: Base sample size (total samples = n_samples * (2n + 2)).

        Returns:
            List of SensitivityResult with S1 and ST indices.
        """
        n = self.n_params
        N = n_samples

        # Generate Saltelli sample matrices
        # A: baseline samples
        # B: perturbed samples
        A = self._generate_sample_matrix(N)
        B = self._generate_sample_matrix(N)

        # Compute baseline model output
        Y_A = np.array([self._evaluate_params(A[i]) for i in range(N)])
        Y_B = np.array([self._evaluate_params(B[i]) for i in range(N)])

        # Estimate variance
        total_variance = np.var(np.concatenate([Y_A, Y_B]))

        if total_variance < 1e-10:
            # Model output is constant
            return [SensitivityResult(
                parameter=param,
                first_order_index=0.0,
                total_order_index=0.0,
                interaction_index=0.0
            ) for param in self.parameters]

        results = []

        for i, param in enumerate(self.parameters):
            # Generate Si sample (A with i-th column from B)
            Si = A.copy()
            Si[:, i] = B[:, i]

            # Compute Si outputs
            Y_Si = np.array([self._evaluate_params(Si[j]) for j in range(N)])

            # First-order index (main effect)
            # E[Y|Xi] = (1/N) * sum(Y_A * Y_Si) / E[Y] - E[Y]
            # Actually: S1 = Var[E[Y|Xi]] / Var[Y]
            # Using formula: S1 = (1/N) * sum(Y_A * Y_Si) - mean(Y)^2 / Var(Y)
            mean_Y = np.mean(np.concatenate([Y_A, Y_B]))
            cov_Y = np.mean(Y_A * Y_Si) - mean_Y ** 2
            S1 = cov_Y / total_variance

            # Total-order index
            # ST = 1 - Var[E[Y|X-i]] / Var[Y]
            # Using formula: ST = (1/(2N)) * sum((Y_A - Y_Si)^2) / Var(Y)
            diff_sq_mean = np.mean((Y_A - Y_Si) ** 2)
            ST = diff_sq_mean / (2 * total_variance)

            # Interaction index
            S2 = ST - S1

            # Clip to valid range
            S1 = np.clip(S1, 0, 1)
            ST = np.clip(ST, S1, 1)
            S2 = np.clip(S2, 0, 1)

            results.append(SensitivityResult(
                parameter=param,
                first_order_index=S1,
                total_order_index=ST,
                interaction_index=S2,
                confidence_first=None,
                confidence_total=None
            ))

        return results

    def _generate_sample_matrix(self, N: int) -> np.ndarray:
        """Generate sample matrix using Sobol' sequence or random."""
        samples = np.zeros((N, self.n_params))

        for j, param in enumerate(self.parameters):
            low, high = self.bounds[param]
            samples[:, j] = np.random.uniform(low, high, N)

        return samples

    def _evaluate_params(self, x: np.ndarray) -> float:
        """Evaluate model with parameter array."""
        params = dict(zip(self.parameters, x))
        return self.model(params)


class MorrisAnalyzer(SensitivityAnalyzer):
    """Morris method for screening parameter sensitivities.

    Computes elementary effects to identify important parameters
    with relatively few model evaluations.
    """

    def __init__(self, model: Callable, parameters: List[str],
                 bounds: Dict[str, Tuple[float, float]],
                 n_trajectories: int = 10):
        """Initialize Morris analyzer.

        Args:
            model: Function that takes parameter dict and returns scalar.
            parameters: List of parameter names.
            bounds: Dictionary mapping parameter names to (min, max) bounds.
            n_trajectories: Number of OAT trajectories to run.
        """
        super().__init__(model, parameters)
        self.bounds = bounds
        self.n_trajectories = n_trajectories

    def analyze(self) -> List[SensitivityResult]:
        """Run Morris screening.

        Returns:
            List of SensitivityResult with mean and std of elementary effects.
        """
        delta = 0.5  # Step size (fraction of range)

        ee_history = []  # Elementary effects for each trajectory

        for _ in range(self.n_trajectories):
            # Random starting point
            x0 = np.array([np.random.uniform(self.bounds[p][0], self.bounds[p][1])
                          for p in self.parameters])

            # Random order of parameter changes
            order = np.random.permutation(self.n_params)

            # Compute elementary effects
            ee = np.zeros(self.n_params)
            x = x0.copy()

            y0 = self.model(dict(zip(self.parameters, x)))

            for i in order:
                # Change parameter i by delta
                low, high = self.bounds[self.parameters[i]]
                step = (high - low) * delta
                x[i] += step
                x[i] = np.clip(x[i], low, high)

                y_new = self.model(dict(zip(self.parameters, x)))
                ee[i] = (y_new - y0) / step

                y0 = y_new

            ee_history.append(ee)

        ee_history = np.array(ee_history)

        # Compute statistics of elementary effects
        results = []
        for i, param in enumerate(self.parameters):
            ee_mean = np.mean(ee_history[:, i])
            ee_std = np.std(ee_history[:, i])

            # Use absolute value for magnitude
            results.append(SensitivityResult(
                parameter=param,
                first_order_index=np.abs(ee_mean),  # mu* (absolute mean)
                total_order_index=ee_std,  # sigma (standard deviation)
                interaction_index=0.0,  # Not computed in Morris
                confidence_first=None,
                confidence_total=None
            ))

        return results


class ParameterRelativeSensitivity:
    """Compute relative sensitivity of model output to parameter changes.

    Uses finite difference approximation of dY/dX.
    """

    def __init__(self, model: Callable):
        """Initialize sensitivity calculator.

        Args:
            model: Function that takes parameter dict and returns scalar.
        """
        self.model = model

    def compute(self, params: Dict[str, float],
                perturbations: Dict[str, float] = None,
                delta_percent: float = 0.1) -> Dict[str, float]:
        """Compute relative sensitivities.

        Args:
            params: Base parameter values.
            perturbations: Explicit perturbation values (optional).
            delta_percent: Fractional change for perturbation (default 10%).

        Returns:
            Dictionary of relative sensitivities (dY/Y) / (dX/X).
        """
        base_output = self.model(params)

        sensitivities = {}

        for param_name, param_value in params.items():
            if perturbations and param_name in perturbations:
                delta_x = perturbations[param_name]
            else:
                delta_x = param_value * delta_percent

            # Perturb up
            params_up = params.copy()
            params_up[param_name] = param_value + delta_x
            y_up = self.model(params_up)

            # Perturb down
            params_down = params.copy()
            params_down[param_name] = param_value - delta_x
            y_down = self.model(params_down)

            # Central difference
            dy = (y_up - y_down) / 2
            dx = 2 * delta_x

            # Relative sensitivity: (dy/y) / (dx/x) = (dy * x) / (y * dx)
            if abs(base_output) > 1e-10 and abs(dx) > 1e-10:
                rel_sens = (dy * param_value) / (base_output * dx)
            else:
                rel_sens = 0.0

            sensitivities[param_name] = rel_sens

        return sensitivities

    def compute_elasticity(self, params: Dict[str, float],
                         delta_percent: float = 0.1) -> Dict[str, float]:
        """Alias for compute() - elasticity is same as relative sensitivity."""
        return self.compute(params, delta_percent=delta_percent)


def analyze_sensitivity(model: Callable, parameters: List[str],
                        bounds: Dict[str, Tuple[float, float]],
                        method: str = "sobol",
                        n_samples: int = 1000) -> List[SensitivityResult]:
    """Convenience function for sensitivity analysis.

    Args:
        model: Model function.
        parameters: Parameter names.
        bounds: Parameter bounds.
        method: 'sobol' or 'morris'.
        n_samples: Number of samples (sobol only).

    Returns:
        List of SensitivityResult.
    """
    if method.lower() == "sobol":
        analyzer = SobolAnalyzer(model, parameters, bounds)
        return analyzer.analyze(n_samples)
    elif method.lower() == "morris":
        analyzer = MorrisAnalyzer(model, parameters, bounds)
        return analyzer.analyze()
    else:
        raise ValueError(f"Unknown method: {method}")
