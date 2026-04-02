"""
Model Ensemble Module

Provides tools for creating and running multi-model ensembles for improved
hydrological predictions with uncertainty estimation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np

from HydroArray.domain.models.base import HydrologyModel
from HydroArray.config.task import ModelConfig


@dataclass
class EnsembleMember:
    """Single member of a model ensemble."""
    name: str
    model: HydrologyModel
    weight: float = 1.0
    config: Optional[ModelConfig] = None

    def __post_init__(self):
        if self.weight < 0:
            self.weight = 1.0


@dataclass
class EnsembleResult:
    """Result from ensemble simulation."""
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    quantiles: Dict[str, np.ndarray]  # e.g., 'q10', 'q90'
    members: Dict[str, np.ndarray]  # Individual member outputs
    weights: Dict[str, float]
    n_members: int


class ModelEnsemble:
    """Multi-model ensemble for hydrological prediction.

    Combines predictions from multiple models (different structures or
    different parameter sets) to produce ensemble forecasts with
    uncertainty estimates.
    """

    def __init__(self, members: Optional[List[EnsembleMember]] = None):
        """Initialize model ensemble.

        Args:
            members: List of EnsembleMember objects.
        """
        self.members: List[EnsembleMember] = members or []
        self._member_outputs: Dict[str, np.ndarray] = {}

    def add_member(self, member: EnsembleMember):
        """Add a model member to the ensemble.

        Args:
            member: EnsembleMember to add.
        """
        self.members.append(member)
        self._member_outputs[member.name] = np.array([])

    def remove_member(self, name: str):
        """Remove a member by name."""
        self.members = [m for m in self.members if m.name != name]
        if name in self._member_outputs:
            del self._member_outputs[name]

    def set_weights(self, weights: Dict[str, float]):
        """Set ensemble member weights.

        Args:
            weights: Dictionary mapping member names to weights.
        """
        total = sum(weights.values())
        for member in self.members:
            if member.name in weights:
                member.weight = weights[member.name] / total
            else:
                member.weight = 1.0 / len(self.members)

    def run(self, inputs, return_individual: bool = False) -> EnsembleResult:
        """Run all ensemble members.

        Args:
            inputs: Input data for the models.
            return_individual: Whether to return individual member outputs.

        Returns:
            EnsembleResult with mean, std, quantiles, etc.
        """
        if not self.members:
            raise ValueError("No ensemble members configured")

        outputs = []
        weights = {}

        for member in self.members:
            try:
                result = member.model.run(inputs)
                member_output = self._extract_output(result)
                outputs.append(member_output)
                weights[member.name] = member.weight
                self._member_outputs[member.name] = member_output
            except Exception as e:
                print(f"Warning: Member {member.name} failed: {e}")
                continue

        if not outputs:
            raise RuntimeError("All ensemble members failed")

        # Stack outputs: (n_members, n_times)
        outputs_array = np.vstack(outputs)

        # Compute ensemble statistics
        ensemble_mean = np.mean(outputs_array, axis=0)
        ensemble_std = np.std(outputs_array, axis=0)
        ensemble_median = np.median(outputs_array, axis=0)

        # Compute quantiles
        quantiles = {
            'q05': np.percentile(outputs_array, 5, axis=0),
            'q10': np.percentile(outputs_array, 10, axis=0),
            'q25': np.percentile(outputs_array, 25, axis=0),
            'q75': np.percentile(outputs_array, 75, axis=0),
            'q90': np.percentile(outputs_array, 90, axis=0),
            'q95': np.percentile(outputs_array, 95, axis=0),
        }

        return EnsembleResult(
            mean=ensemble_mean,
            std=ensemble_std,
            median=ensemble_median,
            quantiles=quantiles,
            members=self._member_outputs if return_individual else {},
            weights=weights,
            n_members=len(outputs)
        )

    def _extract_output(self, result) -> np.ndarray:
        """Extract numpy array from model result."""
        if isinstance(result, np.ndarray):
            return result.flatten()
        elif hasattr(result, 'data'):
            data = result.data
            if isinstance(data, np.ndarray):
                return data.flatten()
            elif isinstance(data, dict):
                # Take first array in dict
                for key, val in data.items():
                    if isinstance(val, np.ndarray):
                        return val.flatten()
        elif hasattr(result, 'values'):
            return np.asarray(result.values).flatten()
        return np.array([result]).flatten()

    def get_member_output(self, name: str) -> Optional[np.ndarray]:
        """Get output from a specific member."""
        return self._member_outputs.get(name)

    def compute_spread(self) -> np.ndarray:
        """Compute ensemble spread (standard deviation of member outputs)."""
        if not self._member_outputs:
            return np.array([])

        outputs = list(self._member_outputs.values())
        if len(outputs) == 1:
            return np.zeros_like(outputs[0])

        return np.std(np.vstack(outputs), axis=0)


class WeightedModelEnsemble(ModelEnsemble):
    """Weighted ensemble that optimizes weights based on historical performance."""

    def __init__(self, members: Optional[List[EnsembleMember]] = None):
        super().__init__(members)
        self._optimal_weights: Optional[Dict[str, float]] = None

    def optimize_weights(self, observations: np.ndarray,
                         metric: str = "nse") -> Dict[str, float]:
        """Optimize ensemble weights based on observations.

        Args:
            observations: Observed values for comparison.
            metric: Metric to use ('nse', 'rmse', 'kge').

        Returns:
            Dictionary of optimized weights.
        """
        from HydroArray.utils.metrics import calculate_nse, calculate_rmse, calculate_kge

        if not self._member_outputs:
            raise ValueError("Run ensemble first to generate member outputs")

        scores = {}

        for name, output in self._member_outputs.items():
            if len(output) != len(observations):
                # Truncate to match length
                min_len = min(len(output), len(observations))
                output = output[:min_len]
                obs = observations[:min_len]
            else:
                obs = observations

            if metric == "nse":
                scores[name] = max(calculate_nse(obs, output), 0)  # Clip to non-negative
            elif metric == "rmse":
                scores[name] = 1.0 / (1.0 + calculate_rmse(obs, output))  # Invert so higher is better
            elif metric == "kge":
                scores[name] = max(calculate_kge(obs, output), 0)
            else:
                scores[name] = 1.0

        # Normalize scores to weights
        total = sum(scores.values())
        if total > 0:
            optimal_weights = {name: score / total for name, score in scores.items()}
        else:
            # Equal weights if all scores zero
            optimal_weights = {name: 1.0 / len(scores) for name in scores}

        self._optimal_weights = optimal_weights

        # Update member weights
        self.set_weights(optimal_weights)

        return optimal_weights

    def get_optimal_weights(self) -> Optional[Dict[str, float]]:
        """Get optimized weights."""
        return self._optimal_weights


class ParameterEnsemble:
    """Ensemble of the same model with different parameter sets.

    Useful for parameter uncertainty analysis and Monte Carlo simulation.
    """

    def __init__(self, base_model: HydrologyModel,
                 parameter_samples: List[Dict[str, float]]):
        """Initialize parameter ensemble.

        Args:
            base_model: Base model to clone for each parameter set.
            parameter_samples: List of parameter dictionaries.
        """
        self.base_model = base_model
        self.parameter_samples = parameter_samples
        self.n_samples = len(parameter_samples)

    def run(self, inputs) -> EnsembleResult:
        """Run model with all parameter samples.

        Args:
            inputs: Input data.

        Returns:
            EnsembleResult with parameter uncertainty bounds.
        """
        outputs = []

        for i, params in enumerate(self.parameter_samples):
            try:
                # Clone model and update parameters
                model = self._clone_model(params)
                result = model.run(inputs)
                output = self._extract_output(result)
                outputs.append(output)
            except Exception as e:
                print(f"Warning: Parameter set {i} failed: {e}")
                continue

        if not outputs:
            raise RuntimeError("All parameter sets failed")

        outputs_array = np.vstack(outputs)

        ensemble_mean = np.mean(outputs_array, axis=0)
        ensemble_std = np.std(outputs_array, axis=0)
        ensemble_median = np.median(outputs_array, axis=0)

        quantiles = {
            'q05': np.percentile(outputs_array, 5, axis=0),
            'q10': np.percentile(outputs_array, 10, axis=0),
            'q25': np.percentile(outputs_array, 25, axis=0),
            'q75': np.percentile(outputs_array, 75, axis=0),
            'q90': np.percentile(outputs_array, 90, axis=0),
            'q95': np.percentile(outputs_array, 95, axis=0),
        }

        # Create member names
        member_names = [f"sample_{i}" for i in range(len(outputs))]
        members = {name: output for name, output in zip(member_names, outputs)}

        return EnsembleResult(
            mean=ensemble_mean,
            std=ensemble_std,
            median=ensemble_median,
            quantiles=quantiles,
            members=members,
            weights={name: 1.0 / len(outputs) for name in member_names},
            n_members=len(outputs)
        )

    def _clone_model(self, params: Dict[str, float]) -> HydrologyModel:
        """Clone base model and set new parameters."""
        import copy
        model = copy.deepcopy(self.base_model)

        if hasattr(model, '_params'):
            for key, value in params.items():
                if hasattr(model._params, key):
                    setattr(model._params, key, value)

        return model

    def _extract_output(self, result) -> np.ndarray:
        """Extract numpy array from model result."""
        if isinstance(result, np.ndarray):
            return result.flatten()
        elif hasattr(result, 'data'):
            data = result.data
            if isinstance(data, np.ndarray):
                return data.flatten()
        return np.array([result]).flatten()

    @staticmethod
    def generate_lhs_samples(param_bounds: Dict[str, Tuple[float, float]],
                             n_samples: int) -> List[Dict[str, float]]:
        """Generate parameter samples using Latin Hypercube Sampling.

        Args:
            param_bounds: Parameter bounds dictionary.
            n_samples: Number of samples to generate.

        Returns:
            List of parameter dictionaries.
        """
        n_params = len(param_bounds)
        param_names = list(param_bounds.keys())

        # Generate LHS samples
        samples = np.zeros((n_samples, n_params))

        for j in range(n_params):
            low, high = param_bounds[param_names[j]]
            # Generate random samples per dimension
            samples[:, j] = np.random.uniform(low, high, n_samples)

        # Simple shuffle to improve coverage (basic LHS approximation)
        for j in range(n_params):
            np.random.shuffle(samples[:, j])

        # Convert to list of dicts
        return [dict(zip(param_names, samples[i])) for i in range(n_samples)]


def create_structural_ensemble(models: List[HydrologyModel],
                               weights: Optional[List[float]] = None) -> ModelEnsemble:
    """Create a structural ensemble from different model types.

    Args:
        models: List of different hydrological models.
        weights: Optional weights for each model.

    Returns:
        ModelEnsemble instance.
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    members = []
    for i, model in enumerate(models):
        member = EnsembleMember(
            name=f"model_{i}_{model.__class__.__name__}",
            model=model,
            weight=weights[i]
        )
        members.append(member)

    return ModelEnsemble(members)
