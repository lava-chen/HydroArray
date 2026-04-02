"""
Model Calibration Module

Provides automatic calibration for hydrological models using various optimization algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
import numpy as np
from scipy import optimize

from HydroArray.utils.metrics import calculate_nse, calculate_rmse, calculate_kge


@dataclass
class CalibrationResult:
    """Result of calibration run."""
    parameters: Dict[str, float]
    objective_value: float
    nse: float
    rmse: float
    kge: float
    n_evaluations: int
    converged: bool
    message: str = ""


class ObjectiveFunction:
    """Wrapper for calibration objective function.

    Supports multiple objective functions:
    - NSE (default, maximize)
    - RMSE (minimize)
    - KGE (maximize)
    - Multi-objective (weighted combination)
    """

    def __init__(self, func: Callable, maximize: bool = True):
        """Initialize objective function.

        Args:
            func: Function that takes parameters dict and returns objective value.
            maximize: Whether to maximize (True) or minimize (False).
        """
        self.func = func
        self.maximize = maximize

    def __call__(self, x: np.ndarray, param_names: List[str]) -> float:
        """Evaluate objective function.

        Args:
            x: Parameter values array.
            param_names: Names corresponding to parameter values.

        Returns:
            Objective value.
        """
        params = dict(zip(param_names, x))
        obj = self.func(params)

        if self.maximize:
            return -obj
        return obj


@dataclass
class ParameterBounds:
    """Bounds for a single parameter."""
    name: str
    min_value: float
    max_value: float
    default_value: Optional[float] = None

    def __post_init__(self):
        if self.default_value is None:
            self.default_value = (self.min_value + self.max_value) / 2


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    parameters: List[ParameterBounds]
    objective: str = "nse"  # 'nse', 'rmse', 'kge', 'multi'
    weights: Optional[Dict[str, float]] = None  # For multi-objective
    max_evaluations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = True


class CalibrationMethod(ABC):
    """Abstract base class for calibration methods."""

    def __init__(self, config: CalibrationConfig):
        self.config = config

    @abstractmethod
    def optimize(self, objective_func: Callable,
                 initial_params: Optional[Dict[str, float]] = None) -> CalibrationResult:
        """Run optimization.

        Args:
            objective_func: Objective function to optimize.
            initial_params: Initial parameter values.

        Returns:
            CalibrationResult.
        """
        pass

    def _validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate parameters are within bounds."""
        for pb in self.config.parameters:
            if pb.name in params:
                val = params[pb.name]
                if val < pb.min_value or val > pb.max_value:
                    return False
        return True

    def _clip_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to bounds."""
        clipped = params.copy()
        for pb in self.config.parameters:
            if pb.name in clipped:
                clipped[pb.name] = np.clip(clipped[pb.name], pb.min_value, pb.max_value)
        return clipped


class GeneticAlgorithm(CalibrationMethod):
    """Genetic Algorithm for hydrological model calibration.

    Implements a simple real-coded genetic algorithm with:
    - Tournament selection
    - Simulated binary crossover (SBX)
    - Polynomial mutation
    """

    def __init__(self, config: CalibrationConfig,
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 tournament_size: int = 3):
        super().__init__(config)
        self.pop_size = population_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.tourney_size = tournament_size

    def _extract_fitness(self, obj_dict: Dict[str, float]) -> float:
        """Extract scalar fitness from objective dict."""
        obj_type = self.config.objective
        if obj_type == "rmse":
            return obj_dict.get('rmse', 9999.0)
        else:  # nse, kge, multi - maximize
            return -obj_dict.get(obj_type, -9999.0)

    def optimize(self, objective_func: Callable,
                 initial_params: Optional[Dict[str, float]] = None) -> CalibrationResult:
        """Run genetic algorithm optimization.

        Args:
            objective_func: Objective function (takes params dict, returns dict with metrics).
            initial_params: Initial parameter values.

        Returns:
            CalibrationResult.
        """
        param_names = [pb.name for pb in self.config.parameters]
        n_params = len(param_names)

        # Initialize population
        bounds = np.array([(pb.min_value, pb.max_value) for pb in self.config.parameters])

        if initial_params:
            pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, n_params))
            # Insert initial params at first position
            init_vals = np.array([initial_params.get(name, (bounds[i, 0] + bounds[i, 1]) / 2)
                                  for i, name in enumerate(param_names)])
            pop[0] = init_vals
        else:
            pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, n_params))

        # Evaluate initial population
        fitness = np.array([self._extract_fitness(objective_func(dict(zip(param_names, ind)))) for ind in pop])

        best_idx = np.argmax(fitness)
        best_params = dict(zip(param_names, pop[best_idx]))
        best_fitness = fitness[best_idx]

        # Evolution
        for gen in range(self.generations):
            # Create offspring
            offspring = np.zeros_like(pop)

            for i in range(self.pop_size):
                # Selection (tournament)
                tourn_idx = np.random.choice(self.pop_size, self.tourney_size, replace=False)
                winner_idx = tourn_idx[np.argmax(fitness[tourn_idx])]
                parent = pop[winner_idx].copy()

                # Crossover
                if np.random.rand() < self.cx_prob:
                    parent2_idx = np.random.choice(self.pop_size)
                    parent2 = pop[parent2_idx]
                    child = self._sbx_crossover(parent, parent2, bounds)
                else:
                    child = parent.copy()

                # Mutation
                if np.random.rand() < self.mut_prob:
                    child = self._polynomial_mutation(child, bounds)

                offspring[i] = child

            pop = offspring
            fitness = np.array([self._extract_fitness(objective_func(dict(zip(param_names, ind)))) for ind in pop])

            # Elitism - keep best
            curr_best_idx = np.argmax(fitness)
            if fitness[curr_best_idx] > best_fitness:
                best_fitness = fitness[curr_best_idx]
                best_params = dict(zip(param_names, pop[curr_best_idx]))

            if self.config.verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best fitness = {-best_fitness:.4f}")

        best_params = self._clip_parameters(best_params)
        final_obj = objective_func(best_params)

        return CalibrationResult(
            parameters=best_params,
            objective_value=-best_fitness if self.config.objective != "rmse" else best_fitness,
            nse=final_obj.get('nse', 0),
            rmse=final_obj.get('rmse', 0),
            kge=final_obj.get('kge', 0),
            n_evaluations=self.pop_size * self.generations,
            converged=True,
            message="Genetic algorithm completed"
        )

    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                       bounds: np.ndarray, eta: float = 15.0) -> np.ndarray:
        """Simulated Binary Crossover (SBX)."""
        child = parent1.copy()
        if np.random.rand() > 0.5:
            parent1, parent2 = parent2, parent1

        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            else:
                child[i] = parent1[i]

        # Clip to bounds
        child = np.clip(child, bounds[:, 0], bounds[:, 1])
        return child

    def _polynomial_mutation(self, individual: np.ndarray,
                             bounds: np.ndarray, eta: float = 20.0) -> np.ndarray:
        """Polynomial mutation."""
        mutated = individual.copy()

        for i in range(len(individual)):
            u = np.random.rand()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

            mutated[i] += delta
            mutated[i] = np.clip(mutated[i], bounds[i, 0], bounds[i, 1])

        return mutated


class SCEOptimizer(CalibrationMethod):
    """Shuffled Complex Evolution (SCE-UA) optimizer.

    A robust global optimization algorithm well-suited for hydrological model calibration.
    """

    def __init__(self, config: CalibrationConfig,
                 n_complexes: int = 5,
                 points_per_complex: int = 10,
                 max_iterations: int = 100):
        super().__init__(config)
        self.n_complexes = n_complexes
        self.ppc = points_per_complex
        self.max_iter = max_iterations

    def _extract_fitness(self, obj_dict: Dict[str, float]) -> float:
        """Extract scalar fitness from objective dict."""
        obj_type = self.config.objective
        if obj_type == "rmse":
            return obj_dict.get('rmse', 9999.0)
        else:  # nse, kge, multi - maximize
            return -obj_dict.get(obj_type, -9999.0)

    def optimize(self, objective_func: Callable,
                 initial_params: Optional[Dict[str, float]] = None) -> CalibrationResult:
        """Run SCE-UI optimization."""
        param_names = [pb.name for pb in self.config.parameters]
        n_params = len(param_names)
        bounds = np.array([(pb.min_value, pb.max_value) for pb in self.config.parameters])

        # Initialize population
        n_points = self.n_complexes * self.ppc
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_points, n_params))

        if initial_params:
            init_vals = np.array([initial_params.get(name, (bounds[i, 0] + bounds[i, 1]) / 2)
                                  for i, name in enumerate(param_names)])
            pop[0] = init_vals

        # Evaluate
        fitness = np.array([self._extract_fitness(objective_func(dict(zip(param_names, p)))) for p in pop])

        # Sort by fitness (maximize NSE/KGE, minimize RMSE)
        if self.config.objective in ['nse', 'kge', 'multi']:
            sorted_idx = np.argsort(-fitness)
        else:
            sorted_idx = np.argsort(fitness)

        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

        best_idx = 0
        best_params = dict(zip(param_names, pop[best_idx]))
        best_fitness = fitness[best_idx]

        if self.config.verbose:
            print(f"Initial best: {-best_fitness:.4f}" if self.config.objective != 'rmse' else f"Initial best: {best_fitness:.4f}")

        # SCE evolution (simplified)
        for iteration in range(self.max_iter):
            # Shuffle and partition into complexes
            indices = np.random.permutation(n_points)
            complexes = [pop[indices[i::self.n_complexes]] for i in range(self.n_complexes)]

            # Evolve each complex
            for cx in range(self.n_complexes):
                complex_pop = complexes[cx]
                n_cx = len(complex_pop)

                # Contract towards centroid
                centroid = np.mean(complex_pop, axis=0)

                # Replace worst point with reflected point
                worst_idx = -1
                worst = complex_pop[worst_idx]
                reflected = 2 * centroid - worst

                # Check bounds
                reflected = np.clip(reflected, bounds[:, 0], bounds[:, 1])

                obj_reflected = self._extract_fitness(objective_func(dict(zip(param_names, reflected))))

                if obj_reflected > (fitness.min() if self.config.objective in ['nse', 'kge', 'multi'] else fitness.max()):
                    complex_pop[worst_idx] = reflected

                complexes[cx] = complex_pop

            # Recombine complexes
            pop = np.vstack(complexes)
            fitness = np.array([self._extract_fitness(objective_func(dict(zip(param_names, p)))) for p in pop])

            # Sort
            if self.config.objective in ['nse', 'kge', 'multi']:
                sorted_idx = np.argsort(-fitness)
            else:
                sorted_idx = np.argsort(fitness)

            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

            curr_best = fitness[0]
            if curr_best > best_fitness:
                best_fitness = curr_best
                best_params = dict(zip(param_names, pop[0]))

            if self.config.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Best = {-best_fitness:.4f}" if self.config.objective != 'rmse' else f"Iter {iteration}: Best = {best_fitness:.4f}")

        best_params = self._clip_parameters(best_params)
        final_obj = objective_func(best_params)

        # Determine objective value for return
        if self.config.objective == "rmse":
            obj_value = best_fitness
        else:
            obj_value = -best_fitness

        return CalibrationResult(
            parameters=best_params,
            objective_value=obj_value,
            nse=final_obj.get('nse', 0),
            rmse=final_obj.get('rmse', 0),
            kge=final_obj.get('kge', 0),
            n_evaluations=self.max_iter * n_points,
            converged=True,
            message="SCE optimization completed"
        )


class ScipyOptimizer(CalibrationMethod):
    """Wrapper for scipy.optimize methods.

    Supports: 'Nelder-Mead', 'Powell', 'L-BFGS-B', 'GA', 'DE', etc.
    """

    def __init__(self, config: CalibrationConfig,
                 method: str = "L-BFGS-B",
                 options: Optional[Dict] = None):
        super().__init__(config)
        self.method = method
        self.options = options or {}

    def optimize(self, objective_func: Callable,
                 initial_params: Optional[Dict[str, float]] = None) -> CalibrationResult:
        """Run scipy optimization."""
        param_names = [pb.name for pb in self.config.parameters]
        n_params = len(param_names)
        bounds = [(pb.min_value, pb.max_value) for pb in self.config.parameters]

        if initial_params:
            x0 = np.array([initial_params.get(name, (bounds[i][0] + bounds[i][1]) / 2)
                           for i, name in enumerate(param_names)])
        else:
            x0 = np.array([pb.default_value for pb in self.config.parameters])

        # Determine which metric to optimize for scipy
        obj_type = self.config.objective

        # Wrap objective for scipy (minimize scalar)
        def wrapped_obj(x):
            params = dict(zip(param_names, x))
            result_dict = objective_func(params)
            # Return the scalar to minimize
            if obj_type == "rmse":
                return result_dict.get('rmse', 9999.0)
            else:  # nse, kge, multi - maximize these, so negate
                return -result_dict.get(obj_type, -9999.0)

        result = optimize.minimize(
            wrapped_obj,
            x0,
            method=self.method,
            bounds=bounds if self.method == 'L-BFGS-B' else None,
            options=self.options
        )

        best_params = dict(zip(param_names, result.x))
        best_params = self._clip_parameters(best_params)
        final_obj = objective_func(best_params)

        # Extract scalar value for the result
        if obj_type == "rmse":
            obj_value = result.fun
        else:
            obj_value = -result.fun

        return CalibrationResult(
            parameters=best_params,
            objective_value=obj_value,
            nse=final_obj.get('nse', 0),
            rmse=final_obj.get('rmse', 0),
            kge=final_obj.get('kge', 0),
            n_evaluations=result.nfev if hasattr(result, 'nfev') else 0,
            converged=result.success,
            message=result.message if hasattr(result, 'message') else ""
        )


def create_objective_function(simulate_func: Callable,
                             observations: np.ndarray,
                             objective: str = "nse",
                             weights: Optional[Dict[str, float]] = None) -> Callable:
    """Create objective function for calibration.

    Args:
        simulate_func: Function that takes parameters dict and returns simulated values.
        observations: Observed values array.
        objective: Objective type ('nse', 'rmse', 'kge', 'multi').
        weights: Weights for multi-objective.

    Returns:
        Objective function that takes parameters dict and returns dict with all metrics.
    """
    def objective_func(params: Dict[str, float]) -> Dict[str, float]:
        try:
            simulated = simulate_func(params)

            nse_val = calculate_nse(observations, simulated)
            rmse_val = calculate_rmse(observations, simulated)
            kge_val = calculate_kge(observations, simulated)

            if objective == "nse":
                return {'nse': nse_val, 'rmse': rmse_val, 'kge': kge_val}
            elif objective == "rmse":
                return {'nse': nse_val, 'rmse': rmse_val, 'kge': kge_val}
            elif objective == "kge":
                return {'nse': nse_val, 'rmse': rmse_val, 'kge': kge_val}
            elif objective == "multi":
                w = weights or {'nse': 0.4, 'rmse': 0.3, 'kge': 0.3}
                multi_val = w.get('nse', 0.4) * nse_val + w.get('rmse', 0.3) * (-rmse_val / 100) + w.get('kge', 0.3) * kge_val
                return {'nse': multi_val, 'rmse': rmse_val, 'kge': kge_val}
            else:
                return {'nse': nse_val, 'rmse': rmse_val, 'kge': kge_val}

        except Exception as e:
            return {'nse': -9999.0, 'rmse': 9999.0, 'kge': -9999.0}

    return objective_func


def calibrate(model, observations: np.ndarray,
             parameters: List[ParameterBounds],
             method: str = "scipy",
             objective: str = "nse",
             initial_params: Optional[Dict[str, float]] = None,
             **kwargs) -> CalibrationResult:
    """Convenience function for model calibration.

    Args:
        model: HydrologyModel instance with a run() or run_step() method.
        observations: Observed flow values.
        parameters: List of ParameterBounds defining calibration parameters.
        method: Optimization method ('scipy', 'ga', 'sce').
        objective: Objective function ('nse', 'rmse', 'kge', 'multi').
        initial_params: Initial parameter values.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        CalibrationResult.
    """
    config = CalibrationConfig(
        parameters=parameters,
        objective=objective,
        max_evaluations=kwargs.get('max_evaluations', 1000)
    )

    # Get forcing data from kwargs
    precip_data = kwargs.get('precip_data', None)
    pet_data = kwargs.get('pet_data', None)

    # Create simulation function
    def simulate_func(params: Dict[str, float]) -> np.ndarray:
        # Update model parameters
        for key, value in params.items():
            if hasattr(model, '_params') and hasattr(model._params, key):
                setattr(model._params, key, value)

        # Run model for all observations
        results = []
        n_obs = len(observations)
        n_nodes = len(kwargs.get('nodes', [0]))

        if precip_data is not None and pet_data is not None:
            for i in range(n_obs):
                p = precip_data[i] if precip_data.ndim > 1 else precip_data[i]
                e = pet_data[i] if pet_data.ndim > 1 else pet_data[i]
                RS, RI, RG, _ = model.water_balance(1.0, np.array([p]), np.array([e]))
                results.append(RS.sum() + RI.sum() + RG.sum())
        else:
            # Use default values if no forcing data provided
            for i in range(n_obs):
                RS, RI, RG, _ = model.water_balance(
                    1.0,
                    np.array([kwargs.get('precip', 10.0)]),
                    np.array([kwargs.get('pet', 2.0)])
                )
                results.append(RS.sum() + RI.sum() + RG.sum())

        return np.array(results)

    obj_func = create_objective_function(simulate_func, observations, objective)

    # Filter kwargs to remove data-related keys that shouldn't go to optimizer
    optimizer_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ['precip_data', 'pet_data', 'nodes']}

    # Create optimizer
    if method == "scipy":
        optimizer = ScipyOptimizer(config, **optimizer_kwargs)
    elif method == "ga":
        optimizer = GeneticAlgorithm(config, **optimizer_kwargs)
    elif method == "sce":
        optimizer = SCEOptimizer(config, **optimizer_kwargs)
    else:
        optimizer = ScipyOptimizer(config, method="L-BFGS-B")

    return optimizer.optimize(obj_func, initial_params)
