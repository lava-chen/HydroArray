"""
Model Calibration Example

Demonstrates how to calibrate hydrological model parameters
using different optimization algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt

import HydroArray as ha
from HydroArray.domain import (
    ParameterBounds,
    ScipyOptimizer,
    GeneticAlgorithm,
)


def create_calibration_data():
    """Create synthetic data for calibration.

    Returns:
        precip: Precipitation data
        pet: PET data
        observations: "Observed" streamflow
    """
    np.random.seed(42)
    n_steps = 200

    # Precipitation
    t = np.arange(n_steps)
    precip_pattern = 10 + 5 * np.sin(2 * np.pi * t / 50)
    precip = np.maximum(precip_pattern + np.random.normal(0, 1, n_steps), 0)

    # PET (lower in winter, higher in summer)
    pet = 2 + 1.5 * np.sin(2 * np.pi * t / 50 - np.pi/3)
    pet = np.maximum(pet + np.random.normal(0, 0.2, n_steps), 0)

    # Generate "true" observations using known parameters
    nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(3)]

    true_params = ha.XinAnjiangParameters(
        WM=120.0,  # True value
        K=0.45,    # True value
        B=0.15,    # True value
        SM=18.0,   # True value
        KI=0.35,   # True value
        KG=0.06    # True value
    )

    model = ha.XinAnjiangModel()
    model.initialize(nodes, true_params)

    observations = []
    for t in range(n_steps):
        RS, RI, RG, _ = model.water_balance(1.0, precip[t], pet[t])
        q = RS.sum() + RI.sum() + RG.sum()
        observations.append(q)

    # Add measurement noise
    observations = np.array(observations) + np.random.normal(0, 0.3, n_steps)

    return precip, pet, observations


def objective_nse(params, model, precip, pet, observations):
    """Objective function based on NSE.

    Args:
        params: Dictionary of parameter values
        model: HydrologyModel instance
        precip: Precipitation
        pet: PET
        observations: Observed flow

    Returns:
        Negative NSE (for minimization)
    """
    try:
        # Update model parameters
        for key, value in params.items():
            if hasattr(model._params, key):
                setattr(model._params, key, value)

        # Run simulation
        simulated = []
        for t in range(len(precip)):
            RS, RI, RG, _ = model.water_balance(1.0, precip[t], pet[t])
            q = RS.sum() + RI.sum() + RG.sum()
            simulated.append(q)

        # Calculate NSE
        nse = ha.calculate_nse(observations, np.array(simulated))

        # Return negative for minimization
        return -nse if nse > -999 else 0

    except Exception:
        return 0


def run_scipy_calibration():
    """Run calibration using Scipy optimizer."""
    print("=" * 60)
    print("Scipy Optimizer Calibration")
    print("=" * 60)

    # Create data
    precip, pet, observations = create_calibration_data()

    # Create model
    nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(3)]
    model = ha.XinAnjiangModel()
    initial_params = ha.XinAnjiangParameters()
    model.initialize(nodes, initial_params)

    # Define calibration parameters
    calib_params = [
        ParameterBounds(name='WM', min_value=50, max_value=200, default_value=100),
        ParameterBounds(name='K', min_value=0.1, max_value=1.0, default_value=0.5),
        ParameterBounds(name='B', min_value=0.01, max_value=0.5, default_value=0.1),
        ParameterBounds(name='SM', min_value=5, max_value=30, default_value=15),
        ParameterBounds(name='KI', min_value=0.1, max_value=0.7, default_value=0.4),
        ParameterBounds(name='KG', min_value=0.01, max_value=0.2, default_value=0.05),
    ]

    print("\nCalibration parameters:")
    for p in calib_params:
        print(f"  {p.name}: [{p.min_value}, {p.max_value}]")

    # Create objective function
    def obj_func(params_dict):
        return objective_nse(params_dict, model, precip, pet, observations)

    # Setup optimizer
    config = ha.CalibrationConfig(
        parameters=calib_params,
        objective='nse',
        max_evaluations=500,
        verbose=True
    )

    optimizer = ScipyOptimizer(config, method='L-BFGS-B')

    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.optimize(obj_func, initial_params=None)

    print("\n--- Calibration Results ---")
    print(f"Converged: {result.converged}")
    print(f"NSE: {-result.objective_value:.4f}")
    print(f"RMSE: {result.rmse:.4f}")
    print("\nOptimal parameters:")
    for name, value in result.parameters.items():
        print(f"  {name}: {value:.4f}")

    return result, model, precip, pet, observations


def run_ga_calibration():
    """Run calibration using Genetic Algorithm."""
    print("\n" + "=" * 60)
    print("Genetic Algorithm Calibration")
    print("=" * 60)

    # Create data
    np.random.seed(123)  # Different seed
    precip, pet, observations = create_calibration_data()

    # Create model
    nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(3)]
    model = ha.XinAnjiangModel()
    model.initialize(nodes, ha.XinAnjiangParameters())

    # Calibration parameters
    calib_params = [
        ParameterBounds(name='WM', min_value=50, max_value=200),
        ParameterBounds(name='K', min_value=0.1, max_value=1.0),
        ParameterBounds(name='B', min_value=0.01, max_value=0.5),
    ]

    def obj_func(params_dict):
        return objective_nse(params_dict, model, precip, pet, observations)

    # GA configuration
    config = ha.CalibrationConfig(
        parameters=calib_params,
        objective='nse',
        verbose=False
    )

    optimizer = GeneticAlgorithm(
        config,
        population_size=30,
        generations=50,
        crossover_prob=0.9,
        mutation_prob=0.1
    )

    print("Running GA optimization...")
    result = optimizer.optimize(obj_func)

    print(f"\n--- GA Results ---")
    print(f"NSE: {-result.objective_value:.4f}")
    print(f"Parameters: {result.parameters}")

    return result


def sensitivity_analysis_example():
    """Run sensitivity analysis on model parameters."""
    print("\n" + "=" * 60)
    print("Sensitivity Analysis")
    print("=" * 60)

    # Create data
    np.random.seed(999)
    precip, pet, observations = create_calibration_data()

    # Create model
    nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(3)]
    model = ha.XinAnjiangModel()
    base_params = ha.XinAnjiangParameters()
    model.initialize(nodes, base_params)

    def model_func(params):
        for key, value in params.items():
            if hasattr(model._params, key):
                setattr(model._params, key, value)

        simulated = []
        for t in range(len(precip)):
            RS, RI, RG, _ = model.water_balance(1.0, precip[t], pet[t])
            q = RS.sum() + RI.sum() + RG.sum()
            simulated.append(q)

        return ha.calculate_nse(observations, np.array(simulated))

    # Parameter bounds
    param_bounds = {
        'WM': (50, 200),
        'K': (0.1, 1.0),
        'B': (0.01, 0.5),
        'SM': (5, 30),
        'KI': (0.1, 0.7),
        'KG': (0.01, 0.2),
    }

    print("\nRunning Morris sensitivity analysis...")
    results = ha.analyze_sensitivity(
        model=model_func,
        parameters=list(param_bounds.keys()),
        bounds=param_bounds,
        method='morris',
        n_samples=100
    )

    print("\nSensitivity Ranking (Morris μ*):")
    sorted_results = sorted(results, key=lambda x: x.first_order_index, reverse=True)
    for r in sorted_results:
        print(f"  {r.parameter}: mu*={r.first_order_index:.4f}, sigma={r.total_order_index:.4f}")

    return results


if __name__ == "__main__":
    # Run calibrations
    scipy_result, model, precip, pet, obs = run_scipy_calibration()
    ga_result = run_ga_calibration()
    sens_results = sensitivity_analysis_example()

    print("\n" + "=" * 60)
    print("Calibration Example Completed!")
    print("=" * 60)
