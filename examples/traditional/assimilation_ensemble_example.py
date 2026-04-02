"""
Data Assimilation and Ensemble Forecasting Example

Demonstrates:
1. Ensemble Kalman Filter for state updating
2. Multi-model ensemble for uncertainty estimation
3. Parameter ensemble for uncertainty analysis
"""

import numpy as np
import matplotlib.pyplot as plt

import HydroArray as ha
from HydroArray.domain import (
    EnsembleKalmanFilter,
    ParticleFilter,
    ModelEnsemble,
    ParameterEnsemble,
)


def generate_test_data(n_steps=100):
    """Generate synthetic precipitation and PET data.

    Returns:
        precip: Precipitation array
        pet: PET array
        true_states: True model states (for comparison)
        observations: Noisy observations
    """
    np.random.seed(42)

    t = np.arange(n_steps)

    # Precipitation - random storm events
    precip = np.zeros(n_steps)
    for i in range(n_steps):
        if np.random.rand() < 0.1:  # 10% chance of rain
            precip[i] = np.random.uniform(5, 20)
        if i > 0 and precip[i-1] > 0:
            precip[i] += precip[i-1] * 0.3  # Some persistence

    # PET - daily cycle
    pet = 3 + 2 * np.sin(2 * np.pi * t / 365 - np.pi/3)
    pet = np.maximum(pet, 0)

    # True states and observations
    nodes = [ha.GridNode(node_id=0, x=0, y=0, area=100)]
    model = ha.XinAnjiangModel()
    model.initialize(nodes, ha.XinAnjiangParameters())

    true_states = []
    observations = []
    RS, RI, RG, _ = model.water_balance(1.0, precip[0], pet[0])
    true_states.append({'W': model._W.copy(), 'S': model._S.copy()})

    for t in range(1, n_steps):
        RS, RI, RG, states = model.water_balance(1.0, precip[t], pet[t])
        true_states.append({'W': model._W.copy(), 'S': model._S.copy()})

    # Generate observations (with noise)
    for t in range(n_steps):
        q = RS.sum() + RI.sum() + RG.sum() if t > 0 else 10.0
        obs = q + np.random.normal(0, 1.0)
        observations.append(obs)

    return precip, pet, true_states, np.array(observations)


def run_enkf_example():
    """Run Ensemble Kalman Filter example."""
    print("=" * 60)
    print("Ensemble Kalman Filter (EnKF) Example")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    n_steps = 100
    n_nodes = 3

    precip, pet, true_states, observations = generate_test_data(n_steps)

    # Create model ensemble
    models = []
    for i in range(10):  # 10 ensemble members
        nodes = [ha.GridNode(node_id=j, x=j, y=0, area=100) for j in range(n_nodes)]
        model = ha.XinAnjiangModel()
        # Vary initial conditions slightly
        params = ha.XinAnjiangParameters()
        params.WM = 100 + np.random.uniform(-10, 10)
        model.initialize(nodes, params)
        models.append(model)

    print(f"\nCreated {len(models)} ensemble members")

    # Initialize EnKF
    state_dim = n_nodes
    obs_dim = 1
    enkf = EnsembleKalmanFilter(state_dim, obs_dim, n_ensemble=50)

    # Initialize ensemble around prior
    initial_state = np.array([80.0, 70.0, 60.0])  # Initial guess
    enkf.initialize_ensemble(initial_state, state_std=10.0)

    # Run assimilation
    print(f"\nRunning EnKF for {n_steps} steps...")
    print("(Assimilating observations every 10 steps)")

    updated_states = []
    predicted_states = []

    for t in range(n_steps):
        # Run each model
        for model in models:
            RS, RI, RG, states = model.water_balance(
                1.0,
                np.array([precip[t]] * n_nodes),
                np.array([pet[t]] * n_nodes)
            )

        # Get mean state from ensemble
        ensemble_states = np.array([m._W for m in models])
        mean_state = np.mean(ensemble_states, axis=0)

        predicted_states.append(mean_state.copy())

        # Assimilate every 10 steps
        if t > 0 and t % 10 == 0:
            obs = observations[t]
            result = enkf.assimilate(
                state=mean_state,
                observation=np.array([obs]),
                obs_error_std=1.0,
                model_error_std=0.1
            )

            # Update model states
            for model in models:
                model._W = result.state + np.random.normal(0, 2, n_nodes)

            updated_states.append((t, result.state.copy()))
            print(f"  Step {t}: assimilated obs={obs:.2f}, updated state={result.state[0]:.2f}")
        else:
            updated_states.append((t, mean_state.copy()))

    print("\nEnKF assimilation completed")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Precipitation
    axes[0].bar(range(n_steps), precip, alpha=0.7, color='blue')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].set_title('Input Precipitation')
    axes[0].set_ylim(0, 25)

    # States
    pred_array = np.array([s for _, s in predicted_states])
    axes[1].plot(range(n_steps), pred_array[:, 0], 'b-', label='Predicted (mean)', alpha=0.7)

    for t, state in updated_states:
        if t % 10 == 0 and t > 0:
            axes[1].axvline(x=t, color='red', linestyle='--', alpha=0.5)

    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('State Variable (mm)')
    axes[1].set_title('EnKF State Updates (first node)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('enkf_results.png', dpi=150)
    print("\nFigure saved as 'enkf_results.png'")


def run_ensemble_forecast_example():
    """Run multi-model ensemble forecast example."""
    print("\n" + "=" * 60)
    print("Multi-Model Ensemble Forecast Example")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    n_steps = 50
    n_nodes = 3

    precip, pet, true_states, observations = generate_test_data(n_steps)

    # Create different model types
    print("\nCreating structural ensemble:")
    xinanjiang = ha.XinAnjiangModel()
    xinanjiang.initialize(
        [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(n_nodes)],
        ha.XinAnjiangParameters()
    )
    print(f"  - XinAnjiang")

    hymod = ha.HyMODModel()
    hymod.initialize(
        [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(n_nodes)],
        ha.HyMODParameters()
    )
    print(f"  - HyMOD")

    sac = ha.SACModel()
    sac.initialize(
        [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(n_nodes)],
        ha.SACParameters()
    )
    print(f"  - SAC-SMA")

    crest = ha.CRESTModel()
    crest.initialize(
        [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(n_nodes)],
        ha.CRESTParameters()
    )
    print(f"  - CREST")

    # Create ensemble
    ensemble = ha.create_structural_ensemble([xinanjiang, hymod, sac, crest])
    print(f"\nCreated ensemble with {len(ensemble.members)} models")

    # Run ensemble
    print("\nRunning ensemble simulation...")

    member_outputs = {}
    for name, model in zip(['xinanjiang', 'hymod', 'sac', 'crest'],
                           [xinanjiang, hymod, sac, crest]):
        outputs = []
        for t in range(n_steps):
            RS, RI, RG, _ = model.water_balance(
                1.0,
                np.array([precip[t]] * n_nodes),
                np.array([pet[t]] * n_nodes)
            )
            outputs.append(RS.sum() + RI.sum() + RG.sum())
        member_outputs[name] = np.array(outputs)

    # Compute ensemble statistics
    outputs_array = np.array(list(member_outputs.values()))
    ensemble_mean = np.mean(outputs_array, axis=0)
    ensemble_std = np.std(outputs_array, axis=0)
    q05 = np.percentile(outputs_array, 5, axis=0)
    q95 = np.percentile(outputs_array, 95, axis=0)

    print("\nEnsemble Statistics (first 5 steps):")
    print("-" * 50)
    print(f"{'Step':<6} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
    for i in range(5):
        print(f"{i:<6} {ensemble_mean[i]:<10.3f} {ensemble_std[i]:<10.3f} [{q05[i]:.2f}, {q95[i]:.2f}]")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Precipitation
    axes[0].bar(range(n_steps), precip, alpha=0.7, color='blue')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].set_title('Input Precipitation')

    # Ensemble forecast
    axes[1].fill_between(range(n_steps), q05, q95, alpha=0.3, color='blue', label='90% CI')
    axes[1].plot(range(n_steps), ensemble_mean, 'b-', linewidth=2, label='Ensemble Mean')

    for name, output in member_outputs.items():
        axes[1].plot(range(n_steps), output, '--', alpha=0.5, label=name)

    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Runoff (mm)')
    axes[1].set_title('Multi-Model Ensemble Forecast')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('ensemble_forecast.png', dpi=150)
    print("\nFigure saved as 'ensemble_forecast.png'")


def run_parameter_ensemble_example():
    """Run parameter ensemble for uncertainty analysis."""
    print("\n" + "=" * 60)
    print("Parameter Ensemble Example")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    n_steps = 50
    n_nodes = 3

    precip, pet, true_states, observations = generate_test_data(n_steps)

    # Create base model
    nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(n_nodes)]
    base_model = ha.XinAnjiangModel()
    base_model.initialize(nodes, ha.XinAnjiangParameters())

    # Generate parameter samples using LHS
    param_bounds = {
        'WM': (80, 150),
        'K': (0.3, 0.7),
        'B': (0.05, 0.25),
        'SM': (10, 25),
        'KI': (0.2, 0.6),
        'KG': (0.02, 0.1),
    }

    print("\nGenerating 20 parameter samples using LHS...")
    param_samples = ParameterEnsemble.generate_lhs_samples(param_bounds, n_samples=20)

    # Create parameter ensemble
    param_ens = ParameterEnsemble(base_model, param_samples)
    print(f"Created parameter ensemble with {param_ens.n_samples} samples")

    # Run simulation
    print("\nRunning parameter ensemble...")
    result = param_ens.run({'precip': precip, 'pet': pet})

    print("\nParameter Ensemble Statistics (first 5 steps):")
    print("-" * 50)
    print(f"{'Step':<6} {'Mean':<10} {'Std':<10} {'Range':<20}")
    for i in range(5):
        range_str = f"[{result.quantiles['q05'][i]:.2f}, {result.quantiles['q95'][i]:.2f}]"
        print(f"{i:<6} {result.mean[i]:<10.3f} {result.std[i]:<10.3f} {range_str}")

    print(f"\nSpread (mean std across all steps): {np.mean(result.std):.3f} mm")


if __name__ == "__main__":
    run_enkf_example()
    run_ensemble_forecast_example()
    run_parameter_ensemble_example()

    print("\n" + "=" * 60)
    print("Data Assimilation & Ensemble Example Completed!")
    print("=" * 60)
