"""
XinAnjiang Model Example

Demonstrates how to use the XinAnjiang three-source water balance model
for rainfall-runoff simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

import HydroArray as ha
from HydroArray.config import ModelConfig, ModelType, RoutingType


def create_sample_data(n_steps=100, n_nodes=5):
    """Create sample precipitation and evaporation data.

    Args:
        n_steps: Number of time steps.
        n_nodes: Number of grid nodes.

    Returns:
        precip: Precipitation array (n_steps, n_nodes)
        pet: Potential evapotranspiration array (n_steps, n_nodes)
    """
    np.random.seed(42)

    # Create seasonal precipitation pattern
    t = np.arange(n_steps)
    precip_pattern = 10 + 8 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n_steps)
    precip_pattern = np.maximum(precip_pattern, 0)

    # Expand to all nodes with some spatial variation
    precip = np.zeros((n_steps, n_nodes))
    for i in range(n_nodes):
        noise = np.random.normal(0, 1, n_steps)
        precip[:, i] = precip_pattern + noise * (1 + i * 0.1)
        precip[:, i] = np.maximum(precip[:, i], 0)

    # PET (potential evapotranspiration) - higher in summer
    pet_pattern = 3 + 2 * np.sin(2 * np.pi * t / 365 - np.pi/3)
    pet = np.zeros((n_steps, n_nodes))
    for i in range(n_nodes):
        pet[:, i] = pet_pattern + np.random.normal(0, 0.3, n_steps)
        pet[:, i] = np.maximum(pet[:, i], 0)

    return precip, pet


def run_xinanjiang_model():
    """Run XinAnjiang model simulation."""
    print("=" * 60)
    print("XinAnjiang Model Simulation Example")
    print("=" * 60)

    # Create grid nodes
    n_nodes = 5
    nodes = [
        ha.GridNode(
            node_id=i,
            x=float(i),
            y=0.0,
            area=100.0,  # km²
            downstream_id=i + 1 if i < n_nodes - 1 else -1
        )
        for i in range(n_nodes)
    ]

    # Mark last node as outlet
    nodes[-1].downstream_id = -1

    print(f"\nCreated {len(nodes)} grid nodes")
    print(f"Outlet node: {n_nodes - 1}")

    # Create and initialize model
    model = ha.XinAnjiangModel()

    # Model parameters (typical values for a Chinese catchment)
    params = ha.XinAnjiangParameters(
        K=0.5,      # Evapotranspiration ratio
        B=0.1,     # Distribution parameter
        IMP=0.0,   # Impermeable fraction
        WM=100.0,  # Maximum tension water storage (mm)
        WUM=20.0,  # Upper zone tension water (mm)
        WLM=80.0,  # Lower zone tension water (mm)
        WDM=0.0,   # Deep zone tension water (mm)
        SM=15.0,   # Free water storage (mm)
        KI=0.4,    # Interflow coefficient
        KG=0.05,   # Groundwater coefficient
        KKX=0.7    # Deep percolation coefficient
    )

    model.initialize(nodes, params)
    print(f"\nModel initialized with parameters:")
    print(f"  WM={params.WM}mm, K={params.K}, B={params.B}")
    print(f"  SM={params.SM}mm, KI={params.KI}, KG={params.KG}")

    # Create input data
    n_steps = 100
    precip, pet = create_sample_data(n_steps, n_nodes)

    print(f"\nRunning simulation for {n_steps} time steps...")

    # Run simulation
    total_runoff = []
    surface_runoff = []
    interflow = []
    baseflow = []

    for t in range(n_steps):
        RS, RI, RG, states = model.water_balance(
            step_hours=1.0,
            precip=precip[t],
            pet=pet[t]
        )

        total_runoff.append(RS.sum() + RI.sum() + RG.sum())
        surface_runoff.append(RS.sum())
        interflow.append(RI.sum())
        baseflow.append(RG.sum())

    total_runoff = np.array(total_runoff)
    surface_runoff = np.array(surface_runoff)
    interflow = np.array(interflow)
    baseflow = np.array(baseflow)

    # Print results
    print("\n--- Results ---")
    print(f"Total runoff volume: {total_runoff.sum():.2f} mm")
    print(f"  Surface runoff: {surface_runoff.sum():.2f} mm ({100*surface_runoff.sum()/total_runoff.sum():.1f}%)")
    print(f"  Interflow: {interflow.sum():.2f} mm ({100*interflow.sum()/total_runoff.sum():.1f}%)")
    print(f"  Baseflow: {baseflow.sum():.2f} mm ({100*baseflow.sum()/total_runoff.sum():.1f}%)")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Precipitation
    axes[0].bar(range(n_steps), precip[:, 0], alpha=0.6, label='Precipitation', color='blue')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].set_title('Input: Precipitation')
    axes[0].set_ylim(0, 25)

    # PET
    axes[1].plot(range(n_steps), pet[:, 0], label='PET', color='orange')
    axes[1].set_ylabel('PET (mm)')
    axes[1].set_title('Input: Potential Evapotranspiration')
    axes[1].legend()

    # Runoff components
    axes[2].stackplot(
        range(n_steps),
        surface_runoff, interflow, baseflow,
        labels=['Surface', 'Interflow', 'Baseflow'],
        colors=['#ff6b6b', '#4dabf7', '#69db7c']
    )
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Runoff (mm)')
    axes[2].set_title('Simulated Runoff Components')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('xinanjiang_results.png', dpi=150)
    print("\nFigure saved as 'xinanjiang_results.png'")

    return total_runoff, precip


def run_with_routing():
    """Run XinAnjiang model with Muskingum routing."""
    print("\n" + "=" * 60)
    print("XinAnjiang Model with Muskingum Routing")
    print("=" * 60)

    # Create model config with routing
    config = ModelConfig(
        model_type=ModelType.XINANJIANG,
        routing=RoutingType.MUSKINGUM,
        parameters={
            'WM': 100.0,
            'K': 0.5,
            'B': 0.1,
            'SM': 15.0,
            'KI': 0.4,
            'KG': 0.05,
            'KE': 10.0,   # Muskingum K (hours)
            'xe': 0.2,    # Muskingum x
        }
    )

    model = ha.create_model(config)
    print(f"\nCreated model: {config.model_type.value}")
    print(f"Routing: {config.routing.value}")


def evaluate_performance():
    """Evaluate model performance with synthetic observations."""
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Generate synthetic observations
    np.random.seed(42)
    precip, pet = create_sample_data(100, 5)

    # Run model
    nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(5)]
    model = ha.XinAnjiangModel()
    model.initialize(nodes, ha.XinAnjiangParameters())

    simulated = []
    for t in range(100):
        RS, RI, RG, _ = model.water_balance(1.0, precip[t], pet[t])
        simulated.append(RS.sum() + RI.sum() + RG.sum())

    simulated = np.array(simulated)

    # Add noise to create "observations"
    observations = simulated + np.random.normal(0, 0.5, 100)

    # Evaluate
    metrics = ha.evaluate_model(
        observed=observations,
        simulated=simulated,
        metrics=('nse', 'rmse', 'kge', 'pbias', 'r2')
    )

    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"NSE:   {metrics['nse']:.4f}")
    print(f"RMSE:  {metrics['rmse']:.4f} mm")
    print(f"KGE:   {metrics['kge']:.4f}")
    print(f"PBias: {metrics['pbias']:.4f}%")
    print(f"R²:    {metrics['r2']:.4f}")

    return metrics


if __name__ == "__main__":
    # Run examples
    runoff, precip = run_xinanjiang_model()
    run_with_routing()
    metrics = evaluate_performance()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
