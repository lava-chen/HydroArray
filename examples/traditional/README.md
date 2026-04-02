# Traditional Hydrological Models Examples

This directory contains examples demonstrating how to use the traditional hydrological models in HydroArray.

## Examples

### 1. XinAnjiang Model (`xinanjiang_example.py`)

Basic usage of the XinAnjiang three-source water balance model.

```bash
python xinanjiang_example.py
```

Topics covered:
- Creating grid nodes for distributed modeling
- Initializing XinAnjiang model with parameters
- Running water balance simulation
- Understanding runoff components (surface, interflow, baseflow)
- Model evaluation with synthetic data

### 2. Model Calibration (`calibration_example.py`)

Automatic calibration of model parameters using optimization algorithms.

```bash
python calibration_example.py
```

Topics covered:
- Defining calibration parameters with bounds
- Using Scipy optimizer (L-BFGS-B)
- Using Genetic Algorithm optimization
- Sensitivity analysis with Morris method
- Interpreting calibration results

### 3. Data Assimilation & Ensemble (`assimilation_ensemble_example.py`)

State updating with Ensemble Kalman Filter and multi-model ensembles.

```bash
python assimilation_ensemble_example.py
```

Topics covered:
- Ensemble Kalman Filter (EnKF) for state updating
- Particle Filter for non-linear assimilation
- Multi-model structural ensemble
- Parameter ensemble for uncertainty analysis
- Visualizing forecast uncertainty

## Running All Examples

```bash
cd examples/traditional
python xinanjiang_example.py
python calibration_example.py
python assimilation_ensemble_example.py
```

## Output Files

The examples generate the following output files:
- `xinanjiang_results.png` - Runoff component visualization
- `enkf_results.png` - EnKF state update visualization
- `ensemble_forecast.png` - Ensemble forecast uncertainty

## Quick Reference

### Creating a Model

```python
import HydroArray as ha

# Grid nodes
nodes = [ha.GridNode(node_id=i, x=i, y=0, area=100) for i in range(5)]

# XinAnjiang model
model = ha.XinAnjiangModel()
params = ha.XinAnjiangParameters(WM=100, K=0.5, B=0.1)
model.initialize(nodes, params)
```

### Running Simulation

```python
precip = np.array([10.0, 8.0, 5.0])
pet = np.array([2.0, 1.5, 1.0])

RS, RI, RG, states = model.water_balance(
    step_hours=1.0,
    precip=precip,
    pet=pet
)
```

### Model Evaluation

```python
metrics = ha.evaluate_model(
    observed=obs_flow,
    simulated=sim_flow,
    metrics=('nse', 'rmse', 'kge')
)
```
