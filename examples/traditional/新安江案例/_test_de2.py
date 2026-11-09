"""验证: 同样的 DE 跑两次, 能否找到同一个解?"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily
from xaj_model import XAJRunner, apply_param_dict
from xaj_config import (
    AREA_KM2, STEP_H_DAILY,
    PARAM_BOUNDS, WARMUP_END, CALIB_END,
    build_initial_params,
)
from xaj_metrics import daily_metrics
import HydroArray as ha

daily = load_daily()
ref = daily.ref
mask = (daily.q_obs_mm.index > pd.Timestamp(WARMUP_END)) & \
       (daily.q_obs_mm.index <= pd.Timestamp(CALIB_END))
p_cal = daily.precip[mask].values
e_cal = daily.pet[mask].values
q_cal = daily.q_obs_mm[mask].values

s_keys = ["K", "SM", "KI", "KG", "CI", "CG"]
s_bounds_list = [PARAM_BOUNDS[k] for k in s_keys]

for trial in range(2):
    print(f"\n[试验 {trial+1}/2] DE popsize=20, maxiter=8, seed=42, init=sobol")
    p0 = build_initial_params(ref)
    x0 = np.array([getattr(p0, k) for k in s_keys], dtype=float)
    x0 = np.array([np.clip(x0[i], s_bounds_list[i][0], s_bounds_list[i][1])
                   for i in range(len(s_keys))], dtype=float)

    def de_obj(x, _p0=p0):
        p_dict = {k: float(x[i]) for i, k in enumerate(s_keys)}
        apply_param_dict(_p0, p_dict)
        runner = XAJRunner(_p0, AREA_KM2)
        q_sim = runner.run(p_cal, e_cal, STEP_H_DAILY)
        m = daily_metrics(q_sim, q_cal)
        nse = m["nse"]
        vol_err = abs(m["volume_err_pct"])
        peak_err = abs(m["peak_err_pct"])
        r2 = m.get("r_squared", 0.0)
        return (-nse + 0.15 * vol_err / 100.0
                + 0.10 * peak_err / 100.0
                + 0.05 * max(0.0, 1.0 - r2))

    res = differential_evolution(
        de_obj, bounds=s_bounds_list, x0=x0,
        maxiter=8, popsize=20, seed=42,
        tol=1e-6, polish=False, init="sobol",
        mutation=(0.5, 1.5), recombination=0.9, updating="deferred",
    )
    print(f"  fun={res.fun:.5f}, nfev={res.nfev}")
    print(f"  x = {[f'{v:.4f}' for v in res.x]}")
