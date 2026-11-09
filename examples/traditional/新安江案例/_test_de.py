"""对比新代码的 DE 是否稳定 + 与未向量化版本对比。"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily
from xaj_model import XAJRunner, apply_param_dict
from xaj_config import (
    AREA_KM2, STEP_H_DAILY,
    PARAM_BOUNDS, DE_CONFIG_DAILY, DE_CONFIG_DAILY_QUICK, QUICK_MODE,
    build_initial_params, WARMUP_END, CALIB_END,
)
from xaj_metrics import daily_metrics
import HydroArray as ha

daily = load_daily()
ref = daily.ref
p0 = build_initial_params(ref)

# 切率定期
mask = (daily.q_obs_mm.index > pd.Timestamp(WARMUP_END)) & \
       (daily.q_obs_mm.index <= pd.Timestamp(CALIB_END))
p_cal = daily.precip[mask].values
e_cal = daily.pet[mask].values
q_cal = daily.q_obs_mm[mask].values

s_keys = ["K", "SM", "KI", "KG", "CI", "CG"]
s_bounds_list = [PARAM_BOUNDS[k] for k in s_keys]

def de_obj_factory():
    def de_obj(x):
        p_dict = {k: float(x[i]) for i, k in enumerate(s_keys)}
        apply_param_dict(p0, p_dict)
        runner = XAJRunner(p0, AREA_KM2)
        q_sim = runner.run(p_cal, e_cal, STEP_H_DAILY)
        m = daily_metrics(q_sim, q_cal)
        nse = m["nse"]
        vol_err = abs(m["volume_err_pct"])
        peak_err = abs(m["peak_err_pct"])
        r2 = m.get("r_squared", 0.0)
        score = (-nse
                 + 0.15 * vol_err / 100.0
                 + 0.10 * peak_err / 100.0
                 + 0.05 * max(0.0, 1.0 - r2))
        return score
    return de_obj

x0 = np.array([getattr(p0, k) for k in s_keys], dtype=float)
x0 = np.array([np.clip(x0[i], s_bounds_list[i][0], s_bounds_list[i][1])
               for i in range(len(s_keys))], dtype=float)

# 跑两次相同的 DE
for trial in range(2):
    print(f"\n[试验 {trial+1}/2] DE popsize=15, maxiter=5, seed=42")
    de_obj = de_obj_factory()
    t0 = time.perf_counter()
    res = differential_evolution(
        de_obj, bounds=s_bounds_list, x0=x0,
        maxiter=5, popsize=15, seed=42,
        tol=1e-6, polish=False, init="sobol",
        mutation=(0.5, 1.5), recombination=0.9, updating="deferred",
    )
    elapsed = time.perf_counter() - t0
    p_dict = {k: float(res.x[i]) for i, k in enumerate(s_keys)}
    apply_param_dict(p0, p_dict)
    runner = XAJRunner(p0, AREA_KM2)
    q_sim = runner.run(p_cal, e_cal, STEP_H_DAILY)
    m = daily_metrics(q_sim, q_cal)
    print(f"  fun={res.fun:.5f}, nfev={res.nfev}, 耗时={elapsed:.1f}s")
    print(f"  率定期 NSE={m['nse']:.4f}, R²={m['r_squared']:.4f}, "
          f"|ΔV|={abs(m['volume_err_pct']):.2f}%, 峰差={m['peak_err_pct']:.1f}%")
    print(f"  最优参数: K={p0.K:.3f}, SM={p0.SM:.2f}, KI={p0.KI:.3f}, "
          f"KG={p0.KG:.3f}, CI={p0.CI:.3f}, CG={p0.CG:.4f}")
