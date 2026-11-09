"""Trace 哪一参数超出 event 边界。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily
from xaj_model import XAJRunner, apply_param_dict, clone_params
from xaj_config import (
    AREA_KM2, STEP_H_DAILY, STEP_H_EVENT,
    PARAM_DEFAULTS, EVENT_CALIB_KEYS, EVENT_CALIB_BOUND, get_calib_bounds,
    build_initial_params,
)
from xaj_calibration import calibrate_daily
import HydroArray as ha

daily = load_daily()

# 跑日模型率定
print("跑日模型率定...")
p_daily, m_daily, q_sim = calibrate_daily(
    daily.precip, daily.pet, daily.q_obs_mm, daily.ref)

print(f"日模型最优 NSE: {m_daily['nse']:.4f}")
print(f"  K={p_daily.K:.3f}, SM={p_daily.SM:.2f}, KI={p_daily.KI:.3f}, "
      f"KG={p_daily.KG:.3f}, CI={p_daily.CI:.3f}, CG={p_daily.CG:.4f}")
print(f"  L={p_daily.L}, KE={p_daily.KE}, xe={p_daily.xe}")

# 复制为 p1 (跟 calibrate_event 一样)
p1 = clone_params(p_daily)
p1.KE = 1.0
print(f"\ncloned p1 for event: KE={p1.KE}, xe={p1.xe}")

# 检查 x0 是否在边界内
print(f"\nEENT_CALIB_KEYS = {EVENT_CALIB_KEYS}")
bounds = get_calib_bounds("event")
print(f"event bounds = {bounds}")
for i, k in enumerate(EVENT_CALIB_KEYS):
    v = getattr(p1, k)
    lo, hi = bounds[i]
    ok = lo <= v <= hi
    print(f"  {k}: x0={v:.4f}, bounds=({lo}, {hi}), {'✓' if ok else '✗ OUT OF BOUNDS'}")
