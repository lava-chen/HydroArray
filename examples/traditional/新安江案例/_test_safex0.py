"""直接调 calibrate_event 测试。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily
from xaj_model import XAJRunner, apply_param_dict, clone_params
from xaj_calibration import calibrate_daily, calibrate_event
from xaj_config import (
    AREA_KM2, STEP_H_DAILY, STEP_H_EVENT,
    PARAM_DEFAULTS, EVENT_CALIB_KEYS, EVENT_CALIB_BOUND, get_calib_bounds,
    build_initial_params, _safe_x0,
)
import HydroArray as ha

daily = load_daily()

print("跑日模型率定...")
p_daily, m_daily, q_sim = calibrate_daily(
    daily.precip, daily.pet, daily.q_obs_mm, daily.ref)

print(f"日模型最优: K={p_daily.K:.4f}, SM={p_daily.SM:.4f}, KI={p_daily.KI:.4f}")

# 复制为 p1
p1 = clone_params(p_daily)
p1.KE = 1.0

# 测试 _safe_x0
s_keys = ["K", "SM", "KI", "KG", "CI", "CG", "L", "xe"]
s_bounds_list = [EVENT_CALIB_BOUND[k] for k in s_keys]
cur = {k: getattr(p1, k) for k in s_keys}
x0 = _safe_x0(cur, s_keys, s_bounds_list)
print(f"\n_safe_x0 输出: {x0}")
for i, (val, (lo, hi)) in enumerate(zip(x0, s_bounds_list)):
    ok = lo <= val <= hi
    print(f"  {s_keys[i]}: x0={val!r} bounds=({lo!r}, {hi!r}) {'✓' if ok else '✗'}")
