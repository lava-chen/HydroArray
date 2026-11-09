"""用上次率定的参数测试,看 NSE 是否还能回到 0.89。"""
import sys
from pathlib import Path
import numpy as np
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily
from xaj_model import XAJRunner, apply_param_dict
from xaj_config import (
    AREA_KM2, STEP_H_DAILY,
    WARMUP_END, CALIB_END,
)
from xaj_metrics import daily_metrics
import HydroArray as ha

daily = load_daily()

# 上次率定的参数
with open("results/calibrated_params.json", encoding="utf-8") as f:
    calib = json.load(f)
p_dict = calib["p_daily"]
print("使用上次率定参数:", p_dict)

p = ha.XinAnjiangParameters(**p_dict)

# 切率定期
mask = (daily.q_obs_mm.index > pd.Timestamp(WARMUP_END)) & \
       (daily.q_obs_mm.index <= pd.Timestamp(CALIB_END))
p_cal = daily.precip[mask].values
e_cal = daily.pet[mask].values
q_cal = daily.q_obs_mm[mask].values

# 跑日模型
runner = XAJRunner(p, AREA_KM2)
q_sim = runner.run(p_cal, e_cal, STEP_H_DAILY)
m = daily_metrics(q_sim, q_cal)
print(f"率定期 NSE = {m['nse']:.4f}, R² = {m['r_squared']:.4f}")
print(f"|ΔV| = {abs(m['volume_err_pct']):.2f}%, 峰差 = {m['peak_err_pct']:.1f}%")

# 整期
runner2 = XAJRunner(p, AREA_KM2)
q_sim_full = runner2.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
m_full = daily_metrics(q_sim_full, daily.q_obs_mm.values)
print(f"整期 NSE = {m_full['nse']:.4f}, R² = {m_full['r_squared']:.4f}")
print(f"|ΔV| = {abs(m_full['volume_err_pct']):.2f}%, 峰差 = {m_full['peak_err_pct']:.1f}%")
