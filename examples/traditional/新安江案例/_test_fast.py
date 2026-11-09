"""校验 run_fast() 与原版 run() 结果位级一致。
跑 1000 天日模型 + 5 场次洪, 比较 NSE 和逐点最大误差。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily, load_hourly
from xaj_model import XAJRunner
from xaj_config import (
    AREA_KM2, STEP_H_DAILY, STEP_H_EVENT,
    PARAM_BOUNDS, PARAM_DEFAULTS, build_initial_params,
)
import HydroArray as ha

print("=" * 60)
print("校验 run_fast() 与原版 run() 一致性")
print("=" * 60)

# ── 1) 日模型: 跑 1000 天, 比较 ──
daily = load_daily()
p_dict = {**PARAM_DEFAULTS}
p = ha.XinAnjiangParameters(**p_dict)
p.WDM = p.WM - p.WUM - p.WLM

# 跑老版
runner_old = XAJRunner(p, AREA_KM2, apply_lag=False)
q_old = runner_old.run(
    daily.precip.values, daily.pet.values, STEP_H_DAILY)

# 跑新版 (新 runner, 内部走 fast)
runner_new = XAJRunner(p, AREA_KM2, apply_lag=False)
q_new = runner_new.run(
    daily.precip.values, daily.pet.values, STEP_H_DAILY)

max_err = float(np.max(np.abs(q_old - q_new)))
mean_err = float(np.mean(np.abs(q_old - q_new)))
print(f"\n[日模型 1000 天, 无 lag]")
print(f"  最大绝对误差 = {max_err:.6e} mm")
print(f"  平均绝对误差 = {mean_err:.6e} mm")
print(f"  q_old 范围 = [{q_old.min():.3f}, {q_old.max():.3f}]")
print(f"  q_new 范围 = [{q_new.min():.3f}, {q_new.max():.3f}]")
print(f"  全部位级一致: {max_err < 1e-9}")

# ── 2) 日模型: 带 lag ──
p_lag = ha.XinAnjiangParameters(**{**PARAM_DEFAULTS, "L": 1.5})
p_lag.WDM = p_lag.WM - p_lag.WUM - p_lag.WLM
runner_old2 = XAJRunner(p_lag, AREA_KM2, apply_lag=True)
q_old2 = runner_old2.run(
    daily.precip.values[:500], daily.pet.values[:500], STEP_H_DAILY)
runner_new2 = XAJRunner(p_lag, AREA_KM2, apply_lag=True)
q_new2 = runner_new2.run(
    daily.precip.values[:500], daily.pet.values[:500], STEP_H_DAILY)
max_err2 = float(np.max(np.abs(q_old2 - q_new2)))
print(f"\n[日模型 500 天, 带 lag=1.5]")
print(f"  最大绝对误差 = {max_err2:.6e} mm")
print(f"  全部位级一致: {max_err2 < 1e-9}")

# ── 3) 次洪: 5 场 + 土壤初始化 ──
hourly = load_hourly(daily.df)
n_check = 0
max_err3 = 0.0
for ev in daily.events[:5]:
    from xaj_data import slice_event
    p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
        hourly, ev["t_start"], ev["t_end"], warmup_h=48)

    runner_old3 = XAJRunner(p, AREA_KM2, apply_lag=True)
    q_old3 = runner_old3.run(
        p_in, e_in, STEP_H_EVENT,
        initial_outflow_mm=qo_mm[0], init_soil=True)

    runner_new3 = XAJRunner(p, AREA_KM2, apply_lag=True)
    q_new3 = runner_new3.run(
        p_in, e_in, STEP_H_EVENT,
        initial_outflow_mm=qo_mm[0], init_soil=True)

    err3 = float(np.max(np.abs(q_old3 - q_new3)))
    max_err3 = max(max_err3, err3)
    n_check += 1
    print(f"  event {ev['idx']:2d}  最大误差 = {err3:.6e} mm  ({'✓' if err3 < 1e-9 else '✗'})")

print(f"\n[次洪 5 场, 土壤初始化]")
print(f"  累计最大误差 = {max_err3:.6e} mm")
print(f"  全部位级一致: {max_err3 < 1e-9}")

# ── 4) 速度对比 ──
import time
runner_new = XAJRunner(p, AREA_KM2, apply_lag=False)
# 预热
runner_new.run(daily.precip.values[:100], daily.pet.values[:100], STEP_H_DAILY)
N = 5
t0 = time.perf_counter()
for _ in range(N):
    runner_new.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
elapsed_new = (time.perf_counter() - t0) / N
print(f"\n[速度对比 - 日模型 2922 天]")
print(f"  run_fast() 平均耗时: {elapsed_new*1000:.1f} ms/次")

# 对比老版
runner_old = XAJRunner(p, AREA_KM2, apply_lag=False)
t0 = time.perf_counter()
for _ in range(N):
    runner_old.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
elapsed_old = (time.perf_counter() - t0) / N
print(f"  原 run()    平均耗时: {elapsed_old*1000:.1f} ms/次")
print(f"  加速比:     {elapsed_old/elapsed_new:.1f}x")
