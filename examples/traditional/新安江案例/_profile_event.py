"""事件级 profile - 看每次 event run 耗时。"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily, load_hourly, slice_event
from xaj_model import XAJRunner
from xaj_config import (
    AREA_KM2, STEP_H_DAILY, STEP_H_EVENT,
    PARAM_DEFAULTS,
)
import HydroArray as ha

daily = load_daily()
hourly = load_hourly(daily.df)
p = ha.XinAnjiangParameters(**PARAM_DEFAULTS)
p.WDM = p.WM - p.WUM - p.WLM

# 单次 event run 耗时
N = 30
for ev in daily.events[:3]:
    p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
        hourly, ev["t_start"], ev["t_end"], warmup_h=48)
    n_steps = len(p_in)
    t0 = time.perf_counter()
    for _ in range(N):
        runner = XAJRunner(p, AREA_KM2, apply_lag=True)
        q = runner.run(p_in, e_in, STEP_H_EVENT,
                       initial_outflow_mm=qo_mm[0], init_soil=True)
    elapsed = (time.perf_counter() - t0) / N * 1000
    print(f"event {ev['idx']:2d} ({n_steps:4d} 步): {elapsed:.2f} ms/次 "
          f"({elapsed/n_steps*1000:.1f} μs/步)")

# 模拟 11 场串联 (de_obj 实际流程)
N = 5
print(f"\n模拟 de_obj (11 场串联), 重复 {N} 次:")
t0 = time.perf_counter()
for _ in range(N):
    for ev in daily.events[:11]:
        p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
            hourly, ev["t_start"], ev["t_end"], warmup_h=48)
        runner = XAJRunner(p, AREA_KM2, apply_lag=True)
        q = runner.run(p_in, e_in, STEP_H_EVENT,
                       initial_outflow_mm=qo_mm[0], init_soil=True)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"  11 场 × 1 = {elapsed:.1f} ms (1 次 de_obj)")

# 预估总耗时
n_evals = 640  # DE 一次跑出 nfev=640
print(f"\n预估: 事件 DE ~{n_evals} evals × {elapsed:.0f} ms = {n_evals*elapsed/1000:.0f} s")
