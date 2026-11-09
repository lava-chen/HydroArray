"""Profile 一下 DE 目标函数各环节的耗时。"""
import sys, time
from pathlib import Path
import numpy as np
import cProfile, pstats, io

sys.path.insert(0, str(Path(__file__).parent))

from xaj_data import load_daily
from xaj_model import XAJRunner, apply_param_dict
from xaj_config import (
    AREA_KM2, STEP_H_DAILY, STEP_H_EVENT,
    PARAM_BOUNDS, PARAM_DEFAULTS, build_initial_params,
)
import HydroArray as ha

daily = load_daily()
p = ha.XinAnjiangParameters(**PARAM_DEFAULTS)
p.WDM = p.WM - p.WUM - p.WLM

# 测试 1: 单次 run 的耗时
N = 50

# a) create runner
t0 = time.perf_counter()
for _ in range(N):
    runner = XAJRunner(p, AREA_KM2)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"XAJRunner() 创建:       {elapsed:.2f} ms/次")

# b) run with new runner
t0 = time.perf_counter()
for _ in range(N):
    runner = XAJRunner(p, AREA_KM2)
    q = runner.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"XAJRunner + run (new):  {elapsed:.2f} ms/次")

# c) run with reused runner
runner = XAJRunner(p, AREA_KM2)
t0 = time.perf_counter()
for _ in range(N):
    q = runner.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"run (reuse runner):     {elapsed:.2f} ms/次")

# d) apply_param_dict
t0 = time.perf_counter()
for _ in range(N):
    apply_param_dict(p, {"K": 0.95, "SM": 25.0, "KI": 0.4, "KG": 0.1, "CI": 0.5, "CG": 0.99})
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"apply_param_dict():     {elapsed:.2f} ms/次")

# e) full de_obj 模拟
s_keys = ["K", "SM", "KI", "KG", "CI", "CG"]
x = np.array([0.95, 25.0, 0.4, 0.1, 0.5, 0.99])

def de_obj_sim(x):
    p_dict = {k: float(x[i]) for i, k in enumerate(s_keys)}
    apply_param_dict(p, p_dict)
    runner = XAJRunner(p, AREA_KM2)
    q = runner.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
    return q

t0 = time.perf_counter()
for _ in range(N):
    de_obj_sim(x)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"de_obj 模拟 (全流程):   {elapsed:.2f} ms/次")

# f) cached runner + setattr
p_alt = ha.XinAnjiangParameters(**PARAM_DEFAULTS)
p_alt.WDM = p_alt.WM - p_alt.WUM - p_alt.WLM
runner_cached = XAJRunner(p_alt, AREA_KM2)

def de_obj_cached(x):
    p_alt.K, p_alt.SM, p_alt.KI, p_alt.KG, p_alt.CI, p_alt.CG = x
    q = runner_cached.run(daily.precip.values, daily.pet.values, STEP_H_DAILY)
    return q

t0 = time.perf_counter()
for _ in range(N):
    de_obj_cached(x)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"de_obj 缓存 (setattr):  {elapsed:.2f} ms/次")

# g) 预估 DE 优化总耗时
print(f"\n预估:")
n_seeds = 1
popsize = 12
maxiter = 4
total_evals = n_seeds * popsize * (maxiter + 1) * len(s_keys)  # DE 每代每维
print(f"  日模型 DE: {n_seeds}×{popsize}×{maxiter} = {n_seeds*popsize*maxiter} evals")
print(f"    当前: {n_seeds*popsize*maxiter * elapsed/1000:.1f} s")
print(f"    缓存版: {n_seeds*popsize*maxiter * elapsed/1000:.1f} s (但 elapsed 还没换)")

# 算事件率定
n_evals_event = 12 * 9 * 4  # popsize × n_params × maxiter
print(f"  事件 DE: ~{n_evals_event} evals × 11 场 = {n_evals_event*11} 模拟")
