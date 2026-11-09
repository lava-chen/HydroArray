"""
新安江模型案例 · 误差统计与指标
================================
- 日模型: 分年统计 + 率定期/验证期/整期统计
- 次洪模型: 分场次 4 指标 + 验证汇总
- SL 250 合格率统计
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import HydroArray as ha

from xaj_config import (
    STEP_H_EVENT, AREA_KM2,
    PERMIT_DAILY_RUNOFF_PCT, PERMIT_EVENT_PEAK_PCT,
    PERMIT_EVENT_VOL_PCT, PERMIT_EVENT_PEAK_LAG_H,
    WARMUP_END, CALIB_START, CALIB_END, VALID_START, VALID_END,
)


# ─────────────────── 日模型指标 ───────────────────

def daily_metrics(q_sim: np.ndarray, q_obs: np.ndarray) -> dict:
    """日模型整期指标 (8 项)。"""
    return ha.regression_summary(q_sim, q_obs)


def daily_metrics_by_year(q_obs_mm: pd.Series, q_sim_mm: np.ndarray) -> pd.DataFrame:
    """日模型分年误差统计。

    列说明:
      - nse, rel_err_pct: 年内日值 NSE 与年径流深相对误差
      - daily_qual_pct: 日值合格率, 限实测>1mm 的天数相对误差≤20%占比
        (低流量天相对误差不稳定, 仅作参考)
      - annual_qualified: 年径流深相对误差绝对值是否 ≤ SL 250 许可 (20%)
      - obs_mm / sim_mm: 年径流深 (mm/a)

    Returns:
        DataFrame, 列: year, n, nse, rel_err_pct, daily_qual_pct,
                       annual_qualified, obs_mm, sim_mm
    """
    sim = pd.Series(q_sim_mm, index=q_obs_mm.index)
    rows = []
    for yr, g_obs in q_obs_mm.groupby(q_obs_mm.index.year):
        g_sim = sim.loc[g_obs.index]
        nse_v = ha.calculate_nse(g_obs.values, g_sim.values)
        rel_err = (g_sim.sum() - g_obs.sum()) / max(g_obs.sum(), 1e-9) * 100.0
        # 日值合格率: 仅对实测 > 1mm 的天计算, 排除低流量不稳定天
        per_day_rel = np.where(
            g_obs.values > 1.0,
            (g_sim.values - g_obs.values) / g_obs.values * 100.0,
            np.nan
        )
        valid_days = np.isfinite(per_day_rel)
        if valid_days.any():
            daily_qual_pct = float(
                np.mean(np.abs(per_day_rel[valid_days]) <= PERMIT_DAILY_RUNOFF_PCT)
                * 100.0
            )
        else:
            daily_qual_pct = float("nan")
        # 年径流深合格判断 (更稳健)
        annual_qualified = bool(abs(rel_err) <= PERMIT_DAILY_RUNOFF_PCT)
        rows.append({
            "year": int(yr),
            "n": int(len(g_obs)),
            "nse": float(nse_v),
            "rel_err_pct": float(rel_err),
            "daily_qual_pct": daily_qual_pct,
            "annual_qualified": annual_qualified,
            "obs_mm": float(g_obs.sum()),
            "sim_mm": float(g_sim.sum()),
        })
    return pd.DataFrame(rows)


def daily_metrics_by_period(q_obs_mm: pd.Series, q_sim_mm: np.ndarray) -> pd.DataFrame:
    """日模型按率定期/验证期/整期统计 (排除 warm-up)。

    列说明:
      - nse, rel_err_pct: 时段内日值 NSE 与时段累计径流深相对误差
      - daily_qual_pct: 日值合格率, 仅实测>1mm 的天
      - annual_qualified: 时段累计径流深相对误差 ≤ SL 250 许可 (20%) 时为 True
      - obs_mm / sim_mm: 时段累计径流深 (mm)
    """
    sim = pd.Series(q_sim_mm, index=q_obs_mm.index)

    # 排除 warm-up 期
    mask_eval = q_obs_mm.index > pd.Timestamp(WARMUP_END)
    obs_eval = q_obs_mm[mask_eval]
    sim_eval = sim[mask_eval]

    periods = [
        ("率定期 1990-1994", CALIB_START, CALIB_END),
        ("验证期 1995-1996", VALID_START, VALID_END),
        ("整期 1990-1996", CALIB_START, VALID_END),
    ]
    rows = []
    for name, ts, te in periods:
        mask = (obs_eval.index >= pd.Timestamp(ts)) & (obs_eval.index <= pd.Timestamp(te))
        o = obs_eval[mask].values
        s = sim_eval[mask].values
        if len(o) == 0:
            continue
        nse_v = ha.calculate_nse(o, s)
        rel_err = (s.sum() - o.sum()) / max(o.sum(), 1e-9) * 100.0
        # 日值合格率: 仅对实测 > 1mm 的天计算
        per_day_rel = np.where(o > 1.0, (s - o) / o * 100.0, np.nan)
        valid_days = np.isfinite(per_day_rel)
        if valid_days.any():
            daily_qual_pct = float(
                np.mean(np.abs(per_day_rel[valid_days]) <= PERMIT_DAILY_RUNOFF_PCT)
                * 100.0
            )
        else:
            daily_qual_pct = float("nan")
        annual_qualified = bool(abs(rel_err) <= PERMIT_DAILY_RUNOFF_PCT)
        rows.append({
            "period": name,
            "n": int(len(o)),
            "nse": float(nse_v),
            "rel_err_pct": float(rel_err),
            "daily_qual_pct": daily_qual_pct,
            "annual_qualified": annual_qualified,
            "obs_mm": float(o.sum()),
            "sim_mm": float(s.sum()),
        })
    return pd.DataFrame(rows)


def classify_event_role(t_start, used_for_calibration: bool = False) -> str:
    """Return the reporting role for one flood event."""
    ts = pd.Timestamp(t_start)
    if used_for_calibration:
        if ts <= pd.Timestamp(WARMUP_END):
            return "预热期(参与率定)"
        return "率定期(参与率定)"
    if ts <= pd.Timestamp(WARMUP_END):
        return "预热期"
    if ts <= pd.Timestamp(CALIB_END):
        return "率定期(未参与率定)"
    return "验证期"


# ─────────────────── 次洪模型指标 ───────────────────

def _is_multipeak(qo_cms: np.ndarray, min_ratio: float = 0.6) -> bool:
    """判断是否复峰: 次大峰 ≥ 主峰 × min_ratio 即视为复峰。"""
    if len(qo_cms) < 4:
        return False
    sorted_vals = np.sort(qo_cms)[::-1]
    if sorted_vals[0] <= 0:
        return False
    return sorted_vals[1] >= sorted_vals[0] * min_ratio


def evaluate_event(q_obs_cms: np.ndarray, q_sim_mm_h: np.ndarray,
                   warmup_h: int) -> dict:
    """对单场洪水计算 4 个指标:
    1. 径流深相对误差 (%)
    2. 洪峰相对误差 (%)
    3. 峰时差 (h) — 复峰洪水采用主峰附近 ±12h 窗口对齐
    4. NSE

    输入 q_obs 单位 m³/s, q_sim 单位 mm/h; warmup_h 之后为事件段。
    """
    # 切掉预热段
    qo_cms = q_obs_cms[warmup_h:]
    qs_mm = q_sim_mm_h[warmup_h:]
    qs_cms = ha.mm_to_cms(qs_mm, area=AREA_KM2, dt_hours=STEP_H_EVENT)

    # 1) 径流深相对误差: 累计 mm
    area_m2 = AREA_KM2 * 1e6
    dt_s = STEP_H_EVENT * 3600.0
    vol_obs_mm = float((qo_cms * dt_s).sum() / area_m2 * 1000.0)
    vol_sim_mm = float(qs_mm.sum())
    if vol_obs_mm > 1e-9:
        vol_rel_err = (vol_sim_mm - vol_obs_mm) / vol_obs_mm * 100.0
    else:
        vol_rel_err = float("nan")

    # 2) 洪峰相对误差 (全场最大值, 不做窗口)
    peak_obs = float(np.max(qo_cms))
    peak_sim = float(np.max(qs_cms))
    if peak_obs > 1e-9:
        peak_rel_err = (peak_sim - peak_obs) / peak_obs * 100.0
    else:
        peak_rel_err = float("nan")

    # 3) 峰时差 (h) — 复峰洪水采用实测主峰附近 ±12h 窗口匹配
    i_obs = int(np.argmax(qo_cms))
    multipeak = _is_multipeak(qo_cms)
    if multipeak:
        lo = max(0, i_obs - 12)
        hi = min(len(qs_cms), i_obs + 13)
        i_sim = lo + int(np.argmax(qs_cms[lo:hi]))
    else:
        i_sim = int(np.argmax(qs_cms))
    peak_lag_h = float(i_sim - i_obs)

    # 4) NSE
    nse_val = float(ha.calculate_nse(qo_cms, qs_cms))

    return {
        "runoff_rel_err_pct": float(vol_rel_err),
        "peak_rel_err_pct": float(peak_rel_err),
        "peak_lag_h": peak_lag_h,
        "nse": nse_val,
        "is_multipeak": bool(multipeak),
    }


def event_qualification(ev_res: pd.DataFrame) -> dict:
    """统计验证期洪水合格率 (SL 250)。"""
    valid = ev_res[ev_res["event_role"] == "验证期"]
    n = len(valid)
    if n == 0:
        return {}
    qual_vol = int((valid["runoff_rel_err_pct"].abs() <= PERMIT_EVENT_VOL_PCT).sum())
    qual_peak = int((valid["peak_rel_err_pct"].abs() <= PERMIT_EVENT_PEAK_PCT).sum())
    qual_lag = int((valid["peak_lag_h"].abs() <= PERMIT_EVENT_PEAK_LAG_H).sum())
    return {
        "valid_count": n,
        "valid_avg_nse": float(valid["nse"].mean()),
        "valid_avg_peak_err": float(valid["peak_rel_err_pct"].mean()),
        "valid_avg_vol_err": float(valid["runoff_rel_err_pct"].mean()),
        "valid_avg_lag": float(valid["peak_lag_h"].mean()),
        "qual_vol": qual_vol,
        "qual_peak": qual_peak,
        "qual_lag": qual_lag,
        "qual_vol_pct": qual_vol / n * 100.0,
        "qual_peak_pct": qual_peak / n * 100.0,
        "qual_lag_pct": qual_lag / n * 100.0,
    }


# ─────────────────── 格式化辅助 ───────────────────

def fmt_pct(x: float, n: int = 2) -> str:
    """格式化百分比, NaN → '—'。"""
    if not np.isfinite(x):
        return "—"
    return f"{x:+.{n}f}%"


def fmt_num(x: float, n: int = 3) -> str:
    """格式化数值, NaN → '—'。"""
    if not np.isfinite(x):
        return "—"
    return f"{x:.{n}f}"
