"""
新安江模型案例 · 参数率定
========================
单阶段 DE 全局率定 + 多起点重启 + L-BFGS-B 精修

关键改进 (相对于旧版两阶段方案):
- 单阶段 DE: 一次性优化所有参数, 避免两阶段策略锁定局部最优
- DE 种群规模: popsize ≥ 5 × n_params (旧版 popsize=3/6 完全无法全局搜索)
- 多起点重启: 5 个不同 seed 各跑一次 DE, 取最优, 排除单次偶然性
- Sobol 初始化: 比均匀随机覆盖度更好
- L-BFGS-B 精修: DE 收尾后用局部优化精修
- 日模型目标函数: 同时考虑 NSE + 水量误差 + 峰差 + R²
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

import HydroArray as ha

from xaj_config import (
    STEP_H_DAILY, STEP_H_EVENT, AREA_KM2,
    WARMUP_END, CALIB_START, CALIB_END,
    PARAM_BOUNDS, DE_CONFIG_DAILY, DE_CONFIG_EVENT,
    PERMIT_EVENT_VOL_PCT, PERMIT_EVENT_PEAK_PCT, PERMIT_EVENT_PEAK_LAG_H,
    KE_DAILY, KE_EVENT,
)
from xaj_model import XAJRunner, apply_param_dict, clone_params
from xaj_metrics import daily_metrics, evaluate_event

# ─────────────────── 率定历史记录 (供收敛曲线绘图) ───────────────────
# 全局字典, key 为 "daily" / "event", value 是 list[dict]
# 每个 dict 至少含:
#   eval:       第几次目标函数评估 (单调递增)
#   score:      de_obj 返回的目标函数值
#   nse / pbias / rmse / r2 / vol_err / peak_err (子集)
# DE 每代 callback 时再补一条 "gen" 级别的记录 (gen, best_score)
CALIB_HISTORY: Dict[str, list] = {"daily": [], "event": []}
CALIB_GEN_HISTORY: Dict[str, list] = {"daily": [], "event": []}


def reset_calibration_history(label: str = "all"):
    """清空率定历史 (避免多次重跑时混淆)。"""
    if label == "all":
        for k in CALIB_HISTORY:
            CALIB_HISTORY[k].clear()
            CALIB_GEN_HISTORY[k].clear()
    else:
        CALIB_HISTORY[label].clear()
        CALIB_GEN_HISTORY[label].clear()


def _slice_calib(q_obs_mm: pd.Series, precip: pd.Series, pet: pd.Series):
    """切出率定期 (排除 warm-up)。"""
    mask = (q_obs_mm.index > pd.Timestamp(WARMUP_END)) & \
           (q_obs_mm.index <= pd.Timestamp(CALIB_END))
    return precip[mask].values, pet[mask].values, q_obs_mm[mask].values


def _eval_on_period(params: ha.XinAnjiangParameters,
                    precip: np.ndarray, pet: np.ndarray,
                    q_obs: np.ndarray, step_hours: float) -> dict:
    """在指定时段上运行模型并返回指标。"""
    # 物理一致性: Muskingum K 应与时间步长同量级 (KE≈dt)
    # 日模型 dt=24h → KE=24h; 次洪 dt=1h → KE=1h
    params.KE = KE_DAILY if step_hours >= 12.0 else KE_EVENT
    runner = XAJRunner(params, AREA_KM2)
    q_sim = runner.run(precip, pet, step_hours)
    return daily_metrics(q_sim, q_obs)


# ─────────────────── DE 多起点重启 ───────────────────

def _multi_seed_de(obj_func, bounds: List[Tuple[float, float]], x0: np.ndarray,
                   cfg: dict, label: str = "",
                   record_callback: bool = False) -> Tuple[np.ndarray, float, int]:
    """在多个不同 seed 下运行 DE, 取其中最优解。

    这是关键改进: 旧版只跑一次 DE (固定 seed=42), 极可能陷入局部最优。
    多起点重启是验证全局最优的标准做法。

    若 record_callback=True, DE 每代 callback 时根据当前 CALIB_HISTORY[label]
    的运行最小值 (best score so far) 追加一条 (gen, best_score) 到
    CALIB_GEN_HISTORY[label], 跨 seed 串接 (gen 号连续累加)。
    """
    best_x = None
    best_fun = np.inf
    best_n = 0
    seeds = list(range(42, 42 + cfg["n_seeds"]))

    for seed in seeds:
        # DE 自带 callback: 每代调用一次, 接收 (xk, convergence)
        de_callback = None
        if record_callback:
            def de_callback(_xk, convergence, _label=label):
                """每代结束时记录一次 best_score so far。
                注意: DE 在 'deferred' 模式下, xk 是当前种群,
                convergence 是种群适应度方差; 我们用历史最小值
                (即所有评估过的目标函数最小值) 作为 best_score。"""
                if CALIB_HISTORY[_label]:
                    cur_best = min(h["score"] for h in CALIB_HISTORY[_label])
                else:
                    cur_best = float("inf")
                CALIB_GEN_HISTORY[_label].append({
                    "gen": len(CALIB_GEN_HISTORY[_label]),
                    "best_score": float(cur_best),
                    "convergence": float(convergence) if convergence is not None else None,
                })
                return False

        res = differential_evolution(
            obj_func, bounds=bounds,
            x0=x0,
            maxiter=cfg["maxiter"],
            popsize=cfg["popsize"],
            seed=seed,
            tol=cfg["tol"],
            polish=cfg["polish"],
            init=cfg["init"],
            mutation=(0.5, 1.5),
            recombination=0.9,
            updating="deferred",  # 批量评估, 显著加速
            callback=de_callback,
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
            best_n = int(res.nfev)
        if label:
            print(f"    [{label}] seed={seed}: fun={res.fun:.5f} nfev={res.nfev}")

    return best_x, best_fun, best_n


# (不再用 _DE_CURRENT_BEST, 改在 de_callback 里直接从 CALIB_HISTORY 算)
_DE_CURRENT_BEST: float = np.inf


def _get_de_current_best() -> float:
    return _DE_CURRENT_BEST


def _final_lbfgsb_refine(obj_func, x0: np.ndarray,
                         bounds: List[Tuple[float, float]],
                         maxiter: int = 100) -> Tuple[np.ndarray, float]:
    """在 DE 收尾后, 再用 L-BFGS-B 做一次局部精修。"""
    res = minimize(
        obj_func, x0, method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "maxfun": maxiter * 3,
                 "ftol": 1e-10, "gtol": 1e-8},
    )
    return res.x, float(res.fun)


# ═══════════════════════════════════════════════════════════════
# 日模型率定
# ═══════════════════════════════════════════════════════════════

def calibrate_daily(precip: pd.Series, pet: pd.Series,
                    q_obs_mm: pd.Series,
                    ref: dict) -> Tuple[ha.XinAnjiangParameters, dict, np.ndarray]:
    """日模型单阶段 DE 全局率定。

    率定 6 个参数 (K, SM, KI, KG, CI, CG)。
    B/WUM/WLM/C/IMP/EX 取 params_ref 固定值, 不参与率定。
    多目标函数同时考虑 NSE + 水量误差 + 峰差, 避免单目标优化带来的偏差。

    改进:
      - 单阶段 DE (无两阶段锁定问题)
      - popsize=30 (6 参数 × 5, 标准推荐)
      - 5 个不同 seed 重启取最优
      - 最后 L-BFGS-B 精修
      - 目标函数: -NSE + 0.15*|ΔV%|/100 + 0.10*|ΔPeak%|/100

    仅在率定期 (1990-1994) 上优化, warm-up 1989 不参与。

    Returns:
        (最优参数, 整期指标, 整期 q_sim)
    """
    print("=" * 60)
    print("日模型 (24h) 单阶段 DE 全局率定 [率定期 1990-1994]")
    print("=" * 60)

    from xaj_config import build_initial_params
    p0 = build_initial_params(ref)

    # 率定期数据
    p_cal, e_cal, q_cal = _slice_calib(q_obs_mm, precip, pet)
    print(f"  率定期: {len(q_cal)} 天, "
          f"平均雨量 {p_cal.mean():.1f} mm/d, 平均径流 {q_cal.mean():.1f} mm/d")

    # 6 个率定参数: K (蒸发折算) + SM/KI/KG (分水源) + CI/CG (退水)
    # B/WUM/WLM/C/IMP/EX 取固定值
    s_keys = ["K", "SM", "KI", "KG", "CI", "CG"]
    s_bounds_list = [PARAM_BOUNDS[k] for k in s_keys]
    print(f"  率定参数: {len(s_keys)} 个")
    print(f"  DE 配置: popsize={DE_CONFIG_DAILY['popsize']}, "
          f"maxiter={DE_CONFIG_DAILY['maxiter']}, "
          f"n_seeds={DE_CONFIG_DAILY['n_seeds']}")

    def de_obj(x):
        """日模型目标函数。同步记录历史到 CALIB_HISTORY['daily']。"""
        p_dict = {k: float(x[i]) for i, k in enumerate(s_keys)}
        apply_param_dict(p0, p_dict)
        m = _eval_on_period(p0, p_cal, e_cal, q_cal, STEP_H_DAILY)
        # 多目标: NSE 为主 (越大越好, 取负数), 水量 + 峰差作为惩罚
        nse = m["nse"]
        vol_err = abs(m["volume_err_pct"])
        peak_err = abs(m["peak_err_pct"])
        # R² 也加入, 鼓励过程线相关性
        r2 = m.get("r_squared", 0.0)
        score = (-nse
                 + 0.15 * vol_err / 100.0
                 + 0.10 * peak_err / 100.0
                 + 0.05 * max(0.0, 1.0 - r2))

        # ── 记录到历史 (供收敛曲线绘图) ──
        CALIB_HISTORY["daily"].append({
            "eval": len(CALIB_HISTORY["daily"]),
            "score": float(score),
            "nse": float(nse),
            "pbias": float(m.get("pbias", vol_err * (1 if score > 0 else -1))),
            "vol_err": float(vol_err),
            "peak_err": float(peak_err),
            "rmse": float(m.get("rmse", 0.0)),
            "r2": float(r2),
        })
        return score

    # 清空日模型历史, 准备本次率定
    reset_calibration_history("daily")

    x0 = np.array([getattr(p0, k) for k in s_keys], dtype=float)
    # 截断 x0 到边界内
    x0 = np.array([np.clip(x0[i], s_bounds_list[i][0], s_bounds_list[i][1])
                   for i in range(len(s_keys))], dtype=float)

    # 多起点 DE 搜索
    print("\n[多起点 DE 搜索]")
    best_x, best_fun, total_n = _multi_seed_de(
        de_obj, s_bounds_list, x0, DE_CONFIG_DAILY, label="daily",
        record_callback=True)

    # 应用最优解
    apply_param_dict(p0, {k: float(best_x[i]) for i, k in enumerate(s_keys)})
    s_m = _eval_on_period(p0, p_cal, e_cal, q_cal, STEP_H_DAILY)
    print(f"\n  DE 最优: NSE={s_m['nse']:.4f}, R²={s_m['r_squared']:.4f}, "
          f"|ΔV|={abs(s_m['volume_err_pct']):.2f}%, "
          f"峰差={s_m['peak_err_pct']:.1f}%")

    # L-BFGS-B 精修
    print("\n[L-BFGS-B 局部精修]")
    final_x, final_fun = _final_lbfgsb_refine(de_obj, best_x, s_bounds_list)
    apply_param_dict(p0, {k: float(final_x[i]) for i, k in enumerate(s_keys)})
    s_m_final = _eval_on_period(p0, p_cal, e_cal, q_cal, STEP_H_DAILY)
    print(f"  精修后: NSE={s_m_final['nse']:.4f}, R²={s_m_final['r_squared']:.4f}, "
          f"|ΔV|={abs(s_m_final['volume_err_pct']):.2f}%, "
          f"峰差={s_m_final['peak_err_pct']:.1f}%")

    # 整期运行 (含 warm-up + 率定 + 验证)
    p0.KE = KE_DAILY   # 日模型步长 24h → KE=24h
    runner = XAJRunner(p0, AREA_KM2)
    q_sim_full = runner.run(precip.values, pet.values, STEP_H_DAILY)
    m_full = daily_metrics(q_sim_full, q_obs_mm.values)
    print(f"\n日模型整期 (1989-1996, 含 warm-up) 指标:")
    for k, v in m_full.items():
        print(f"  {k:18s} = {v:8.3f}")

    return p0, m_full, q_sim_full


# ═══════════════════════════════════════════════════════════════
# 次洪模型率定
# ═══════════════════════════════════════════════════════════════

def calibrate_event(hourly, events: List[dict], ref: dict,
                    base_params: ha.XinAnjiangParameters
                    ) -> Tuple[ha.XinAnjiangParameters, dict]:
    """次洪模型率定: 用前 11 场 (event 1-11, 1989-1994) 的多场洪水优化小时尺度参数。

    率定 8 个参数 (K, SM, KI, KG, CI, CG, L, xe)。
    B 固定 0.3 不参与率定 (日→小时尺度不改变产流非线性)。
    单阶段 DE 全局率定:
      - 多起点重启 (3 seeds) 取最优
      - L-BFGS-B 精修
      - 目标函数: 多场 NSE/水量/洪峰/峰时综合得分 + 差场惩罚 + 稳健性约束

    Muskingum 稳定性条件: 2*KE*xe <= dt <= 2*KE*(1-xe), dt=1h。
    对 290 km² 流域, 洪水波传播时间约 1-3 h, KE 取此范围。

    Args:
        hourly: HourlyData 数据包
        events: 场次配置列表
        ref: 参数参考表
        base_params: 日模型率定后的参数

    Returns:
        (event_params, train_meta)
    """
    print("\n" + "=" * 60)
    print("次洪模型 (1h) 单阶段 DE 全局率定 [率定期 1990-1994 多场]")
    print("=" * 60)

    from xaj_data import slice_event

    # 选取前 11 场 (event 1-11, 1989-1994) 用于率定
    train_evs = [e for e in events if int(e["idx"]) <= 11]
    if not train_evs:
        train_evs = events[:1]

    print(f"  率定场次: {len(train_evs)} 场")
    for ev in train_evs:
        print(f"    event {ev['idx']}  {ev['t_start']} ~ {ev['t_end']}")

    train_events = [int(ev["idx"]) for ev in train_evs]

    p1 = clone_params(base_params)

    def _apply(p_dict):
        apply_param_dict(p1, p_dict)

    def _eval_single(ev):
        """评估单场洪水, 返回 (metrics, p_in, e_in, qo_cms, q_sim_mm, t_idx, warm)。"""
        p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
            hourly, ev["t_start"], ev["t_end"], warmup_h=48)
        # 物理一致性: 次洪 dt=1h, KE=1h
        p1.KE = KE_EVENT
        # 使用简单的 _initialize_discharge (只设退水水库和河道),
        # 不反推土壤含水量 (避免引入新误差)
        runner = XAJRunner(p1, AREA_KM2, apply_lag=True)
        q_sim_mm = runner.run(
            p_in, e_in, STEP_H_EVENT,
            initial_outflow_mm=qo_mm[0],
            init_soil=False,
        )
        m = evaluate_event(qo_cms, q_sim_mm, warm)
        return m, p_in, e_in, qo_cms, q_sim_mm, t_idx, warm

    def _eval_all():
        """评估所有率定场次, 返回 (avg_metrics, per_event_list)。"""
        ms = []
        for ev in train_evs:
            m, *_ = _eval_single(ev)
            ms.append({**m, "event": int(ev["idx"])})
        avg = {
            "runoff_rel_err_pct": float(np.mean([m["runoff_rel_err_pct"] for m in ms])),
            "peak_rel_err_pct": float(np.mean([m["peak_rel_err_pct"] for m in ms])),
            "peak_lag_h": float(np.mean([m["peak_lag_h"] for m in ms])),
            "nse": float(np.mean([m["nse"] for m in ms])),
        }
        return avg, ms

    # 8 个率定变量: K (蒸发折算) + SM/KI/KG (分水源) + CI/CG (退水)
    #                + L (滞时) + xe (Muskingum 权重)
    # B 固定 0.3, KE 固定为步长 1h, 均不参与率定
    s_keys = ["K", "SM", "KI", "KG", "CI", "CG", "L", "xe"]
    s_bounds_list = [PARAM_BOUNDS[k] for k in s_keys]
    print(f"  率定参数: {len(s_keys)} 个")
    print(f"  DE 配置: popsize={DE_CONFIG_EVENT['popsize']}, "
          f"maxiter={DE_CONFIG_EVENT['maxiter']}, "
          f"n_seeds={DE_CONFIG_EVENT['n_seeds']}")

    cur = {k: getattr(p1, k) for k in s_keys}

    def de_obj(x):
        """次洪目标函数 (SL 250 硬约束 + min-max). 同步记录历史到 CALIB_HISTORY['event']。

        关键设计:
          - 硬约束: 任何一场超过 SL 250 许可, 施加重罚 (不允许超过)
          - min-max: 在满足硬约束的前提下, 最小化"最差场次"的 NSE 违反量
          - 不再用平均指标, 因为平均会掩盖差场
        """
        p_dict = {k: float(x[i]) for i, k in enumerate(s_keys)}
        _apply(p_dict)

        # Muskingum 稳定性硬约束: 2*KE*xe <= dt <= 2*KE*(1-xe)
        ke_v = float(p1.KE)
        xe_v = p_dict["xe"]
        dt = STEP_H_EVENT
        lo_stab = 2.0 * ke_v * xe_v
        hi_stab = 2.0 * ke_v * (1.0 - xe_v)
        stab_penalty = 0.0
        if lo_stab > dt:
            stab_penalty += (lo_stab - dt) * 10.0
        if hi_stab < dt:
            stab_penalty += (dt - hi_stab) * 10.0
        if (p_dict["KI"] + p_dict["KG"]) > 0.7:
            stab_penalty += ((p_dict["KI"] + p_dict["KG"]) - 0.7) * 50.0

        _, ms = _eval_all()

        # ── SL 250 硬约束 (关键!) ──
        # 每场都必须在许可范围内; 越界施加重罚
        sl_violation = 0.0
        for m in ms:
            vol_v = max(0.0, abs(m["runoff_rel_err_pct"]) - PERMIT_EVENT_VOL_PCT)
            peak_v = max(0.0, abs(m["peak_rel_err_pct"]) - PERMIT_EVENT_PEAK_PCT)
            lag_v = max(0.0, abs(m["peak_lag_h"]) - PERMIT_EVENT_PEAK_LAG_H)
            # 越界惩罚: 每超 1% 加 0.5 (远大于其他项)
            sl_violation += vol_v * 0.5 + peak_v * 0.5 + lag_v * 1.0
            if m["nse"] < 0.0:
                sl_violation += abs(m["nse"]) * 2.0

        # ── 主目标: min-max (最差场次优先) ──
        # 在满足硬约束前提下, 优化"最差场次 NSE"
        worst_nse = float(np.min([m["nse"] for m in ms]))
        mean_nse = float(np.mean([m["nse"] for m in ms]))
        mean_vol = float(np.mean([abs(m["runoff_rel_err_pct"]) for m in ms]))
        mean_peak = float(np.mean([abs(m["peak_rel_err_pct"]) for m in ms]))
        mean_lag = float(np.mean([abs(m["peak_lag_h"]) for m in ms]))

        # 主目标 = 最差场次 NSE (鼓励最差的先达标)
        # 辅目标 = 平均 NSE + 各项许可内误差
        primary = -worst_nse          # 越大越好 (取负数越小越好)
        secondary = (-mean_nse
                     + mean_vol / 100.0 * 0.15
                     + mean_peak / 100.0 * 0.15
                     + mean_lag / 6.0 * 0.05)

        # 加权: 硬约束惩罚 >> 主目标 > 辅目标
        score = (sl_violation            # SL 250 越界惩罚 (绝对优先)
                + 5.0 * primary         # 最差场次
                + 1.0 * secondary       # 平均质量
                + stab_penalty)

        # ── 记录到历史 (供收敛曲线绘图) ──
        CALIB_HISTORY["event"].append({
            "eval": len(CALIB_HISTORY["event"]),
            "score": float(score),
            "nse": float(mean_nse),
            "worst_nse": float(worst_nse),
            "vol_err": float(mean_vol),
            "peak_err": float(mean_peak),
            "peak_lag": float(mean_lag),
            "sl_violation": float(sl_violation),
        })
        return score

    # 清空次洪历史, 准备本次率定
    reset_calibration_history("event")

    x0 = np.array([np.clip(cur[k], s_bounds_list[i][0], s_bounds_list[i][1])
                   for i, k in enumerate(s_keys)], dtype=float)

    # 多起点 DE 搜索
    print("\n[多起点 DE 搜索]")
    best_x, best_fun, total_n = _multi_seed_de(
        de_obj, s_bounds_list, x0, DE_CONFIG_EVENT, label="event",
        record_callback=True)

    _apply({k: float(best_x[i]) for i, k in enumerate(s_keys)})
    avg_m, per_ms = _eval_all()
    print(f"\n  DE 最优 ({len(train_evs)} 场平均): "
          f"NSE={avg_m['nse']:.4f}, "
          f"径流深={avg_m['runoff_rel_err_pct']:+.2f}%, "
          f"洪峰={avg_m['peak_rel_err_pct']:+.2f}%, "
          f"峰时={avg_m['peak_lag_h']:+.2f}h")

    # L-BFGS-B 精修
    print("\n[L-BFGS-B 局部精修]")
    final_x, final_fun = _final_lbfgsb_refine(de_obj, best_x, s_bounds_list)
    _apply({k: float(final_x[i]) for i, k in enumerate(s_keys)})
    avg_m, per_ms = _eval_all()
    print(f"  精修后: NSE={avg_m['nse']:.4f}, "
          f"径流深={avg_m['runoff_rel_err_pct']:+.2f}%, "
          f"洪峰={avg_m['peak_rel_err_pct']:+.2f}%, "
          f"峰时={avg_m['peak_lag_h']:+.2f}h")

    print(f"\n  关键参数: K={p1.K:.3f}, B={p1.B:.3f}, SM={p1.SM:.1f}, "
          f"KI={p1.KI:.3f}, KG={p1.KG:.3f}, "
          f"CI={p1.CI:.3f}, CG={p1.CG:.4f}, "
          f"L={p1.L:.2f}h, KE={p1.KE:.2f}, xe={p1.xe:.3f}")

    # 验证 Muskingum 稳定性
    dt = STEP_H_EVENT
    lo_stab = 2.0 * p1.KE * p1.xe
    hi_stab = 2.0 * p1.KE * (1.0 - p1.xe)
    stab_ok = lo_stab <= dt <= hi_stab
    print(f"  Muskingum 稳定性: 2*KE*xe={lo_stab:.3f} <= dt={dt} <= 2*KE*(1-xe)={hi_stab:.3f}"
          f"  → {'✓ 稳定' if stab_ok else '✗ 不稳定'}")

    # 保留第一场的切片用于绘图
    p1.KE = KE_EVENT   # 次洪模型步长 1h → KE=1h
    m0, p_in, e_in, qo_cms, q_sim_mm, t_idx, warm = _eval_single(train_evs[0])

    return p1, {
        "train_events": train_events,
        "train_metrics": avg_m,
        "train_per_event": per_ms,
        "train_slice": (p_in, e_in, qo_cms, q_sim_mm, t_idx, warm),
    }


def run_event_validation(hourly, events: List[dict],
                         params: ha.XinAnjiangParameters,
                         train_events: List[int] | None = None) -> pd.DataFrame:
    """对所有场次跑一次, 返回每场指标 DataFrame。

    Args:
        hourly: HourlyData 数据包
        events: 场次配置列表
        params: 次洪率定后的参数
    """
    from xaj_data import slice_event
    from xaj_metrics import classify_event_role

    train_events = set(train_events or [])
    rows = []
    # 次洪验证: KE=1h (与次洪步长一致)
    params.KE = KE_EVENT
    for ev in events:
        p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
            hourly, ev["t_start"], ev["t_end"], warmup_h=48)
        runner = XAJRunner(params, AREA_KM2, apply_lag=True)
        q_sim_mm = runner.run(
            p_in, e_in, STEP_H_EVENT,
            initial_outflow_mm=qo_mm[0],
            init_soil=False,
        )
        m = evaluate_event(qo_cms, q_sim_mm, warm)
        used_for_calibration = int(ev["idx"]) in train_events
        rows.append({
            "event": ev["idx"],
            "purpose": ev.get("purpose", ""),
            "t_start": ev["t_start"],
            "t_end": ev["t_end"],
            "used_for_calibration": used_for_calibration,
            "event_role": classify_event_role(
                ev["t_start"], used_for_calibration=used_for_calibration
            ),
            **m,
        })
    return pd.DataFrame(rows)
