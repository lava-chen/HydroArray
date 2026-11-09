"""
新安江模型案例 · 绘图
====================
日模型:
  - 降雨-径流过程线 (率定期/验证期分图)
  - 流量历时曲线 FDC
  - 年径流深散点图 (45° 线) [老师要求]
  - 分年误差柱状图 (单一年内 obs vs sim, 不混合量纲)

次洪模型:
  - 一场一图 (实测/计算不同线型)
  - 分面总览图
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import HydroArray as ha

from xaj_config import (
    OUT, AREA_KM2, STEP_H_EVENT, STEP_H_DAILY,
    WARMUP_END, CALIB_START, CALIB_END, VALID_START, VALID_END,
)
from xaj_data import HourlyData, slice_event
from xaj_model import XAJRunner
from xaj_calibration import CALIB_HISTORY, CALIB_GEN_HISTORY

# 中文字体
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                    "Noto Sans SC", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


# ─────────────────── 日模型绘图 ───────────────────

def plot_daily_rainfall_runoff(df: pd.DataFrame, precip: pd.Series,
                                q_obs_mm: pd.Series, q_sim_mm: np.ndarray):
    """日模型降雨-径流过程线 (率定期 + 验证期 分两子图)。"""
    sim = pd.Series(q_sim_mm, index=q_obs_mm.index)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    periods = [
        ("率定期 1990-1994", CALIB_START, CALIB_END),
        ("验证期 1995-1996", VALID_START, VALID_END),
    ]
    for ax, (name, ts, te) in zip(axes, periods):
        mask = (q_obs_mm.index >= pd.Timestamp(ts)) & \
               (q_obs_mm.index <= pd.Timestamp(te))
        t = q_obs_mm.index[mask]
        p = precip.values[mask]
        o = q_obs_mm.values[mask]
        s = sim.values[mask]

        # 双轴: 上为降雨柱 (朝下), 下为径流
        ax2 = ax.twinx()
        ax2.bar(t, p, width=1.0, color="#74a9cf", alpha=0.6, label="面雨量")
        ax2.invert_yaxis()
        ax2.set_ylabel("降雨 (mm/d)", color="#74a9cf")

        ax.plot(t, o, color="black", lw=0.8, label="实测")
        ax.plot(t, s, color="#d7301f", lw=0.8, alpha=0.85, label="计算")
        ax.set_ylabel("径流深 (mm/d)")
        ax.set_title(f"{name} 降雨-径流过程")
        ax.grid(True, alpha=0.3)
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()
    path = OUT / "daily_rainfall_runoff.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_daily_fdc(q_obs_mm: pd.Series, q_sim_mm: np.ndarray):
    """日模型流量历时曲线 (FDC)。"""
    ha.plot_flow_duration_curve(
        observed=q_obs_mm.values, simulated=q_sim_mm,
        flow_unit="mm/d", title="日模型流量历时曲线 (FDC)",
        save_path=OUT / "daily_fdc.png",
    )
    return OUT / "daily_fdc.png"


def plot_annual_scatter(q_obs_mm: pd.Series, q_sim_mm: np.ndarray):
    """年径流深散点图 (计算 vs 实测, 45° 线, 标年份)。

    老师要求: 计算径流深(年)与实测径流深(年)对比散点图, 带 45° 线。
    """
    annual_obs = q_obs_mm.resample("YE").sum()
    annual_sim = pd.Series(q_sim_mm, index=q_obs_mm.index).resample("YE").sum()
    years = annual_obs.index.year

    fig, ax = plt.subplots(figsize=(7, 6.5))
    # 颜色: warm-up 灰, 率定期蓝, 验证期红
    colors = []
    for yr in years:
        if yr <= 1989:
            colors.append("#bdbdbd")     # warm-up
        elif yr <= 1994:
            colors.append("#2c7fb8")     # 率定
        else:
            colors.append("#d7301f")     # 验证

    ax.scatter(annual_obs.values, annual_sim.values, s=80,
               c=colors, edgecolor="white", zorder=3)
    for x, y, yr in zip(annual_obs.values, annual_sim.values, years):
        ax.annotate(str(yr), (x, y), xytext=(5, 4),
                    textcoords="offset points", fontsize=9, color="#333")

    # 45° 线
    lo = float(min(annual_obs.min(), annual_sim.min()))
    hi = float(max(annual_obs.max(), annual_sim.max()))
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="black", lw=0.8, ls="--", label="45° 线 (y=x)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("实测年径流深 (mm/a)")
    ax.set_ylabel("计算年径流深 (mm/a)")
    ax.set_title("年径流深: 计算 vs 实测 (含 45° 线)")
    ax.grid(True, alpha=0.3)
    # 图例: 颜色含义
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bdbdbd",
               markersize=8, label="预热期 1989"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7fb8",
               markersize=8, label="率定期 1990-1994"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d7301f",
               markersize=8, label="验证期 1995-1996"),
        Line2D([0], [0], color="black", ls="--", lw=0.8, label="45° 线 (y=x)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9)
    plt.tight_layout()
    path = OUT / "daily_annual_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_daily_yearly_errors(q_obs_mm: pd.Series, q_sim_mm: np.ndarray):
    """分年误差柱状图: 每年实测 vs 计算径流深 (mm/a), 同一量纲可比较。

    替代旧版混合量纲的指标柱状图。
    """
    annual_obs = q_obs_mm.resample("YE").sum()
    annual_sim = pd.Series(q_sim_mm, index=q_obs_mm.index).resample("YE").sum()
    years = annual_obs.index.year.values

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    w = 0.35
    ax.bar(x - w/2, annual_obs.values, w, color="#2c7fb8", label="实测")
    ax.bar(x + w/2, annual_sim.values, w, color="#d7301f", label="计算")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("年份")
    ax.set_ylabel("年径流深 (mm/a)")
    ax.set_title("分年实测 vs 计算径流深对比")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = OUT / "daily_yearly_runoff_compare.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ─────────────────── 次洪模型绘图 ───────────────────

def plot_single_event(hourly: HourlyData, params: ha.XinAnjiangParameters,
                      t_start: str, t_end: str,
                      title_prefix: str, save_path,
                      annotate: Optional[dict] = None):
    """一场一图: 上为降雨柱 (mm/h, 朝下), 下为流量 (m³/s, 实测/计算不同线型)。"""
    p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
        hourly, t_start, t_end, warmup_h=48)
    p_plot = hourly.precip_h.reindex(t_idx).values
    runner = XAJRunner(params, AREA_KM2, apply_lag=True)
    q_sim_mm = runner.run(
        p_in, e_in, STEP_H_EVENT, initial_outflow_mm=qo_mm[0]
    )
    qs_cms = ha.mm_to_cms(q_sim_mm, area=AREA_KM2, dt_hours=STEP_H_EVENT)

    # 切掉预热段
    t_evt = t_idx[warm:]
    p_evt = p_plot[warm:]
    qo_evt = qo_cms[warm:]
    qs_evt = qs_cms[warm:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2]})
    ax1.bar(t_evt, p_evt, width=0.04, color="#74a9cf", label="面雨量")
    ax1.invert_yaxis()
    ax1.set_ylabel("降雨 (mm/h)")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title(f"{title_prefix} ({t_evt[0]:%Y-%m-%d %H} ~ {t_evt[-1]:%Y-%m-%d %H})")

    # 实测/计算不同线型
    ax2.plot(t_evt, qo_evt, color="black", lw=1.0, ls="-", label="实测")
    ax2.plot(t_evt, qs_evt, color="#d7301f", lw=1.0, ls="--", label="计算")
    ax2.set_ylabel("流量 (m³/s)")
    ax2.set_xlabel("时间")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, alpha=0.3)
    if annotate is not None:
        m = annotate
        txt = (f"洪峰相对误差: {m['peak_rel_err_pct']:+.2f}%\n"
               f"径流深相对误差: {m['runoff_rel_err_pct']:+.2f}%\n"
               f"峰时差: {m['peak_lag_h']:+.1f} h\n"
               f"NSE: {m['nse']:.3f}")
        ax2.text(0.02, 0.97, txt, transform=ax2.transAxes,
                 va="top", ha="left", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="#888", alpha=0.85))
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_event_outputs(hourly: HourlyData, params: ha.XinAnjiangParameters,
                       ev_res: pd.DataFrame):
    """次洪成果图: 每场一图 + 分面总览。"""
    role_map = {
        "预热期": ("preheat", "预热期场次"),
        "预热期(参与率定)": ("calib", "预热期率定场次"),
        "率定期(参与率定)": ("calib", "率定期场次"),
        "率定期(未参与率定)": ("calibcheck", "率定期检验场次"),
        "验证期": ("valid", "验证期场次"),
    }

    for _, r in ev_res.iterrows():
        idx = int(r["event"])
        tag, title = role_map.get(r["event_role"], ("event", "洪水场次"))
        plot_single_event(
            hourly, params, r["t_start"], r["t_end"],
            title_prefix=f"{title} #{idx}",
            save_path=OUT / f"event_{idx:02d}_{tag}.png",
            annotate=r,
        )

    # 分面对比图 (带 role 区分: 率定/校核/验证/预热)
    panel = {}
    panel_roles = {}
    role_to_tag = {
        "预热期": "preheat",
        "预热期(参与率定)": "calib",
        "率定期(参与率定)": "calib",
        "率定期(未参与率定)": "calibcheck",
        "验证期": "valid",
    }
    for _, r in ev_res.iterrows():
        idx = int(r["event"])
        p_in, e_in, qo_cms, qo_mm, t_idx, warm = slice_event(
            hourly, r["t_start"], r["t_end"], warmup_h=48)
        runner = XAJRunner(params, AREA_KM2, apply_lag=True)
        q_sim_mm = runner.run(
            p_in, e_in, STEP_H_EVENT, initial_outflow_mm=qo_mm[0]
        )
        t_evt = t_idx[warm:]
        panel[idx] = (t_evt, qo_mm[warm:], q_sim_mm[warm:])
        panel_roles[idx] = role_to_tag.get(r.get("event_role", ""), "calib")
    ha.plot_event_panel(
        panel, ncols=3, flow_unit="mm/h",
        title="次洪模型分面对比 (实测 vs 模拟, 蓝框=率定/红框=验证)",
        save_path=OUT / "event_panel.png",
        event_roles=panel_roles,
    )


# ─────────────────── 参数率定收敛曲线 ───────────────────

def _running_best(xs):
    """对一组指标值, 计算 running best (单调下降/上升)。"""
    best = []
    cur = None
    for x in xs:
        if cur is None or x < cur:
            cur = x
        best.append(cur)
    return np.array(best)


def plot_calibration_convergence(label: str = "daily",
                                 title: str = "",
                                 save_path=None):
    """绘制参数率定收敛曲线 (DE 目标函数值 + 各分项指标随评估次数的变化)。

    label: "daily" 或 "event"
    title: 图标题
    save_path: 默认 OUT / f"calibration_convergence_{label}.png"

    图内容 (2×2):
      左上: 综合目标函数 score (含 running best)
      右上: NSE 随评估次数变化 (含 running best)
      左下: 误差指标 (水量/洪峰/洪量等) 随评估次数变化
      右下: DE 每代 best_score 收敛曲线 (反映"代际收敛速度")
    """
    eval_hist = CALIB_HISTORY.get(label, [])
    gen_hist = CALIB_GEN_HISTORY.get(label, [])
    if not eval_hist:
        print(f"[plot_calibration_convergence] {label}: 无历史, 跳过。")
        return None

    df = pd.DataFrame(eval_hist)
    if not title:
        title = "日模型 DE 率定收敛曲线" if label == "daily" else "次洪模型 DE 率定收敛曲线"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # ─ 左上: 综合目标 score ─
    ax = axes[0, 0]
    ax.plot(df["eval"], df["score"], color="#cccccc", lw=0.5,
            alpha=0.7, label="每次评估")
    ax.plot(df["eval"], _running_best(df["score"].values),
            color="#d7301f", lw=1.5, label="Running best")
    ax.set_xlabel("目标函数评估次数")
    ax.set_ylabel("Score (越低越好)")
    ax.set_title(f"{title} · 综合目标收敛")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ─ 右上: NSE 随评估次数 ─
    ax = axes[0, 1]
    nse_col = "nse" if "nse" in df.columns else None
    if nse_col:
        ax.plot(df["eval"], df[nse_col], color="#cccccc", lw=0.5,
                alpha=0.7, label="每次评估")
        # running best NSE = running max
        rb = []
        cur = None
        for v in df[nse_col]:
            if cur is None or v > cur:
                cur = v
            rb.append(cur)
        ax.plot(df["eval"], rb, color="#2c7fb8", lw=1.5, label="Running best")
        ax.set_ylabel("NSE (越高越好)")
    ax.set_xlabel("目标函数评估次数")
    ax.set_title(f"{title} · NSE 收敛")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ─ 左下: 误差分项 ─
    ax = axes[1, 0]
    if label == "daily":
        ax.plot(df["eval"], df["vol_err"], color="#74a9cf",
                lw=0.8, alpha=0.7, label="|径流深相对误差| (%)")
        ax.plot(df["eval"], df["peak_err"], color="#fd8d3c",
                lw=0.8, alpha=0.7, label="|洪峰相对误差| (%)")
    else:  # event
        ax.plot(df["eval"], df["vol_err"], color="#74a9cf",
                lw=0.8, alpha=0.7, label="平均 |径流深相对误差| (%)")
        ax.plot(df["eval"], df["peak_err"], color="#fd8d3c",
                lw=0.8, alpha=0.7, label="平均 |洪峰相对误差| (%)")
        if "peak_lag" in df.columns:
            ax.plot(df["eval"], df["peak_lag"].abs(), color="#33a02c",
                    lw=0.8, alpha=0.7, label="平均 |峰时差| (h)")
    ax.set_xlabel("目标函数评估次数")
    ax.set_ylabel("误差指标")
    ax.set_title(f"{title} · 误差分项")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─ 右下: DE 每代 best_score (代际收敛) ─
    ax = axes[1, 1]
    if gen_hist:
        gd = pd.DataFrame(gen_hist)
        ax.plot(gd["gen"], gd["best_score"], color="#d7301f", lw=1.5)
        ax.set_xlabel("DE 代 (含多起点)")
        ax.set_ylabel("累计最优 Score")
        ax.set_title(f"{title} · DE 代际收敛")
    else:
        ax.text(0.5, 0.5, "无 DE 代际数据", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    if save_path is None:
        save_path = OUT / f"calibration_convergence_{label}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 收敛曲线: {save_path}")
    return save_path
