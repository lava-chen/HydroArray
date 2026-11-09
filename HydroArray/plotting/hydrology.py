"""
通用水文过程图。

所有函数只假设:
  - x: 时间轴(任意 dtype, 通常 DatetimeIndex 或数值步长)
  - obs/sim: 一维 ndarray,同长度
不耦合任何特定模型、不耦合任何特定尺度(小时/日/月都成)。
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt

from HydroArray.plotting.styles import use_hydro_style, get_colors
from HydroArray.plotting.export import save_figure


__all__ = [
    "plot_hydrograph",
    "plot_precip_hydrograph",
    "plot_event_panel",
    "plot_flow_duration_curve",
    "plot_metric_bars",
]


# ─────────────────── 双序列对比 (水文过程线最常用) ───────────────────
def plot_hydrograph(
    time_index,
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    obs_label: str = "Observed",
    sim_label: str = "Simulated",
    ylabel: str = "Discharge",
    title: str = "Hydrograph",
    annotate_peaks: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    dpi: int = 300,
    svg: bool = True,
    keep_svg: bool = True,
    show: bool = False,
):
    """单条水文过程线: obs + sim 双线,可标注峰"""
    use_hydro_style(style)
    colors = get_colors(style)
    obs = np.asarray(observed).flatten()
    sim = np.asarray(simulated).flatten()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time_index, obs, color="black", lw=1.0, label=obs_label)
    ax.plot(time_index, sim, color=colors[1], lw=1.0, alpha=0.85, label=sim_label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    if annotate_peaks:
        i_obs = int(np.argmax(obs))
        i_sim = int(np.argmax(sim))
        ax.scatter([time_index[i_obs]], [obs[i_obs]],
                   color="black", s=60, zorder=5, marker="v")
        ax.scatter([time_index[i_sim]], [sim[i_sim]],
                   color=colors[1], s=60, zorder=5, marker="^")
        ax.axvline(time_index[i_obs], color="black", ls=":", lw=0.7, alpha=0.5)
        ax.axvline(time_index[i_sim], color=colors[1], ls=":", lw=0.7, alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, dpi=dpi, svg=svg, keep_svg=keep_svg)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ─────────────────── 降雨 + 径流 双轴 ───────────────────
def plot_precip_hydrograph(
    time_index,
    precipitation: np.ndarray,
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    precip_label: str = "Precipitation",
    obs_label: str = "Observed",
    sim_label: str = "Simulated",
    precip_unit: str = "mm",
    flow_unit: str = "m³/s",
    title: str = "Rainfall-Runoff",
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    dpi: int = 300,
    svg: bool = True,
    keep_svg: bool = True,
    show: bool = False,
):
    """上: 降雨柱(朝下)  下: 径流线。任何时空尺度通用。"""
    use_hydro_style(style)
    colors = get_colors(style)
    obs = np.asarray(observed).flatten()
    sim = np.asarray(simulated).flatten()
    prc = np.asarray(precipitation).flatten()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    ax1.bar(time_index, prc, color=colors[0], width=1.0, label=precip_label)
    ax1.set_ylabel(f"{precip_label} ({precip_unit})")
    ax1.invert_yaxis()
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title(title)

    ax2.plot(time_index, obs, color="black", lw=0.9, label=obs_label)
    ax2.plot(time_index, sim, color=colors[1], lw=0.9, alpha=0.85, label=sim_label)
    ax2.set_ylabel(f"Discharge ({flow_unit})")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, dpi=dpi, svg=svg, keep_svg=keep_svg)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ─────────────────── 多事件分面 ───────────────────
def plot_event_panel(
    events: Mapping,
    *,
    ncols: int = 3,
    flow_unit: str = "m³/s",
    title: str = "Event Comparison",
    obs_label: str = "Observed",
    sim_label: str = "Simulated",
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    dpi: int = 300,
    svg: bool = True,
    keep_svg: bool = True,
    show: bool = False,
    event_roles: Optional[Mapping] = None,
    role_styles: Optional[Mapping] = None,
):
    """多事件分面图。

    Args:
        events: {key: (time_index, observed, simulated)} 的字典。
                key 通常是事件 ID / 年份,用于子图标题。
        ncols:  每行子图数
        event_roles: {key: role_str} 可选, 用于在标题上标 [Calib]/[Valid]
                     并对子图边框着色以视觉区分
        role_styles: {role_str: {"title_tag", "spine_color", "bg_color"}} 可选,
                     自定义每种 role 的样式。默认提供 4 种:
                       "calib" / "calibcheck" / "preheat" / "valid"
    """
    use_hydro_style(style)
    colors = get_colors(style)
    n = len(events)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # 默认 role 样式
    default_role_styles = {
        "preheat":     {"title_tag": " [预热]",  "spine_color": "#999999", "bg_color": "#f0f0f0"},
        "calib":       {"title_tag": " [率定]",  "spine_color": "#2c7fb8", "bg_color": "#ffffff"},
        "calibcheck":  {"title_tag": " [校核]",  "spine_color": "#7fcdbb", "bg_color": "#ffffff"},
        "valid":       {"title_tag": " [验证]",  "spine_color": "#d7301f", "bg_color": "#fff5f0"},
    }
    if role_styles:
        default_role_styles.update(role_styles)
    role_styles = default_role_styles

    for ax, (k, (t, obs, sim)) in zip(axes, events.items()):
        ax.plot(t, np.asarray(obs).flatten(), color="black", lw=0.9, label=obs_label)
        ax.plot(t, np.asarray(sim).flatten(), color=colors[1], lw=0.9, alpha=0.85, label=sim_label)
        # 子图标题加 role 标签 + 边框着色
        role = (event_roles or {}).get(k, None)
        if role and role in role_styles:
            st = role_styles[role]
            ax.set_title(f"Event {k}{st['title_tag']}")
            for spine in ax.spines.values():
                spine.set_edgecolor(st["spine_color"])
                spine.set_linewidth(1.5)
            ax.set_facecolor(st["bg_color"])
        else:
            ax.set_title(f"Event {k}")
        ax.set_ylabel(flow_unit)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        save_figure(fig, save_path, dpi=dpi, svg=svg, keep_svg=keep_svg)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ─────────────────── 流量历时曲线 ───────────────────
def plot_flow_duration_curve(
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    flow_unit: str = "m³/s",
    obs_label: str = "Observed",
    sim_label: str = "Simulated",
    title: str = "Flow Duration Curve",
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    dpi: int = 300,
    svg: bool = True,
    keep_svg: bool = True,
    show: bool = False,
):
    """FDC: 横轴超越概率,纵轴流量 (semilog-y)"""
    use_hydro_style(style)
    colors = get_colors(style)

    def _exc(q):
        q = np.sort(np.asarray(q).flatten())[::-1]
        p = np.arange(1, len(q) + 1) / (len(q) + 1) * 100
        return p, q

    p_obs, v_obs = _exc(observed)
    p_sim, v_sim = _exc(simulated)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(p_obs, v_obs, color="black", lw=1.0, label=obs_label)
    ax.semilogy(p_sim, v_sim, color=colors[1], lw=1.0, alpha=0.85, label=sim_label)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel(f"Discharge ({flow_unit})")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, dpi=dpi, svg=svg, keep_svg=keep_svg)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ─────────────────── 通用指标柱状图 ───────────────────
def plot_metric_bars(
    metrics: Mapping,
    *,
    title: str = "Metrics",
    ylabel: str = "Value",
    ylim: Optional[tuple] = None,
    threshold: Optional[Sequence[tuple]] = None,
    save_path: Optional[Union[str, Path]] = None,
    style: str = "hess",
    dpi: int = 300,
    svg: bool = True,
    keep_svg: bool = True,
    show: bool = False,
):
    """通用柱状图:展示一组指标 (key -> 数值)。

    Args:
        threshold: 可选的 [(value, label, color), ...] 水平参考线,
                   例: [(20, "20% limit", "red"), (-20, "", "red")]
                   简化形式 (value, label) 也接受,默认红色。
    """
    use_hydro_style(style)
    colors = get_colors(style)
    names = list(metrics.keys())
    values = [float(v) for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names) + 2), 4))
    bars = ax.bar(names, values, color=colors[1])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    if threshold:
        for th in threshold:
            v = th[0]
            lbl = th[1] if len(th) >= 2 else f"{v}"
            c = th[2] if len(th) >= 3 else "red"
            ax.axhline(v, color=c, ls="--", lw=0.8, label=lbl)
        ax.legend(frameon=False)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, dpi=dpi, svg=svg, keep_svg=keep_svg)
    if show:
        plt.show()
    plt.close(fig)
    return fig
