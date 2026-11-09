"""
新安江模型案例 · 主入口
======================
仅负责编排: 数据加载 → 日模型率定 → 次洪模型率定 → 绘图 → Markdown 报告 → Word 报告。

模块化结构:
  xaj_config.py      — 配置与常量
  xaj_data.py        — 数据加载
  xaj_model.py       — 模型运行 (三水源 + 退水 + Muskingum)
  xaj_calibration.py — 率定 (日模型分层 + 次洪)
  xaj_metrics.py     — 误差统计
  xaj_plotting.py    — 绘图
  xaj_report.py      — Markdown 报告生成
  xaj_word.py        — Word (.docx) 报告生成

入口: python main.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Windows GBK 终端兼容
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from xaj_config import OUT, AREA_KM2, STEP_H_DAILY, QUICK_MODE
from xaj_data import load_daily, load_hourly
from xaj_calibration import (
    calibrate_daily, calibrate_event, run_event_validation,
)
from xaj_metrics import (
    daily_metrics_by_year, daily_metrics_by_period,
    event_qualification,
)
from xaj_plotting import (
    plot_daily_rainfall_runoff, plot_daily_fdc,
    plot_annual_scatter, plot_daily_yearly_errors,
    plot_event_outputs, plot_calibration_convergence,
)
from xaj_report import write_report
from xaj_word import write_word_report


def main():
    # ── 1) 加载日资料 ──
    print("=" * 60)
    print("新安江模型案例 · 呈村流域 (290 km²)")
    print("=" * 60)
    daily = load_daily()
    print(f"加载日资料: {len(daily.df)} 天, "
          f"{daily.precip.mean():.1f} mm/d 平均雨量, "
          f"{daily.q_obs_mm.mean():.1f} mm/d 平均径流")

    # ── 2) 日模型单阶段 DE 全局率定 (率定期 1990-1994) ──
    p_daily, daily_m, q_sim_full = calibrate_daily(
        daily.precip_subbasin, daily.pet, daily.q_obs_mm, daily.ref
    )

    # ── 2.5) 日模型率定收敛曲线 ──
    print("\n绘制日模型率定收敛曲线...")
    plot_calibration_convergence(label="daily")

    # ── 3) 日模型误差统计 ──
    df_yearly = daily_metrics_by_year(daily.q_obs_mm, q_sim_full)
    df_period = daily_metrics_by_period(daily.q_obs_mm, q_sim_full)
    print("\n日模型率定期/验证期统计:")
    print(df_period.to_string(index=False))

    # ── 4) 日模型绘图 ──
    print("\n绘制日模型图表...")
    plot_daily_rainfall_runoff(daily.df, daily.precip,
                                daily.q_obs_mm, q_sim_full)
    plot_daily_fdc(daily.q_obs_mm, q_sim_full)
    plot_annual_scatter(daily.q_obs_mm, q_sim_full)
    plot_daily_yearly_errors(daily.q_obs_mm, q_sim_full)

    # ── 5) 加载小时资料 + 次洪率定 ──
    hourly = load_hourly(daily.df)
    print(f"\n加载小时资料: {len(hourly.df)} h, "
          f"{hourly.precip_h.mean():.2f} mm/h 平均雨量")

    p_event, train_meta = calibrate_event(
        hourly, daily.events, daily.ref, p_daily
    )
    train_events = train_meta["train_events"]

    # ── 5.5) 次洪模型率定收敛曲线 ──
    print("\n绘制次洪模型率定收敛曲线...")
    plot_calibration_convergence(label="event")

    # ── 6) 次洪验证 ──
    ev_res = run_event_validation(
        hourly, daily.events, p_event, train_events=train_events
    )
    ev_res.to_csv(OUT / "event_metrics.csv", index=False, encoding="utf-8-sig")
    df_yearly.to_csv(OUT / "daily_metrics_by_year.csv",
                     index=False, encoding="utf-8-sig")
    df_period.to_csv(OUT / "daily_metrics_by_period.csv",
                     index=False, encoding="utf-8-sig")
    ev_summary = event_qualification(ev_res)
    print(f"\n次洪验证汇总: 平均 NSE={ev_summary['valid_avg_nse']:.3f}, "
          f"洪量合格 {ev_summary['qual_vol']}/{ev_summary['valid_count']}, "
          f"洪峰合格 {ev_summary['qual_peak']}/{ev_summary['valid_count']}")

    # ── 7) 次洪绘图 ──
    print("\n绘制次洪模型图表...")
    plot_event_outputs(hourly, p_event, ev_res)

    # ── 8) 生成 Markdown 报告 ──
    print("\n生成技术报告 (Markdown)...")
    report_path = write_report(
        daily_m, p_daily, p_event, daily.ref,
        df_yearly, df_period, ev_res, train_events, ev_summary
    )
    print(f"  Markdown: {report_path}")

    # ── 8.5) 把率定参数写到 JSON (给 regen_report.py 用, 在写 Word 之前避免文件锁) ──
    import json
    calib_dict = {
        "p_daily": {k: float(getattr(p_daily, k))
                    for k in ["B", "C", "IMP", "EX", "CS",
                              "WM", "WUM", "WLM", "WDM",
                              "K", "SM", "KI", "KG", "CI", "CG",
                              "L", "KE", "xe"]},
        "p_event": {k: float(getattr(p_event, k))
                    for k in ["B", "C", "IMP", "EX", "CS",
                              "WM", "WUM", "WLM", "WDM",
                              "K", "SM", "KI", "KG", "CI", "CG",
                              "L", "KE", "xe"]},
    }
    json_path = OUT / "calibrated_params.json"
    json_path.write_text(json.dumps(calib_dict, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"  Params:   {json_path}")

    # ── 9) 生成 Word 报告 (写到 v3 避免被锁住) ──
    print("\n生成技术报告 (Word)...")
    word_path = write_word_report(
        daily_m, p_daily, p_event, daily.ref,
        df_yearly, df_period, ev_res, train_events, ev_summary,
        out_path=OUT / "新安江模型技术报告_v3.docx",
    )
    print(f"  Word:     {word_path}")

    print(f"\n图表目录:   {OUT}")
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)

    return daily_m, ev_res, report_path, word_path


if __name__ == "__main__":
    main()
